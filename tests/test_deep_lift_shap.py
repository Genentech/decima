import warnings
import weakref

import pytest
import torch
from torch import nn

pytest.importorskip("tangermeme", minversion="1.6.0")

from grelu.model.heads import ConvHead
from grelu.model.layers import Attention, relative_shift
from grelu.model.trunks.borzoi import BorzoiTrunk
from grelu.transforms.prediction_transforms import Aggregate, Specificity
from tangermeme.utils import random_one_hot

from decima.interpret import _deep_lift_shap_engine as engine_module
from decima.interpret import deep_lift_shap as dls
from decima.interpret._deep_lift_shap_engine import Engine, UnsupportedArchitecture, relative_shift_transpose
from decima.interpret.attributer import get_attribution_method
from decima.interpret.deep_lift_shap import HookableAttention, deep_lift_shap, hookable_attention, task_weights

LENGTH = 128 * 24
N_TASKS = 6


class SmallDecima(nn.Module):
    """A model built as `DecimaModel` is, small enough to attribute on a CPU in float64."""

    def __init__(self, seed=0, act_func="gelu"):
        super().__init__()
        torch.manual_seed(seed)
        self.embedding = BorzoiTrunk(
            stem_channels=16,
            stem_kernel_size=15,
            init_channels=32,
            n_conv=7,
            kernel_size=5,
            channels=64,
            n_transformers=2,
            key_len=8,
            value_len=12,
            pos_dropout=0.0,
            attn_dropout=0.0,
            ff_dropout=0.2,
            n_heads=2,
            n_pos_features=8,
            crop_len=4,
            flash_attn=False,
            norm_kwargs={"eps": 1e-5},
            act_func=act_func,
        )
        self.head = ConvHead(n_tasks=N_TASKS, in_channels=80, pool_func="avg")
        conv = self.embedding.conv_tower.blocks[0].conv
        conv.weight = nn.Parameter(torch.cat([conv.weight, torch.randn(16, 1, 15) * 0.1], dim=1))

        # Untrained normalizations are close to the identity: give them statistics and weights of both signs.
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.running_mean.normal_(0, 0.3)
                module.running_var.uniform_(0.5, 2.0)
                module.weight.data.normal_(0, 1.0)
                module.bias.data.normal_(0, 0.3)
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.normal_(1, 0.3)
                module.bias.data.normal_(0, 0.3)

    def forward(self, x):
        return self.head(self.embedding(x))


class SmallLightning(nn.Module):
    """What `deep_lift_shap` reads of a Decima `LightningModel`."""

    def __init__(self, seed=0, **kwargs):
        super().__init__()
        self.model = SmallDecima(seed, **kwargs)
        self.model_params = {"n_tasks": N_TASKS}


class SmallEnsemble(nn.Module):
    def __init__(self):
        super().__init__()
        self.models = nn.ModuleList([SmallLightning(0), SmallLightning(1)])
        self.model_params = {"n_tasks": N_TASKS}


def make_inputs(n, seed=1):
    x = random_one_hot((n, 4, LENGTH), random_state=seed).double()
    mask = torch.zeros(n, 1, LENGTH, dtype=torch.float64)
    mask[:, :, 1000:1500] = 1
    return torch.cat([x, mask], dim=1)


@pytest.fixture(scope="module")
def model():
    return SmallLightning().double().eval()


def test_get_attribution_method():
    assert get_attribution_method("deepliftshap") is deep_lift_shap


def test_task_weights(model):
    y = torch.randn(3, N_TASKS, 1, dtype=torch.float64)

    weights = task_weights(model, tasks=[0, 1], off_tasks=[3, 4, 5], transform="specificity")
    expected = Specificity(on_tasks=[0, 1], off_tasks=[3, 4, 5], compare_func="subtract")(y)
    assert torch.allclose(y[:, :, 0] @ weights.double(), expected.flatten(), atol=1e-6)

    weights = task_weights(model, tasks=[0], off_tasks=None, transform="specificity")
    assert torch.allclose(weights, torch.tensor([1.0, -0.2, -0.2, -0.2, -0.2, -0.2]))

    weights = task_weights(model, tasks=[1, 2], transform="aggregate")
    expected = Aggregate(tasks=[1, 2], task_aggfunc="mean")(y)
    assert torch.allclose(y[:, :, 0] @ weights.double(), expected.flatten(), atol=1e-6)

    with pytest.raises(ValueError):
        task_weights(model, tasks=[0], off_tasks=[1], transform="aggregate")
    with pytest.raises(ValueError):
        task_weights(model, tasks=[0], transform="other")


def test_hookable_attention(model):
    inputs = make_inputs(2)
    with torch.no_grad():
        expected = model.model(inputs)
        with hookable_attention(model.model):
            assert sum(isinstance(m, HookableAttention) for m in model.model.modules()) == 2
            assert torch.equal(model.model(inputs), expected)

    assert not any(isinstance(m, HookableAttention) for m in model.model.modules())
    assert sum(isinstance(m, Attention) for m in model.model.modules()) == 2


def test_relative_shift_transpose():
    grad = torch.randn(2, 3, 5, 5, dtype=torch.float64)
    x = torch.randn(2, 3, 5, 9, dtype=torch.float64, requires_grad=True)
    (expected,) = torch.autograd.grad(relative_shift(x), x, grad)
    assert torch.equal(relative_shift_transpose(grad), expected)


def test_engine_forward(model):
    inputs = make_inputs(2)
    engine = Engine(model.model, torch.ones(N_TASKS))
    with torch.no_grad():
        assert torch.allclose(engine.forward(inputs).logits, model.model(inputs).squeeze(-1), atol=1e-12)


@pytest.mark.parametrize("output", ["logits", "expression"])
@pytest.mark.parametrize("act_func", ["gelu", "gelu_borzoi"])
def test_engine_matches_tangermeme(output, act_func):
    model = SmallDecima(act_func=act_func).double().eval()
    inputs = make_inputs(1)
    x, mask = inputs[:, :4], inputs[:, 4:]
    weights = torch.tensor([0.5, 0.5, -1 / 3, -1 / 3, -1 / 3, 0.0])
    references = dls._references(x[0], 5, 0).double()

    attributions, y, y_ref, deltas = Engine(model, weights, output=output, pairs_per_step=2).attribute(
        x[0], mask[0], references
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        expected = dls._default([model], weights, output, x, mask, references[None], 2, 1e-9, "cpu", False)[0]

    # The attributions of every pair sum to the difference between its two outputs.
    assert y_ref.shape == deltas.shape == (5,)
    assert deltas.max() < 1e-10 * (y - y_ref).abs().mean() + 1e-12
    pearson, rel_l2 = dls._agreement(attributions, expected)
    assert pearson > 1 - 1e-9
    assert rel_l2 < 1e-5
    assert not model.training


def test_engine_unsupported():
    model = SmallDecima(act_func="relu").eval()
    with pytest.raises(UnsupportedArchitecture):
        Engine(model, torch.ones(N_TASKS))
    with pytest.raises(UnsupportedArchitecture):
        Engine(SmallDecima().train(), torch.ones(N_TASKS))


@pytest.mark.parametrize("hypothetical", [False, True])
def test_deep_lift_shap(model, hypothetical):
    inputs = make_inputs(2)
    kwargs = dict(tasks=[0, 1], off_tasks=[2, 3], n_shuffles=4, random_state=0, hypothetical=hypothetical)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fast = deep_lift_shap(model, inputs, **kwargs)
        default = deep_lift_shap(model, inputs, fast=False, **kwargs)

    assert fast.shape == default.shape == (2, 4, LENGTH)
    pearson, rel_l2 = dls._agreement(fast, default)
    assert pearson > 1 - 1e-9
    assert rel_l2 < 1e-5
    if not hypothetical:
        assert torch.all(fast[inputs[:, :4] == 0] == 0)
    assert not model.training
    assert not any(isinstance(m, HookableAttention) for m in model.modules())


def test_deep_lift_shap_references(model):
    inputs = make_inputs(2)
    references = torch.stack([dls._references(x, 3, 7) for x in inputs[:, :4]])
    kwargs = dict(tasks=[0], transform="aggregate", output="expression")
    given = deep_lift_shap(model, inputs, references=references, **kwargs)
    drawn = deep_lift_shap(model, inputs, n_shuffles=3, random_state=7, **kwargs)
    assert torch.allclose(given, drawn, rtol=0, atol=1e-12)

    with pytest.raises(ValueError):
        deep_lift_shap(model, inputs, references=references[:1], **kwargs)
    with pytest.raises(ValueError):
        deep_lift_shap(model, inputs[:, :4], **kwargs)
    with pytest.raises(ValueError):
        deep_lift_shap(model, inputs, **{**kwargs, "output": "other"})


def test_deep_lift_shap_ensemble():
    ensemble = SmallEnsemble().double().eval()
    inputs = make_inputs(1)
    kwargs = dict(tasks=[0], off_tasks=[1], n_shuffles=3, random_state=0)
    attributions = deep_lift_shap(ensemble, inputs, **kwargs)
    members = [deep_lift_shap(m, inputs, **kwargs) for m in ensemble.models]
    assert torch.allclose(attributions, (members[0] + members[1]) / 2, rtol=0, atol=1e-12)
    assert not torch.allclose(members[0], members[1])


def test_deep_lift_shap_falls_back():
    model = SmallLightning(act_func="relu").double().eval()
    inputs = make_inputs(1)
    kwargs = dict(tasks=[0], off_tasks=[1], n_shuffles=3, random_state=0)
    with pytest.warns(RuntimeWarning, match="using the default path"):
        attributions = deep_lift_shap(model, inputs, **kwargs)
    assert torch.equal(attributions, deep_lift_shap(model, inputs, fast=False, **kwargs))


def test_deep_lift_shap_disagreement(model, monkeypatch):
    inputs = make_inputs(1)
    kwargs = dict(tasks=[0], off_tasks=[1], n_shuffles=3, random_state=0)
    expected = deep_lift_shap(model, inputs, fast=False, **kwargs)

    monkeypatch.setattr(dls, "_CHECKED", weakref.WeakKeyDictionary())
    attribute = Engine.attribute

    def scaled(self, *args):
        attributions, *rest = attribute(self, *args)
        return (attributions * 1.5, *rest)

    monkeypatch.setattr(Engine, "attribute", scaled)
    with pytest.warns(RuntimeWarning, match="relative L2 distance"):
        attributions = deep_lift_shap(model, inputs, **kwargs)
    assert torch.equal(attributions, expected)


def test_deep_lift_shap_unchecked(model, monkeypatch):
    inputs = make_inputs(1)
    kwargs = dict(tasks=[0], off_tasks=[1], n_shuffles=3, random_state=0)
    expected = deep_lift_shap(model, inputs, **kwargs)

    calls = []

    def out_of_memory(*args, **kwargs):
        calls.append(1)
        raise torch.cuda.OutOfMemoryError("CUDA out of memory.")

    monkeypatch.setattr(dls, "_CHECKED", weakref.WeakKeyDictionary())
    monkeypatch.setattr(dls, "_UNCHECKABLE", weakref.WeakKeyDictionary())
    monkeypatch.setattr(dls, "_default", out_of_memory)
    for _ in range(2):  # the warning repeats; the attempt does not
        with pytest.warns(RuntimeWarning, match="could not check the fast path"):
            attributions = deep_lift_shap(model, inputs, **kwargs)
        assert torch.equal(attributions, expected)
    assert len(calls) == 1


def test_deep_lift_shap_checks_once(model, monkeypatch):
    inputs = make_inputs(1)
    kwargs = dict(tasks=[0], off_tasks=[1], n_shuffles=3, random_state=0)
    calls, default = [], dls._default

    def counted(*args):
        calls.append(args[2])
        return default(*args)

    monkeypatch.setattr(dls, "_CHECKED", weakref.WeakKeyDictionary())
    monkeypatch.setattr(dls, "_default", counted)
    first = deep_lift_shap(model, inputs, **kwargs)
    assert torch.equal(deep_lift_shap(model, inputs, **kwargs), first)
    assert calls == ["logits"]

    deep_lift_shap(model, inputs, output="expression", **kwargs)
    assert calls == ["logits", "expression"]


def test_prefetch():
    seen = []

    def draw(i):
        seen.append(i)
        return i * i

    assert list(dls._prefetch(draw, 4)) == [0, 1, 4, 9]
    assert seen == [0, 1, 2, 3]


def test_compiled_rule_falls_back():
    def double(x):
        return x * 2

    def broken(*args):
        raise RuntimeError("no compiler")

    rule = engine_module._Compiled.__new__(engine_module._Compiled)
    rule.function, rule.compiled = double, broken
    assert rule(torch.ones(2)).tolist() == [2.0, 2.0]
    assert rule.compiled is None


def test_engine_does_not_compile_on_cpu():
    engine = Engine(SmallDecima().eval(), torch.ones(N_TASKS), compile=True)
    assert not engine.compile
    assert engine._rule(engine_module.max_pool_rule) is engine_module.max_pool_rule


def test_tf32_matmul_is_restored(model):
    for previous in (False, True):
        torch.backends.cuda.matmul.allow_tf32 = previous
        with dls._tf32_matmul():
            assert torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cuda.matmul.allow_tf32 is previous

    torch.backends.cuda.matmul.allow_tf32 = False
    deep_lift_shap(model, make_inputs(1), tasks=[0], off_tasks=[1], n_shuffles=3, random_state=0)
    assert not torch.backends.cuda.matmul.allow_tf32


def test_engine_recomputes_logits():
    model = SmallDecima().double().eval()
    inputs = make_inputs(1)
    x, mask = inputs[0, :4], inputs[0, 4:]
    references = dls._references(x, 3, 0).double()
    kept = Engine(model, torch.ones(N_TASKS)).attribute(x, mask, references)
    engine = Engine(model, torch.ones(N_TASKS), keep_logits=False)
    with torch.no_grad():
        assert all(t.logits is None for t in engine.forward(inputs).transformers)
    assert torch.allclose(engine.attribute(x, mask, references)[0], kept[0], rtol=0, atol=1e-14)


def test_engine_drops_logits_when_out_of_memory(monkeypatch):
    model = SmallDecima().double().eval()
    inputs = make_inputs(1)
    x, mask = inputs[0, :4], inputs[0, 4:]
    references = dls._references(x, 3, 0).double()
    expected = Engine(model, torch.ones(N_TASKS)).attribute(x, mask, references)[0]

    backward = Engine.backward

    def limited(self, s, r):
        if s.transformers[0].logits is not None:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory.")
        return backward(self, s, r)

    monkeypatch.setattr(Engine, "backward", limited)
    engine = Engine(model, torch.ones(N_TASKS), pairs_per_step=2)
    attributions = engine.attribute(x, mask, references)[0]
    assert not engine.keep_logits and engine.pairs_per_step == 1
    assert torch.allclose(attributions, expected, rtol=0, atol=1e-14)
