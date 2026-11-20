"""DeepLIFT/SHAP attributions of Decima models.

Examples:
    >>> model = load_decima_model(
    ...     "v1_rep0",
    ...     device="cuda",
    ... )
    >>> attrs = deep_lift_shap(
    ...     model,
    ...     inputs,
    ...     tasks=tasks,
    ...     off_tasks=off_tasks,
    ...     random_state=0,
    ... )

`tangermeme.deep_lift_shap.deep_lift_shap` attaches a rule to each module type it knows how to correct and treats
everything else as linear. Two parts of a Decima model are neither modules nor linear: the attention of gReLU's
transformer blocks, whose softmax and two bilinear products are function calls, and the exponential that turns the
logits into expression. Without rules for them the attributions do not sum to the change in the prediction. Here the
attention runs through `HookableAttention` for the duration of a call, which computes the same values with modules
tangermeme has rules for, and the exponential is a module with the rescale rule.
"""

import contextlib
import gc
import logging
import warnings
import weakref
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Union

import numpy as np
import torch
from einops import rearrange
from grelu.model.layers import Attention, relative_shift
from grelu.transforms.prediction_transforms import Aggregate, Specificity
from torch import Tensor, nn
from tqdm import tqdm

try:
    from tangermeme._deep_lift_utils import _nonlinear
    from tangermeme.deep_lift_shap import BilinearOp
    from tangermeme.deep_lift_shap import deep_lift_shap as _tangermeme_deep_lift_shap
    from tangermeme.ersatz import dinucleotide_shuffle

    from decima.interpret._deep_lift_shap_engine import Engine
except ImportError as exc:  # the rules for LayerNorm, softmax and bilinear products
    raise ImportError("DeepLIFT/SHAP attributions of Decima models require tangermeme>=1.6.0.") from exc

logger = logging.getLogger("decima")

_CHECK_REFERENCES = 1
"""`deep_lift_shap` attributes the first sequence against this many of its references with the default path, to
check the fast path against."""

_CHECK_PEARSON_MIN = 0.9999
_CHECK_REL_L2_MAX = 1e-2
"""The fast path agrees with the default path when the Pearson correlation of their attributions is at least
`_CHECK_PEARSON_MIN` and their relative L2 distance at most `_CHECK_REL_L2_MAX`."""

_COMPILE_MIN_PAIRS = 160
"""`deep_lift_shap` compiles the fast path's rules for a call with at least this many sequence-reference pairs
over its replicates, and from then on in the process. On an H200 the compiled rules save 0.07 seconds per pair and
their first calls take 6 seconds."""

_COMPILE = False

_CHECKED = weakref.WeakKeyDictionary()
"""For each replicate whose fast path has agreed with the default path in this process, the (output, length, device
type, compiled) combinations it agreed for. `deep_lift_shap` does not check those again."""


class HookableAttention(nn.Module):
    """gReLU's `Attention`, with its softmax and its two bilinear products as modules.

    The layers are those of the wrapped module and the operations are the same, so the output is the same. The
    products of the queries with the positional keys stay a function call, since the positional keys do not depend on
    the input and the product is therefore linear.

    Args:
        attention: The `grelu.model.layers.Attention` to run.
    """

    def __init__(self, attention: Attention):
        super().__init__()
        self.attention = attention
        self.content = BilinearOp("b h i d, b h j d -> b h i j")
        self.softmax = nn.Softmax(dim=-1)
        self.mix = BilinearOp("b h i j, b h j d -> b h i d")

    def forward(self, x: Tensor) -> Tensor:
        attention = self.attention
        q, k, v = attention.to_q(x), attention.to_k(x), attention.to_v(x)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=attention.n_heads), (q, k, v))
        q = q / (attention.key_len**0.5)

        content_logits = self.content(q + attention.rel_content_bias, k)
        pos_k = attention._get_pos_k(x)
        pos_logits = torch.einsum("b h i d, h j d -> b h i j", q + attention.rel_pos_bias, pos_k)
        attn = attention.attn_dropout(self.softmax(content_logits + relative_shift(pos_logits)))

        out = rearrange(self.mix(attn, v), "b h n d -> b n (h d)")
        return attention.to_out(out)


class _Exp(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(x)


class _Attributed(nn.Module):
    """The output `deep_lift_shap` attributes: a weighted sum of a model's logits or of their exponentials, of shape
    (n, 1), from a one-hot sequence (n, 4, length) and its gene mask (n, 1, length)."""

    def __init__(self, model: nn.Module, weights: Tensor, output: str):
        super().__init__()
        self.model = model
        self.exp = _Exp() if output == "expression" else nn.Identity()
        self.register_buffer("weights", weights[:, None], persistent=False)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        logits = self.model(torch.cat([x, mask], dim=1)).squeeze(-1)
        return self.exp(logits) @ self.weights.to(logits.dtype)


@contextlib.contextmanager
def hookable_attention(model: nn.Module):
    """Run every `grelu.model.layers.Attention` of a model through `HookableAttention` inside the context."""
    swapped = []
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, Attention):
                setattr(parent, name, HookableAttention(child))
                swapped.append((parent, name, child))
    try:
        yield model
    finally:
        for parent, name, child in swapped:
            setattr(parent, name, child)


def task_weights(
    model,
    tasks: Optional[List] = None,
    off_tasks: Optional[List] = None,
    transform: str = "specificity",
) -> Tensor:
    """The weight of each task in the output that `DecimaAttributer` attributes with a transform.

    "specificity" is the mean over `tasks` minus the mean over `off_tasks`, and "aggregate" the mean over `tasks`.
    Both are linear in the tasks, so each is a vector of weights: the transform of each task's indicator.

    Args:
        model: The Decima model, which resolves task names.
        tasks: The tasks to attribute.
        off_tasks: The tasks to contrast them with. None means all other tasks. Only with "specificity".
        transform: "specificity" or "aggregate".

    Returns:
        The weights, of shape (n_tasks,).
    """
    if transform == "specificity":
        function = Specificity(on_tasks=tasks, off_tasks=off_tasks, model=model, compare_func="subtract")
    elif transform == "aggregate":
        if off_tasks is not None:
            raise ValueError("`off_tasks` is not allowed with `aggregate` transform. Please use `specificity` instead.")
        function = Aggregate(tasks=tasks, task_aggfunc="mean", model=model)
    else:
        raise ValueError(f"Invalid transform: {transform}. Must be `specificity` or `aggregate`.")

    n_tasks = model.model_params["n_tasks"]
    return function(torch.eye(n_tasks, dtype=torch.float64)[:, :, None]).reshape(n_tasks).float()


def _replicates(model) -> List[nn.Module]:
    """The `DecimaModel` of each replicate of a model."""
    models = getattr(model, "models", None)
    return [m.model for m in models] if models is not None else [model.model]


def _references(x: Tensor, n_shuffles: int, random_state: int) -> Tensor:
    """The references tangermeme's `deep_lift_shap` draws for one sequence (4, length): reference k is its
    dinucleotide shuffle with the seed ``random_state + k``. Returns (n_shuffles, 4, length)."""
    seeds = [random_state + k for k in range(n_shuffles)]
    return dinucleotide_shuffle(x[None].expand(n_shuffles, -1, -1), n=1, random_state=seeds)[:, 0]


def _prefetch(function, n: int):
    """Yield function(0), ..., function(n - 1), computing each on a thread while the caller uses the one before."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(function, 0)
        for i in range(n):
            current = pending.result()
            if i + 1 < n:
                pending = pool.submit(function, i + 1)
            yield current


def _default(replicates, weights, output, x, mask, references, batch_size, warning_threshold, device, verbose):
    """The hypothetical attributions of the sequences x (n, 4, length) averaged over their references
    (n, K, 4, length) and over the replicates, with tangermeme's `deep_lift_shap`."""
    total = 0
    for replicate in replicates:
        with hookable_attention(replicate):
            # tangermeme restores the training mode of the module it is given, so the wrapper must share the
            # replicate's evaluation mode or the replicate would leave in training mode.
            total = total + _tangermeme_deep_lift_shap(
                _Attributed(replicate, weights, output).eval(),
                x,
                args=(mask,),
                batch_size=batch_size,
                references=references,
                hypothetical=True,
                warning_threshold=warning_threshold,
                additional_nonlinear_ops={_Exp: _nonlinear},
                device=device,
                verbose=verbose,
            )
    return total / len(replicates)


def _agreement(ours: Tensor, stock: Tensor):
    """The Pearson correlation and the relative L2 distance, ||ours - stock|| / ||stock||, of two attributions, in
    float64. A NaN in either gives NaN."""
    a, b = ours.double().flatten().cpu(), stock.double().flatten().cpu()
    rel_l2 = (a - b).norm() / b.norm().clamp_min(1e-30)
    a, b = a - a.mean(), b - b.mean()
    pearson = (a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-300)
    return float(pearson), float(rel_l2)


def deep_lift_shap(
    model,
    inputs: Tensor,
    tasks: Optional[List] = None,
    off_tasks: Optional[List] = None,
    transform: str = "specificity",
    output: str = "logits",
    references: Optional[Tensor] = None,
    n_shuffles: int = 20,
    hypothetical: bool = False,
    batch_size: int = 1,
    warning_threshold: Optional[float] = 0.001,
    random_state: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    fast: bool = True,
    verbose: bool = False,
) -> Tensor:
    """DeepLIFT/SHAP attributions of a Decima model.

    The attributed output is a weighted sum over the tasks, with the weights of `task_weights`: the mean over `tasks`
    minus the mean over `off_tasks` ("specificity") or the mean over `tasks` ("aggregate"). With `output="logits"`
    the sum is over the model's logits, the logarithm of the predicted expression, where the attributions of each
    sequence-reference pair sum to the difference between the two outputs up to floating-point rounding. With
    `output="expression"` the sum is over the predicted expression, the output the other attribution methods of
    Decima attribute, and the exponential gets the rescale rule. For an ensemble, the attributions are the mean over
    its replicates.

    There are two paths. The default path is tangermeme's `deep_lift_shap`, which attributes through backward hooks
    on the model's modules, with the attention run through `HookableAttention`. The fast path computes the same
    attributions from a forward and a backward pass written out for Decima's architecture, with tangermeme's rules
    and references: each sequence is forwarded once rather than once per reference, and the backward pass runs over
    one row per sequence-reference pair rather than two. The two paths agree up to floating-point rounding.

    The fast path is used when `fast` is True and the model is built as the fast path expects. The first time a model
    is attributed in a process, for each `output`, input length and device type, the first sequence is attributed
    against its first reference with both paths. When the two disagree beyond rounding, with a
    Pearson correlation below 0.9999 or a relative L2 distance above 1e-2, or when the fast path raises, a
    `RuntimeWarning` gives the reason and the default path attributes everything. The default path needs several
    times the memory of the fast path (68 GB against 15 GB for a 524,288 bp input). When it runs out of GPU memory
    during the check, a `RuntimeWarning` says so and the fast path is used unchecked.

    On CUDA, a call with at least 160 sequence-reference pairs over the model's replicates runs the fast path's
    softmax and max-pooling rules under `torch.compile`, as do later calls in the process. Their first calls
    compile.

    Both paths draw the same references: reference k of every sequence is its dinucleotide shuffle drawn with the
    seed ``random_state + k``. With `random_state` None, a seed is drawn once per call.

    Args:
        model: A Decima `LightningModel` or `EnsembleLightningModel`, as `load_decima_model` returns. It is put in
            evaluation mode and moved to `device`.
        inputs: The one-hot sequences with their gene masks, of shape (n, 5, length), without unknown characters.
        tasks: The tasks to attribute, as names or indices.
        off_tasks: The tasks to contrast them with. None means all other tasks. Only with "specificity".
        transform: "specificity" or "aggregate". Default is "specificity".
        output: "logits" or "expression". Default is "logits".
        references: One-hot references of shape (n, n_references, 4, length), or None to draw `n_shuffles`
            dinucleotide shuffles of each sequence.
        n_shuffles: The number of references drawn per sequence when `references` is None. Default is 20.
        hypothetical: Whether to return the hypothetical attributions of every character at every position, rather
            than those multiplied by the sequence. Default is False.
        batch_size: The number of sequence-reference pairs per batch on the default path. A pair takes two rows. The
            fast path sizes its steps itself. Default is 1.
        warning_threshold: A `RuntimeWarning` is given when a pair's convergence delta exceeds this: per batch on the
            default path, once per call on the fast path. None gives no such warning. Default is 0.001.
        random_state: The seed of the references. Default is None.
        device: The device to compute on. None means the model's device.
        fast: Whether to use the fast path where it applies. False always uses tangermeme's `deep_lift_shap`.
            Default is True.
        verbose: Whether to show progress bars. Default is False.

    Returns:
        The attributions, of shape (n, 4, length), averaged over each sequence's references, on the CPU.
    """
    if output not in ("logits", "expression"):
        raise ValueError(f"Invalid output: {output}. Must be `logits` or `expression`.")
    if inputs.ndim != 3 or inputs.shape[1] != 5:
        raise ValueError("`inputs` must have shape (n, 5, length): a one-hot encoded sequence and a gene mask.")

    weights = task_weights(model, tasks, off_tasks, transform)
    replicates = _replicates(model)
    device = torch.device(device) if device is not None else next(model.parameters()).device
    model.to(device).eval()

    inputs = inputs.detach().cpu()
    x, mask = inputs[:, :4].float(), inputs[:, 4:].float()
    n = len(x)
    given = references is not None
    if given:
        if references.ndim != 4 or references.shape[0] != n or references.shape[2:] != x.shape[1:]:
            raise ValueError("`references` must have shape (n, n_references, 4, length).")
        references = references.detach().cpu()
        n_shuffles = references.shape[1]
    if random_state is None:
        random_state = int(np.random.randint(0, 9_999_999))
    threshold = float("inf") if warning_threshold is None else warning_threshold

    def references_of(i):
        return references[i] if given else _references(x[i], n_shuffles, random_state)

    def default():
        attributions = []
        for i in tqdm(range(n), disable=not verbose, desc="DeepLIFT/SHAP"):
            attributions.append(
                _default(
                    replicates,
                    weights,
                    output,
                    x[i : i + 1],
                    mask[i : i + 1],
                    references_of(i)[None].float(),
                    batch_size,
                    threshold,
                    device,
                    False,
                )
            )
        return torch.cat(attributions)

    def finish(attributions):
        return attributions if hypothetical else attributions * x

    if not fast or n * n_shuffles <= _CHECK_REFERENCES:
        return finish(default())

    reason = None
    try:
        global _COMPILE
        _COMPILE = _COMPILE or n * n_shuffles * len(replicates) >= _COMPILE_MIN_PAIRS
        engines = [Engine(r, weights, output=output, device=device, compile=_COMPILE) for r in replicates]

        def engine(x_, mask_, references_):
            results = [e.attribute(x_, mask_, references_) for e in engines]
            attributions = sum(r[0] for r in results) / len(results)
            return attributions.cpu(), [r[1:] for r in results]

        # Reference draws run one sequence ahead of the device, on a thread.
        drawn = _prefetch(references_of, n)
        first = next(drawn)
        check = min(_CHECK_REFERENCES, n_shuffles)
        key = (output, x.shape[-1], device.type, _COMPILE)
        pearson, rel_l2 = 1.0, 0.0
        if not all(key in _CHECKED.get(replicate, ()) for replicate in replicates):
            # The default path keeps both rows of a pair for every hooked module, several times the fast path's
            # memory, so it can run out of memory where the fast path fits. The error is handled outside the except
            # clause: its traceback holds the failed batch's tensors, which must be freed before the cache is emptied.
            stock = None
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    stock = _default(
                        replicates,
                        weights,
                        output,
                        x[:1],
                        mask[:1],
                        first[None, :check].float(),
                        1,
                        threshold,
                        device,
                        False,
                    )[0]
            except torch.cuda.OutOfMemoryError:
                pass

            if stock is None:
                gc.collect()
                torch.cuda.empty_cache()
                warnings.warn(
                    "deep_lift_shap could not check the fast path against the default path, which ran out of GPU "
                    "memory. The fast path's convergence deltas are still checked.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                pearson, rel_l2 = _agreement(engine(x[0], mask[0], first[:check])[0], stock)
                if pearson >= _CHECK_PEARSON_MIN and rel_l2 <= _CHECK_REL_L2_MAX:
                    for replicate in replicates:
                        _CHECKED.setdefault(replicate, set()).add(key)

        if not (pearson >= _CHECK_PEARSON_MIN and rel_l2 <= _CHECK_REL_L2_MAX):
            reason = (
                f"its attributions of the first sequence against {check} references had a Pearson correlation of "
                f"{pearson:.6f} and a relative L2 distance of {rel_l2:.3g} with the default path's"
            )
        else:
            attributions, deltas, differences = [], [], []
            for i in tqdm(range(n), disable=not verbose, desc="DeepLIFT/SHAP"):
                attribution, outputs = engine(x[i], mask[i], first if i == 0 else next(drawn))
                attributions.append(attribution)
                for y, y_ref, delta in outputs:
                    deltas.append(delta.cpu())
                    differences.append((y - y_ref).abs().cpu())
            attributions = torch.stack(attributions)
    except Exception as exc:
        reason = "it raised {}: {}".format(type(exc).__name__, str(exc).strip().split("\n")[0])

    if reason is not None:
        warnings.warn(
            f"deep_lift_shap is using the default path because the fast path {reason}. "
            "Pass fast=False to skip the fast path.",
            RuntimeWarning,
            stacklevel=2,
        )
        return finish(default())

    deltas, differences = torch.cat(deltas), torch.cat(differences)
    if bool((deltas > threshold).any()):
        warnings.warn(
            "Convergence deltas too high: {} of {} pairs > {:g}, the largest {:.4g}".format(
                int((deltas > threshold).sum()), deltas.numel(), threshold, float(deltas.max())
            ),
            RuntimeWarning,
            stacklevel=2,
        )
    relative = float(deltas.double().mean() / differences.double().mean().clamp_min(1e-12))
    if relative > 0.01:
        warnings.warn(
            f"Convergence deltas are {relative:.1%} of the output differences over {deltas.numel()} pairs: the "
            "attributions do not sum to the change in the output.",
            RuntimeWarning,
            stacklevel=2,
        )
    return finish(attributions)
