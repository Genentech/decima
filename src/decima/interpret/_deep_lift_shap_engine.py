"""A fast DeepLIFT/SHAP engine for Decima models.

`tangermeme.deep_lift_shap.deep_lift_shap` attributes a model through backward hooks, over batches that hold each
sequence-reference pair as two rows: every sequence is forwarded once per reference, and the backward pass runs over
both halves of the batch. This engine computes the same multipliers from a forward and a backward pass written out
for Decima's architecture (the Borzoi trunk of gReLU and a convolutional head):

* Each sequence is forwarded once, and what the rules read from it is kept while its references are processed in
  chunks.
* The backward pass runs over one row per pair. tangermeme's rules give both rows of a pair the same multipliers, so
  the reference half carries no additional information.
* Only the inputs of the rules are kept from the forward pass: the output of every pooled convolution, and the
  attention logits and a few small tensors per transformer block.
* The rules are tangermeme's: the rescale rule for every GELU and ReLU (and for the exponential, when the expression
  is attributed), the max-pooling rule, the closed-form LayerNorm rule, the softmax rule and the midpoint rule for
  the two bilinear products of the attention. BatchNorm in evaluation mode, the convolutions, the linear layers, the
  upsampling, the crop and the average pooling are linear and take their ordinary gradients.

This module is private. `decima.interpret.deep_lift_shap.deep_lift_shap` runs it as its fast path and compares it
with tangermeme's `deep_lift_shap` on every call that uses it.
"""

import gc
import logging
from typing import List, NamedTuple, Optional

import torch
import torch.nn.functional as F
from grelu.model.layers import Attention, relative_shift
from tangermeme._deep_lift_utils import (
    _layer_normalization_grad,
    _layer_normalization_stats,
    _layer_normalization_terms,
    _rescale,
)
from torch import Tensor, nn
from torch.nn.grad import conv1d_input

logger = logging.getLogger("decima")


class UnsupportedArchitecture(ValueError):
    """The model is not built the way the engine's forward and backward passes are written."""


def _require(condition: bool, what: str) -> None:
    if not condition:
        raise UnsupportedArchitecture(f"the fast path does not support this model: {what}")


class _BatchNorm(NamedTuple):
    """An evaluation-mode BatchNorm1d."""

    mean: Tensor
    var: Tensor
    weight: Optional[Tensor]
    bias: Optional[Tensor]
    eps: float
    scale: Tensor  # (C, 1): the factor the layer multiplies its input by

    def __call__(self, x: Tensor) -> Tensor:
        return F.batch_norm(x, self.mean, self.var, self.weight, self.bias, False, 0.0, self.eps)


class _Conv(NamedTuple):
    """A stride-1, "same"-padded Conv1d."""

    weight: Tensor
    bias: Optional[Tensor]
    padding: int
    groups: int

    def __call__(self, x: Tensor) -> Tensor:
        return F.conv1d(x, self.weight, self.bias, padding=self.padding, groups=self.groups)

    def transpose(self, grad: Tensor) -> Tensor:
        """The gradient at the convolution's input given the one at its output."""
        shape = (grad.shape[0], self.weight.shape[1] * self.groups, grad.shape[-1])
        return conv1d_input(shape, self.weight, grad, padding=self.padding, groups=self.groups)


class _NormActConv(NamedTuple):
    """BatchNorm, GELU and a convolution, in that order: gReLU's "NACDR" and "NACD" blocks in evaluation mode."""

    norm: _BatchNorm
    approximate: str
    conv: _Conv

    def __call__(self, x: Tensor) -> Tensor:
        return self.conv(F.gelu(self.norm(x), approximate=self.approximate))

    def backward(self, grad: Tensor, x: Tensor, x_ref: Tensor) -> Tensor:
        """The gradient at the block's input, given the one at its output and the inputs of both rows of the pairs."""
        grad = gelu_rule(self.conv.transpose(grad), self.norm(x), self.norm(x_ref), self.approximate)
        return grad.mul_(self.norm.scale)


class _Transformer(NamedTuple):
    """One of gReLU's TransformerBlocks: LayerNorm, attention with relative positions, and a feed-forward network."""

    norm_weight: Tensor
    norm_bias: Tensor
    norm_eps: float
    to_q: Tensor
    to_k: Tensor
    to_v: Tensor
    to_out_weight: Tensor
    to_out_bias: Tensor
    rel_content_bias: Tensor
    rel_pos_bias: Tensor
    attention: Attention  # for the positional keys, which depend on the length only
    n_heads: int
    key_len: int
    ffn_norm_weight: Tensor
    ffn_norm_bias: Tensor
    ffn_norm_eps: float
    linear1_weight: Tensor
    linear1_bias: Tensor
    linear2_weight: Tensor
    linear2_bias: Tensor


class _TransformerSaved(NamedTuple):
    """What one transformer block's backward rules read from a forward pass."""

    x: Tensor  # (R, N, C) the block's input
    q: Tensor  # (R, h, N, d) the scaled queries
    k: Tensor  # (R, h, N, d)
    v: Tensor  # (R, h, N, dv)
    logits: Tensor  # (R, h, N, N) the softmax's input
    x_ffn: Tensor  # (R, N, C) the feed-forward network's input
    u: Tensor  # (R, N, 2C) the ReLU's input


class _Saved(NamedTuple):
    """What the backward rules read from a forward pass over R rows."""

    z: List[Tensor]  # the output of each convolution of the convolutional tower, before pooling
    transformers: List[_TransformerSaved]
    unet: List[Tensor]  # the input of each U-net block
    x_pointwise: Tensor  # the input of the pointwise convolution block
    e: Tensor  # the input of the final GELU
    logits: Tensor  # (R, n_tasks)


def _batch_norm(layer: nn.Module, device: torch.device) -> _BatchNorm:
    _require(isinstance(layer, nn.BatchNorm1d), f"expected BatchNorm1d, found {type(layer).__name__}")
    _require(layer.track_running_stats and layer.running_mean is not None, "BatchNorm1d without running statistics")
    mean, var = layer.running_mean.detach().to(device), layer.running_var.detach().to(device)
    weight = None if layer.weight is None else layer.weight.detach().to(device)
    bias = None if layer.bias is None else layer.bias.detach().to(device)
    scale = (var + layer.eps).rsqrt()
    if weight is not None:
        scale = scale * weight
    return _BatchNorm(mean, var, weight, bias, layer.eps, scale[:, None])


def _conv(layer: nn.Module, device: torch.device) -> _Conv:
    _require(isinstance(layer, nn.Conv1d), f"expected Conv1d, found {type(layer).__name__}")
    kernel_size = layer.kernel_size[0]
    _require(layer.stride[0] == 1 and layer.dilation[0] == 1 and kernel_size % 2 == 1, "a strided or dilated Conv1d")
    _require(layer.padding in ("same", (kernel_size // 2,)), "a Conv1d that changes the length")
    bias = None if layer.bias is None else layer.bias.detach().to(device)
    return _Conv(layer.weight.detach().to(device), bias, kernel_size // 2, layer.groups)


def _gelu(activation: nn.Module) -> str:
    layer = activation.layer
    _require(isinstance(layer, nn.GELU), f"expected GELU, found {type(layer).__name__}")
    return layer.approximate


def _norm_act_conv(block: nn.Module, conv: nn.Module, device: torch.device) -> _NormActConv:
    _require(block.order.startswith("NAC"), f"a block of order {block.order}")
    _require(not getattr(block, "residual", False), "a residual convolutional block")
    return _NormActConv(_batch_norm(block.norm.layer, device), _gelu(block.act), _conv(conv, device))


def _is_pool(pool: nn.Module) -> bool:
    layer = pool.layer
    return (
        isinstance(layer, nn.MaxPool1d)
        and layer.kernel_size == 2
        and layer.stride == 2
        and layer.padding == 0
        and layer.dilation == 1
    )


def _transformer(block: nn.Module, device: torch.device) -> _Transformer:
    attention, ffn = block.mha, block.ffn
    _require(isinstance(attention, Attention), f"expected Attention, found {type(attention).__name__}")
    norm, ffn_norm = block.norm.layer, ffn.dense1.norm.layer
    _require(isinstance(norm, nn.LayerNorm) and isinstance(ffn_norm, nn.LayerNorm), "a transformer without LayerNorm")
    _require(norm.elementwise_affine and ffn_norm.elementwise_affine, "a LayerNorm without weights")
    _require(len(norm.normalized_shape) == 1 and len(ffn_norm.normalized_shape) == 1, "a LayerNorm over two axes")
    _require(isinstance(ffn.dense1.act.layer, nn.ReLU), "a feed-forward network without ReLU")
    _require(isinstance(ffn.dense2.norm.layer, nn.Identity), "a normalized second feed-forward layer")
    _require(isinstance(ffn.dense2.act.layer, nn.Identity), "an activation after the feed-forward network")

    def get(t):
        return t.detach().to(device)

    return _Transformer(
        get(norm.weight),
        get(norm.bias),
        norm.eps,
        get(attention.to_q.weight),
        get(attention.to_k.weight),
        get(attention.to_v.weight),
        get(attention.to_out.weight),
        get(attention.to_out.bias),
        get(attention.rel_content_bias),
        get(attention.rel_pos_bias),
        attention,
        attention.n_heads,
        attention.key_len,
        get(ffn_norm.weight),
        get(ffn_norm.bias),
        ffn_norm.eps,
        get(ffn.dense1.linear.weight),
        get(ffn.dense1.linear.bias),
        get(ffn.dense2.linear.weight),
        get(ffn.dense2.linear.bias),
    )


def gelu_rule(grad: Tensor, x: Tensor, x_ref: Tensor, approximate: str) -> Tensor:
    """The rescale rule through a GELU.

    Args:
        grad: The gradient of each pair at the GELU's output, of shape (P, ...).
        x: The GELU's input for the sequence, of shape (1, ...).
        x_ref: The GELU's input for each reference, of shape (P, ...).
        approximate: The GELU's approximation, "none" or "tanh".

    Returns:
        The gradient at the GELU's input: `grad` times the secant, or, where the input barely moves, the ordinary
        gradient of the sequence row, as `tangermeme._deep_lift_utils._nonlinear` applies the rule.
    """
    gradient = torch.ops.aten.gelu_backward(grad, x.expand_as(grad), approximate=approximate)
    delta_out = F.gelu(x, approximate=approximate) - F.gelu(x_ref, approximate=approximate)
    return _rescale(x - x_ref, delta_out, gradient, grad)


def relu_rule(grad: Tensor, x: Tensor, x_ref: Tensor) -> Tensor:
    """The rescale rule through a ReLU, with the arguments of `gelu_rule`."""
    return _rescale(x - x_ref, F.relu(x) - F.relu(x_ref), grad * (x > 0), grad)


def exp_rule(grad: Tensor, x: Tensor, x_ref: Tensor) -> Tensor:
    """The rescale rule through an exponential, with the arguments of `gelu_rule`."""
    exp = torch.exp(x)
    return _rescale(x - x_ref, exp - torch.exp(x_ref), grad * exp, grad)


def max_pool_rule(grad: Tensor, z: Tensor, z_ref: Tensor) -> Tensor:
    """tangermeme's rule for a max pooling of width and stride 2.

    The change in each pooled output is routed to the position that won the maximum in the sequence and to the one
    that won it in the reference, and each position's share is divided by the change in its input. Where the input
    does not change, the ordinary gradient of the sequence row is used.

    Args:
        grad: The gradient of each pair at the pooled output, of shape (P, C, L / 2).
        z: The pooling's input for the sequence, of shape (1, C, L).
        z_ref: The pooling's input for each reference, of shape (P, C, L).

    Returns:
        The gradient at the pooling's input, of shape (P, C, L).
    """
    out, index = F.max_pool1d(z, 2, return_indices=True)
    out_ref, index_ref = F.max_pool1d(z_ref, 2, return_indices=True)

    # A window starts at an even position, so the lowest bit of an index says which of its two positions won.
    second = (index & 1).bool()
    second_ref = (index_ref & 1).bool()
    del index, index_ref

    highest = torch.maximum(out, out_ref)
    share = grad * (highest - out_ref)
    share_ref = grad * (out - highest)
    del highest, out, out_ref

    zero = torch.zeros((), dtype=grad.dtype, device=grad.device)
    routed = torch.stack(
        [
            torch.where(second, zero, share) + torch.where(second_ref, zero, share_ref),
            torch.where(second, share, zero) + torch.where(second_ref, share_ref, zero),
        ],
        dim=-1,
    ).flatten(-2)
    del share, share_ref, second_ref

    gradient = torch.stack([torch.where(second, zero, grad), torch.where(second, grad, zero)], dim=-1).flatten(-2)
    delta = z - z_ref
    unchanged = delta == 0
    return torch.where(unchanged, gradient, routed.div_(delta.masked_fill_(unchanged, 1)))


def layer_norm_rule(grad: Tensor, x: Tensor, x_ref: Tensor, weight: Tensor, eps: float) -> Tensor:
    """tangermeme's closed-form rule for a LayerNorm over the last axis.

    Args:
        grad: The gradient of each pair at the normalization's output, of shape (P, N, C).
        x: The normalization's input for the sequence, of shape (1, N, C).
        x_ref: The normalization's input for each reference, of shape (P, N, C).
        weight: The normalization's weight, of shape (C,).
        eps: The normalization's epsilon.

    Returns:
        The gradient at the normalization's input, of shape (P, N, C).
    """
    mu, v = _layer_normalization_stats(x, 1, eps)
    mu_ref, v_ref = _layer_normalization_stats(x_ref, 1, eps)
    terms = _layer_normalization_terms(x, mu, v, x_ref, mu_ref, v_ref, 1)
    return _layer_normalization_grad(grad * weight, *terms, 1)


def softmax_terms(logits: Tensor, logits_ref: Tensor):
    """The terms of tangermeme's softmax rule that depend only on the activations.

    The softmax over the last axis is decomposed as ``a_i = exp(x_i - c)``, ``v = 1 / sum_i a_i`` and
    ``y_i = a_i * v``, with the same ``c`` for a sequence and its reference.

    Args:
        logits: The softmax's input for the sequence, of shape (1, ..., N).
        logits_ref: The softmax's input for each reference, of shape (P, ..., N).

    Returns:
        The midpoint of the two softmax outputs, and the three factors `softmax_rule` combines with a gradient.
    """
    c = torch.maximum(logits.amax(dim=-1, keepdim=True), logits_ref.amax(dim=-1, keepdim=True))
    a = torch.exp(logits - c)
    a_ref = torch.exp(logits_ref - c)
    del c

    tiny = torch.finfo(a.dtype).tiny
    v = a.sum(dim=-1, keepdim=True).clamp_min(tiny).reciprocal()
    v_ref = a_ref.sum(dim=-1, keepdim=True).clamp_min(tiny).reciprocal()
    y = a * v
    y_ref = a_ref * v_ref

    delta_x = logits - logits_ref
    mult_x_to_a = torch.where(delta_x.abs() > 1e-6, (a - a_ref).div_(delta_x), a_ref)
    del a, a_ref

    delta_v = v - v_ref
    delta_log_v = torch.log(v) - torch.log(v_ref)
    delta_log_y = delta_x.add_(delta_log_v)
    del delta_x
    delta_y_over_delta_log_y = torch.where(delta_log_y.abs() > 1e-6, (y - y_ref).div_(delta_log_y), y_ref)
    del delta_log_y

    delta_log_v_over_delta_v = torch.where(delta_v.abs() > 1e-6, delta_log_v / delta_v, v_ref.reciprocal())
    denominator = delta_log_v_over_delta_v * (-v * v_ref)
    midpoint = y.add_(y_ref).mul_(0.5)
    return midpoint, delta_y_over_delta_log_y, mult_x_to_a, denominator


def softmax_rule(grad: Tensor, delta_y_over_delta_log_y: Tensor, mult_x_to_a: Tensor, denominator: Tensor) -> Tensor:
    """tangermeme's softmax rule: the gradient at the logits given the one at the softmax's output, of shape
    (P, ..., N), and the last three terms of `softmax_terms`."""
    numerator = grad * delta_y_over_delta_log_y
    total = numerator.sum(dim=-1, keepdim=True).mul_(denominator)
    return numerator.addcmul_(mult_x_to_a, total)


def softmax_backward(grad: Tensor, logits: Tensor, logits_ref: Tensor):
    """tangermeme's softmax rule in one call: the gradient at the logits, and the midpoint of the two softmax
    outputs, given the gradient at the softmax's output and the logits of both rows of the pairs."""
    midpoint, *terms = softmax_terms(logits, logits_ref)
    return softmax_rule(grad, *terms), midpoint


class _Compiled:
    """A rule under `torch.compile`, which fuses its elementwise passes over the largest tensors of a step into a
    few kernels. It is compiled for any shape, so that the max-pooling rule compiles once and not once per level of
    the convolutional tower. A rule that fails to compile runs as written from then on."""

    def __init__(self, function):
        self.function = function
        self.compiled = torch.compile(function, fullgraph=True, dynamic=True)

    def __call__(self, *args):
        if self.compiled is not None:
            try:
                return self.compiled(*args)
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as exc:
                logger.warning(
                    "compiling %s failed (%s: %s); running it uncompiled",
                    self.function.__name__,
                    type(exc).__name__,
                    str(exc).strip().split("\n")[0],
                )
                self.compiled = None
        return self.function(*args)


_COMPILED = {}
"""The compiled rules of this process, by name, shared by every engine."""


def _compiled(function):
    if function.__name__ not in _COMPILED:
        _COMPILED[function.__name__] = _Compiled(function)
    return _COMPILED[function.__name__]


def relative_shift_transpose(grad: Tensor) -> Tensor:
    """The transpose of `relative_shift`: the gradient at its input, of shape (..., N, 2N - 1), given the one at its
    output, of shape (..., N, N)."""
    *batch, n, _ = grad.shape
    grad = F.pad(grad, (0, n - 1)).reshape(*batch, 2 * n - 1, n)
    grad = F.pad(grad, (0, 0, 1, 0)).reshape(*batch, n, 2 * n)
    return grad[..., 1:]


def _heads(x: Tensor, n_heads: int) -> Tensor:
    """(R, N, h * d) -> (R, h, N, d)."""
    return x.view(*x.shape[:2], n_heads, -1).transpose(1, 2)


def _merge(x: Tensor) -> Tensor:
    """(R, h, N, d) -> (R, N, h * d)."""
    return x.transpose(1, 2).flatten(2)


class Engine:
    """DeepLIFT/SHAP multipliers of one Decima model for a weighted sum of its tasks.

    Args:
        model: The `DecimaModel`, with an `embedding` (gReLU's `BorzoiTrunk`) and a `head` (gReLU's `ConvHead`), in
            evaluation mode. It is not changed.
        weights: The weight of each task in the attributed output, of shape (n_tasks,).
        output: "logits", to attribute the weighted sum of the logits, or "expression", to attribute the weighted
            sum of their exponentials.
        device: Where the passes run. None means the model's device.
        pairs_per_step: The number of references of a sequence that are forwarded and back-propagated together. On
            an H200 and a 524,288 bp input, the time per reference does not depend on it and each takes about 13 GB.
            An out-of-memory error halves it.
        compile: Whether to run the softmax and max-pooling rules under `torch.compile` on CUDA. Their first calls
            in a process then compile, which took 6 seconds on an H200, and 20 references take 5.1 seconds rather
            than 6.6.
    """

    def __init__(
        self,
        model: nn.Module,
        weights: Tensor,
        output: str = "logits",
        device: Optional[torch.device] = None,
        pairs_per_step: int = 1,
        compile: bool = False,
    ):
        _require(not model.training, "it is in training mode")
        _require(output in ("logits", "expression"), f"output {output}")
        trunk, head = model.embedding, model.head
        if device is None:
            device = next(model.parameters()).device
        self.device = device = torch.device(device)
        self.output = output
        self.pairs_per_step = pairs_per_step
        self.compile = compile and device.type == "cuda"

        tower = trunk.conv_tower.blocks
        stem = tower[0]
        _require(isinstance(stem.act.layer, nn.Identity), "an activation in the stem")
        _require(all(_is_pool(block.pool) for block in tower), "a pooling other than MaxPool1d(2)")
        _require(
            [getattr(block, "return_pre_pool", False) for block in tower] == [False] * (len(tower) - 2) + [True] * 2,
            "U-net connections from other than the last two convolutional blocks",
        )
        self.stem = _conv(stem.conv, device)
        self.tower = [_norm_act_conv(block, block.conv, device) for block in tower[1:]]
        self.transformers = [_transformer(block, device) for block in trunk.transformer_tower.blocks]

        self.unet = []
        for block in trunk.unet_tower.blocks:
            _require(
                block.upsample.scale_factor == 2 and block.upsample.mode == "nearest", "an upsampling other than x2"
            )
            _require(block.sconv.depthwise.bias is None, "a depthwise convolution with a bias")
            self.unet.append(
                (
                    _norm_act_conv(block.conv, block.conv.conv, device),
                    _norm_act_conv(block.channel_transform, block.channel_transform.conv.layer, device),
                    _conv(block.sconv.depthwise, device),
                    _conv(block.sconv.pointwise, device),
                )
            )
        _require(len(self.unet) == 2, f"{len(self.unet)} U-net blocks")

        _require(isinstance(trunk.pointwise_conv.pool.layer, nn.Identity), "a pooling after the pointwise convolution")
        self.pointwise = _norm_act_conv(trunk.pointwise_conv, trunk.pointwise_conv.conv, device)
        self.approximate = _gelu(trunk.act)
        crop = trunk.crop.layer
        _require(isinstance(crop, (nn.Identity, nn.ConstantPad1d)), f"a crop of type {type(crop).__name__}")
        self.crop = 0 if isinstance(crop, nn.Identity) else -crop.padding[0]

        transform = head.channel_transform
        _require(isinstance(transform.norm.layer, nn.Identity), "a normalization in the head")
        _require(isinstance(transform.act.layer, nn.Identity), "an activation in the head")
        _require(isinstance(head.pool.layer, nn.AdaptiveAvgPool1d), "a head that does not average over positions")
        head_conv = _conv(transform.conv.layer, device)
        _require(head_conv.padding == 0 and head_conv.bias is not None, "a head convolution wider than one position")
        self.head_weight, self.head_bias = head_conv.weight[:, :, 0], head_conv.bias

        self.weights = weights.detach().to(device=device, dtype=self.head_weight.dtype)
        self._pos_k = {}

    def _positional_keys(self, t: _Transformer, n: int) -> Tensor:
        """The positional keys of a block for a length-n input, of shape (h, 2n - 1, d). They depend on nothing else,
        so they are computed once per block and length."""
        key = (id(t.attention), n)
        if key not in self._pos_k:
            x = torch.zeros(1, n, t.to_q.shape[1], dtype=t.to_q.dtype, device=self.device)
            self._pos_k[key] = t.attention._get_pos_k(x).detach()
        return self._pos_k[key]

    def _logits(self, t: _Transformer, q: Tensor, k: Tensor) -> Tensor:
        """The attention logits of a block, as gReLU's `Attention.get_attn_scores` computes them."""
        content = torch.einsum("b h i d, b h j d -> b h i j", q + t.rel_content_bias, k)
        positional = torch.einsum("b h i d, h j d -> b h i j", q + t.rel_pos_bias, self._positional_keys(t, q.shape[2]))
        return content.add_(relative_shift(positional))

    def _rule(self, function):
        """A rule, compiled when the engine compiles."""
        return _compiled(function) if self.compile else function

    def forward(self, x: Tensor) -> _Saved:
        """Run the model on x, of shape (R, 5, L), keeping what the backward rules read. Every operation is row-wise."""
        z = [self.stem(x)]
        for block in self.tower:
            z.append(block(F.max_pool1d(z[-1], 2)))

        h = F.max_pool1d(z[-1], 2).transpose(1, 2)
        transformers = []
        for t in self.transformers:
            normed = F.layer_norm(h, h.shape[-1:], t.norm_weight, t.norm_bias, t.norm_eps)
            q = _heads(F.linear(normed, t.to_q), t.n_heads) / (t.key_len**0.5)
            k = _heads(F.linear(normed, t.to_k), t.n_heads)
            v = _heads(F.linear(normed, t.to_v), t.n_heads)
            logits = self._logits(t, q, k)
            mixed = _merge(torch.einsum("b h i j, b h j d -> b h i d", logits.softmax(dim=-1), v))
            x_ffn = h + F.linear(mixed, t.to_out_weight, t.to_out_bias)

            normed = F.layer_norm(x_ffn, x_ffn.shape[-1:], t.ffn_norm_weight, t.ffn_norm_bias, t.ffn_norm_eps)
            u = F.linear(normed, t.linear1_weight, t.linear1_bias)
            transformers.append(_TransformerSaved(h, q, k, v, logits, x_ffn, u))
            h = x_ffn + F.linear(F.relu(u), t.linear2_weight, t.linear2_bias)

        h = h.transpose(1, 2)
        unet = []
        for (conv, channel_transform, depthwise, pointwise), y in zip(self.unet, (z[-1], z[-2])):
            unet.append(h)
            h = F.interpolate(conv(h), scale_factor=2, mode="nearest") + channel_transform(y)
            h = pointwise(depthwise(h))

        e = self.pointwise(h)
        pooled = F.gelu(e[..., self.crop : e.shape[-1] - self.crop], approximate=self.approximate).mean(dim=-1)
        logits = F.linear(pooled, self.head_weight, self.head_bias)
        return _Saved(z, transformers, unet, h, e, logits)

    def outputs(self, saved: _Saved) -> Tensor:
        """The attributed output of each row of a forward pass, of shape (R,)."""
        logits = saved.logits.exp() if self.output == "expression" else saved.logits
        return logits @ self.weights

    def _transformer_backward(self, grad: Tensor, t: _Transformer, s: _TransformerSaved, r: _TransformerSaved):
        """The backward pass of one transformer block: the gradient of each pair at the block's output, of shape
        (P, N, C), -> the one at its input."""
        # The feed-forward network: linear2, the ReLU's rescale rule, linear1 and the LayerNorm rule.
        g = relu_rule(grad @ t.linear2_weight, s.u, r.u)
        g = layer_norm_rule(g @ t.linear1_weight, s.x_ffn, r.x_ffn, t.ffn_norm_weight, t.ffn_norm_eps)
        grad = grad + g

        # The attention, with the midpoint rule for its two bilinear products and the softmax rule between them.
        g_mixed = _heads(grad @ t.to_out_weight, t.n_heads)
        g = g_mixed @ (0.5 * (s.v + r.v)).transpose(-1, -2)
        g, midpoint = self._rule(softmax_backward)(g, s.logits, r.logits)
        g_v = midpoint.transpose(-1, -2) @ g_mixed
        del midpoint, g_mixed

        g_k = g.transpose(-1, -2) @ (0.5 * (s.q + r.q) + t.rel_content_bias)
        g_q = g @ (0.5 * (s.k + r.k))
        positional = relative_shift_transpose(g)
        del g
        g_q += torch.einsum("b h i j, h j d -> b h i d", positional, self._positional_keys(t, s.q.shape[2]))
        del positional

        g = _merge(g_q) @ (t.to_q / (t.key_len**0.5))
        g += _merge(g_k) @ t.to_k
        g += _merge(g_v) @ t.to_v
        return grad + layer_norm_rule(g, s.x, r.x, t.norm_weight, t.norm_eps)

    def backward(self, s: _Saved, r: _Saved) -> Tensor:
        """The multipliers of each pair of a sequence and one of its references.

        Args:
            s: The forward pass of the sequence, one row.
            r: The forward pass of its references, P rows.

        Returns:
            The gradient of the attributed output at the one-hot sequence under tangermeme's rules, of shape
            (P, 4, L).
        """
        n_pairs = r.logits.shape[0]

        # The head: the weighted sum of the tasks, the exponential's rescale rule when the expression is attributed,
        # the 1x1 convolution and the average over the cropped positions, which every position shares.
        grad = self.weights.expand(n_pairs, -1)
        if self.output == "expression":
            grad = exp_rule(grad, s.logits, r.logits)
        length = s.e.shape[-1]
        crop = slice(self.crop, length - self.crop)
        grad = (grad @ self.head_weight / (length - 2 * self.crop))[:, :, None]
        cropped = gelu_rule(
            grad.expand(-1, -1, length - 2 * self.crop), s.e[..., crop], r.e[..., crop], self.approximate
        )
        grad = cropped.new_zeros(n_pairs, cropped.shape[1], length)
        grad[..., crop] = cropped
        del cropped
        grad = self.pointwise.backward(grad, s.x_pointwise, r.x_pointwise)

        # The U-net, last block first. Each block adds the upsampled stream to a transformed output of the
        # convolutional tower, whose gradient is kept until the tower's backward pass reaches that output.
        skips = []
        for (conv, channel_transform, depthwise, pointwise), x, x_ref, y, y_ref in zip(
            reversed(self.unet), reversed(s.unet), reversed(r.unet), (s.z[-2], s.z[-1]), (r.z[-2], r.z[-1])
        ):
            grad = depthwise.transpose(pointwise.transpose(grad))
            skips.append(channel_transform.backward(grad, y, y_ref))
            grad = conv.backward(grad.view(*grad.shape[:2], -1, 2).sum(dim=-1), x, x_ref)

        grad = grad.transpose(1, 2)
        for t, st, rt in zip(reversed(self.transformers), reversed(s.transformers), reversed(r.transformers)):
            grad = self._transformer_backward(grad, t, st, rt)
        grad = grad.transpose(1, 2)

        # The convolutional tower, last block first: the max-pooling rule, then BatchNorm, GELU and the convolution.
        skips = {len(s.z) - 1: skips[1], len(s.z) - 2: skips[0]}
        for i in range(len(s.z) - 1, -1, -1):
            grad = self._rule(max_pool_rule)(grad, s.z[i], r.z[i])
            if i in skips:
                grad += skips.pop(i)
            if i > 0:
                grad = self.tower[i - 1].backward(grad, F.max_pool1d(s.z[i - 1], 2), F.max_pool1d(r.z[i - 1], 2))

        return self.stem.transpose(grad)[:, :4]

    def attribute(self, x: Tensor, mask: Tensor, references: Tensor):
        """The hypothetical attributions of one sequence, averaged over its references.

        Args:
            x: The one-hot sequence, of shape (4, L).
            mask: The gene mask, of shape (1, L).
            references: The one-hot references, of shape (K, 4, L).

        Returns:
            The attributions, of shape (4, L), on the engine's device; the attributed output of the sequence; the
            attributed output of each reference, of shape (K,); and the convergence delta of each pair, of shape
            (K,).
        """
        dtype = self.head_weight.dtype
        x = x.to(device=self.device, dtype=dtype)[None]
        mask = mask.to(device=self.device, dtype=dtype)[None]
        n_references = references.shape[0]

        with torch.no_grad():
            s = self.forward(torch.cat([x, mask], dim=1))
            y = self.outputs(s)[0]

            total = torch.zeros_like(x[0])
            y_ref, deltas = [], []
            step, start = max(1, min(self.pairs_per_step, n_references)), 0
            while start < n_references:
                # The error is handled outside the except clause: its traceback holds the failed step's tensors,
                # which must be freed before the cache is emptied.
                failed = None
                try:
                    ref = references[start : start + step].to(device=self.device, dtype=dtype)
                    r = self.forward(torch.cat([ref, mask.expand(len(ref), -1, -1)], dim=1))
                    multipliers = self.backward(s, r)
                    y_ref.append(self.outputs(r))
                    del r
                except torch.cuda.OutOfMemoryError as exc:
                    if step == 1:
                        raise
                    failed = str(exc).strip().split("\n")[0]

                if failed is not None:
                    r = multipliers = ref = None
                    gc.collect()
                    torch.cuda.empty_cache()
                    logger.warning(
                        "out of GPU memory with %d references per step; retrying with %d (%s)", step, step // 2, failed
                    )
                    step //= 2
                    self.pairs_per_step = step
                    continue

                deltas.append(((y - y_ref[-1]) - ((x - ref) * multipliers).sum(dim=(1, 2))).abs())
                # tangermeme's `hypothetical_attributions`: character i gets the multipliers of replacing the
                # reference's character with it, summed over the characters.
                total += (multipliers - (ref * multipliers).sum(dim=1, keepdim=True)).sum(dim=0)
                del multipliers, ref
                start += step

        # Compiling leaves the traced tensors in reference cycles, which only the cyclic garbage collector frees.
        if self.compile:
            gc.collect()

        return total / n_references, y, torch.cat(y_ref), torch.cat(deltas)
