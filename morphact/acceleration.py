"""Optional pointwise ``torch.compile`` acceleration for MorphAct generators.

Only stateless callables are compiled.  Keeping the owning modules intact
preserves parameter identities, optimizer references, and checkpoint keys.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .network import MultiHeadAttention, MultiHeadCrossAttention, apply_cross_rope, apply_rope


@functools.cache
def _compiled_self_rope() -> Callable[..., tuple[Tensor, Tensor]]:
    return torch.compile(apply_rope, dynamic=True, fullgraph=True)


@functools.cache
def _compiled_cross_rope() -> Callable[..., tuple[Tensor, Tensor]]:
    return torch.compile(apply_cross_rope, dynamic=True, fullgraph=True)


def _rms_norm(x: Tensor, weight: Tensor | None, eps: float | None) -> Tensor:
    return F.rms_norm(x, (x.shape[-1],), weight, eps)


@functools.cache
def _compiled_rms_norm() -> Callable[[Tensor, Tensor | None, float | None], Tensor]:
    return torch.compile(_rms_norm, dynamic=True, fullgraph=True)


def _rms_forward(module: nn.RMSNorm, x: Tensor) -> Tensor:
    return _compiled_rms_norm()(x, module.weight, module.eps)


def enable_pointwise_compilation(generator: nn.Module) -> None:
    """Compile MorphAct RoPE and RMSNorm pointwise operations in place.

    This modifies only function attributes on existing children.  It does not
    wrap modules, so parameters, optimizer references, and ``state_dict`` keys
    stay unchanged.  Repeated calls reuse the cached compiled functions.
    """

    for child in generator.modules():
        if isinstance(child, MultiHeadCrossAttention):
            child._rotary_fn = _compiled_cross_rope()
        elif isinstance(child, MultiHeadAttention):
            child._rotary_fn = _compiled_self_rope()
        elif isinstance(child, nn.RMSNorm) and not getattr(child, "_morphact_pointwise_compiled", False):
            child.forward = functools.partial(_rms_forward, child)
            child._morphact_pointwise_compiled = True
