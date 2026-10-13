"""Scripted FP32 low-rank updates shared by the Pi and X-VLA executors."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .factors import LowRankFactors, batched_lora_linear


def _matrix_delta(x: Tensor, a: Tensor, b: Tensor, scale: Tensor) -> Tensor:
    a = a.to(device=x.device, dtype=torch.float32)
    b = b.to(device=x.device, dtype=torch.float32)
    hidden = torch.einsum("b...i,bir->b...r", x.float(), a)
    delta = torch.einsum("b...r,bro->b...o", hidden, b)
    return delta * scale.to(device=x.device, dtype=torch.float32)


def _qkv_delta(x: Tensor, a: Tensor, b: Tensor, scale: Tensor) -> Tensor:
    a = a.to(device=x.device, dtype=torch.float32)
    b = b.to(device=x.device, dtype=torch.float32)
    hidden = torch.einsum("b...i,bhir->b...hr", x.float(), a)
    delta = torch.einsum("b...hr,bhro->b...ho", hidden, b).flatten(start_dim=-2)
    return delta * scale.to(device=x.device, dtype=torch.float32)


def _o_delta(x: Tensor, a: Tensor, b: Tensor, scale: Tensor) -> Tensor:
    a = a.to(device=x.device, dtype=torch.float32)
    b = b.to(device=x.device, dtype=torch.float32)
    head_inputs = x.float().unflatten(-1, (a.shape[1], a.shape[2]))
    hidden = torch.einsum("b...hi,bhir->b...hr", head_inputs, a)
    delta = torch.einsum("b...hr,bhro->b...o", hidden, b)
    return delta * scale.to(device=x.device, dtype=torch.float32)


@torch.jit.script
def _matrix_update(x: Tensor, base: Tensor, a: Tensor, b: Tensor, scale: Tensor) -> Tensor:
    return (base.float() + _matrix_delta(x, a, b, scale)).to(base.dtype)


@torch.jit.script
def _qkv_update(x: Tensor, base: Tensor, a: Tensor, b: Tensor, scale: Tensor) -> Tensor:
    return (base.float() + _qkv_delta(x, a, b, scale)).to(base.dtype)


@torch.jit.script
def _o_update(x: Tensor, base: Tensor, a: Tensor, b: Tensor, scale: Tensor) -> Tensor:
    return (base.float() + _o_delta(x, a, b, scale)).to(base.dtype)


_NATIVE_DELTAS = {"matrix": _matrix_delta, "q": _qkv_delta, "kv": _qkv_delta, "o": _o_delta}
_DELTAS = {name: torch.jit.script(function) for name, function in _NATIVE_DELTAS.items()}
_UPDATES = {"matrix": _matrix_update, "q": _qkv_update, "kv": _qkv_update, "o": _o_update}


def scripted_lora_linear(inputs: Tensor, linear, factors: LowRankFactors) -> Tensor:
    # Keep the base Linear and its autocast context unchanged. Training stays eager.
    if torch.is_grad_enabled():
        return batched_lora_linear(inputs, linear, factors)
    base = F.linear(inputs.to(linear.weight.dtype), linear.weight, linear.bias)
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        return _UPDATES[factors.projection.factor_geometry](
            inputs, base, factors.lora_a, factors.lora_b, factors.scale
        )


def scripted_headwise_delta(inputs: Tensor, factors: LowRankFactors) -> Tensor:
    # X-VLA has a fused base QKV projection and adds these individual deltas.
    functions = _NATIVE_DELTAS if torch.is_grad_enabled() else _DELTAS
    function = functions[factors.projection.factor_geometry]
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        return function(inputs, factors.lora_a, factors.lora_b, factors.scale)
