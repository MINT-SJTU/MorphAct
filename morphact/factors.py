from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from .config import GeneratorConfig, ProjectionSpec


@dataclass(frozen=True, slots=True)
class TargetInfo:
    layer_index: int
    suffix: str
    in_features: int
    out_features: int


@dataclass(frozen=True, slots=True)
class LowRankFactors:
    """Per-sample A/B factors, scalar alpha/rank scale, and projection geometry.

    Matrix factors use ``[B, in, rank]`` and ``[B, rank, out]``. Head-wise
    factors insert a head axis after B; an output projection sums head results.
    """

    lora_a: Tensor
    lora_b: Tensor
    scale: Tensor
    projection: ProjectionSpec


def build_target_manifest(layers: Iterable[nn.Module], config: GeneratorConfig) -> tuple[TargetInfo, ...]:
    layers = tuple(layers)
    if len(layers) != config.num_base_model_layers:
        raise ValueError(f"Expected {config.num_base_model_layers} expert layers, got {len(layers)}")
    manifest = []
    for layer_index, layer in enumerate(layers):
        for projection in config.projections:
            try:
                module = layer.get_submodule(projection.suffix)
            except AttributeError as exc:
                raise ValueError(f"Missing target layer {layer_index}.{projection.suffix}") from exc
            if not isinstance(module, nn.Linear):
                raise ValueError(f"Target {layer_index}.{projection.suffix} is not nn.Linear")
            actual = (module.in_features, module.out_features)
            expected = (projection.in_features, projection.out_features)
            if actual != expected:
                raise ValueError(f"Target {layer_index}.{projection.suffix} has shape {actual}, expected {expected}")
            manifest.append(TargetInfo(layer_index, projection.suffix, *actual))
    return tuple(manifest)


def validate_generated_state(state: dict[str, Tensor], config: GeneratorConfig, batch_size: int) -> None:
    expected: dict[str, tuple[int, ...]] = {}
    for layer_index in range(config.num_base_model_layers):
        for projection in config.projections:
            expected[config.factor_key(layer_index, projection.suffix, "lora_A")] = (
                batch_size,
                *(
                    (projection.num_heads, projection.factor_in_features, config.rank)
                    if projection.factor_geometry != "matrix"
                    else (projection.factor_in_features, config.rank)
                ),
            )
            expected[config.factor_key(layer_index, projection.suffix, "lora_B")] = (
                batch_size,
                *(
                    (projection.num_heads, config.rank, projection.factor_out_features)
                    if projection.factor_geometry != "matrix"
                    else (config.rank, projection.factor_out_features)
                ),
            )
    if set(state) != set(expected) | {"scale"}:
        raise ValueError("Generated factor keys do not match the target manifest")
    scale = state["scale"].detach()
    if scale.numel() != 1:
        raise ValueError("Generated LoRA scale does not match the configuration")
    if scale.is_floating_point():
        # alpha/rank may not be exactly representable in the generated dtype.
        expected_scale = scale.new_tensor(config.scale)
        scale_matches = torch.equal(scale.reshape(()), expected_scale)
    else:
        scale_matches = float(scale) == config.scale
    if not scale_matches:
        raise ValueError("Generated LoRA scale does not match the configuration")
    for key, shape in expected.items():
        if tuple(state[key].shape) != shape:
            raise ValueError(f"Generated factor {key} has shape {tuple(state[key].shape)}, expected {shape}")


def get_lora_factors(
    state: dict[str, Tensor], config: GeneratorConfig, layer_index: int, suffix: str
) -> LowRankFactors:
    projection = next((item for item in config.projections if item.suffix == suffix), None)
    if projection is None:
        raise ValueError(f"Unknown LoRA projection: {suffix}")
    return LowRankFactors(
        lora_a=state[config.factor_key(layer_index, suffix, "lora_A")],
        lora_b=state[config.factor_key(layer_index, suffix, "lora_B")],
        scale=state["scale"],
        projection=projection,
    )


def batched_lora_linear(
    inputs: Tensor,
    linear: nn.Linear,
    factors: LowRankFactors,
) -> Tensor:
    """Apply frozen base compute plus a per-sample FP32 LoRA residual."""
    base_inputs = inputs.to(dtype=linear.weight.dtype)
    base = F.linear(base_inputs, linear.weight, linear.bias)
    lora_a, lora_b, scale, projection = (
        factors.lora_a,
        factors.lora_b,
        factors.scale,
        factors.projection,
    )
    if (linear.in_features, linear.out_features) != (projection.in_features, projection.out_features):
        raise ValueError(f"LoRA projection {projection.suffix} does not match target linear shape")
    if inputs.shape[0] != lora_a.shape[0] or inputs.shape[0] != lora_b.shape[0]:
        raise ValueError("LoRA factor batch size does not match inputs")
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        fp32_inputs = inputs.float()
        lora_a = lora_a.to(device=inputs.device, dtype=torch.float32)
        lora_b = lora_b.to(device=inputs.device, dtype=torch.float32)
        if projection.factor_geometry == "matrix":
            expected_a = (inputs.shape[0], projection.in_features, lora_a.shape[-1])
            expected_b = (inputs.shape[0], lora_a.shape[-1], projection.out_features)
            if tuple(lora_a.shape) != expected_a or tuple(lora_b.shape) != expected_b:
                raise ValueError(f"Invalid matrix LoRA factor shapes for {projection.suffix}")
            hidden = torch.einsum("b...i,bir->b...r", fp32_inputs, lora_a)
            delta = torch.einsum("b...r,bro->b...o", hidden, lora_b)
        elif projection.factor_geometry in {"q", "kv"}:
            expected_a = (inputs.shape[0], projection.num_heads, projection.in_features, lora_a.shape[-1])
            expected_b = (inputs.shape[0], projection.num_heads, lora_a.shape[-1], projection.head_dim)
            if tuple(lora_a.shape) != expected_a or tuple(lora_b.shape) != expected_b:
                raise ValueError(f"Invalid Q/K/V LoRA factor shapes for {projection.suffix}")
            hidden = torch.einsum("b...i,bhir->b...hr", fp32_inputs, lora_a)
            delta = torch.einsum("b...hr,bhro->b...ho", hidden, lora_b).flatten(start_dim=-2)
        elif projection.factor_geometry == "o":
            expected_a = (inputs.shape[0], projection.num_heads, projection.head_dim, lora_a.shape[-1])
            expected_b = (inputs.shape[0], projection.num_heads, lora_a.shape[-1], projection.out_features)
            if tuple(lora_a.shape) != expected_a or tuple(lora_b.shape) != expected_b:
                raise ValueError(f"Invalid O LoRA factor shapes for {projection.suffix}")
            head_inputs = fp32_inputs.unflatten(-1, (projection.num_heads, projection.head_dim))
            hidden = torch.einsum("b...hi,bhir->b...hr", head_inputs, lora_a)
            delta = torch.einsum("b...hr,bhro->b...o", hidden, lora_b)
        else:
            raise ValueError(f"Unsupported LoRA factor geometry: {projection.factor_geometry}")
        adapted = base.float() + delta * scale.to(device=inputs.device, dtype=torch.float32)
    return adapted.to(dtype=base.dtype)
