"""Backbone-independent condition batching for observation-conditioned MORPHACT."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class ConditionSequence:
    """Tokens ``[B, T, D]`` and mask ``[B, T]`` with True for valid tokens."""

    tokens: Tensor
    valid_mask: Tensor


@dataclass(frozen=True, slots=True)
class ConditionBucket:
    """Equal-length condition tokens and their positions in the original batch."""

    indices: Tensor
    tokens: Tensor


def validate_condition_sequence(tokens: Tensor, valid_mask: Tensor) -> None:
    if tokens.ndim != 3:
        raise ValueError(f"Condition tokens must be rank 3, got {tokens.shape}")
    if valid_mask.ndim != 2 or valid_mask.shape != tokens.shape[:2]:
        raise ValueError("Condition mask must have shape [batch, sequence]")


def build_condition_buckets(sequence: ConditionSequence) -> tuple[ConditionBucket, ...]:
    """Group valid tokens by length without detaching their feature gradients."""
    validate_condition_sequence(sequence.tokens, sequence.valid_mask)
    valid_mask = sequence.valid_mask.to(device=sequence.tokens.device, dtype=torch.bool)
    lengths = valid_mask.sum(dim=1)
    if bool((lengths == 0).any()):
        raise ValueError("Every condition sample must contain at least one valid token")

    buckets = []
    for length in sorted({int(value) for value in lengths.detach().cpu().tolist()}):
        indices = torch.nonzero(lengths == length, as_tuple=False).flatten()
        compact = torch.stack([sequence.tokens[index][valid_mask[index]] for index in indices])
        buckets.append(ConditionBucket(indices=indices, tokens=compact))
    return tuple(buckets)


def generate_state(
    parameter_generator: nn.Module,
    buckets: tuple[ConditionBucket, ...],
    *,
    batch_size: int,
) -> tuple[dict[str, Tensor], int]:
    """Generate one factor state per sample and restore the original batch order."""
    if not buckets:
        raise ValueError("At least one condition bucket is required")
    if batch_size < 0:
        raise ValueError("Batch size must be non-negative")

    bucket_indices: list[Tensor] = []
    seen: set[int] = set()
    for bucket in buckets:
        if not isinstance(bucket.indices, Tensor) or bucket.indices.ndim != 1:
            raise ValueError("Condition bucket indices must be a rank-1 tensor")
        if bucket.indices.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise ValueError("Condition bucket indices must be an integer tensor")
        if not isinstance(bucket.tokens, Tensor) or bucket.tokens.ndim != 3:
            raise ValueError("Condition bucket tokens must be rank 3")
        if bucket.tokens.shape[0] != bucket.indices.shape[0]:
            raise ValueError("Condition bucket tokens must match the index batch dimension")

        indices = bucket.indices.detach().to(device="cpu", dtype=torch.long)
        for batch_index in indices.tolist():
            if batch_index < 0 or batch_index >= batch_size or batch_index in seen:
                raise ValueError("Condition buckets do not form a valid batch partition")
            seen.add(batch_index)
        bucket_indices.append(indices)

    if seen != set(range(batch_size)):
        raise ValueError("Condition buckets do not cover the full batch")

    expected_keys: set[str] | None = None
    factor_keys: tuple[str, ...] | None = None
    factor_shapes: dict[str, torch.Size] = {}
    scale: Tensor | None = None
    generated_by_bucket: list[dict[str, Tensor]] = []
    parameter_dtype = next(parameter_generator.parameters()).dtype

    for bucket, indices in zip(buckets, bucket_indices, strict=True):
        generated = parameter_generator(bucket.tokens.to(dtype=parameter_dtype))
        keys = set(generated) - {"scale"}
        if expected_keys is None:
            expected_keys = keys
            factor_keys = tuple(key for key in generated if key != "scale")
        elif keys != expected_keys:
            raise ValueError("ParameterGenerator returned different keys across buckets")

        bucket_scale = generated.get("scale")
        if not isinstance(bucket_scale, Tensor) or bucket_scale.numel() != 1:
            raise ValueError("ParameterGenerator must return a scalar scale")
        if scale is None:
            scale = bucket_scale
        elif not torch.equal(scale.detach().cpu(), bucket_scale.detach().cpu()):
            raise ValueError("ParameterGenerator returned different scales across buckets")

        for key in keys:
            value = generated[key]
            if not isinstance(value, Tensor) or value.ndim == 0 or value.shape[0] != indices.shape[0]:
                raise ValueError(f"Generated factor {key} has an invalid batch dimension")
            if key not in factor_shapes:
                factor_shapes[key] = value.shape[1:]
            elif value.shape[1:] != factor_shapes[key]:
                raise ValueError(f"Generated factor {key} has an inconsistent shape")
        generated_by_bucket.append(generated)

    assert factor_keys is not None
    assert scale is not None
    if len(buckets) == 1 and torch.equal(bucket_indices[0], torch.arange(batch_size)):
        factors = {key: generated_by_bucket[0][key].contiguous() for key in factor_keys}
    else:
        restore_indices = torch.argsort(torch.cat(bucket_indices))
        indices_by_device = {}
        factors = {}
        for key in factor_keys:
            compact = torch.cat([generated[key] for generated in generated_by_bucket], dim=0)
            if compact.device not in indices_by_device:
                indices_by_device[compact.device] = restore_indices.to(device=compact.device)
            factors[key] = compact.index_select(0, indices_by_device[compact.device])
    factors["scale"] = scale
    return factors, len(buckets)
