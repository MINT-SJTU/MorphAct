"""Pi0-specific MORPHACT condition construction plus shared condition utilities."""

from __future__ import annotations

from morphact.condition import ConditionBucket
from morphact.condition import ConditionSequence
from morphact.condition import build_condition_buckets
from morphact.condition import generate_state
from morphact.condition import validate_condition_sequence
import torch
from torch import Tensor

__all__ = [
    "ConditionBucket",
    "ConditionSequence",
    "build_action_condition",
    "build_condition_buckets",
    "build_vlm_condition",
    "generate_state",
]

# Retain the former private helper for downstream callers that imported it.
_validate_sequence = validate_condition_sequence


def build_action_condition(
    *,
    final_tokens: Tensor,
    prefix_valid_mask: Tensor,
    state_token: Tensor,
) -> ConditionSequence:
    """Append normalized Pi0 state to every sample's complete final VLM sequence."""
    _validate_sequence(final_tokens, prefix_valid_mask)
    if state_token.shape != (final_tokens.shape[0], 1, final_tokens.shape[2]):
        raise ValueError("Action MORPHACT requires one state token per sample")
    state_mask = torch.ones(state_token.shape[:2], dtype=torch.bool, device=state_token.device)
    return ConditionSequence(
        torch.cat((final_tokens, state_token), dim=1),
        torch.cat((prefix_valid_mask.detach(), state_mask), dim=1),
    )


def build_vlm_condition(*, prefix_embeddings: Tensor, prefix_valid_mask: Tensor) -> ConditionSequence:
    """Keep the visual gradient path through the full Pi0 multimodal prefix."""
    _validate_sequence(prefix_embeddings, prefix_valid_mask)
    return ConditionSequence(prefix_embeddings, prefix_valid_mask.detach())
