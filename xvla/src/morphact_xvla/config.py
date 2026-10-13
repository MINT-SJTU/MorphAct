from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class XVLAMorphActConfig:
    """Serializable capacity and precision settings for the two XVLA MORPHACT modules."""

    generator_dim: int = 768
    generator_head_dim: int = 128
    generator_token_dim: int = 768
    dim_accumulation: int = 4
    vlm_generator_blocks: int = 6
    action_generator_blocks: int = 4
    vlm_rank: int = 8
    vlm_alpha: int = 16
    action_rank: int = 8
    action_alpha: int = 16
    factor_init_std: float = 0.01
    compute_precision: Literal["float32", "bfloat16"] = "bfloat16"
    gradient_checkpointing: bool = True

    def __post_init__(self) -> None:
        positive = {
            "generator_dim": self.generator_dim,
            "generator_head_dim": self.generator_head_dim,
            "generator_token_dim": self.generator_token_dim,
            "dim_accumulation": self.dim_accumulation,
            "vlm_generator_blocks": self.vlm_generator_blocks,
            "action_generator_blocks": self.action_generator_blocks,
            "vlm_rank": self.vlm_rank,
            "vlm_alpha": self.vlm_alpha,
            "action_rank": self.action_rank,
            "action_alpha": self.action_alpha,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"XVLA MORPHACT dimensions must be positive: {invalid}")
        if self.generator_dim % self.generator_head_dim:
            raise ValueError("generator_dim must be divisible by generator_head_dim")
        if self.generator_dim % self.dim_accumulation:
            raise ValueError("generator_dim must be divisible by dim_accumulation")
        if self.vlm_rank % self.dim_accumulation or self.action_rank % self.dim_accumulation:
            raise ValueError("Both ranks must be divisible by dim_accumulation")
        if self.compute_precision not in {"float32", "bfloat16"}:
            raise ValueError("compute_precision must be float32 or bfloat16")
        if self.factor_init_std <= 0:
            raise ValueError("factor_init_std must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> XVLAMorphActConfig:
        return cls(**dict(values))
