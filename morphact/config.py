from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ProjectionSpec:
    """Target projection geometry; attention ranks are defined per head block."""

    suffix: str
    in_features: int
    out_features: int
    factor_geometry: Literal["matrix", "q", "kv", "o"] = "matrix"
    num_heads: int = 1
    head_dim: int | None = None

    def __post_init__(self) -> None:
        if not self.suffix or self.in_features <= 0 or self.out_features <= 0 or self.num_heads <= 0:
            raise ValueError(f"Invalid LoRA projection: {self}")
        if self.factor_geometry == "matrix":
            if self.num_heads != 1 or self.head_dim is not None:
                raise ValueError("Matrix LoRA projections must use one implicit head")
            return
        if self.factor_geometry not in {"q", "kv", "o"} or self.head_dim is None or self.head_dim <= 0:
            raise ValueError(f"Invalid headwise LoRA geometry: {self}")
        if self.factor_geometry in {"q", "kv"} and self.out_features != self.num_heads * self.head_dim:
            raise ValueError("Q/K/V LoRA output width must equal num_heads * head_dim")
        if self.factor_geometry == "o" and self.in_features != self.num_heads * self.head_dim:
            raise ValueError("O LoRA input width must equal num_heads * head_dim")

    @property
    def factor_count(self) -> int:
        return self.num_heads if self.factor_geometry != "matrix" else 1

    @property
    def factor_in_features(self) -> int:
        return self.head_dim if self.factor_geometry == "o" else self.in_features

    @property
    def factor_out_features(self) -> int:
        return self.head_dim if self.factor_geometry in {"q", "kv"} else self.out_features

    def generator_suffix(self, head_index: int = 0) -> str:
        if self.factor_geometry == "matrix":
            if head_index:
                raise ValueError("Matrix LoRA projections have no head index")
            return self.suffix
        if not 0 <= head_index < self.num_heads:
            raise ValueError(f"Invalid LoRA head index {head_index} for {self.suffix}")
        return f"{self.suffix}.__head_{head_index}"


@dataclass(frozen=True, slots=True)
class GeneratorConfig:
    """Generator capacity, target projections, residual scale, and compute precision."""

    prefix: str
    projections: tuple[ProjectionSpec, ...]
    input_dim: int = 2048
    token_dim: int = 1024
    output_dim: int = 1024
    rank: int = 16
    alpha: int = 32
    dim_accumulation: int = 4
    d_model: int = 1024
    head_dim: int = 128
    num_base_model_layers: int = 18
    num_pg_layers: int = 8
    state_dim: int | None = None
    factor_init_std: float = 0.01
    compute_precision: Literal["float32", "bfloat16"] = "float32"

    def __post_init__(self) -> None:
        if self.compute_precision not in {"float32", "bfloat16"}:
            raise ValueError("MORPHACT compute_precision must be float32 or bfloat16")
        dimensions = {
            "input_dim": self.input_dim,
            "token_dim": self.token_dim,
            "output_dim": self.output_dim,
            "rank": self.rank,
            "alpha": self.alpha,
            "dim_accumulation": self.dim_accumulation,
            "d_model": self.d_model,
            "head_dim": self.head_dim,
            "num_base_model_layers": self.num_base_model_layers,
            "num_pg_layers": self.num_pg_layers,
        }
        invalid = [name for name, value in dimensions.items() if value <= 0]
        if invalid:
            raise ValueError(f"MORPHACT dimensions must be positive: {invalid}")
        if self.state_dim is not None and self.state_dim <= 0:
            raise ValueError("state_dim must be positive when present")
        if not math.isfinite(self.factor_init_std) or self.factor_init_std <= 0:
            raise ValueError("factor_init_std must be finite and positive")
        if not self.prefix.endswith(".") or not self.projections:
            raise ValueError("MORPHACT prefix must end with a dot and projections cannot be empty")
        if len({projection.suffix for projection in self.projections}) != len(self.projections):
            raise ValueError("MORPHACT projection suffixes must be unique")
        if self.d_model % self.head_dim:
            raise ValueError("d_model must be divisible by head_dim")
        if self.d_model % self.dim_accumulation or self.rank % self.dim_accumulation:
            raise ValueError("d_model and rank must be divisible by dim_accumulation")
        if self.token_dim != self.output_dim:
            raise ValueError("parameter_generator tokenizer token_dim must equal its output_dim")

    @property
    def scale(self) -> float:
        return self.alpha / self.rank

    @property
    def target_count(self) -> int:
        return self.num_base_model_layers * len(self.projections)

    @property
    def generated_value_count(self) -> int:
        per_layer = sum(
            projection.factor_count * self.rank * (projection.factor_in_features + projection.factor_out_features)
            for projection in self.projections
        )
        return self.num_base_model_layers * per_layer

    @property
    def pg_mapping(self) -> dict[str, dict[str, int]]:
        return {
            projection.generator_suffix(head_index): {
                "lora_A_dim": projection.factor_in_features,
                "lora_B_dim": projection.factor_out_features,
            }
            for projection in self.projections
            for head_index in range(projection.factor_count)
        }

    def factor_key(self, layer_index: int, suffix: str, factor: str) -> str:
        if factor not in {"lora_A", "lora_B"}:
            raise ValueError(f"Unknown LoRA factor: {factor}")
        return f"{self.prefix}{layer_index}.{suffix}.{factor}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> GeneratorConfig:
        values = dict(values)
        values["projections"] = tuple(ProjectionSpec(**item) for item in values["projections"])
        return cls(**values)
