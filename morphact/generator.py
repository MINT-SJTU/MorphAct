from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ._core import StructuredGenerator
from .config import GeneratorConfig


class ParameterGenerator(nn.Module):
    """Generate per-sample low-rank factors from context tokens.

    ``forward`` accepts compact tokens of shape ``[batch, tokens, input_dim]``.
    Outputs contain ``lora_A``/``lora_B`` tensors for each target and a scalar
    ``scale``. Factors and residual arithmetic use FP32; generator attention can
    use BF16. Robot state is projected separately by ``state_token``.
    """

    def __init__(self, config: GeneratorConfig) -> None:
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.state_projection = nn.Linear(config.state_dim, config.input_dim) if config.state_dim is not None else None
        self.core = StructuredGenerator(config)
        # Both nonzero readouts let the native loss reach the generator trunk.
        readout_std = config.factor_init_std / math.sqrt(self.core.model.proj_out_A.in_features)
        nn.init.normal_(self.core.model.proj_out_A.weight, mean=0.0, std=readout_std)
        nn.init.zeros_(self.core.model.proj_out_A.bias)
        nn.init.normal_(self.core.model.proj_out_B.weight, mean=0.0, std=readout_std)
        nn.init.zeros_(self.core.model.proj_out_B.bias)
        self.to(dtype=torch.float32)

    def state_token(self, state: Tensor) -> Tensor:
        if self.state_projection is None:
            raise RuntimeError("This generator has no state projection")
        if state.ndim != 2 or state.shape[1] != self.config.state_dim:
            raise ValueError(f"Expected normalized state shape [batch, {self.config.state_dim}], got {state.shape}")
        with torch.autocast(device_type=state.device.type, enabled=False):
            return self.state_projection(state.float())[:, None, :]

    def forward(self, condition: Tensor) -> dict[str, Tensor]:
        if condition.ndim != 3 or condition.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"Expected condition shape [batch, tokens, {self.config.input_dim}], got {condition.shape}"
            )
        if condition.shape[0] == 0 or condition.shape[1] == 0:
            raise ValueError("Condition batch and token dimensions must be nonempty")
        with torch.autocast(
            device_type=condition.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.compute_precision == "bfloat16",
        ):
            self.core.model.gradient_checkpointing = self.gradient_checkpointing
            generated = self.core(condition.float())
        state = {"scale": generated["scale"]}
        for layer_index in range(self.config.num_base_model_layers):
            for projection in self.config.projections:
                factor_a = []
                factor_b = []
                for head_index in range(projection.factor_count):
                    raw_prefix = f"{self.config.prefix}{layer_index}.{projection.generator_suffix(head_index)}"
                    factor_a.append(generated[f"{raw_prefix}.lora_A"].transpose(-1, -2))
                    factor_b.append(generated[f"{raw_prefix}.lora_B"].transpose(-1, -2))
                key_a = self.config.factor_key(layer_index, projection.suffix, "lora_A")
                key_b = self.config.factor_key(layer_index, projection.suffix, "lora_B")
                if projection.factor_geometry == "matrix":
                    state[key_a] = factor_a[0]
                    state[key_b] = factor_b[0]
                else:
                    state[key_a] = torch.stack(factor_a, dim=1)
                    state[key_b] = torch.stack(factor_b, dim=1)
        return state
