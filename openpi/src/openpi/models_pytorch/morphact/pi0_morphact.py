from __future__ import annotations

from collections.abc import Iterator

from morphact import ParameterGenerator
from morphact.factors import build_target_manifest
from morphact.factors import validate_generated_state
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional

from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

from .condition import build_action_condition
from .condition import build_condition_buckets
from .condition import build_vlm_condition
from .condition import generate_state
from .config import ACTION_MORPHACT_CONFIG
from .config import VISION_MODULE_NAMES
from .config import VLM_MORPHACT_CONFIG
from .config import GeneratorConfig
from .config import action_interface_module_names
from .executor import Pi0MorphActExecutor
from .executor import PrefixEncoding


def _compact_prefix(embeddings: Tensor, valid_mask: Tensor, attention_mask: Tensor):
    """Drop columns unused by every sample before the expensive VLM layers."""
    keep = valid_mask.any(dim=0)
    # Preserve attention groups even if a removed column starts a group.
    groups = attention_mask.long().cumsum(dim=1)[:, keep]
    compact_attention = torch.diff(groups, dim=1, prepend=groups.new_zeros((groups.shape[0], 1))) > 0
    return embeddings[:, keep], valid_mask[:, keep], compact_attention


class Pi0MorphAct(nn.Module):
    """Joint Pi0/Pi0.5 adaptation with frozen Transformers and trainable vision/interfaces."""

    def __init__(
        self,
        base_model: nn.Module,
        *,
        vlm_config: GeneratorConfig = VLM_MORPHACT_CONFIG,
        action_config: GeneratorConfig = ACTION_MORPHACT_CONFIG,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        if vlm_config.state_dim is not None:
            raise ValueError("VLM MORPHACT cannot have a state projection")
        if action_config.state_dim != base_model.config.action_dim:
            raise ValueError("Action MORPHACT state_dim must match Pi0 action_dim")
        self.base_model = base_model
        self.action_interface_module_names = action_interface_module_names(
            pi05=getattr(base_model.config, "pi05", False)
        )
        self.vlm_generator_config = vlm_config
        self.action_generator_config = action_config
        self.gradient_checkpointing = gradient_checkpointing
        base_model.requires_grad_(requires_grad=False)
        for module_name in (*VISION_MODULE_NAMES, *self.action_interface_module_names):
            module = base_model.get_submodule(module_name)
            module.float().requires_grad_(requires_grad=True)

        joint_model = base_model.paligemma_with_expert
        self.vlm_target_manifest = build_target_manifest(joint_model.paligemma.language_model.layers, vlm_config)
        self.action_target_manifest = build_target_manifest(joint_model.gemma_expert.model.layers, action_config)
        # These registered names define the version-1 adapter tensor keys.
        self.vlm_generator = ParameterGenerator(vlm_config)
        self.action_generator = ParameterGenerator(action_config)
        self.vlm_generator.gradient_checkpointing = gradient_checkpointing
        self.action_generator.gradient_checkpointing = gradient_checkpointing
        self.executor = Pi0MorphActExecutor(joint_model, action_config, vlm_config=vlm_config)
        self.executor.gradient_checkpointing = gradient_checkpointing
        vision_tower = base_model.get_submodule(VISION_MODULE_NAMES[0])
        if gradient_checkpointing:
            vision_tower.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            vision_tower.gradient_checkpointing_disable()
        self.last_metrics: dict[str, float] = {}
        self._inference_graph = None
        self.train()
        self.assert_trainable_state()

    @property
    def config(self):
        return self.base_model.config

    def configure_inference(self, enabled: bool = True) -> None:
        """Enable acceleration after loading weights; call again after replacing them."""
        from morphact.inference import configure_inference
        from morphact.inference_graph import CUDAGraphRunner

        configure_inference(self, [self.vlm_generator, self.action_generator], enabled=enabled)
        self._inference_graph = CUDAGraphRunner(self._denoise_actions) if enabled else None

    def adaptation_modules(self) -> dict[str, nn.Module]:
        return {
            "vlm_generator": self.vlm_generator,
            "action_generator": self.action_generator,
            **{
                f"base_model.{name}": self.base_model.get_submodule(name)
                for name in (*VISION_MODULE_NAMES, *self.action_interface_module_names)
            },
        }

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.trainable_parameters())

    @property
    def vision_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.vision_parameters())

    @property
    def action_interface_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.action_interface_parameters())

    @property
    def vlm_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.vlm_generator.parameters())

    @property
    def action_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.action_generator.parameters())

    def train(self, mode: bool = True):  # noqa: FBT001, FBT002
        super().train(mode)
        self.base_model.eval()
        for module in self.adaptation_modules().values():
            module.train(mode)
        self.executor.training = mode
        return self

    def vision_parameters(self) -> Iterator[nn.Parameter]:
        for name in VISION_MODULE_NAMES:
            yield from self.base_model.get_submodule(name).parameters()

    def action_interface_parameters(self) -> Iterator[nn.Parameter]:
        for name in self.action_interface_module_names:
            yield from self.base_model.get_submodule(name).parameters()

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        for module in self.adaptation_modules().values():
            yield from module.parameters()

    def assert_trainable_state(self) -> None:
        expected = {id(parameter) for parameter in self.trainable_parameters()}
        actual = {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        if expected != actual:
            raise RuntimeError("Only both MORPHACT modules, vision and the native action interfaces must be trainable")
        if any(parameter.dtype != torch.float32 for parameter in self.trainable_parameters()):
            raise RuntimeError("Trainable MORPHACT, vision and interface parameters must stay float32")

    def _record_generation_metrics(
        self,
        name: str,
        condition_mask: Tensor,
        generator_calls: int,
    ) -> None:
        self.last_metrics[f"{name}_generator_calls"] = float(generator_calls)
        self.last_metrics[f"{name}_condition_tokens_mean"] = float(condition_mask.sum(dim=1).float().mean())

    def _generate_vlm_state(self, prefix_embeddings: Tensor, prefix_valid_mask: Tensor) -> dict[str, Tensor]:
        condition = build_vlm_condition(prefix_embeddings=prefix_embeddings, prefix_valid_mask=prefix_valid_mask)
        generated_state, generator_calls = generate_state(
            self.vlm_generator, build_condition_buckets(condition), batch_size=prefix_embeddings.shape[0]
        )
        validate_generated_state(generated_state, self.vlm_generator_config, prefix_embeddings.shape[0])
        self._record_generation_metrics("vlm", condition.valid_mask, generator_calls)
        return generated_state

    def _generate_action_state(
        self, final_tokens: Tensor, prefix_valid_mask: Tensor, state: Tensor
    ) -> dict[str, Tensor]:
        condition = build_action_condition(
            final_tokens=final_tokens,
            prefix_valid_mask=prefix_valid_mask,
            state_token=self.action_generator.state_token(state),
        )
        generated_state, generator_calls = generate_state(
            self.action_generator, build_condition_buckets(condition), batch_size=final_tokens.shape[0]
        )
        validate_generated_state(generated_state, self.action_generator_config, final_tokens.shape[0])
        self._record_generation_metrics("action", condition.valid_mask, generator_calls)
        return generated_state

    def _prepare_prefix(self, images, image_masks, language_tokens, language_masks, state):
        prefix_embeddings, prefix_pad_masks, prefix_att_masks = self.base_model.embed_prefix(
            images, image_masks, language_tokens, language_masks
        )
        prefix_embeddings, prefix_pad_masks, prefix_att_masks = _compact_prefix(
            prefix_embeddings, prefix_pad_masks, prefix_att_masks
        )
        attention_mask = self.base_model._prepare_attention_masks_4d(  # noqa: SLF001
            make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        )
        position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        vlm_state = self._generate_vlm_state(prefix_embeddings, prefix_pad_masks)
        encoding = self.executor.encode_prefix(prefix_embeddings, attention_mask, position_ids, vlm_state)
        action_state = self._generate_action_state(encoding.final_tokens, prefix_pad_masks, state)
        return encoding, prefix_pad_masks, action_state

    def _decode(self, state, x_t, timestep, prefix_encoding, prefix_pad_masks, action_state):
        # Native Pi0/Pi0.5 suffix encoding owns state, timestep and action geometry.
        suffix_embeddings, suffix_pad_masks, suffix_att_masks, adarms_cond = self.base_model.embed_suffix(
            state, x_t, timestep
        )
        suffix_length = suffix_pad_masks.shape[1]
        prefix_mask = prefix_pad_masks[:, None, :].expand(-1, suffix_length, -1)
        suffix_mask = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        attention_mask = self.base_model._prepare_attention_masks_4d(  # noqa: SLF001
            torch.cat((prefix_mask, suffix_mask), dim=2)
        )
        prefix_offsets = prefix_pad_masks.sum(dim=-1, keepdim=True)
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        suffix_output = self.executor.decode_suffix(
            suffix_embeddings, prefix_encoding, attention_mask, position_ids, adarms_cond, action_state
        )
        suffix_output = suffix_output[:, -self.config.action_horizon :].float()
        return self.base_model.action_out_proj(suffix_output)

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        self.last_metrics = {}
        images, image_masks, language_tokens, language_masks, state = self.base_model._preprocess_observation(  # noqa: SLF001
            observation, train=True
        )
        noise = self.base_model.sample_noise(actions.shape, actions.device) if noise is None else noise
        time = self.base_model.sample_time(actions.shape[0], actions.device) if time is None else time
        expanded_time = time[:, None, None]
        x_t = expanded_time * noise + (1 - expanded_time) * actions
        target_velocity = noise - actions
        prefix_encoding, prefix_pad_masks, action_state = self._prepare_prefix(
            images, image_masks, language_tokens, language_masks, state
        )
        predicted_velocity = self._decode(state, x_t, time, prefix_encoding, prefix_pad_masks, action_state)
        return functional.mse_loss(target_velocity, predicted_velocity, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        self.last_metrics = {}
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        device = torch.device(device)
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = self.base_model.sample_noise(
                (batch_size, self.config.action_horizon, self.config.action_dim), device
            )
        images, image_masks, language_tokens, language_masks, state = self.base_model._preprocess_observation(  # noqa: SLF001
            observation, train=False
        )
        prefix_encoding, prefix_pad_masks, action_state = self._prepare_prefix(
            images, image_masks, language_tokens, language_masks, state
        )
        # PG and noise sampling stay outside capture. Every request supplies fresh inputs.
        denoise = self._inference_graph if self._inference_graph is not None and not self.training else self._denoise_actions
        return denoise(state, noise, prefix_pad_masks, prefix_encoding.key_values, action_state, num_steps)

    def _denoise_actions(self, state, noise, prefix_pad_masks, key_values, action_state, num_steps):
        prefix_encoding = PrefixEncoding(None, key_values)  # PG has already consumed the final tokens.
        step_size = -1.0 / num_steps
        x_t = noise
        for step in range(num_steps):
            timestep = torch.full((state.shape[0],), 1.0 + step * step_size, dtype=torch.float32, device=state.device)
            x_t = x_t + step_size * self._decode(state, x_t, timestep, prefix_encoding, prefix_pad_masks, action_state)
        return x_t
