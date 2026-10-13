from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch
from torch import Tensor, nn

from morphact import ParameterGenerator, GeneratorConfig, ProjectionSpec
from morphact.condition import ConditionSequence, build_condition_buckets, generate_state
from morphact.factors import validate_generated_state

from .config import XVLAMorphActConfig
from .executor import XVLAMorphActExecutor
from .native.models.modeling_xvla import XVLA


class XVLAMorphAct(nn.Module):
    """Native X-VLA with observation-conditioned MORPHACT factors on both Transformers."""

    def __init__(self, base_model: XVLA, config: XVLAMorphActConfig | None = None) -> None:
        super().__init__()
        self.base_model = base_model
        self.morphact_config = config or XVLAMorphActConfig()
        self.vlm_morphact_config = self._build_vlm_morphact_config()
        self.action_morphact_config = self._build_action_morphact_config()
        self.vlm_morphact = ParameterGenerator(self.vlm_morphact_config)
        self.action_morphact = ParameterGenerator(self.action_morphact_config)
        self.vlm_morphact.gradient_checkpointing = self.morphact_config.gradient_checkpointing
        self.action_morphact.gradient_checkpointing = self.morphact_config.gradient_checkpointing
        self.executor = XVLAMorphActExecutor(base_model, self.vlm_morphact_config, self.action_morphact_config)
        self._inference_graph = None
        self.last_metrics: dict[str, float] = {}
        self._configure_trainable_parameters()
        self.train()
        self.assert_trainable_state()

    @classmethod
    def from_pretrained(
        cls,
        base_model_path: str,
        *,
        config: XVLAMorphActConfig | None = None,
        torch_dtype: torch.dtype | None = None,
        **kwargs: Any,
    ) -> XVLAMorphAct:
        base_model = XVLA.from_pretrained(base_model_path, torch_dtype=torch_dtype, **kwargs)
        return cls(base_model, config=config)

    @property
    def native(self) -> XVLA:
        return self.base_model

    @property
    def config(self):
        return self.base_model.config

    def configure_inference(self, enabled: bool = True) -> None:
        """Select serving optimizations without changing training initialization."""
        from morphact.inference import configure_inference
        from morphact.inference_graph import CUDAGraphRunner

        configure_inference(self, [self.vlm_morphact, self.action_morphact], enabled=enabled)
        self._inference_graph = CUDAGraphRunner(self._denoise_actions) if enabled else None

    def _common_morphact_kwargs(self, *, blocks: int, rank: int, alpha: int) -> dict[str, Any]:
        return {
            "input_dim": int(self.base_model.vlm.config.projection_dim),
            "token_dim": self.morphact_config.generator_token_dim,
            "output_dim": self.morphact_config.generator_token_dim,
            "rank": rank,
            "alpha": alpha,
            "dim_accumulation": self.morphact_config.dim_accumulation,
            "d_model": self.morphact_config.generator_dim,
            "head_dim": self.morphact_config.generator_head_dim,
            "num_pg_layers": blocks,
            "factor_init_std": self.morphact_config.factor_init_std,
            "compute_precision": self.morphact_config.compute_precision,
        }

    def _build_vlm_morphact_config(self) -> GeneratorConfig:
        language_config = self.base_model.vlm.config.text_config
        hidden_size = language_config.d_model
        heads = language_config.encoder_attention_heads
        head_dim = hidden_size // heads
        projections = (
            ProjectionSpec("self_attn.q_proj", hidden_size, hidden_size, "q", heads, head_dim),
            ProjectionSpec("self_attn.k_proj", hidden_size, hidden_size, "kv", heads, head_dim),
            ProjectionSpec("self_attn.v_proj", hidden_size, hidden_size, "kv", heads, head_dim),
            ProjectionSpec("self_attn.out_proj", hidden_size, hidden_size, "o", heads, head_dim),
            ProjectionSpec("fc1", hidden_size, language_config.encoder_ffn_dim),
            ProjectionSpec("fc2", language_config.encoder_ffn_dim, hidden_size),
        )
        return GeneratorConfig(
            prefix="vlm.encoder.layers.",
            projections=projections,
            num_base_model_layers=len(self.base_model.vlm.language_model.model.encoder.layers),
            **self._common_morphact_kwargs(
                blocks=self.morphact_config.vlm_generator_blocks,
                rank=self.morphact_config.vlm_rank,
                alpha=self.morphact_config.vlm_alpha,
            ),
        )

    def _build_action_morphact_config(self) -> GeneratorConfig:
        transformer = self.base_model.transformer
        hidden_size = transformer.hidden_size
        attention = transformer.blocks[0].attn
        heads, head_dim = attention.num_heads, attention.head_dim
        mlp_width = transformer.blocks[0].mlp.fc1.out_features
        projections = (
            ProjectionSpec("attn.q", hidden_size, hidden_size, "q", heads, head_dim),
            ProjectionSpec("attn.k", hidden_size, hidden_size, "kv", heads, head_dim),
            ProjectionSpec("attn.v", hidden_size, hidden_size, "kv", heads, head_dim),
            ProjectionSpec("attn.proj", hidden_size, hidden_size, "o", heads, head_dim),
            ProjectionSpec("mlp.fc1", hidden_size, mlp_width),
            ProjectionSpec("mlp.fc2", mlp_width, hidden_size),
        )
        dim_proprio = self.base_model.transformer.action_encoder.input_size
        dim_proprio -= self.base_model.action_space.dim_action + transformer.dim_time
        return GeneratorConfig(
            prefix="action.blocks.",
            projections=projections,
            num_base_model_layers=len(transformer.blocks),
            state_dim=dim_proprio,
            **self._common_morphact_kwargs(
                blocks=self.morphact_config.action_generator_blocks,
                rank=self.morphact_config.action_rank,
                alpha=self.morphact_config.action_alpha,
            ),
        )

    def _trainable_base_modules(self) -> tuple[nn.Module, ...]:
        vlm = self.base_model.vlm
        transformer = self.base_model.transformer
        modules: list[nn.Module] = [
            vlm.vision_tower,
            vlm.image_proj_norm,
            vlm.image_pos_embed,
            vlm.visual_temporal_embed,
            transformer.vlm_proj,
            transformer.aux_visual_proj,
            transformer.action_encoder,
            transformer.action_decoder,
        ]
        if transformer.len_soft_prompts > 0:
            modules.append(transformer.soft_prompt_hub)
        return tuple(modules)

    def _configure_trainable_parameters(self) -> None:
        self.base_model.requires_grad_(False)
        for module in self._trainable_base_modules():
            module.float().requires_grad_(True)
        self.base_model.vlm.image_projection.data = self.base_model.vlm.image_projection.data.float()
        self.base_model.vlm.image_projection.requires_grad_(True)
        self.vlm_morphact.float().requires_grad_(True)
        self.action_morphact.float().requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_model.eval()
        for module in self._trainable_base_modules():
            module.train(mode)
        self.vlm_morphact.train(mode)
        self.action_morphact.train(mode)
        self.executor.training = mode
        return self

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.vlm_morphact.parameters()
        yield from self.action_morphact.parameters()
        yield self.base_model.vlm.image_projection
        for module in self._trainable_base_modules():
            yield from module.parameters()

    def assert_trainable_state(self) -> None:
        expected = {id(parameter) for parameter in self.trainable_parameters()}
        actual = {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        if expected != actual:
            raise RuntimeError("Only MORPHACT, vision, image projection and action interfaces may be trainable")
        if any(parameter.dtype != torch.float32 for parameter in self.trainable_parameters()):
            raise RuntimeError("All trainable XVLA MORPHACT parameters must remain float32")

    def _prepare_observation(
        self, input_ids: Tensor, image_input: Tensor, image_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch_size, views = image_input.shape[:2]
        flat_mask = image_mask.reshape(-1).bool()
        if not bool(flat_mask.any()):
            raise ValueError("At least one image view must be valid per batch")
        flat_images = image_input.flatten(0, 1)
        valid_features = self.base_model.vlm._encode_image(flat_images[flat_mask])
        image_tokens, hidden_size = valid_features.shape[1:]
        image_features = valid_features.new_zeros((batch_size * views, image_tokens, hidden_size))
        image_features[flat_mask] = valid_features
        image_features = image_features.view(batch_size, views, image_tokens, hidden_size)
        language_features = self.base_model.vlm.get_input_embeddings()(input_ids)
        merged, attention_mask = self.base_model.vlm._merge_input_ids_with_image_features(
            image_features[:, 0], language_features
        )
        return image_features, language_features, merged, attention_mask, flat_mask

    def _generate(
        self, morphact: ParameterGenerator, config: GeneratorConfig, condition: ConditionSequence, metric_name: str
    ) -> dict[str, Tensor]:
        state, calls = generate_state(
            morphact, build_condition_buckets(condition), batch_size=condition.tokens.shape[0]
        )
        validate_generated_state(state, config, condition.tokens.shape[0])
        self.last_metrics[f"{metric_name}_generator_calls"] = float(calls)
        return state

    def _encode_observation(
        self, input_ids: Tensor, image_input: Tensor, image_mask: Tensor, proprio: Tensor
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        image_features, language_features, merged, attention_mask, _ = self._prepare_observation(
            input_ids, image_input, image_mask
        )
        batch_size, _, image_tokens, hidden_size = image_features.shape
        vlm_condition_tokens = torch.cat((image_features.flatten(1, 2), language_features), dim=1)
        image_valid = image_mask.bool()[:, :, None].expand(-1, -1, image_tokens).flatten(1)
        language_valid = torch.ones(batch_size, language_features.shape[1], device=input_ids.device, dtype=torch.bool)
        vlm_state = self._generate(
            self.vlm_morphact,
            self.vlm_morphact_config,
            ConditionSequence(vlm_condition_tokens, torch.cat((image_valid, language_valid), dim=1)),
            "vlm",
        )
        vlm_features = self.executor.encode_florence(merged, attention_mask, vlm_state)
        aux_features = image_features[:, 1:].reshape(batch_size, -1, hidden_size)
        action_tokens = torch.cat((vlm_features, aux_features, self.action_morphact.state_token(proprio)), dim=1)
        aux_valid = image_mask.bool()[:, 1:, None].expand(-1, -1, image_tokens).flatten(1)
        action_valid = torch.cat(
            (
                torch.ones(batch_size, vlm_features.shape[1], device=input_ids.device, dtype=torch.bool),
                aux_valid,
                torch.ones(batch_size, 1, device=input_ids.device, dtype=torch.bool),
            ),
            dim=1,
        )
        action_state = self._generate(
            self.action_morphact,
            self.action_morphact_config,
            ConditionSequence(action_tokens, action_valid),
            "action",
        )
        return vlm_features, aux_features, action_state

    def _predict(
        self,
        *,
        domain_id: Tensor,
        vlm_features: Tensor,
        aux_features: Tensor,
        action_with_noise: Tensor,
        proprio: Tensor,
        time: Tensor,
        action_state: dict[str, Tensor],
    ) -> Tensor:
        return self.executor.decode_action(
            domain_id=domain_id,
            vlm_features=vlm_features,
            aux_visual_inputs=aux_features,
            action_with_noise=action_with_noise,
            proprio=proprio,
            time=time,
            state=action_state,
        )

    def forward(
        self,
        input_ids: Tensor,
        image_input: Tensor,
        image_mask: Tensor,
        domain_id: Tensor,
        proprio: Tensor,
        action: Tensor,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> dict[str, Tensor]:
        self.last_metrics = {}
        batch_size = input_ids.shape[0]
        if time is None:
            time = (
                torch.rand(1, device=input_ids.device) + torch.arange(batch_size, device=input_ids.device) / batch_size
            )
            time = time % (1 - 1e-5)
        if noise is None:
            noise = torch.randn_like(action)
        action_noisy = noise * time[:, None, None] + action * (1 - time[:, None, None])
        proprio_model, action_noisy_model = self.base_model.action_space.preprocess(proprio, action_noisy)
        vlm_features, aux_features, action_state = self._encode_observation(
            input_ids, image_input, image_mask, proprio_model
        )
        prediction = self._predict(
            domain_id=domain_id,
            vlm_features=vlm_features,
            aux_features=aux_features,
            action_with_noise=action_noisy_model,
            proprio=proprio_model,
            time=time,
            action_state=action_state,
        )
        return self.base_model.action_space.compute_loss(prediction, action)

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: Tensor,
        image_input: Tensor,
        image_mask: Tensor,
        domain_id: Tensor,
        proprio: Tensor,
        steps: int = 10,
        noise: Tensor | None = None,
    ) -> Tensor:
        self.last_metrics = {}
        self.eval()
        steps = max(1, int(steps))
        batch_size = input_ids.shape[0]
        dim_action = self.base_model.action_space.dim_action
        if noise is None:
            noise = torch.randn(
                batch_size,
                self.base_model.num_actions,
                dim_action,
                device=proprio.device,
                dtype=proprio.dtype,
            )
        proprio_model, _ = self.base_model.action_space.preprocess(proprio, noise)
        vlm_features, aux_features, action_state = self._encode_observation(
            input_ids, image_input, image_mask, proprio_model
        )
        # Sampling and observation encoding stay outside the reusable denoising graph.
        denoise = self._inference_graph if self._inference_graph is not None else self._denoise_actions
        action = denoise(noise, proprio, domain_id, vlm_features, aux_features, action_state, steps)
        return self.base_model.action_space.postprocess(action)

    def _denoise_actions(
        self,
        noise: Tensor,
        proprio: Tensor,
        domain_id: Tensor,
        vlm_features: Tensor,
        aux_features: Tensor,
        action_state: dict[str, Tensor],
        steps: int,
    ) -> Tensor:
        batch_size = proprio.shape[0]
        action = torch.zeros_like(noise)
        for step in range(steps, 0, -1):
            time = torch.full((batch_size,), step / steps, device=proprio.device, dtype=proprio.dtype)
            action_noisy = noise * time[:, None, None] + action * (1 - time[:, None, None])
            proprio_model, action_noisy_model = self.base_model.action_space.preprocess(proprio, action_noisy)
            action = self._predict(
                domain_id=domain_id,
                vlm_features=vlm_features,
                aux_features=aux_features,
                action_with_noise=action_noisy_model,
                proprio=proprio_model,
                time=time,
                action_state=action_state,
            )
        return action
