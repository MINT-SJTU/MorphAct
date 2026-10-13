from __future__ import annotations

from dataclasses import dataclass

from morphact.factors import batched_lora_linear
from morphact.factors import get_lora_factors
import torch
from torch import Tensor
import torch.utils.checkpoint
from transformers.models.gemma import modeling_gemma

from .config import GeneratorConfig


@dataclass(frozen=True, slots=True)
class PrefixEncoding:
    final_tokens: Tensor | None
    key_values: tuple[tuple[Tensor, Tensor], ...]


class Pi0MorphActExecutor:
    """Run Pi0 with optional dynamic LoRA in the VLM and action expert."""

    def __init__(
        self,
        joint_model,
        action_config: GeneratorConfig,
        *,
        vlm_config: GeneratorConfig | None = None,
    ) -> None:
        self.joint_model = joint_model
        self.action_config = action_config
        self.vlm_config = vlm_config
        self.gradient_checkpointing = False
        self.training = False
        self.linear = batched_lora_linear

    @property
    def vlm(self):
        return self.joint_model.paligemma.language_model

    @property
    def expert(self):
        return self.joint_model.gemma_expert.model

    @staticmethod
    def _base_attention_projections(hidden_states, attention):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)
        query = attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key = attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        return query, key, value

    def _adapted_attention_projections(
        self,
        hidden_states,
        attention,
        layer_index,
        generated_state,
        config,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)
        query = (
            self.linear(
                hidden_states,
                attention.q_proj,
                get_lora_factors(generated_state, config, layer_index, "self_attn.q_proj"),
            )
            .view(hidden_shape)
            .transpose(1, 2)
        )
        key = (
            self.linear(
                hidden_states,
                attention.k_proj,
                get_lora_factors(generated_state, config, layer_index, "self_attn.k_proj"),
            )
            .view(hidden_shape)
            .transpose(1, 2)
        )
        value = (
            self.linear(
                hidden_states,
                attention.v_proj,
                get_lora_factors(generated_state, config, layer_index, "self_attn.v_proj"),
            )
            .view(hidden_shape)
            .transpose(1, 2)
        )
        return query, key, value

    @staticmethod
    def _rotary(rotary_embedding, query, key, position_ids):
        dummy = torch.zeros(query.shape[0], query.shape[2], query.shape[-1], device=query.device, dtype=query.dtype)
        cos, sin = rotary_embedding(dummy, position_ids)
        return modeling_gemma.apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=1)

    @staticmethod
    def _attention_output(attention, query, key, value, attention_mask):
        output, _ = modeling_gemma.eager_attention_forward(
            attention,
            query,
            key,
            value,
            attention_mask,
            attention.scaling,
            dropout=0.0,
        )
        return output.reshape(query.shape[0], query.shape[2], -1)

    def _adapted_mlp(self, hidden_states, mlp, layer_index, generated_state, config):
        gate_output = self.linear(
            hidden_states,
            mlp.gate_proj,
            get_lora_factors(generated_state, config, layer_index, "mlp.gate_proj"),
        )
        up_output = self.linear(
            hidden_states,
            mlp.up_proj,
            get_lora_factors(generated_state, config, layer_index, "mlp.up_proj"),
        )
        mlp_output = mlp.act_fn(gate_output) * up_output
        return self.linear(
            mlp_output,
            mlp.down_proj,
            get_lora_factors(generated_state, config, layer_index, "mlp.down_proj"),
        )

    def _use_gradient_checkpointing(self) -> bool:
        return self.gradient_checkpointing and self.training and torch.is_grad_enabled()

    def _encode_layer(
        self,
        hidden_states: Tensor,
        layer,
        layer_index: int,
        attention_mask: Tensor,
        position_ids: Tensor,
        generated_state: dict[str, Tensor] | None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        residual = hidden_states
        normalized, gate = layer.input_layernorm(hidden_states, cond=None)
        if generated_state is None:
            query, key, value = self._base_attention_projections(normalized, layer.self_attn)
        else:
            query, key, value = self._adapted_attention_projections(
                normalized,
                layer.self_attn,
                layer_index,
                generated_state,
                self.vlm_config,
            )
        query, key = self._rotary(self.vlm.rotary_emb, query, key, position_ids)
        attention_output = self._attention_output(layer.self_attn, query, key, value, attention_mask)
        if generated_state is None:
            attention_output = layer.self_attn.o_proj(attention_output)
        else:
            attention_output = self.linear(
                attention_output,
                layer.self_attn.o_proj,
                get_lora_factors(generated_state, self.vlm_config, layer_index, "self_attn.o_proj"),
            )
        hidden_states = modeling_gemma._gated_residual(residual, attention_output, gate)  # noqa: SLF001

        residual = hidden_states
        normalized, gate = layer.post_attention_layernorm(hidden_states, cond=None)
        if generated_state is None:
            mlp_output = layer.mlp(normalized.to(dtype=layer.mlp.up_proj.weight.dtype))
        else:
            mlp_output = self._adapted_mlp(
                normalized,
                layer.mlp,
                layer_index,
                generated_state,
                self.vlm_config,
            )
        hidden_states = modeling_gemma._gated_residual(residual, mlp_output, gate)  # noqa: SLF001
        return hidden_states, key, value

    def _decode_layer(
        self,
        hidden_states: Tensor,
        prefix_key: Tensor,
        prefix_value: Tensor,
        layer,
        layer_index: int,
        attention_mask: Tensor,
        position_ids: Tensor,
        adarms_cond: Tensor | None,
        generated_state: dict[str, Tensor] | None,
    ) -> Tensor:
        residual = hidden_states
        normalized, gate = layer.input_layernorm(hidden_states, cond=adarms_cond)
        if generated_state is None:
            query, key, value = self._base_attention_projections(normalized, layer.self_attn)
        else:
            query, key, value = self._adapted_attention_projections(
                normalized,
                layer.self_attn,
                layer_index,
                generated_state,
                self.action_config,
            )
        query, key = self._rotary(self.expert.rotary_emb, query, key, position_ids)
        key = torch.cat((prefix_key, key), dim=2)
        value = torch.cat((prefix_value, value), dim=2)
        attention_output = self._attention_output(layer.self_attn, query, key, value, attention_mask)
        if generated_state is None:
            attention_output = layer.self_attn.o_proj(attention_output)
        else:
            attention_output = self.linear(
                attention_output,
                layer.self_attn.o_proj,
                get_lora_factors(generated_state, self.action_config, layer_index, "self_attn.o_proj"),
            )
        hidden_states = modeling_gemma._gated_residual(residual, attention_output, gate)  # noqa: SLF001

        residual = hidden_states
        normalized, gate = layer.post_attention_layernorm(hidden_states, cond=adarms_cond)
        if generated_state is None:
            mlp_output = layer.mlp(normalized.to(dtype=layer.mlp.up_proj.weight.dtype))
        else:
            mlp_output = self._adapted_mlp(
                normalized,
                layer.mlp,
                layer_index,
                generated_state,
                self.action_config,
            )
        return modeling_gemma._gated_residual(residual, mlp_output, gate)  # noqa: SLF001

    def encode_prefix(
        self,
        prefix_embeddings: Tensor,
        attention_mask: Tensor,
        position_ids: Tensor,
        generated_state: dict[str, Tensor] | None = None,
    ) -> PrefixEncoding:
        if generated_state is not None and self.vlm_config is None:
            raise ValueError("VLM generated state requires a VLM MORPHACT config")
        hidden_states = prefix_embeddings
        key_values = []
        for layer_index, layer in enumerate(self.vlm.layers):
            if self._use_gradient_checkpointing():

                def encode_layer(hidden_states, layer=layer, layer_index=layer_index):
                    return self._encode_layer(
                        hidden_states,
                        layer,
                        layer_index,
                        attention_mask,
                        position_ids,
                        generated_state,
                    )

                hidden_states, key, value = torch.utils.checkpoint.checkpoint(
                    encode_layer,
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states, key, value = self._encode_layer(
                    hidden_states,
                    layer,
                    layer_index,
                    attention_mask,
                    position_ids,
                    generated_state,
                )
            key_values.append((key, value))

        final_tokens, _ = self.vlm.norm(hidden_states, cond=None)
        return PrefixEncoding(final_tokens=final_tokens, key_values=tuple(key_values))

    def decode_suffix(
        self,
        suffix_embeddings: Tensor,
        prefix_encoding: PrefixEncoding,
        attention_mask: Tensor,
        position_ids: Tensor,
        adarms_cond: Tensor | None,
        generated_state: dict[str, Tensor] | None,
    ) -> Tensor:
        hidden_states = suffix_embeddings
        for layer_index, (layer, prefix_key_value) in enumerate(
            zip(self.expert.layers, prefix_encoding.key_values, strict=True)
        ):
            prefix_key, prefix_value = prefix_key_value
            if self._use_gradient_checkpointing():

                def decode_layer(hidden_states, prefix_key, prefix_value, layer=layer, layer_index=layer_index):
                    return self._decode_layer(
                        hidden_states,
                        prefix_key,
                        prefix_value,
                        layer,
                        layer_index,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        generated_state,
                    )

                hidden_states = torch.utils.checkpoint.checkpoint(
                    decode_layer,
                    hidden_states,
                    prefix_key,
                    prefix_value,
                    use_reentrant=False,
                )
            else:
                hidden_states = self._decode_layer(
                    hidden_states,
                    prefix_key,
                    prefix_value,
                    layer,
                    layer_index,
                    attention_mask,
                    position_ids,
                    adarms_cond,
                    generated_state,
                )

        hidden_states, _ = self.expert.norm(hidden_states, cond=adarms_cond)
        return hidden_states
