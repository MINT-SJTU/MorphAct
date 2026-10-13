from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from morphact.config import GeneratorConfig
from morphact.factors import LowRankFactors, batched_lora_linear, get_lora_factors

from .native.models.transformer import timestep_embedding


def _headwise_delta(inputs: Tensor, factors: LowRankFactors) -> Tensor:
    """Return the FP32 LoRA residual for one logical projection."""
    projection = factors.projection
    with torch.autocast(device_type=inputs.device.type, enabled=False):
        lora_a = factors.lora_a.to(device=inputs.device, dtype=torch.float32)
        lora_b = factors.lora_b.to(device=inputs.device, dtype=torch.float32)
        scale = factors.scale.to(device=inputs.device, dtype=torch.float32)
        fp32_inputs = inputs.float()
        if projection.factor_geometry in {"q", "kv"}:
            hidden = torch.einsum("b...i,bhir->b...hr", fp32_inputs, lora_a)
            delta = torch.einsum("b...hr,bhro->b...ho", hidden, lora_b).flatten(start_dim=-2)
        elif projection.factor_geometry == "o":
            head_inputs = fp32_inputs.unflatten(-1, (projection.num_heads, projection.head_dim))
            hidden = torch.einsum("b...hi,bhir->b...hr", head_inputs, lora_a)
            delta = torch.einsum("b...hr,bhro->b...o", hidden, lora_b)
        else:
            hidden = torch.einsum("b...i,bir->b...r", fp32_inputs, lora_a)
            delta = torch.einsum("b...r,bro->b...o", hidden, lora_b)
        return delta * scale


class XVLAMorphActExecutor:
    """Explicit MORPHACT execution over native Florence and action Transformer modules."""

    def __init__(self, base_model: nn.Module, vlm_config: GeneratorConfig, action_config: GeneratorConfig) -> None:
        self.base_model = base_model
        self.vlm_config = vlm_config
        self.action_config = action_config
        self.training = False
        self.linear = batched_lora_linear
        self.delta = _headwise_delta

    def _florence_attention(
        self,
        hidden_states: Tensor,
        attention: nn.Module,
        layer_index: int,
        attention_mask: Tensor | None,
        state: dict[str, Tensor],
    ) -> Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        q = self.linear(
            hidden_states,
            attention.q_proj,
            get_lora_factors(state, self.vlm_config, layer_index, "self_attn.q_proj"),
        )
        k = self.linear(
            hidden_states,
            attention.k_proj,
            get_lora_factors(state, self.vlm_config, layer_index, "self_attn.k_proj"),
        )
        v = self.linear(
            hidden_states,
            attention.v_proj,
            get_lora_factors(state, self.vlm_config, layer_index, "self_attn.v_proj"),
        )
        q = q.view(batch_size, sequence_length, attention.num_heads, attention.head_dim).transpose(1, 2)
        k = k.view(batch_size, sequence_length, attention.num_heads, attention.head_dim).transpose(1, 2)
        v = v.view(batch_size, sequence_length, attention.num_heads, attention.head_dim).transpose(1, 2)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=attention.dropout if attention.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, attention.embed_dim)
        return self.linear(
            output,
            attention.out_proj,
            get_lora_factors(state, self.vlm_config, layer_index, "self_attn.out_proj"),
        )

    def encode_florence(
        self,
        inputs_embeds: Tensor,
        attention_mask: Tensor,
        state: dict[str, Tensor],
    ) -> Tensor:
        encoder = self.base_model.vlm.language_model.model.encoder
        # Native forward_vlm passes an all-valid mask returned by Florence's merge helper.
        inputs_embeds = inputs_embeds.to(dtype=encoder.layernorm_embedding.weight.dtype)
        position_input = inputs_embeds[:, :, -1]
        positions = encoder.embed_positions(position_input).to(inputs_embeds.device)
        hidden_states = encoder.layernorm_embedding(inputs_embeds + positions)
        hidden_states = F.dropout(hidden_states, p=encoder.dropout, training=encoder.training)
        mask = None
        if attention_mask is not None:
            mask = (1.0 - attention_mask[:, None, None, :].to(dtype=inputs_embeds.dtype)) * torch.finfo(
                inputs_embeds.dtype
            ).min
            mask = mask.expand(-1, 1, inputs_embeds.shape[1], -1)

        for layer_index, layer in enumerate(encoder.layers):
            residual = hidden_states
            attention_output = self._florence_attention(hidden_states, layer.self_attn, layer_index, mask, state)
            attention_output = F.dropout(attention_output, p=layer.dropout, training=layer.training)
            hidden_states = layer.self_attn_layer_norm(residual + attention_output)

            residual = hidden_states
            hidden_states = self.linear(
                hidden_states,
                layer.fc1,
                get_lora_factors(state, self.vlm_config, layer_index, "fc1"),
            )
            hidden_states = layer.activation_fn(hidden_states)
            hidden_states = F.dropout(hidden_states, p=layer.activation_dropout, training=layer.training)
            hidden_states = self.linear(
                hidden_states,
                layer.fc2,
                get_lora_factors(state, self.vlm_config, layer_index, "fc2"),
            )
            hidden_states = F.dropout(hidden_states, p=layer.dropout, training=layer.training)
            hidden_states = layer.final_layer_norm(residual + hidden_states)
        return hidden_states

    def _action_attention(
        self,
        hidden_states: Tensor,
        attention: nn.Module,
        layer_index: int,
        state: dict[str, Tensor],
    ) -> Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        # Keep the native fused base projection, then add independently generated Q/K/V residuals.
        qkv = attention.qkv(hidden_states).float()
        residuals = []
        for suffix in ("attn.q", "attn.k", "attn.v"):
            factors = get_lora_factors(state, self.action_config, layer_index, suffix)
            residuals.append(self.delta(hidden_states, factors))
        qkv = qkv + torch.cat(residuals, dim=-1)
        qkv = qkv.to(dtype=hidden_states.dtype)
        qkv = qkv.reshape(batch_size, sequence_length, 3, attention.num_heads, attention.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = attention.q_norm(q), attention.k_norm(k)
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=attention.attn_drop.p if attention.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(batch_size, sequence_length, hidden_size)
        output = self.linear(
            output,
            attention.proj,
            get_lora_factors(state, self.action_config, layer_index, "attn.proj"),
        )
        return attention.proj_drop(output)

    def _action_block(
        self,
        hidden_states: Tensor,
        block: nn.Module,
        layer_index: int,
        state: dict[str, Tensor],
    ) -> Tensor:
        hidden_states = hidden_states.to(dtype=block.norm1.weight.dtype)
        hidden_states = hidden_states + self._action_attention(
            block.norm1(hidden_states), block.attn, layer_index, state
        )
        mlp_input = block.norm2(hidden_states)
        mlp_output = self.linear(
            mlp_input,
            block.mlp.fc1,
            get_lora_factors(state, self.action_config, layer_index, "mlp.fc1"),
        )
        mlp_output = block.mlp.act(mlp_output)
        mlp_output = block.mlp.drop1(mlp_output)
        mlp_output = block.mlp.norm(mlp_output)
        mlp_output = self.linear(
            mlp_output,
            block.mlp.fc2,
            get_lora_factors(state, self.action_config, layer_index, "mlp.fc2"),
        )
        mlp_output = block.mlp.drop2(mlp_output)
        return hidden_states + mlp_output

    def decode_action(
        self,
        *,
        domain_id: Tensor,
        vlm_features: Tensor,
        aux_visual_inputs: Tensor,
        action_with_noise: Tensor,
        proprio: Tensor,
        time: Tensor,
        state: dict[str, Tensor],
    ) -> Tensor:
        transformer = self.base_model.transformer
        batch_size, num_actions = action_with_noise.shape[:2]
        time_embedding = timestep_embedding(time, transformer.dim_time)
        time_tokens = time_embedding[:, None, :].expand(batch_size, num_actions, -1)
        proprio_tokens = proprio[:, None, :].expand(batch_size, num_actions, -1)
        action_tokens = torch.cat((action_with_noise, proprio_tokens, time_tokens), dim=-1)
        hidden_states = transformer.action_encoder(action_tokens.float(), domain_id)
        if transformer.use_hetero_proj:
            vlm_tokens = transformer.vlm_proj(vlm_features.float(), domain_id)
            aux_tokens = transformer.aux_visual_proj(aux_visual_inputs.float(), domain_id)
        else:
            vlm_tokens = transformer.vlm_proj(vlm_features.float())
            aux_tokens = transformer.aux_visual_proj(aux_visual_inputs.float())
        hidden_states = torch.cat((hidden_states, vlm_tokens, aux_tokens), dim=1)
        sequence_length = hidden_states.shape[1]
        if sequence_length > transformer.pos_emb.shape[1]:
            raise ValueError(f"Sequence length {sequence_length} exceeds max_len_seq={transformer.pos_emb.shape[1]}")
        hidden_states = hidden_states + transformer.pos_emb[:, :sequence_length]
        if transformer.len_soft_prompts > 0:
            prompts = transformer.soft_prompt_hub(domain_id).view(
                batch_size, transformer.len_soft_prompts, transformer.hidden_size
            )
            hidden_states = torch.cat((hidden_states, prompts), dim=1)
        for layer_index, block in enumerate(transformer.blocks):
            hidden_states = self._action_block(hidden_states, block, layer_index, state)
        decoded = transformer.norm(hidden_states[:, :num_actions]).float()
        return transformer.action_decoder(decoded, domain_id)
