from torch import Tensor, nn

from .config import GeneratorConfig
from .network import TransformerModel
from .tokenizer import FactorTokenizer


class StructuredGenerator(nn.Module):
    """Decode structured queries into per-projection low-rank factors."""

    def __init__(self, config: GeneratorConfig):
        super().__init__()
        if config.head_dim % 4:
            raise ValueError("head_dim must be divisible by 4 for two-axis rotary embeddings")
        self.config = config

        self.tokenizer = FactorTokenizer(
            token_dim=config.token_dim,
            rank=config.rank,
            alpha=config.alpha,
            pg_mapping=config.pg_mapping,
        )
        self.lora_A_token_count = int(self.tokenizer.lora_A_token_count)
        self.lora_B_token_count = int(self.tokenizer.lora_B_token_count)

        self.model = TransformerModel(
            d_model=config.d_model,
            num_base_model_layers=config.num_base_model_layers,
            num_token_per_layer=self.lora_A_token_count + self.lora_B_token_count,
            lora_rank=config.rank,
            output_dim=config.output_dim,
            head_dim=config.head_dim,
            num_blocks=config.num_pg_layers,
            dim_accumulation=config.dim_accumulation,
            lora_A_token_count=self.lora_A_token_count,
            lora_B_token_count=self.lora_B_token_count,
        )

        self.layer_num = config.num_base_model_layers
        self.prefix = config.prefix
        self.lora_rank = config.rank

        self.hidden_in = nn.Linear(config.input_dim, config.d_model)

    def forward(self, condition: Tensor) -> dict[str, Tensor]:
        embeddings = self.hidden_in(condition)

        # (batch_size, num_layers, token_num, token_size_0, token_size_1)
        output = self.model(encoder_hidden_states=embeddings)

        return self.decode_parameters(output)

    def decode_parameters(self, output: Tensor) -> dict[str, Tensor]:
        batch_size, layer_count = output.shape[:2]
        if layer_count != self.layer_num:
            raise ValueError("Generated parameter layer count does not match the model")
        decoded = self.tokenizer.detokenize(self.tokenizer.shape_state_dict, output.flatten(0, 1))
        layers = {key: value.unflatten(0, (batch_size, layer_count)).unbind(dim=1) for key, value in decoded.items()}
        result = {
            f"{self.prefix}{layer_index}.{key}": values[layer_index]
            for layer_index in range(layer_count)
            for key, values in layers.items()
        }
        result["scale"] = output.new_full((), self.tokenizer.lora_scale)
        return result
