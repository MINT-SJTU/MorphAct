from collections import OrderedDict

import numpy as np
import torch
from torch import Tensor


class FactorTokenizer:
    """Partition structured output tokens into ordered A/B parameter blocks."""

    def __init__(
        self,
        token_dim: int,
        rank: int,
        alpha: int,
        pg_mapping: dict[str, dict[str, int]],
    ):
        self.token_size = [rank, token_dim]
        assert rank > 0, "rank should be a positive integer."
        assert alpha > 0, "alpha should be a positive integer."
        self.rank = rank
        self.alpha = alpha

        self.pg_mapping = pg_mapping
        self._module_names = list(pg_mapping)

    def get_key_order(self, key: str) -> int:
        """Order all A blocks before B blocks, preserving projection order."""
        # Match the complete module name: head_1 must not match head_10.
        module_name, _, factor = key.rpartition(".")
        if factor == "lora_A":
            major_order = 0
        elif factor == "lora_B":
            major_order = 1
        else:
            raise ValueError(f"Key {key} does not have lora_A or lora_B.")
        try:
            minor_order = self._module_names.index(module_name)
        except ValueError as exc:
            raise ValueError(f"Key {key} does not match any known module in pg_mapping.") from exc

        return major_order * len(self._module_names) + minor_order

    def detokenize(self, shape_state_dict: dict[str, tuple[int, ...]], tokens: Tensor) -> OrderedDict[str, Tensor]:
        shape_state_dict = OrderedDict(sorted(shape_state_dict.items(), key=lambda item: self.get_key_order(item[0])))
        batch_size = tokens.shape[0]
        token_area = self.token_size[0] * self.token_size[1]
        parameter_counts = [int(np.prod(shape)) for shape in shape_state_dict.values()]
        token_counts = [(count + token_area - 1) // token_area for count in parameter_counts]
        segments = torch.split(tokens, token_counts, dim=1)
        result = OrderedDict()
        for (key, shape), count, segment in zip(shape_state_dict.items(), parameter_counts, segments, strict=True):
            flattened = segment.view(batch_size, -1)
            parameter = flattened if flattened.shape[1] == count else flattened.narrow(1, 0, count)
            result[key] = parameter.view(batch_size, *shape)
        return result

    @property
    def lora_scale(self) -> float:
        return self.alpha / self.rank

    @property
    def shape_state_dict(self) -> OrderedDict[str, tuple]:
        shape_state_dict = {}
        for module in self._module_names:
            dims = self.pg_mapping[module]
            lora_A_shape = (self.rank, dims["lora_A_dim"])
            lora_B_shape = (dims["lora_B_dim"], self.rank)
            shape_state_dict[f"{module}.lora_A"] = lora_A_shape
            shape_state_dict[f"{module}.lora_B"] = lora_B_shape

        shape_state_dict = OrderedDict(sorted(shape_state_dict.items(), key=lambda x: self.get_key_order(x[0])))
        return shape_state_dict

    def _count_tokens(self, prefix: str) -> int:
        token_area = self.token_size[0] * self.token_size[1]
        count = 0
        for key, shape in self.shape_state_dict.items():
            if prefix in key:
                param_count = np.prod(shape)
                count += (param_count + token_area - 1) // token_area
        return count

    @property
    def lora_A_token_count(self) -> int:
        return self._count_tokens("lora_A")

    @property
    def lora_B_token_count(self) -> int:
        return self._count_tokens("lora_B")
