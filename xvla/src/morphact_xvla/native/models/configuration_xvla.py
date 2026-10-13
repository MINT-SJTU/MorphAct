from .configuration_florence2 import Florence2Config
from transformers.configuration_utils import PretrainedConfig


class XVLAConfig(PretrainedConfig):
    """
    Configuration class for the **XVLA (Extended Vision-Language-Action)** model.

    This configuration defines all submodules of XVLA in a single place:
      - The visual-language backbone (Florence2)
      - The temporal/action transformer
      - The action/proprio setup
    """

    model_type = "xvla"

    def __init__(
        # === Florence backbone ===
        self,
        florence_config: dict | None = None,

        # === Transformer head ===
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_domains: int = 30,
        len_soft_prompts: int = 32,
        dim_time: int = 32,
        max_len_seq: int = 512,
        use_hetero_proj: bool = False,
        soft_prompt_length: int = 32,

        # === Action & proprio ===
        max_action_dim: int = 20,  # Maximum action dimension for padding (used by "auto" action mode)
        real_action_dim: int = 20,
        num_actions: int = 30,
        action_mode: str = "ee6d",
        use_proprio: bool = True,

        **kwargs,
    ):
        # Florence2 backbone configuration
        if isinstance(florence_config, dict):
            self.florence_config = Florence2Config(**florence_config)
        elif isinstance(florence_config, Florence2Config):
            self.florence_config = florence_config
        else:
            self.florence_config = Florence2Config()

        # Transformer hyperparameters
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_domains = num_domains
        self.len_soft_prompts = len_soft_prompts
        self.dim_time = dim_time
        self.max_len_seq = max_len_seq
        self.use_hetero_proj = use_hetero_proj
        self.soft_prompt_length = soft_prompt_length

        # Action/proprioception settings
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.use_proprio = use_proprio

        self.real_action_dim = real_action_dim
        self.max_action_dim = max_action_dim

        # Initialize base HF config attributes (e.g. name_or_path)
        super().__init__(**kwargs)

    # -------------------------------------------------------------------------
    # Serialization helpers
    # -------------------------------------------------------------------------
    def to_dict(self):
        """
        Convert this configuration (and its Florence sub-config)
        into a fully serializable dictionary for HF save/load.
        """
        output = super().to_dict()
        output["florence_config"] = self.florence_config.to_dict()
        return output
