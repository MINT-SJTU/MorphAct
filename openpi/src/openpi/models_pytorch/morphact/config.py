from morphact import GeneratorConfig
from morphact import ProjectionSpec

VISION_MODULE_NAMES = (
    "paligemma_with_expert.paligemma.model.vision_tower",
    "paligemma_with_expert.paligemma.model.multi_modal_projector",
)
ACTION_INTERFACE_MODULE_NAMES = (
    "state_proj",
    "action_in_proj",
    "action_time_mlp_in",
    "action_time_mlp_out",
    "action_out_proj",
)
PI05_ACTION_INTERFACE_MODULE_NAMES = (
    "action_in_proj",
    "time_mlp_in",
    "time_mlp_out",
    "action_out_proj",
)


def action_interface_module_names(*, pi05: bool = False) -> tuple[str, ...]:
    return PI05_ACTION_INTERFACE_MODULE_NAMES if pi05 else ACTION_INTERFACE_MODULE_NAMES


def adaptation_module_names(*, pi05: bool = False) -> tuple[str, ...]:
    return (
        "vlm_generator",
        "action_generator",
        *(f"base_model.{name}" for name in (*VISION_MODULE_NAMES, *action_interface_module_names(pi05=pi05))),
    )


ACTION_MORPHACT_CONFIG = GeneratorConfig(
    prefix="expert.layers.",
    projections=(
        ProjectionSpec("self_attn.q_proj", 1024, 2048, factor_geometry="q", num_heads=8, head_dim=256),
        ProjectionSpec("self_attn.k_proj", 1024, 256, factor_geometry="kv", head_dim=256),
        ProjectionSpec("self_attn.v_proj", 1024, 256, factor_geometry="kv", head_dim=256),
        ProjectionSpec("self_attn.o_proj", 2048, 1024, factor_geometry="o", num_heads=8, head_dim=256),
        ProjectionSpec("mlp.gate_proj", 1024, 4096),
        ProjectionSpec("mlp.up_proj", 1024, 4096),
        ProjectionSpec("mlp.down_proj", 4096, 1024),
    ),
    state_dim=32,
    rank=32,
    alpha=32,
)

VLM_MORPHACT_CONFIG = GeneratorConfig(
    prefix="vlm.layers.",
    projections=(
        ProjectionSpec("self_attn.q_proj", 2048, 2048, factor_geometry="q", num_heads=8, head_dim=256),
        ProjectionSpec("self_attn.k_proj", 2048, 256, factor_geometry="kv", head_dim=256),
        ProjectionSpec("self_attn.v_proj", 2048, 256, factor_geometry="kv", head_dim=256),
        ProjectionSpec("self_attn.o_proj", 2048, 2048, factor_geometry="o", num_heads=8, head_dim=256),
        ProjectionSpec("mlp.gate_proj", 2048, 16384),
        ProjectionSpec("mlp.up_proj", 2048, 16384),
        ProjectionSpec("mlp.down_proj", 16384, 2048),
    ),
    num_pg_layers=12,
    alpha=16,
)
