import dataclasses

import einops
import numpy as np

from openpi import transforms

EVO1_DATASET_REPO_ID = "MINT-SJTU/Evo1_MetaWorld_Dataset"
EVO1_DATASET_REVISION = "612c4e83d0e3b4e0be5227970bd8f5f07fc0c488"
EVO1_VIDEO_BACKEND = "pyav"


def make_metaworld_example() -> dict:
    """Create an Evo-1 MetaWorld policy input example."""
    return {
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(4),
        "prompt": "complete the MetaWorld task",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class MetaWorldInputs(transforms.DataTransformFn):
    """Convert the Evo-1 MetaWorld image, state, and actions to OpenPI fields."""

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        inputs = {
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
            "state": np.asarray(data["observation/state"]),
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            prompt = data["prompt"]
            inputs["prompt"] = prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class MetaWorldOutputs(transforms.DataTransformFn):
    """Convert padded model actions back to the four-dimensional MetaWorld space."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :4]}
