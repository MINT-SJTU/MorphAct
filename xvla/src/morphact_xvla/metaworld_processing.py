"""Versioned MetaWorld-only preprocessing; never used by the LIBERO policy."""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from .model import XVLAMorphAct


def validate_processing(config):
    if not isinstance(config, dict):
        raise ValueError("MetaWorld processing configuration must be an object")
    config = {key: value for key, value in config.items() if key != "stats_sha256"}
    expected = {
        "version",
        "normalization",
        "bounds",
        "mixed_tasks",
        "refresh_episodes",
        "mask_unused_actions",
    }
    if set(config) != expected or config["version"] != 2 or config["normalization"] != "minmax":
        raise ValueError("Unknown MetaWorld processing configuration")
    if config["mask_unused_actions"] is not True:
        raise ValueError("Normalized MetaWorld requires masked unused action dimensions")
    if not 1 <= config["refresh_episodes"] <= config["mixed_tasks"] <= 50:
        raise ValueError("Invalid mixed trajectory pool")
    for key in ("state", "action"):
        lo = np.asarray(config["bounds"][key]["min"], dtype=np.float64)
        hi = np.asarray(config["bounds"][key]["max"], dtype=np.float64)
        if lo.shape != (4,) or hi.shape != (4,) or not np.isfinite([lo, hi]).all() or not (hi > lo).all():
            raise ValueError("Expected finite, nondegenerate 4D bounds")
    return config


def verify_source_stats(config, root):
    path = Path(root) / "meta/episodes_stats.jsonl"
    rows = [json.loads(x)["stats"] for x in path.read_text().splitlines()]
    for key, source in [("state", "observation.state"), ("action", "action")]:
        for bound, fn in [("min", np.min), ("max", np.max)]:
            actual = fn(np.asarray([r[source][bound] for r in rows]), axis=0)
            np.testing.assert_array_equal(actual, config["bounds"][key][bound])


class BoundsNormalizer:
    def __init__(self, config):
        self.config = validate_processing(config)

    def normalize(self, value, key):
        value = np.asarray(value, dtype=np.float32)
        if value.shape[-1] != 4 or not np.isfinite(value).all():
            raise ValueError("Expected finite 4D values before normalization")
        low = np.asarray(self.config["bounds"][key]["min"], dtype=np.float32)
        high = np.asarray(self.config["bounds"][key]["max"], dtype=np.float32)
        return np.clip(2 * (value - low) / (high - low) - 1, -1, 1).astype(np.float32)

    def denormalize_action(self, value):
        if value.shape[-1] != 4:
            raise ValueError("Expected 4D action output")
        bounds = self.config["bounds"]["action"]
        if torch.is_tensor(value):
            lo = torch.as_tensor(bounds["min"], device=value.device, dtype=torch.float32)
            hi = torch.as_tensor(bounds["max"], device=value.device, dtype=torch.float32)
            return (value.float() + 1) * 0.5 * (hi - lo) + lo
        lo = np.asarray(bounds["min"], dtype=np.float32)
        hi = np.asarray(bounds["max"], dtype=np.float32)
        return (np.asarray(value, dtype=np.float32) + 1) * 0.5 * (hi - lo) + lo


class MetaWorldPolicy(XVLAMorphAct):
    """Mask only clean padding predictions, retaining the existing t*noise term."""

    def _predict(self, **kwargs):
        prediction = super()._predict(**kwargs)
        space = self.base_model.action_space
        if self.base_model.action_mode != "auto" or space.real_dim != 4 or prediction.shape[-1] != 20:
            raise ValueError("This policy subclass is only for 4D MetaWorld auto actions")
        return torch.cat((prediction[..., :4], torch.zeros_like(prediction[..., 4:])), dim=-1)
