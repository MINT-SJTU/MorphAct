from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import torch
from PIL import Image


def decode_image_from_bytes(value) -> Image.Image:
    if isinstance(value, np.ndarray) and value.ndim == 3 and value.shape[-1] == 3:
        if value.dtype != np.uint8:
            raise ValueError("RGB image arrays must have dtype uint8")
        return Image.fromarray(value)
    if isinstance(value, (bytes, bytearray)):
        value = np.frombuffer(value, dtype=np.uint8)
    rgb = cv2.imdecode(value, cv2.IMREAD_COLOR)
    if rgb is None:
        rgb = np.frombuffer(value, dtype=np.uint8)
        if rgb.size == 2764800:
            rgb = rgb.reshape(720, 1280, 3)
        elif rgb.size == 921600:
            rgb = rgb.reshape(480, 640, 3)
    return Image.fromarray(rgb)


def action_slice(
    abs_traj: torch.Tensor,
    idx_for_delta: Sequence[int] = (),
    idx_for_mask_proprio: Sequence[int] = (),
) -> dict[str, torch.Tensor]:
    if not isinstance(abs_traj, torch.Tensor):
        raise TypeError("abs_traj must be a torch.Tensor")
    if abs_traj.ndim != 2 or abs_traj.size(0) < 2:
        raise ValueError("abs_traj must be [H+1, D] with H>=1")

    proprio = abs_traj[0].clone()
    action = abs_traj[1:].clone()
    if idx_for_delta:
        idx = torch.as_tensor(idx_for_delta, dtype=torch.long, device=abs_traj.device)
        action[:, idx] -= proprio[idx]
    if idx_for_mask_proprio:
        idx = torch.as_tensor(idx_for_mask_proprio, dtype=torch.long, device=abs_traj.device)
        proprio[idx] = 0.0
    return {"proprio": proprio, "action": action}
