from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any

import safetensors.torch
import torch

from openpi.models import pi0_config
from openpi.models_pytorch.morphact.config import GeneratorConfig
from openpi.models_pytorch.morphact.pi0_morphact import Pi0MorphAct
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.policies import policy
from openpi.shared import download
import openpi.shared.normalize as normalize
from openpi.training import config as training_config
from openpi.training.morphact_checkpoint import METHOD
from openpi.training.morphact_checkpoint import load_morphact_checkpoint
from openpi.training.morphact_checkpoint import read_metadata
from openpi.training.morphact_checkpoint import verify_base_checkpoint
import openpi.transforms as transforms


def _validate_pi0_config(model_config: pi0_config.Pi0Config) -> None:
    if (
        "lora" in str(model_config.paligemma_variant).lower()
        or "lora" in str(model_config.action_expert_variant).lower()
    ):
        raise ValueError("Single-stage generated-parameter policy requires a non-LoRA Pi0 or Pi0.5 model")


def create_morphact_policy(
    train_config: training_config.TrainConfig,
    morphact_checkpoint: str | Path,
    *,
    base_checkpoint: str | Path | None = None,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    device: str | None = None,
    inference_acceleration: bool = True,
) -> policy.Policy:
    morphact_checkpoint = Path(download.maybe_download(str(morphact_checkpoint)))
    metadata = read_metadata(morphact_checkpoint)
    if metadata["method"] != METHOD:
        raise ValueError("Unsupported adapter checkpoint method")
    if not isinstance(train_config.model, pi0_config.Pi0Config):
        raise ValueError("Single-stage generated-parameter policy requires a Pi0Config")
    _validate_pi0_config(train_config.model)
    model_type = "pi05" if train_config.model.pi05 else "pi0"
    if metadata["model_type"] != model_type:
        raise ValueError("Checkpoint model_type does not match the policy model config")

    if base_checkpoint is None:
        raise ValueError("base_checkpoint is required to load the adapter's base weights")
    base_checkpoint = verify_base_checkpoint(metadata, base_checkpoint)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_config = dataclasses.replace(
        train_config.model,
        dtype="bfloat16" if device.startswith("cuda") else "float32",
        pytorch_compile_mode=None,
    )
    logging.info("Loading %s base from %s", model_type, base_checkpoint)
    base_model = PI0Pytorch(model_config).to(device)
    safetensors.torch.load_model(base_model, base_checkpoint, device=device)
    model = Pi0MorphAct(
        base_model,
        vlm_config=GeneratorConfig.from_dict(metadata["vlm_generator_config"]),
        action_config=GeneratorConfig.from_dict(metadata["action_generator_config"]),
        gradient_checkpointing=True,
    ).to(device)
    load_morphact_checkpoint(
        morphact_checkpoint,
        model=model,
        restore_rng=False,
    )
    model.eval()
    model.configure_inference(inference_acceleration)

    data_config = train_config.data.create(train_config.assets_dirs, model_config)
    if norm_stats is None:
        asset_id = metadata.get("norm_stats_asset_id") or data_config.asset_id
        if asset_id is None:
            raise ValueError("Asset id is required to load adapter normalization statistics")
        norm_stats = normalize.load(morphact_checkpoint / "assets" / asset_id)
    repack_transforms = repack_transforms or transforms.Group()
    policy_metadata = {
        **(train_config.policy_metadata or {}),
        "method": METHOD,
        "model_type": model_type,
        "generator_type": METHOD,
        "vlm_generator_rank": model.vlm_generator_config.rank,
        "action_generator_rank": model.action_generator_config.rank,
        "action_horizon": model_config.action_horizon,
        "action_dim": model_config.action_dim,
        "vlm_target_count": len(model.vlm_target_manifest),
        "action_target_count": len(model.action_target_manifest),
        "inference_acceleration": inference_acceleration,
    }
    return policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=policy_metadata,
        is_pytorch=True,
        pytorch_device=device,
    )
