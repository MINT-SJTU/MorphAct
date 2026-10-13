from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from dataclasses import field
import math
import os
from pathlib import Path
from typing import Literal

from openpi.models import pi0_config
from openpi.training import config as training_config
from openpi.training import optimizer as optimizer_config

LIBERO_SPATIAL_TASK_INDICES = tuple(range(30, 40))
LIBERO_FULL_TASK_INDICES = tuple(range(40))
METAWORLD_TASK_INDICES = tuple(range(50))


def _artifact_dir(name: str) -> str:
    repository = Path(__file__).resolve().parents[4]
    root = Path(os.environ.get("MORPHACT_ARTIFACT_ROOT", repository / "artifacts"))
    return str(root / name)


@dataclass(frozen=True, slots=True)
class MorphActTrainSettings:
    base_checkpoint: str
    config_name: str = "pi0_libero"
    exp_name: str = "morphact300m-libero-50k-gb64"
    task_indices: tuple[int, ...] | None = None
    assets_base_dir: str = field(default_factory=lambda: _artifact_dir("openpi/assets"))
    checkpoint_base_dir: str = field(default_factory=lambda: _artifact_dir("checkpoints"))
    num_devices: int = 4
    per_device_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    num_workers: int = 8
    loader_workers_override: int | None = None
    num_train_steps: int = 50_000
    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_lr: float = 2.5e-6
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 1e-10
    max_grad_norm: float = 1.0
    explosion_grad_norm: float = 1e6
    log_interval: int = 10
    save_interval: int = 2_500
    seed: int = 42
    sampler_seed: int = 0
    precision: Literal["bfloat16", "float32"] = "bfloat16"
    morphact_compute_precision: Literal["bfloat16", "float32"] = "bfloat16"
    compile_morphact_pointwise: bool = False
    morphact_width: int = 1024
    vlm_morphact_layers: int = 8
    action_morphact_layers: int = 6
    gradient_checkpointing: bool = True
    resume_checkpoint: str | None = None
    resume_norm_stats_source: Literal["checkpoint", "assets"] = "checkpoint"
    allow_global_batch_change: bool = False
    overwrite: bool = False
    swanlab_enabled: bool = True
    swanlab_mode: Literal["local", "cloud"] = "local"
    swanlab_project: str = "morphact"
    swanlab_workspace: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.base_checkpoint, str) or not self.base_checkpoint.strip():
            raise ValueError("base_checkpoint is required")
        positive_integers = {
            "num_devices": self.num_devices,
            "per_device_batch_size": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_train_steps": self.num_train_steps,
            "warmup_steps": self.warmup_steps,
            "log_interval": self.log_interval,
            "save_interval": self.save_interval,
            "morphact_width": self.morphact_width,
            "vlm_morphact_layers": self.vlm_morphact_layers,
            "action_morphact_layers": self.action_morphact_layers,
        }
        invalid_integers = [
            name
            for name, value in positive_integers.items()
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ]
        if invalid_integers:
            raise ValueError(f"MORPHACT training integer values must be positive: {invalid_integers}")
        if self.loader_workers_override is not None and self.loader_workers_override < 0:
            raise ValueError("loader_workers_override must be nonnegative")
        if isinstance(self.sampler_seed, bool) or not isinstance(self.sampler_seed, int) or self.sampler_seed < 0:
            raise ValueError("sampler_seed must be a non-negative integer")
        if self.morphact_width % 128:
            raise ValueError("morphact_width must be divisible by the unchanged MORPHACT head_dim 128")
        if self.num_workers < 0 or isinstance(self.num_workers, bool) or not isinstance(self.num_workers, int):
            raise ValueError("num_workers must be a non-negative integer")
        if self.warmup_steps >= self.num_train_steps:
            raise ValueError("warmup_steps must be smaller than num_train_steps")

        positive_floats = {
            "peak_lr": self.peak_lr,
            "decay_lr": self.decay_lr,
            "adam_eps": self.adam_eps,
            "max_grad_norm": self.max_grad_norm,
            "explosion_grad_norm": self.explosion_grad_norm,
        }
        invalid_floats = [
            name
            for name, value in positive_floats.items()
            if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(value) or value <= 0
        ]
        if invalid_floats:
            raise ValueError(f"MORPHACT training floating-point values must be finite and positive: {invalid_floats}")
        if (
            not isinstance(self.weight_decay, int | float)
            or isinstance(self.weight_decay, bool)
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("weight_decay must be finite and non-negative")
        if not 0 <= self.adam_beta1 < 1 or not 0 <= self.adam_beta2 < 1:
            raise ValueError("Adam betas must be in [0, 1)")
        if self.precision not in {"bfloat16", "float32"}:
            raise ValueError("precision must be 'bfloat16' or 'float32'")
        if self.morphact_compute_precision not in {"bfloat16", "float32"}:
            raise ValueError("morphact_compute_precision must be 'bfloat16' or 'float32'")
        if not isinstance(self.gradient_checkpointing, bool):
            raise ValueError("gradient_checkpointing must be a boolean")
        if not isinstance(self.compile_morphact_pointwise, bool):
            raise ValueError("compile_morphact_pointwise must be a boolean")
        if self.swanlab_mode not in {"local", "cloud"}:
            raise ValueError("swanlab_mode must be 'local' or 'cloud'")
        if self.resume_checkpoint is not None and self.overwrite:
            raise ValueError("Cannot resume and overwrite together")
        if self.resume_norm_stats_source not in {"checkpoint", "assets"}:
            raise ValueError("resume_norm_stats_source must be 'checkpoint' or 'assets'")
        if not isinstance(self.allow_global_batch_change, bool):
            raise ValueError("allow_global_batch_change must be a boolean")

    @property
    def effective_batch_size(self) -> int:
        return self.num_devices * self.per_device_batch_size * self.gradient_accumulation_steps

    def resume_contract(self) -> dict:
        contract = dataclasses.asdict(self)
        if self.task_indices is None and self.config_name == "pi0_libero_spatial":
            contract["task_indices"] = LIBERO_SPATIAL_TASK_INDICES
        for key in (
            "base_checkpoint",
            "assets_base_dir",
            "checkpoint_base_dir",
            "exp_name",
            "resume_checkpoint",
            "resume_norm_stats_source",
            "allow_global_batch_change",
            "overwrite",
            "swanlab_enabled",
            "swanlab_mode",
            "swanlab_project",
            "swanlab_workspace",
            "log_interval",
            "save_interval",
            "num_workers",
            "loader_workers_override",
            "compile_morphact_pointwise",
        ):
            contract.pop(key)
        return contract

    def checkpoint_contract(self) -> dict:
        return self.resume_contract()

    def create_openpi_config(self) -> training_config.TrainConfig:
        config = training_config.get_config(self.config_name)
        if not isinstance(config.model, pi0_config.Pi0Config):
            raise ValueError("Single-stage MORPHACT requires a Pi0Config")
        if (
            "lora" in str(config.model.paligemma_variant).lower()
            or "lora" in str(config.model.action_expert_variant).lower()
        ):
            raise ValueError("Single-stage MORPHACT requires the non-LoRA Pi0 model")
        data = config.data
        if self.task_indices is not None:
            if not isinstance(
                data,
                training_config.LeRobotLiberoDataConfig | training_config.LeRobotMetaWorldDataConfig,
            ):
                raise ValueError(
                    "task_indices require a supported LeRobot data config; set task_indices=None otherwise"
                )
            data = dataclasses.replace(data, task_indices=self.task_indices)
        model = dataclasses.replace(config.model, dtype=self.precision, pytorch_compile_mode=None)
        lr_schedule = optimizer_config.CosineDecaySchedule(
            warmup_steps=self.warmup_steps,
            peak_lr=self.peak_lr,
            decay_steps=self.num_train_steps,
            decay_lr=self.decay_lr,
        )
        optimizer = optimizer_config.AdamW(
            b1=self.adam_beta1,
            b2=self.adam_beta2,
            eps=self.adam_eps,
            weight_decay=self.weight_decay,
            clip_gradient_norm=self.max_grad_norm,
        )
        return dataclasses.replace(
            config,
            model=model,
            data=data,
            exp_name=self.exp_name,
            assets_base_dir=self.assets_base_dir,
            checkpoint_base_dir=self.checkpoint_base_dir,
            batch_size=self.num_devices * self.per_device_batch_size,
            num_workers=self.num_workers if self.loader_workers_override is None else self.loader_workers_override,
            num_train_steps=self.num_train_steps,
            log_interval=self.log_interval,
            save_interval=self.save_interval,
            seed=self.seed,
            sampler_seed=self.sampler_seed,
            pytorch_training_precision=self.precision,
            lr_schedule=lr_schedule,
            optimizer=optimizer,
            ema_decay=None,
            resume=self.resume_checkpoint is not None,
            overwrite=self.overwrite,
            wandb_enabled=False,
        )
