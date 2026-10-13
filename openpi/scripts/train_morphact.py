"""Train MorphAct adapters for PyTorch Pi0 and Pi0.5."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from dataclasses import replace
import logging
import math
import os
from pathlib import Path
import random
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import swanlab
import torch
import tqdm
import tyro

from openpi.models_pytorch.morphact.config import ACTION_MORPHACT_CONFIG
from openpi.models_pytorch.morphact.config import VLM_MORPHACT_CONFIG
from openpi.models_pytorch.morphact.pi0_morphact import Pi0MorphAct
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.training.morphact_checkpoint import capture_rng_state
from openpi.training.morphact_checkpoint import load_morphact_checkpoint
from openpi.training.morphact_checkpoint import METHOD
from openpi.training.morphact_checkpoint import read_metadata
from openpi.training.morphact_checkpoint import resolve_base_checkpoint
from openpi.training.morphact_checkpoint import resolve_morphact_checkpoint
from openpi.training.morphact_checkpoint import save_morphact_checkpoint
from openpi.training.morphact_checkpoint import verify_base_checkpoint
from openpi.training.morphact_config import MorphActTrainSettings


class TrainingStabilityError(RuntimeError):
    def __init__(self, *, step: int, micro_step: int, reason: str, value: float | None = None) -> None:
        super().__init__(f"Training failed at optimizer step {step}, micro-step {micro_step}: {reason}={value}")
        self.step = step
        self.micro_step = micro_step
        self.reason = reason
        self.value = value


@dataclass(frozen=True, slots=True)
class DistributedRuntime:
    world_size: int
    rank: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _setup_runtime(settings: MorphActTrainSettings) -> DistributedRuntime:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != settings.num_devices:
        raise RuntimeError(f"Configured num_devices={settings.num_devices}, but torchrun WORLD_SIZE={world_size}")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if world_size > 1 and not torch.distributed.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        kwargs = {"device_id": device} if device.type == "cuda" else {}
        torch.distributed.init_process_group(backend=backend, init_method="env://", **kwargs)
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    _set_seed(settings.seed)
    return DistributedRuntime(world_size=world_size, rank=rank, local_rank=local_rank, device=device)


def _barrier() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def _cleanup_runtime() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def _checkpoint_root(config) -> Path:
    return Path(config.checkpoint_dir)


def _initialize_output(config, settings: MorphActTrainSettings) -> None:
    root = _checkpoint_root(config)
    if root.exists():
        if not root.is_dir():
            raise NotADirectoryError(f"Checkpoint output path is not a directory: {root}")
        if settings.resume_checkpoint is None:
            if not settings.overwrite:
                raise FileExistsError(f"Checkpoint directory exists: {root}")
            shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)


def _validate_resume_checkpoint(checkpoint: Path) -> dict:
    required_files = ("adapter.safetensors", "metadata.json", "training_state.pt")
    missing = [name for name in required_files if not (checkpoint / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Resume checkpoint is incomplete: {checkpoint}; missing {', '.join(missing)}")
    metadata = read_metadata(checkpoint)
    if metadata["method"] != METHOD:
        raise ValueError("Unsupported MorphAct checkpoint")
    return metadata


def _validate_base_match(metadata: Mapping, base_checkpoint: Path) -> None:
    verify_base_checkpoint(metadata, base_checkpoint)


def _validate_setup_paths(config, settings: MorphActTrainSettings) -> tuple[Path, Path | None, dict | None]:
    if not settings.base_checkpoint:
        raise ValueError("base_checkpoint is required for MorphAct training")
    base_checkpoint = resolve_base_checkpoint(settings.base_checkpoint)
    output_root = _checkpoint_root(config)
    if output_root.exists() and not output_root.is_dir():
        raise NotADirectoryError(f"Checkpoint output path is not a directory: {output_root}")
    if settings.resume_checkpoint is None:
        # Only rank 0 owns fresh output creation; other ranks may arrive later.
        if int(os.environ.get("RANK", "0")) == 0 and output_root.exists() and not settings.overwrite:
            raise FileExistsError(f"Checkpoint directory exists: {output_root}")
        return base_checkpoint, None, None

    resume_checkpoint = resolve_morphact_checkpoint(settings.resume_checkpoint)
    metadata = _validate_resume_checkpoint(resume_checkpoint)
    _validate_base_match(metadata, base_checkpoint)
    return base_checkpoint, resume_checkpoint, metadata


def _init_swanlab(config, settings: MorphActTrainSettings):
    if not settings.swanlab_enabled:
        return None
    if settings.swanlab_mode == "cloud":
        api_key = os.environ.get("SWANLAB_API_KEY")
        if not api_key:
            raise RuntimeError("Cloud SwanLab is enabled but SWANLAB_API_KEY is not set")
        swanlab.login(api_key=api_key)
    return swanlab.init(
        project=settings.swanlab_project,
        workspace=settings.swanlab_workspace or None,
        experiment_name=config.exp_name,
        config={**settings.checkpoint_contract(), "compile_morphact_pointwise": settings.compile_morphact_pointwise},
        logdir=os.environ.get("OPENPI_MORPHACT_SWANLAB_LOG_DIR", str(config.checkpoint_dir / "swanlab")),
        mode=settings.swanlab_mode,
        settings=swanlab.Settings(requirements_collect=False),
    )


def learning_rate(step: int, settings: MorphActTrainSettings) -> float:
    if step < 0:
        raise ValueError("Learning-rate schedule step must be non-negative")
    if step <= settings.warmup_steps:
        return settings.peak_lr * (step + 1) / (settings.warmup_steps + 1)
    progress = min(
        1.0,
        (step - settings.warmup_steps) / max(1, settings.num_train_steps - settings.warmup_steps),
    )
    return settings.decay_lr + 0.5 * (settings.peak_lr - settings.decay_lr) * (1 + math.cos(math.pi * progress))


def learning_rate_multiplier(step: int, settings: MorphActTrainSettings) -> float:
    return learning_rate(step, settings) / settings.peak_lr


def _gradient_groups(model: Pi0MorphAct) -> dict[str, tuple[torch.nn.Parameter, ...]]:
    return {
        "vlm": tuple(model.vlm_generator.parameters()),
        "action": tuple(model.action_generator.parameters()),
        "vision": tuple(model.vision_parameters()),
        "interface": tuple(model.action_interface_parameters()),
    }


def _optimizer_parameter_groups(model: Pi0MorphAct, settings: MorphActTrainSettings) -> list[dict]:
    parameters = tuple(model.trainable_parameters())
    if not parameters:
        raise ValueError("Pi0MorphAct has no trainable parameters")
    return [{"name": "trainable", "params": parameters, "lr": settings.peak_lr, "peak_lr": settings.peak_lr}]


def _make_optimizer(model: Pi0MorphAct, settings: MorphActTrainSettings) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        _optimizer_parameter_groups(model, settings),
        betas=(settings.adam_beta1, settings.adam_beta2),
        eps=settings.adam_eps,
        weight_decay=settings.weight_decay,
    )


def _scheduled_group_lrs(
    optimizer: torch.optim.Optimizer, settings: MorphActTrainSettings, step: int
) -> dict[str, float]:
    current_lr = learning_rate(step, settings)
    scheduled = {}
    for group in optimizer.param_groups:
        group["lr"] = current_lr
        group["peak_lr"] = settings.peak_lr
        group_name = group.get("name", "trainable")
        scheduled[group_name] = current_lr
    return scheduled


def _gradient_norm(parameters) -> float:
    norms = [parameter.grad.detach().float().norm(2) for parameter in parameters if parameter.grad is not None]
    if not norms:
        return 0.0
    return float(torch.stack(norms).norm(2))


def _morphact_gradient_norms(model: Pi0MorphAct) -> dict[str, float]:
    return {name: _gradient_norm(parameters) for name, parameters in _gradient_groups(model).items()}


def _trainable_parameter_names(model: Pi0MorphAct) -> tuple[str, ...]:
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    return tuple(names[id(parameter)] for parameter in model.trainable_parameters())


def _validate_gradient_coverage(model: Pi0MorphAct, *, step: int, micro_step: int) -> None:
    parameters = tuple(model.trainable_parameters())
    missing = [index for index, parameter in enumerate(parameters) if parameter.grad is None]
    if missing:
        parameter_names = _trainable_parameter_names(model)
        missing_names = [parameter_names[index] for index in missing]
        raise TrainingStabilityError(
            step=step,
            micro_step=micro_step,
            reason=f"missing_trainable_gradient:{missing_names}",
            value=float(len(missing)),
        )


def _validate_and_clip_gradients(model: Pi0MorphAct, settings: MorphActTrainSettings, step: int) -> float:
    parameters = tuple(model.trainable_parameters())
    _validate_gradient_coverage(
        model,
        step=step,
        micro_step=settings.gradient_accumulation_steps,
    )
    grad_norm = _gradient_norm(parameters)
    if not math.isfinite(grad_norm):
        raise TrainingStabilityError(
            step=step,
            micro_step=settings.gradient_accumulation_steps,
            reason="grad_norm",
            value=grad_norm,
        )
    if grad_norm >= settings.explosion_grad_norm:
        raise TrainingStabilityError(
            step=step,
            micro_step=settings.gradient_accumulation_steps,
            reason="grad_norm_threshold",
            value=grad_norm,
        )
    torch.nn.utils.clip_grad_norm_(parameters, settings.max_grad_norm, error_if_nonfinite=True)
    return grad_norm


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().float())


def _mean_metrics(metrics: dict[str, float], runtime: DistributedRuntime) -> dict[str, float]:
    if runtime.world_size == 1:
        return metrics
    keys = sorted(metrics)
    values = torch.tensor([float(metrics[key]) for key in keys], dtype=torch.float64, device=runtime.device)
    torch.distributed.all_reduce(values, op=torch.distributed.ReduceOp.SUM)
    values /= runtime.world_size
    reduced = dict(zip(keys, values.cpu().tolist(), strict=True))
    # Keep chart creation order while reducing identical key order on every rank.
    return {key: reduced[key] for key in metrics}


def _gather_rng_states(runtime: DistributedRuntime):
    local_state = capture_rng_state()
    if runtime.world_size == 1:
        return [local_state]
    gathered = [None] * runtime.world_size if runtime.is_main else None
    torch.distributed.gather_object(local_state, gathered, dst=0)
    return gathered


def _next_batch(data_iterator, loader, epoch: int):
    try:
        return next(data_iterator), data_iterator, epoch
    except StopIteration:
        epoch += 1
        data_iterator = iter(loader)
        return next(data_iterator), data_iterator, epoch


def _make_model(base_model: PI0Pytorch, settings: MorphActTrainSettings) -> Pi0MorphAct:
    vlm_config = replace(
        VLM_MORPHACT_CONFIG,
        compute_precision=settings.morphact_compute_precision,
        d_model=settings.morphact_width,
        num_pg_layers=settings.vlm_morphact_layers,
    )
    action_config = replace(
        ACTION_MORPHACT_CONFIG,
        compute_precision=settings.morphact_compute_precision,
        d_model=settings.morphact_width,
        num_pg_layers=settings.action_morphact_layers,
    )
    model = Pi0MorphAct(
        base_model,
        vlm_config=vlm_config,
        action_config=action_config,
        gradient_checkpointing=settings.gradient_checkpointing,
    )
    if settings.compile_morphact_pointwise:
        from morphact.acceleration import enable_pointwise_compilation

        enable_pointwise_compilation(model.vlm_generator)
        enable_pointwise_compilation(model.action_generator)
        logging.info("MORPHACT pointwise compilation enabled; checkpoint parameter layout is unchanged")
    return model


def _wrap_distributed_model(
    model: Pi0MorphAct,
    runtime: DistributedRuntime,
    device: torch.device,
) -> torch.nn.Module:
    if runtime.world_size <= 1:
        return model
    return torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[runtime.local_rank] if device.type == "cuda" else None,
        broadcast_buffers=False,
        find_unused_parameters=False,
        gradient_as_bucket_view=True,
        static_graph=False,
    )


def _restore_task_selection(settings: MorphActTrainSettings) -> MorphActTrainSettings:
    """An omitted task override inherits the saved selection when resuming."""
    if settings.resume_checkpoint is not None and settings.task_indices is None:
        metadata = read_metadata(resolve_morphact_checkpoint(settings.resume_checkpoint))
        saved = metadata["run_config"].get("task_indices")
        if saved is not None:
            return replace(settings, task_indices=tuple(saved))
    return settings


def _should_save_checkpoint(step: int, settings: MorphActTrainSettings) -> bool:
    return step % settings.save_interval == 0 or step == settings.num_train_steps


def _configure_resume_norm_stats(config, settings: MorphActTrainSettings, checkpoint: Path | None, metadata):
    if checkpoint is None or settings.resume_norm_stats_source == "assets" or config.data.repo_id == "fake":
        return config
    asset_id = metadata.get("norm_stats_asset_id") or config.data.assets.asset_id or config.data.repo_id
    if not isinstance(asset_id, str) or not asset_id:
        raise ValueError(
            "Cannot locate resume checkpoint normalization statistics: no asset id. "
            "Set --resume-norm-stats-source assets to explicitly use the current data assets."
        )
    assets_dir = checkpoint / "assets"
    stats_path = assets_dir / asset_id / "norm_stats.json"
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"Resume checkpoint normalization statistics not found: {stats_path}. "
            "Restore its assets directory, or explicitly use --resume-norm-stats-source assets "
            "with --assets-base-dir pointing to the intended statistics."
        )
    logging.info("Resuming normalization statistics from %s", stats_path)
    assets = replace(config.data.assets, assets_dir=str(assets_dir), asset_id=asset_id)
    return replace(config, data=replace(config.data, assets=assets))


def _resume_samples_seen(metadata: Mapping) -> int:
    samples_seen = metadata["metrics"].get("samples_seen")
    if (
        isinstance(samples_seen, int | float)
        and not isinstance(samples_seen, bool)
        and math.isfinite(samples_seen)
        and samples_seen >= 0
        and int(samples_seen) == samples_seen
    ):
        return int(samples_seen)
    saved = metadata["run_config"]
    batch = 1
    for key in ("num_devices", "per_device_batch_size", "gradient_accumulation_steps"):
        value = saved.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"Checkpoint {key} must be a positive integer to restore samples_seen")
        batch *= value
    return metadata["global_step"] * batch


def train(settings: MorphActTrainSettings) -> None:
    settings = _restore_task_selection(settings)
    config = settings.create_openpi_config()
    base_checkpoint, resume_checkpoint, resume_metadata = _validate_setup_paths(config, settings)
    config = _configure_resume_norm_stats(config, settings, resume_checkpoint, resume_metadata)
    runtime = _setup_runtime(settings)
    device = runtime.device
    # Finish path validation on every rank before rank 0 creates the output.
    _barrier()
    if runtime.is_main:
        _initialize_output(config, settings)
    _barrier()
    swanlab_run = _init_swanlab(config, settings) if runtime.is_main else None

    if runtime.is_main:
        logging.info("Loading Pi0 base from %s", base_checkpoint)
    base_model = PI0Pytorch(config.model).to(device)
    safetensors.torch.load_model(base_model, base_checkpoint, device=str(device))
    model = _make_model(base_model, settings).to(device)
    model.assert_trainable_state()
    trainable_names = _trainable_parameter_names(model)
    if runtime.is_main:
        logging.info(
            "Trainable parameters: total=%d vlm=%d action=%d vision=%d interface=%d",
            model.trainable_parameter_count,
            model.vlm_parameter_count,
            model.action_parameter_count,
            model.vision_parameter_count,
            model.action_interface_parameter_count,
        )
        logging.info("Actual trainable parameter list (%d tensors): %s", len(trainable_names), trainable_names)

    optimizer = _make_optimizer(model, settings)
    global_step = 0
    micro_batches_seen = 0
    samples_seen = 0
    if resume_checkpoint is not None:
        metadata = load_morphact_checkpoint(
            resume_checkpoint,
            model=model,
            optimizer=optimizer,
            restore_rng=True,
            rng_rank=runtime.rank,
            expected_run_config=settings.resume_contract(),
            allow_global_batch_change=settings.allow_global_batch_change,
        )
        global_step = int(metadata["global_step"])
        micro_batches_seen = int(metadata["micro_batches_seen"])
        samples_seen = _resume_samples_seen(metadata)
        if not 0 <= global_step <= config.num_train_steps:
            raise ValueError(f"Resume global_step={global_step} is outside [0, {config.num_train_steps}]")
        if runtime.is_main:
            logging.info(
                "Resumed MorphAct checkpoint at optimizer step %d; loader cursor is not recorded",
                global_step,
            )
    elif resume_metadata is not None:
        raise RuntimeError("Unexpected resume metadata without a resume checkpoint")

    model.assert_trainable_state()
    training_model: torch.nn.Module = model
    training_model = _wrap_distributed_model(model, runtime, device)
    if resume_checkpoint is None:
        _set_seed(settings.seed + runtime.rank)

    from openpi.training import data_loader

    loader = data_loader.create_data_loader(config, framework="pytorch", shuffle=True)
    dataset_size = loader.dataset_size
    logging.info("Selected dataset examples=%s; epoch counts cumulative sample exposure", dataset_size)
    logging.info("Runtime data-loader workers=%d", config.num_workers)
    data_config = loader.data_config()
    training_model.train()
    progress = tqdm.tqdm(
        total=config.num_train_steps,
        initial=global_step,
        desc="MorphAct Pi0/Pi0.5",
        disable=not runtime.is_main,
    )
    data_iterator = iter(loader)
    epoch = 0
    last_metrics: dict[str, float] = {}
    interval_start = time.monotonic()
    last_log_step = global_step
    try:
        while global_step < config.num_train_steps:
            schedule_step = global_step
            current_lr = learning_rate(schedule_step, settings)
            _scheduled_group_lrs(optimizer, settings, schedule_step)
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = torch.zeros((), dtype=torch.float32, device=device)
            for micro_step in range(1, settings.gradient_accumulation_steps + 1):
                (observation, actions), data_iterator, epoch = _next_batch(data_iterator, loader, epoch)
                observation = jax.tree.map(lambda value: value.to(device), observation)
                # actions stay native [batch, horizon, action_dim] control commands.
                actions = actions.to(device=device, dtype=torch.float32)
                autocast = torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda" and settings.precision == "bfloat16",
                )
                sync_context = (
                    training_model.no_sync()
                    if runtime.world_size > 1 and micro_step < settings.gradient_accumulation_steps
                    else nullcontext()
                )
                with sync_context:
                    with autocast:
                        micro_loss = training_model(observation, actions).mean()
                        scaled_loss = micro_loss / settings.gradient_accumulation_steps
                    if not bool(torch.isfinite(micro_loss.detach())):
                        raise TrainingStabilityError(
                            step=global_step,
                            micro_step=micro_step,
                            reason="loss",
                            value=float(micro_loss.detach()),
                        )
                    scaled_loss.backward()
                accumulated_loss += micro_loss.detach().float() / settings.gradient_accumulation_steps
                micro_batches_seen += 1

            group_grad_norms = _morphact_gradient_norms(model)
            grad_norm = _validate_and_clip_gradients(model, settings, global_step)
            optimizer.step()
            global_step += 1
            samples_seen += settings.effective_batch_size

            last_metrics = {
                "loss": _scalar(accumulated_loss),
                **({"epoch": samples_seen / dataset_size} if dataset_size else {}),
                "lr": current_lr,
                "lr_schedule_step": float(schedule_step),
                "optimizer_step": float(global_step),
                "grad_norm_raw": grad_norm,
                "vlm_grad_norm_raw": group_grad_norms["vlm"],
                "action_grad_norm_raw": group_grad_norms["action"],
                "vision_grad_norm_raw": group_grad_norms["vision"],
                "interface_grad_norm_raw": group_grad_norms["interface"],
                "effective_batch_size": float(settings.effective_batch_size),
                "samples_per_optimizer_step": float(settings.effective_batch_size),
                "samples_seen": float(samples_seen),
                "micro_batches_seen": float(micro_batches_seen),
                "data_loader_workers": float(config.num_workers),
                **model.last_metrics,
            }
            last_metrics = _mean_metrics(last_metrics, runtime)

            if runtime.is_main and (global_step == 1 or global_step % config.log_interval == 0):
                elapsed = time.monotonic() - interval_start
                interval_steps = global_step - last_log_step
                seconds_per_step = elapsed / max(1, interval_steps)
                last_metrics["seconds_per_optimizer_step"] = seconds_per_step
                last_metrics["samples_per_second"] = settings.effective_batch_size * interval_steps / max(elapsed, 1e-9)
                logging.info(
                    "optimizer_step=%d samples_seen=%d loss=%.5f grad=%.4f lr=%.3e "
                    "vlm_grad=%.4f action_grad=%.4f vision_grad=%.4f interface_grad=%.4f "
                    "samples_per_second=%.2f",
                    global_step,
                    int(last_metrics["samples_seen"]),
                    last_metrics["loss"],
                    last_metrics["grad_norm_raw"],
                    last_metrics["lr"],
                    last_metrics["vlm_grad_norm_raw"],
                    last_metrics["action_grad_norm_raw"],
                    last_metrics["vision_grad_norm_raw"],
                    last_metrics["interface_grad_norm_raw"],
                    last_metrics["samples_per_second"],
                )
                if swanlab_run is not None:
                    swanlab.log(last_metrics, step=global_step)
                interval_start = time.monotonic()
                last_log_step = global_step

            if _should_save_checkpoint(global_step, settings):
                rng_states = _gather_rng_states(runtime)
                if runtime.is_main:
                    save_morphact_checkpoint(
                        config.checkpoint_dir / str(global_step),
                        model=model,
                        optimizer=optimizer,
                        global_step=global_step,
                        micro_batches_seen=micro_batches_seen,
                        base_checkpoint=str(base_checkpoint),
                        metrics=last_metrics,
                        run_config=settings.resume_contract(),
                        rng_states=rng_states,
                        norm_stats=data_config.norm_stats,
                        asset_id=data_config.asset_id,
                    )
                _barrier()

            if runtime.is_main:
                progress.update(1)
                progress.set_postfix(loss=f"{last_metrics['loss']:.4f}")
    finally:
        progress.close()
        if swanlab_run is not None:
            swanlab.finish()
        _cleanup_runtime()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True)
    train(tyro.cli(MorphActTrainSettings))


if __name__ == "__main__":
    main()
