from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
import logging
import math
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any

import numpy as np
from safetensors.torch import load_file
from safetensors.torch import save_file
import torch
from torch import nn

from openpi.models_pytorch.morphact.config import GeneratorConfig
from openpi.models_pytorch.morphact.config import adaptation_module_names
from openpi.shared import download
import openpi.shared.normalize as normalize

METHOD = "morphact_pi"
FORMAT_VERSION = 1
ADAPTER_FILENAME = "adapter.safetensors"
METADATA_FILENAME = "metadata.json"
TRAINING_STATE_FILENAME = "training_state.pt"
ADAPTATION_MODULE_NAMES = adaptation_module_names()


def _model_type(model: nn.Module) -> str:
    config = getattr(getattr(model, "base_model", None), "config", None)
    return "pi05" if getattr(config, "pi05", False) else "pi0"


def resolve_base_checkpoint(path: str | Path) -> Path:
    resolved = Path(download.maybe_download(str(path)))
    if resolved.is_dir():
        resolved = resolved / "model.safetensors"
    if not resolved.is_file():
        raise FileNotFoundError(f"Pi0/Pi0.5 PyTorch checkpoint not found: {resolved}")
    return resolved


def verify_base_checkpoint(metadata: Mapping[str, Any], path: str | Path) -> Path:
    """Resolve the base file; model loading checks tensor keys and shapes."""
    del metadata
    return resolve_base_checkpoint(path)


def resolve_morphact_checkpoint(path: str | Path) -> Path:
    resolved = Path(download.maybe_download(str(path)))
    if not resolved.is_dir():
        raise FileNotFoundError(f"Pi0 adapter checkpoint not found: {resolved}")
    return resolved


def _unwrap(model: nn.Module) -> nn.Module:
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module
    if not isinstance(model, nn.Module):
        raise TypeError(f"Expected a PyTorch module, got {type(model).__name__}")
    return model


def _adaptation_modules(model: nn.Module) -> dict[str, nn.Module]:
    method = getattr(model, "adaptation_modules", None)
    if not callable(method):
        raise TypeError("Pi0MorphAct must provide adaptation_modules()")
    modules = method()
    if not isinstance(modules, Mapping):
        raise TypeError("Pi0MorphAct adaptation_modules() must return a mapping")
    modules = dict(modules)
    names = adaptation_module_names(pi05=_model_type(model) == "pi05")
    expected = set(names)
    missing = sorted(expected - set(modules))
    extra = sorted(set(modules) - expected)
    if missing or extra:
        raise ValueError(f"Pi0MorphAct adaptation module paths mismatch; missing={missing}, extra={extra}")
    for name in names:
        if not isinstance(modules[name], nn.Module):
            raise TypeError(f"Pi0MorphAct adaptation module {name} is not a PyTorch module")
    return {name: modules[name] for name in names}


def _adaptation_state_spec(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for module_name, module in _adaptation_modules(model).items():
        for key, value in module.state_dict().items():
            if not torch.is_tensor(value):
                raise TypeError(f"Pi0MorphAct adaptation tensor {module_name}.{key} is not a tensor")
            full_key = f"{module_name}.{key}"
            if full_key in state:
                raise ValueError(f"Duplicate Pi0MorphAct adaptation tensor key: {full_key}")
            state[full_key] = value
    return state


def morphact_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    model = _unwrap(model)
    state = _adaptation_state_spec(model)
    if not state:
        raise RuntimeError("Pi0MorphAct adaptation_modules() has no state")
    return {key: value.detach().cpu().contiguous() for key, value in state.items()}


def _target_metadata(manifest) -> list[dict[str, Any]]:
    result = []
    for target in manifest:
        if isinstance(target, Mapping):
            result.append(dict(target))
        else:
            result.append(asdict(target))
    return result


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))


def capture_rng_state() -> dict[str, Any]:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _validate_rng_state(state: Mapping[str, Any]) -> None:
    required = {"torch", "cuda", "numpy", "python"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"Checkpoint RNG state is missing keys: {missing}")
    torch_state = state["torch"]
    if not torch.is_tensor(torch_state) or torch_state.dtype != torch.uint8 or torch_state.ndim != 1:
        raise ValueError("Checkpoint torch RNG state must be a one-dimensional uint8 tensor")
    cuda_state = state["cuda"]
    if cuda_state is not None:
        cuda_states = (
            cuda_state if isinstance(cuda_state, Sequence) and not torch.is_tensor(cuda_state) else [cuda_state]
        )
        if not cuda_states:
            raise ValueError("Checkpoint CUDA RNG states must not be empty")
        if any(not torch.is_tensor(value) or value.dtype != torch.uint8 or value.ndim != 1 for value in cuda_states):
            raise ValueError("Checkpoint CUDA RNG states must be one-dimensional uint8 tensors")
    numpy_state = state["numpy"]
    try:
        numpy_probe = np.random.RandomState()
        numpy_probe.set_state(numpy_state)
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint NumPy RNG state is invalid") from exc
    python_state = state["python"]
    try:
        random.Random().setstate(python_state)
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint Python RNG state is invalid") from exc


def restore_rng_state(state: Mapping[str, Any]) -> None:
    _validate_rng_state(state)
    cuda_state = state["cuda"] if torch.cuda.is_available() else None
    if isinstance(cuda_state, Sequence) and not torch.is_tensor(cuda_state):
        # Legacy checkpoints saved every visible device in each rank. Restore
        # only this rank's training device, ignoring unused GPU states.
        device_index = torch.cuda.current_device()
        if device_index >= len(cuda_state):
            raise ValueError("Legacy checkpoint has no RNG state for the current CUDA device")
        cuda_state = cuda_state[device_index]
    torch.set_rng_state(state["torch"])
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state)
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def _config_metadata(config: GeneratorConfig, name: str) -> dict[str, Any]:
    to_dict = getattr(config, "to_dict", None)
    if not callable(to_dict):
        raise TypeError(f"Pi0MorphAct {name} config must provide to_dict()")
    values = to_dict()
    if not isinstance(values, Mapping):
        raise TypeError(f"Pi0MorphAct {name} config to_dict() must return a mapping")
    return _jsonable(dict(values))


def _validate_model_metadata(model: nn.Module, metadata: Mapping[str, Any]) -> None:
    if metadata["model_type"] != _model_type(model):
        raise ValueError("Checkpoint model_type does not match the current model")
    _adaptation_modules(model)
    for name in ("action_generator_config", "vlm_generator_config"):
        config_values = metadata.get(name)
        if not isinstance(config_values, Mapping):
            raise ValueError(f"Checkpoint is missing {name}")
        try:
            checkpoint_config = GeneratorConfig.from_dict(dict(config_values))
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(f"Checkpoint {name} is invalid") from exc
        model_config = getattr(model, name, None)
        if not isinstance(model_config, GeneratorConfig) or checkpoint_config != model_config:
            raise ValueError(f"Checkpoint {name} does not match the current model")

    for name in ("action_target_manifest", "vlm_target_manifest"):
        model_manifest = _jsonable(_target_metadata(getattr(model, name, ())))
        if metadata.get(name) != model_manifest:
            raise ValueError(f"Checkpoint {name} does not match the current model")


def _validate_adaptation_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    expected = _adaptation_state_spec(model)
    provided = dict(state)
    missing = sorted(set(expected) - set(provided))
    extra = sorted(set(provided) - set(expected))
    if missing or extra:
        raise ValueError(f"Checkpoint adaptation tensor keys mismatch; missing={missing}, extra={extra}")
    shape_mismatches = {
        key: (tuple(value.shape), tuple(expected[key].shape))
        for key, value in provided.items()
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(expected[key].shape)
    }
    if shape_mismatches:
        raise ValueError(f"Checkpoint adaptation tensor shapes mismatch: {shape_mismatches}")


def _load_adaptation_state(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    _validate_adaptation_state(model, state)
    for module_name, module in _adaptation_modules(model).items():
        prefix = f"{module_name}."
        module_state = {key.removeprefix(prefix): value for key, value in state.items() if key.startswith(prefix)}
        module.load_state_dict(module_state, strict=True)


def _model_trainable_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    method = getattr(model, "trainable_parameters", None)
    if not callable(method):
        raise TypeError("Pi0MorphAct must provide trainable_parameters()")
    parameters = tuple(method())
    if not parameters or any(not isinstance(parameter, nn.Parameter) for parameter in parameters):
        raise ValueError("Pi0MorphAct trainable_parameters() must return parameters")
    if len({id(parameter) for parameter in parameters}) != len(parameters):
        raise ValueError("Pi0MorphAct trainable_parameters() contains duplicate parameters")
    return parameters


def _validate_optimizer_coverage(model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    expected = {id(parameter) for parameter in _model_trainable_parameters(model)}
    actual_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    actual = {id(parameter) for parameter in actual_parameters}
    if len(actual) != len(actual_parameters) or actual != expected:
        raise ValueError("Optimizer parameters must exactly match Pi0MorphAct trainable_parameters()")


def _validate_optimizer_state(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    state: Mapping[str, Any],
    *,
    global_step: int,
) -> None:
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("Single-stage adapter checkpoints require torch.optim.AdamW")
    if not isinstance(state, Mapping):
        raise ValueError("Checkpoint optimizer state must be a mapping")
    saved_groups = state.get("param_groups")
    current_state = optimizer.state_dict()
    current_groups = current_state.get("param_groups")
    if not isinstance(saved_groups, list) or not isinstance(current_groups, list):
        raise ValueError("Checkpoint optimizer state has invalid parameter groups")
    if len(saved_groups) != len(current_groups):
        raise ValueError("Checkpoint optimizer parameter-group count does not match the current optimizer")

    saved_to_parameter: dict[int, nn.Parameter] = {}
    for saved_group, current_group in zip(saved_groups, optimizer.param_groups, strict=True):
        saved_parameters = saved_group.get("params")
        current_parameters = current_group.get("params")
        if not isinstance(saved_parameters, list) or not isinstance(current_parameters, list | tuple):
            raise ValueError("Checkpoint optimizer parameter groups do not match the current optimizer")
        if len(saved_parameters) != len(current_parameters):
            raise ValueError("Checkpoint optimizer parameter groups do not match the current optimizer")
        for saved_id, parameter in zip(saved_parameters, current_parameters, strict=True):
            if not isinstance(saved_id, int):
                raise ValueError("Checkpoint optimizer parameter ids must be integers")
            if saved_id in saved_to_parameter:
                raise ValueError(f"Checkpoint optimizer parameter id is duplicated: {saved_id}")
            saved_to_parameter[saved_id] = parameter

    if len(saved_to_parameter) != len(_model_trainable_parameters(model)):
        raise ValueError("Checkpoint optimizer parameters do not cover all Pi0MorphAct trainable parameters")

    saved_state = state.get("state")
    if not isinstance(saved_state, Mapping):
        raise ValueError("Checkpoint optimizer state has invalid slots")
    unknown_ids = sorted(set(saved_state) - set(saved_to_parameter), key=str)
    if unknown_ids:
        raise ValueError(f"Checkpoint optimizer state has unknown parameter ids: {unknown_ids}")
    if global_step > 0:
        missing_ids = sorted(set(saved_to_parameter) - set(saved_state), key=str)
        if missing_ids:
            raise ValueError(f"Checkpoint optimizer state is missing parameters: {missing_ids}")
    for saved_id, slots in saved_state.items():
        if not isinstance(slots, Mapping):
            raise ValueError(f"Checkpoint optimizer slots for parameter {saved_id} are invalid")
        expected_slots = {"step", "exp_avg", "exp_avg_sq"}
        missing_slots = sorted(expected_slots - set(slots))
        extra_slots = sorted(set(slots) - expected_slots)
        if missing_slots or extra_slots:
            raise ValueError(
                f"Checkpoint AdamW slots mismatch for parameter {saved_id}; "
                f"missing={missing_slots}, extra={extra_slots}"
            )
        parameter = saved_to_parameter[saved_id]
        step = slots["step"]
        if not torch.is_tensor(step) or tuple(step.shape) != ():
            raise ValueError(f"Checkpoint optimizer step shape mismatch: {getattr(step, 'shape', None)} != ()")
        for slot_name in ("exp_avg", "exp_avg_sq"):
            value = slots[slot_name]
            if not torch.is_tensor(value) or tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"Checkpoint optimizer slot shape mismatch for {slot_name}: "
                    f"{getattr(value, 'shape', None)} != {tuple(parameter.shape)}"
                )


def _validate_training_state(training_state: Mapping[str, Any], world_size: int) -> list[Mapping[str, Any]]:
    rng_states = training_state.get("rng_states")
    if not isinstance(rng_states, Sequence) or isinstance(rng_states, str | bytes):
        raise ValueError("Checkpoint RNG states must be a sequence")
    rng_states = list(rng_states)
    if len(rng_states) != world_size:
        raise ValueError(
            f"Checkpoint contains {len(rng_states)} RNG states, but metadata declares world size {world_size}"
        )
    for state in rng_states:
        if not isinstance(state, Mapping):
            raise ValueError("Checkpoint RNG state must be a mapping")
        _validate_rng_state(state)
    return rng_states


def _metadata(
    model: nn.Module,
    *,
    global_step: int,
    micro_batches_seen: int,
    base_checkpoint: str | Path,
    metrics: Mapping[str, float],
    run_config: Mapping[str, Any],
    world_size: int,
    norm_stats_asset_id: str | None,
) -> dict[str, Any]:
    return {
        "method": METHOD,
        "format_version": FORMAT_VERSION,
        "model_type": _model_type(model),
        "adaptation_modules": list(_adaptation_modules(model)),
        "global_step": global_step,
        "micro_batches_seen": micro_batches_seen,
        "base_checkpoint": resolve_base_checkpoint(base_checkpoint).name,
        "world_size": world_size,
        "action_target_manifest": _jsonable(_target_metadata(model.action_target_manifest)),
        "vlm_target_manifest": _jsonable(_target_metadata(model.vlm_target_manifest)),
        "action_generator_config": _config_metadata(model.action_generator_config, "action_generator"),
        "vlm_generator_config": _config_metadata(model.vlm_generator_config, "vlm_generator"),
        "metrics": {key: float(value) for key, value in metrics.items()},
        "run_config": _jsonable(dict(run_config)),
        "norm_stats_asset_id": norm_stats_asset_id,
    }


def save_morphact_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    global_step: int,
    micro_batches_seen: int,
    base_checkpoint: str | Path,
    metrics: Mapping[str, float],
    run_config: Mapping[str, Any],
    rng_states: Sequence[Mapping[str, Any]] | None = None,
    norm_stats=None,
    asset_id: str | None = None,
) -> Path:
    model = _unwrap(model)
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step < 0:
        raise ValueError("global_step must be a non-negative integer")
    if isinstance(micro_batches_seen, bool) or not isinstance(micro_batches_seen, int) or micro_batches_seen < 0:
        raise ValueError("micro_batches_seen must be a non-negative integer")
    if not isinstance(metrics, Mapping) or not isinstance(run_config, Mapping):
        raise ValueError("metrics and run_config must be mappings")
    if norm_stats is not None and asset_id is None:
        raise ValueError("asset_id is required when norm_stats are provided")
    if asset_id is not None and (not isinstance(asset_id, str) or not asset_id):
        raise ValueError("asset_id must be a non-empty string")

    _validate_optimizer_coverage(model, optimizer)
    _validate_optimizer_state(model, optimizer, optimizer.state_dict(), global_step=global_step)
    assert_trainable_state = getattr(model, "assert_trainable_state", None)
    if not callable(assert_trainable_state):
        raise TypeError("Pi0MorphAct must provide assert_trainable_state()")
    assert_trainable_state()
    base_checkpoint = resolve_base_checkpoint(base_checkpoint)
    run_config = _jsonable(dict(run_config))
    rng_states = [capture_rng_state()] if rng_states is None else list(rng_states)
    if not rng_states:
        raise ValueError("At least one per-rank RNG state is required")
    world_size = len(rng_states)
    for state in rng_states:
        if not isinstance(state, Mapping):
            raise ValueError("Each RNG state must be a mapping")
        _validate_rng_state(state)
    expected_world_size = run_config.get("num_devices", run_config.get("world_size"))
    if expected_world_size is not None and expected_world_size != world_size:
        raise ValueError(f"run_config world size {expected_world_size} does not match RNG state count {world_size}")

    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.exists():
        raise FileExistsError(f"Checkpoint already exists: {checkpoint_dir}")
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{checkpoint_dir.name}-", dir=checkpoint_dir.parent))
    try:
        save_file(morphact_state_dict(model), temporary / ADAPTER_FILENAME)
        metadata = _metadata(
            model,
            global_step=global_step,
            micro_batches_seen=micro_batches_seen,
            base_checkpoint=base_checkpoint,
            metrics=metrics,
            run_config=run_config,
            world_size=world_size,
            norm_stats_asset_id=asset_id if norm_stats is not None else None,
        )
        (temporary / METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "rng_states": list(rng_states),
                "world_size": world_size,
            },
            temporary / TRAINING_STATE_FILENAME,
        )
        if norm_stats is not None:
            normalize.save(temporary / "assets" / asset_id, norm_stats)
        temporary.rename(checkpoint_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return checkpoint_dir


def read_metadata(checkpoint_dir: str | Path) -> dict[str, Any]:
    checkpoint_dir = resolve_morphact_checkpoint(checkpoint_dir)
    metadata_path = checkpoint_dir / METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Adapter checkpoint metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        not isinstance(metadata, dict)
        or metadata.get("method") != METHOD
        or type(metadata.get("format_version")) is not int
        or metadata["format_version"] != FORMAT_VERSION
    ):
        raise ValueError("Unsupported MorphAct checkpoint format")
    model_type = metadata.get("model_type")
    if model_type not in ("pi0", "pi05"):
        raise ValueError("Checkpoint model_type is unsupported")
    if metadata.get("adaptation_modules") != list(adaptation_module_names(pi05=model_type == "pi05")):
        raise ValueError("Checkpoint adaptation module paths are unsupported or incomplete")
    required = (
        "global_step",
        "micro_batches_seen",
        "base_checkpoint",
        "world_size",
        "action_target_manifest",
        "vlm_target_manifest",
        "action_generator_config",
        "vlm_generator_config",
        "metrics",
        "run_config",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"Adapter checkpoint metadata is missing keys: {missing}")
    for key in ("global_step", "micro_batches_seen", "world_size"):
        if isinstance(metadata[key], bool) or not isinstance(metadata[key], int):
            raise ValueError(f"Adapter checkpoint metadata {key} must be an integer")
    if metadata["global_step"] < 0 or metadata["micro_batches_seen"] < 0 or metadata["world_size"] <= 0:
        raise ValueError("Adapter checkpoint metadata has invalid progress or world size")
    if not isinstance(metadata["base_checkpoint"], str) or not metadata["base_checkpoint"]:
        raise ValueError("Adapter checkpoint metadata base_checkpoint must be a non-empty string")
    if not isinstance(metadata["action_target_manifest"], list) or not isinstance(
        metadata["vlm_target_manifest"], list
    ):
        raise ValueError("Adapter checkpoint target manifests must be lists")
    if not isinstance(metadata["metrics"], dict) or not isinstance(metadata["run_config"], dict):
        raise ValueError("Adapter checkpoint metrics and run_config must be mappings")
    return metadata


def load_morphact_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = False,
    expected_run_config: Mapping[str, Any] | None = None,
    rng_rank: int = 0,
    allow_global_batch_change: bool = False,
) -> dict[str, Any]:
    model = _unwrap(model)
    if not isinstance(allow_global_batch_change, bool):
        raise ValueError("allow_global_batch_change must be a boolean")
    if allow_global_batch_change and expected_run_config is None:
        raise ValueError("expected_run_config is required when allowing a global batch change")
    if isinstance(rng_rank, bool) or not isinstance(rng_rank, int) or rng_rank < 0:
        raise ValueError("rng_rank must be a non-negative integer")
    if optimizer is None and restore_rng:
        raise ValueError("Optimizer is required when restore_rng=True")
    checkpoint_dir = resolve_morphact_checkpoint(checkpoint_dir)
    metadata = read_metadata(checkpoint_dir)
    _validate_model_metadata(model, metadata)
    batch_change = None
    if expected_run_config is not None:
        expected = _jsonable(dict(expected_run_config))
        saved = dict(metadata["run_config"])
        if metadata["world_size"] != saved.get("num_devices"):
            raise ValueError("Checkpoint world size does not match its run configuration")
        # Older checkpoints included these runtime-only settings. Normalize
        # both sides so logging, checkpoint I/O and loader parallelism can change.
        for key in ("log_interval", "save_interval", "num_workers"):
            expected.pop(key, None)
            saved.pop(key, None)
        # Before this field existed, DistributedSampler always used its default seed 0.
        expected.setdefault("sampler_seed", 0)
        saved.setdefault("sampler_seed", 0)
        if allow_global_batch_change:
            batch_keys = ("num_devices", "per_device_batch_size", "gradient_accumulation_steps")
            for contract in (saved, expected):
                for key in batch_keys:
                    value = contract.get(key)
                    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                        raise ValueError(f"Resume contract {key} must be a positive integer for batch migration")
            saved_batch = tuple(saved[key] for key in batch_keys)
            expected_batch = tuple(expected[key] for key in batch_keys)
            if saved_batch != expected_batch:
                batch_change = (math.prod(saved_batch), math.prod(expected_batch))
            for key in batch_keys[1:]:
                saved.pop(key)
                expected.pop(key)
        if saved != expected:
            raise ValueError("Training settings differ from the checkpoint resume contract")
    if rng_rank >= metadata["world_size"]:
        raise ValueError(f"Checkpoint has {metadata['world_size']} ranks; cannot use rank {rng_rank}")

    state = load_file(checkpoint_dir / ADAPTER_FILENAME, device="cpu")
    _validate_adaptation_state(model, state)

    training_state = None
    rng_states = None
    if optimizer is not None:
        training_state_path = checkpoint_dir / TRAINING_STATE_FILENAME
        if not training_state_path.is_file():
            raise FileNotFoundError(f"Adapter training state not found: {training_state_path}")
        training_state = torch.load(training_state_path, map_location="cpu", weights_only=False)
        if not isinstance(training_state, Mapping):
            raise ValueError("Adapter training state must be a mapping")
        if training_state.get("world_size") != metadata["world_size"]:
            raise ValueError("Checkpoint training-state world size does not match metadata")
        if "optimizer" not in training_state:
            raise ValueError("Adapter training state is missing optimizer")
        _validate_optimizer_coverage(model, optimizer)
        _validate_optimizer_state(
            model,
            optimizer,
            training_state["optimizer"],
            global_step=metadata["global_step"],
        )
        rng_states = _validate_training_state(training_state, metadata["world_size"])

    _load_adaptation_state(model, state)
    if optimizer is not None:
        assert training_state is not None
        optimizer.load_state_dict(training_state["optimizer"])
    if restore_rng:
        assert rng_states is not None
        restore_rng_state(rng_states[rng_rank])

    assert_trainable_state = getattr(model, "assert_trainable_state", None)
    if not callable(assert_trainable_state):
        raise TypeError("Pi0MorphAct must provide assert_trainable_state()")
    assert_trainable_state()
    if batch_change is not None:
        logging.info("Resumed MorphAct with effective global batch size %d -> %d", *batch_change)
    return metadata
