"""Checkpointing for native XVLA models with MORPHACT adapters.

Only parameters that are currently marked trainable are stored. The base model
architecture is validated when loading; its source and Hub revision are recorded
as provenance, not a weight-file whitelist. Frozen weights are not copied.

The optional ``training_state.pt`` file contains optimizer and RNG state.  It
is loaded with ``weights_only=False`` because Python and NumPy RNG states are
part of the trusted local resume artifact.  Callers should pass only local
checkpoint paths they trust to :func:`load_checkpoint` with
``load_training_state=True``.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

CHECKPOINT_FORMAT = "morphact_xvla"
CHECKPOINT_VERSION = 2
ADAPTER_FILENAME = "adapter.safetensors"
METADATA_FILENAME = "metadata.json"
TRAINING_STATE_FILENAME = "training_state.pt"
_BASE_CONFIG_RUNTIME_FIELDS = frozenset(
    {
        "torch_dtype",
        "transformers_version",
        "_name_or_path",
        "name_or_path",
        "model_name_or_path",
        "_attn_implementation_internal",
        "_attn_implementation_autoset",
        "_commit_hash",
        "revision",
    }
)
_TRAINING_PATH_FIELDS = frozenset(
    {"base_model", "train_metas", "morphact_config", "processing_config", "output_dir", "resume"}
)


def _json_value(value: Any, *, field: str) -> Any:
    """Return a JSON-normalized copy and reject non-serializable values."""

    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be JSON serializable") from exc
    return json.loads(encoded)


def _config_dict(config: Any, *, field: str) -> dict[str, Any]:
    if isinstance(config, Mapping):
        raw = dict(config)
    else:
        to_dict = getattr(config, "to_dict", None)
        if not callable(to_dict):
            raise TypeError(f"{field} must expose to_dict()")
        raw = to_dict()
    normalized = _json_value(raw, field=field)
    if not isinstance(normalized, dict):
        raise TypeError(f"{field}.to_dict() must return a mapping")
    return normalized


def _optional_config_dict(config: Any, *, field: str) -> dict[str, Any] | None:
    if config is None:
        return None
    return _config_dict(config, field=field)


def _model_base(model: Any) -> Any | None:
    for attribute in ("base_model", "native"):
        value = getattr(model, attribute, None)
        if value is not None:
            return value
    return None


def _base_source(base_model: Any) -> str | None:
    if isinstance(base_model, (str, os.PathLike)):
        source = os.fspath(base_model)
        return source if source else None

    for attribute in ("name_or_path", "model_name_or_path", "_name_or_path"):
        value = getattr(base_model, attribute, None)
        if isinstance(value, str) and value:
            return value

    config = getattr(base_model, "config", None)
    if config is not None:
        for attribute in ("name_or_path", "_name_or_path"):
            value = getattr(config, attribute, None)
            if isinstance(value, str) and value:
                return value
    return None


def _base_config(base_model: Any) -> dict[str, Any]:
    config = getattr(base_model, "config", None)
    if config is None:
        raise ValueError("base_model must expose config.to_dict()")
    return _config_dict(config, field="base_config")


def _base_identity(base_model: Any, source: str) -> dict[str, Any]:
    """Record source provenance without storing local paths or hashing files."""

    path = Path(source)
    if path.is_dir() or path.is_absolute():
        return {"kind": "local"}
    config = getattr(base_model, "config", None)
    revision = getattr(config, "_commit_hash", None) or getattr(config, "revision", None)
    identity = {"kind": "hub", "id": source}
    if isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision):
        identity["revision"] = revision
    return identity


def base_pretrained_kwargs(source: str, metadata: Mapping[str, Any] | None) -> dict[str, str]:
    """Reuse a recorded Hub revision when loading the same Hub source."""

    if metadata is None or Path(source).is_dir() or Path(source).is_absolute():
        return {}
    identity = metadata.get("base_identity", {})
    revision = identity.get("revision")
    if (
        identity.get("kind") == "hub"
        and identity.get("id", metadata.get("base_model")) == source
        and isinstance(revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", revision)
    ):
        return {"revision": revision}
    return {}


def canonical_training_config(config: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Compare model/optimizer settings; data comes from the current invocation."""

    if config is None:
        return None
    result = {key: value for key, value in config.items() if key not in _TRAINING_PATH_FIELDS}
    for name, legacy_field in (
        ("metaworld_processing", "stats_sha256"),
        ("metaworld_contract", "processing_sha256"),
    ):
        if isinstance(result.get(name), Mapping):
            result[name] = {key: value for key, value in result[name].items() if key != legacy_field}
    return result


def _canonical_base_config(value: Any) -> Any:
    """Strip HF runtime/source metadata while retaining architecture fields."""

    if isinstance(value, Mapping):
        return {
            key: _canonical_base_config(item) for key, item in value.items() if key not in _BASE_CONFIG_RUNTIME_FIELDS
        }
    if isinstance(value, list):
        return [_canonical_base_config(item) for item in value]
    return value


def _morphact_config(model: Any) -> dict[str, Any]:
    config = getattr(model, "morphact_config", None)
    if config is None:
        raise ValueError("model must expose morphact_config.to_dict()")
    return _config_dict(config, field="morphact_config")


def _assert_model_contract(model: Any) -> None:
    checker = getattr(model, "assert_trainable_state", None)
    if callable(checker):
        try:
            result = checker()
        except AssertionError as exc:
            raise ValueError("model.assert_trainable_state() rejected the model") from exc
        if result is False:
            raise ValueError("model.assert_trainable_state() rejected the model")


def _trainable_parameters(model: Any) -> list[tuple[str, torch.nn.Parameter]]:
    named = list(model.named_parameters())
    trainable = [(name, parameter) for name, parameter in named if parameter.requires_grad]
    if not trainable:
        raise ValueError("model has no requires_grad=True parameters")
    if len({name for name, _ in trainable}) != len(trainable):
        raise ValueError("model has duplicate trainable parameter names")
    non_fp32 = [name for name, parameter in trainable if parameter.dtype != torch.float32]
    if non_fp32:
        raise ValueError(f"Trainable parameters must be FP32: {non_fp32}")
    return trainable


def _validate_trainable_interface(
    model: Any,
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> None:
    provider = getattr(model, "trainable_parameters", None)
    if not callable(provider):
        return
    provided = list(provider())
    provided_parameters: list[torch.nn.Parameter] = []
    for value in provided:
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], torch.nn.Parameter):
            value = value[1]
        if not isinstance(value, torch.nn.Parameter):
            raise TypeError("model.trainable_parameters() must yield Parameters")
        provided_parameters.append(value)
    expected_ids = {id(parameter) for _, parameter in named_parameters}
    if {id(parameter) for parameter in provided_parameters} != expected_ids:
        raise ValueError("model.trainable_parameters() does not match requires_grad parameters")


def _optimizer_topology(
    optimizer: torch.optim.Optimizer,
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> list[list[str]]:
    names_by_id = {id(parameter): name for name, parameter in named_parameters}
    topology: list[list[str]] = []
    for group_index, group in enumerate(optimizer.param_groups):
        names: list[str] = []
        for parameter in group["params"]:
            name = names_by_id.get(id(parameter))
            if name is None:
                raise ValueError(f"Optimizer group {group_index} contains a parameter outside the model")
            names.append(name)
        topology.append(names)
    flattened = [name for group in topology for name in group]
    expected = {name for name, _ in named_parameters}
    if len(set(flattened)) != len(flattened) or set(flattened) != expected:
        raise ValueError("Optimizer parameters must cover each trainable model parameter exactly once")
    return topology


def _capture_rng_state() -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - only relevant to minimal installs
        raise RuntimeError("NumPy is required when saving resumable training state") from exc

    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _checkpoint_metadata(
    *,
    base_source: str,
    base_config: dict[str, Any],
    base_identity: dict[str, Any],
    morphact_config: dict[str, Any],
    step: int,
    trainable_keys: list[str],
    training_config: dict[str, Any] | None,
    optimizer_topology: list[list[str]] | None,
) -> dict[str, Any]:
    return {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "base_model": base_source if base_identity.get("kind") == "hub" else "local",
        "base_config": _canonical_base_config(base_config),
        "base_identity": base_identity,
        "morphact_config": morphact_config,
        "step": step,
        "trainable_keys": list(trainable_keys),
        "training_config": canonical_training_config(training_config),
        "optimizer_topology": optimizer_topology,
        "has_training_state": optimizer_topology is not None,
    }


def save_checkpoint(
    model: Any,
    path: str | os.PathLike[str],
    *,
    base_model: Any,
    step: int = 0,
    optimizer: torch.optim.Optimizer | None = None,
    training_config: Mapping[str, Any] | Any | None = None,
) -> None:
    """Atomically save a native XVLA MORPHACT checkpoint.

    ``base_model`` may be a model object exposing ``config.to_dict()`` and a
    source name/path, or an explicit source string/Path.  When a source string
    is passed, the base configuration is read from ``model.base_model`` or
    ``model.native``.

    Existing checkpoint directories are never overwritten.  Only trainable
    model parameters are written to ``adapter.safetensors``; frozen base weights
    are not duplicated.
    """

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("step must be a non-negative integer")

    trainable = _trainable_parameters(model)
    _validate_trainable_interface(model, trainable)
    _assert_model_contract(model)
    trainable_keys = [name for name, _ in trainable]

    base_object = base_model if not isinstance(base_model, (str, os.PathLike)) else _model_base(model)
    if base_object is None:
        raise ValueError("base_model source string requires model.base_model or model.native")
    base_source = _base_source(base_model)
    if base_source is None:
        raise ValueError("base_model must be a non-empty source path or Hub ID")
    base_config = _base_config(base_object)
    base_identity = _base_identity(base_object, base_source)
    morphact_config = _morphact_config(model)
    normalized_training_config = _optional_config_dict(training_config, field="training_config")

    optimizer_topology = None
    training_state: dict[str, Any] | None = None
    if optimizer is not None:
        optimizer_topology = _optimizer_topology(optimizer, trainable)
        training_state = {
            "step": step,
            "optimizer": optimizer.state_dict(),
            **_capture_rng_state(),
        }

    metadata = _checkpoint_metadata(
        base_source=base_source,
        base_config=base_config,
        base_identity=base_identity,
        morphact_config=morphact_config,
        step=step,
        trainable_keys=trainable_keys,
        training_config=normalized_training_config,
        optimizer_topology=optimizer_topology,
    )
    tensors = {name: parameter.detach().cpu().contiguous() for name, parameter in trainable}

    target = Path(path)
    if os.path.lexists(target):
        raise FileExistsError(f"Checkpoint path already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        save_file(tensors, str(temporary / ADAPTER_FILENAME))
        _write_json(temporary / METADATA_FILENAME, metadata)
        if training_state is not None:
            torch.save(training_state, temporary / TRAINING_STATE_FILENAME)
        if os.path.lexists(target):
            raise FileExistsError(f"Checkpoint path appeared during save: {target}")
        os.rename(temporary, target)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _read_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path / METADATA_FILENAME
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    try:
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid checkpoint metadata: {metadata_path}") from exc
    if not isinstance(metadata, dict):
        raise TypeError("Checkpoint metadata must be a JSON object")

    required = {
        "format",
        "version",
        "base_model",
        "base_config",
        "base_identity",
        "morphact_config",
        "step",
        "trainable_keys",
        "training_config",
        "optimizer_topology",
        "has_training_state",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"Checkpoint metadata is missing keys: {missing}")
    if metadata["format"] != CHECKPOINT_FORMAT or metadata["version"] != CHECKPOINT_VERSION:
        raise ValueError("Unsupported MORPHACT checkpoint format or version")
    if not isinstance(metadata["base_model"], str) or not metadata["base_model"]:
        raise ValueError("Checkpoint base_model must be a non-empty string")
    if isinstance(metadata["step"], bool) or not isinstance(metadata["step"], int) or metadata["step"] < 0:
        raise ValueError("Checkpoint step must be a non-negative integer")
    if not isinstance(metadata["trainable_keys"], list) or any(
        not isinstance(key, str) for key in metadata["trainable_keys"]
    ):
        raise ValueError("Checkpoint trainable_keys must be a list of strings")
    if len(set(metadata["trainable_keys"])) != len(metadata["trainable_keys"]):
        raise ValueError("Checkpoint trainable_keys contains duplicates")
    if not isinstance(metadata["base_config"], dict) or not isinstance(metadata["morphact_config"], dict):
        raise TypeError("Checkpoint base_config and morphact_config must be objects")
    if not isinstance(metadata["base_identity"], dict):
        raise TypeError("Checkpoint base_identity must be an object")
    if metadata["optimizer_topology"] is not None and (
        not isinstance(metadata["optimizer_topology"], list)
        or any(
            not isinstance(group, list) or any(not isinstance(name, str) for name in group)
            for group in metadata["optimizer_topology"]
        )
    ):
        raise ValueError("Checkpoint optimizer_topology is invalid")
    if not isinstance(metadata["has_training_state"], bool):
        raise TypeError("Checkpoint has_training_state must be a boolean")
    if metadata["has_training_state"] != (metadata["optimizer_topology"] is not None):
        raise ValueError("Checkpoint training-state metadata is inconsistent")
    return metadata


def _validate_base_model(
    model: Any,
    metadata: Mapping[str, Any],
    expected_base_model: Any | None,
) -> None:
    current_base = _model_base(model)
    expected_object = expected_base_model if expected_base_model is not None else current_base
    if expected_object is None:
        raise ValueError("Model must expose base_model or native for checkpoint validation")

    current_config = _base_config(current_base) if current_base is not None else None
    expected_config = (
        _base_config(expected_object) if not isinstance(expected_object, (str, os.PathLike)) else current_config
    )
    if current_config is None:
        current_config = expected_config
    if current_config is None:
        raise ValueError("Unable to obtain current base model config")
    if _canonical_base_config(current_config) != _canonical_base_config(metadata["base_config"]):
        raise ValueError("Checkpoint base_config does not match the current base model")
    if expected_config is not None and _canonical_base_config(expected_config) != _canonical_base_config(
        metadata["base_config"]
    ):
        raise ValueError("Checkpoint base_config does not match expected_base_model")


def _validate_optimizer_state(
    optimizer_state: Any,
    topology: list[list[str]],
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
) -> None:
    if not isinstance(optimizer_state, Mapping):
        raise TypeError("Checkpoint optimizer state must be a mapping")
    saved_groups = optimizer_state.get("param_groups")
    if not isinstance(saved_groups, list) or len(saved_groups) != len(topology):
        raise ValueError("Checkpoint optimizer parameter-group topology does not match metadata")

    parameter_by_name = dict(named_parameters)
    saved_id_to_name: dict[Any, str] = {}
    for group, expected_names in zip(saved_groups, topology, strict=True):
        if not isinstance(group, Mapping) or group.get("params") is None:
            raise ValueError("Checkpoint optimizer parameter group is invalid")
        parameter_ids = group["params"]
        if not isinstance(parameter_ids, list) or len(parameter_ids) != len(expected_names):
            raise ValueError("Checkpoint optimizer parameter-group topology does not match metadata")
        for parameter_id, name in zip(parameter_ids, expected_names, strict=True):
            if parameter_id in saved_id_to_name or name not in parameter_by_name:
                raise ValueError("Checkpoint optimizer parameter topology is invalid")
            saved_id_to_name[parameter_id] = name

    saved_state = optimizer_state.get("state", {})
    if not isinstance(saved_state, Mapping):
        raise TypeError("Checkpoint optimizer state entries are invalid")
    for parameter_id, state in saved_state.items():
        name = saved_id_to_name.get(parameter_id)
        if name is None or not isinstance(state, Mapping):
            raise ValueError("Checkpoint optimizer state references an unknown parameter")
        parameter = parameter_by_name[name]
        for value in state.values():
            if torch.is_tensor(value) and value.numel() != 1 and tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(f"Optimizer state shape does not match parameter {name}")


def _validate_rng_state(state: Mapping[str, Any]) -> None:
    import numpy as np

    required = {"torch", "cuda", "numpy", "python"}
    missing = sorted(required - state.keys())
    if missing:
        raise ValueError(f"Training state is missing RNG keys: {missing}")
    if not torch.is_tensor(state["torch"]) or state["torch"].dtype != torch.uint8 or state["torch"].ndim != 1:
        raise ValueError("Training state torch RNG must be a tensor")
    cuda_values = state["cuda"] if isinstance(state["cuda"], list) else [state["cuda"]]
    if any(
        value is not None and (not torch.is_tensor(value) or value.dtype != torch.uint8 or value.ndim != 1)
        for value in cuda_values
    ):
        raise ValueError("Training state CUDA RNG must contain byte tensors")
    try:
        torch.Generator(device="cpu").set_state(state["torch"])
        np.random.RandomState().set_state(state["numpy"])
        random.Random().setstate(state["python"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Training state RNG payload is invalid") from exc


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    import numpy as np

    torch.set_rng_state(state["torch"])
    cuda_state = state["cuda"]
    if torch.cuda.is_available():
        if isinstance(cuda_state, list):
            # Legacy checkpoints saved all visible devices. Restore only the
            # current device; visibility may differ from the original machine.
            index = torch.cuda.current_device()
            cuda_state = cuda_state[index if index < len(cuda_state) else 0] if cuda_state else None
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state)
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def load_checkpoint(
    model: Any,
    path: str | os.PathLike[str],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    expected_base_model: Any | None = None,
    load_training_state: bool = False,
) -> dict[str, Any]:
    """Validate and load a native XVLA MORPHACT checkpoint.

    All metadata, trainable tensors, and requested training-state topology are
    validated before any model parameter is copied.  ``expected_base_model``
    may be a model object or an explicit source string/Hub ID.
    """

    checkpoint = Path(path)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    metadata = _read_metadata(checkpoint)

    trainable = _trainable_parameters(model)
    _validate_trainable_interface(model, trainable)
    _assert_model_contract(model)
    current_keys = {name for name, _ in trainable}
    metadata_keys = set(metadata["trainable_keys"])
    if current_keys != metadata_keys:
        missing = sorted(metadata_keys - current_keys)
        extra = sorted(current_keys - metadata_keys)
        raise ValueError(f"Trainable key set mismatch (missing={missing}, extra={extra})")

    _validate_base_model(model, metadata, expected_base_model)
    if _morphact_config(model) != metadata["morphact_config"]:
        raise ValueError("Checkpoint morphact_config does not match the current model")

    weights_path = checkpoint / ADAPTER_FILENAME
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    tensors = load_file(str(weights_path), device="cpu")
    if set(tensors) != metadata_keys:
        raise ValueError("Checkpoint tensor keys do not match metadata trainable_keys")

    for name, parameter in trainable:
        tensor = tensors[name]
        if tensor.dtype != torch.float32:
            raise ValueError(f"Checkpoint tensor {name} is not FP32")
        if tuple(tensor.shape) != tuple(parameter.shape):
            raise ValueError(f"Checkpoint tensor shape does not match parameter {name}")
        if parameter.dtype != torch.float32:
            raise ValueError(f"Current trainable parameter {name} is not FP32")

    training_state: Mapping[str, Any] | None = None
    if load_training_state:
        if not metadata["has_training_state"]:
            raise ValueError("Checkpoint does not contain resumable training state")
        if optimizer is None:
            raise ValueError("optimizer is required when load_training_state=True")
        state_path = checkpoint / TRAINING_STATE_FILENAME
        if not state_path.is_file():
            raise FileNotFoundError(state_path)
        # This is deliberately an explicit trusted-local load: the artifact
        # contains Python/NumPy RNG tuples that weights_only=True cannot restore.
        loaded_state = torch.load(state_path, map_location="cpu", weights_only=False)
        if not isinstance(loaded_state, Mapping):
            raise ValueError("Training state must be a mapping")
        if loaded_state.get("step") != metadata["step"]:
            raise ValueError("Training state step does not match metadata")
        _validate_rng_state(loaded_state)
        topology = metadata["optimizer_topology"]
        current_topology = _optimizer_topology(optimizer, trainable)
        if current_topology != topology:
            raise ValueError("Optimizer parameter topology does not match checkpoint metadata")
        _validate_optimizer_state(loaded_state.get("optimizer"), topology, trainable)
        training_state = loaded_state

    # No validation below this point should be capable of rejecting the model
    # tensor payload.  This keeps failed loads from partially updating params.
    with torch.no_grad():
        for name, parameter in trainable:
            parameter.copy_(tensors[name].to(device=parameter.device))

    if training_state is not None:
        optimizer.load_state_dict(training_state["optimizer"])
        _restore_rng_state(training_state)

    return deepcopy(metadata)
