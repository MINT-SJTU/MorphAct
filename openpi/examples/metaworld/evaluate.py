"""Headless MT50 evaluation for the Evo-1 MetaWorld dataset.

The dataset task labels are not all the same as MetaWorld's public V3 keys.
``MT50_TASKS`` is therefore deliberately explicit: evaluation resolves an
environment by its official key and selects its benchmark tasks by
``Task.env_name``.  It never uses a dataset task index as an environment id.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import importlib.metadata
import json
import logging
import os
import pathlib
import random
import sys
import time
import typing
import uuid

import numpy as np
from morphact.eval_seed import SEED_SCHEME, episode_seed
from morphact.eval_queue import task_episodes


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    """A dataset task and the corresponding official MetaWorld V3 key."""

    reference_index: int
    slug: str
    env_name: str
    prompt: str
    difficulty: str


# This order and the prompts are taken from mt50_order.json and tasks.jsonl in
# the Evo-1 evaluation bundle. The five non-identity env_name entries are the
# corresponding official keys in MetaWorld's MT50 V3 API.
MT50_TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(4, "button-press-topdown-v3", "button-press-topdown-v3", "Press a button from the top", "easy"),
    TaskSpec(
        5,
        "button-press-topdown-wall-v3",
        "button-press-topdown-wall-v3",
        "Bypass a wall and press a button from the top",
        "easy",
    ),
    TaskSpec(6, "button-press-v3", "button-press-v3", "Press a button", "easy"),
    TaskSpec(7, "button-press-wall-v3", "button-press-wall-v3", "Bypass a wall and press a button", "easy"),
    TaskSpec(8, "coffee-button-v3", "coffee-button-v3", "Push a button on the coffee machine", "easy"),
    TaskSpec(11, "dial-turn-v3", "dial-turn-v3", "Rotate a dial 180 degrees", "easy"),
    TaskSpec(13, "door-close-v3", "door-close-v3", "Close a door with a revolving joint", "easy"),
    TaskSpec(14, "door-lock-v3", "door-lock-v3", "Lock the door by rotating the lock clockwise", "easy"),
    TaskSpec(15, "door-v3", "door-open-v3", "Open a door with a revolving joint", "easy"),
    TaskSpec(16, "door-unlock-v3", "door-unlock-v3", "Unlock the door by rotating the lock counter-clockwise", "easy"),
    TaskSpec(18, "drawer-close-v3", "drawer-close-v3", "Push and close a drawer", "easy"),
    TaskSpec(19, "drawer-open-v3", "drawer-open-v3", "Open a drawer", "easy"),
    TaskSpec(20, "faucet-open-v3", "faucet-open-v3", "Rotate the faucet counter-clockwise", "easy"),
    TaskSpec(21, "faucet-close-v3", "faucet-close-v3", "Rotate the faucet clockwise", "easy"),
    TaskSpec(23, "handle-press-side-v3", "handle-press-side-v3", "Press a handle down sideways", "easy"),
    TaskSpec(24, "handle-press-v3", "handle-press-v3", "Press a handle down", "easy"),
    TaskSpec(25, "handle-pull-side-v3", "handle-pull-side-v3", "Pull a handle up sideways", "easy"),
    TaskSpec(26, "handle-pull-v3", "handle-pull-v3", "Pull a handle up", "easy"),
    TaskSpec(27, "lever-pull-v3", "lever-pull-v3", "Pull a lever down 90 degrees", "easy"),
    TaskSpec(31, "plate-slide-v3", "plate-slide-v3", "Slide a plate into a cabinet", "easy"),
    TaskSpec(32, "plate-slide-side-v3", "plate-slide-side-v3", "Slide a plate into a cabinet sideways", "easy"),
    TaskSpec(33, "plate-slide-back-v3", "plate-slide-back-v3", "Get a plate from the cabinet", "easy"),
    TaskSpec(
        34, "plate-slide-back-side-v3", "plate-slide-back-side-v3", "Get a plate from the cabinet sideways", "easy"
    ),
    TaskSpec(36, "peg-unplug-side-v3", "peg-unplug-side-v3", "Unplug a peg sideways", "easy"),
    TaskSpec(43, "reach-v3", "reach-v3", "Reach a goal position", "easy"),
    TaskSpec(44, "reach-wall-v3", "reach-wall-v3", "Bypass a wall and reach a goal", "easy"),
    TaskSpec(48, "window-open-v3", "window-open-v3", "Push and open a window", "easy"),
    TaskSpec(49, "window-close-v3", "window-close-v3", "Push and close a window", "easy"),
    TaskSpec(1, "basketball-v3", "basketball-v3", "Dunk the basketball into the basket", "medium"),
    TaskSpec(
        2, "bin-picking-v3", "bin-picking-v3", "Grasp the puck from one bin and place it into another bin", "medium"
    ),
    TaskSpec(3, "box-close-v3", "box-close-v3", "Grasp the cover and close the box with it", "medium"),
    TaskSpec(9, "coffee-pull-v3", "coffee-pull-v3", "Pull a mug from a coffee machine", "medium"),
    TaskSpec(10, "coffee-push-v3", "coffee-push-v3", "Push a mug under a coffee machine", "medium"),
    TaskSpec(22, "hammer-v3", "hammer-v3", "Hammer a screw on the wall", "medium"),
    TaskSpec(35, "peg-insertion-side-v3", "peg-insert-side-v3", "Insert a peg sideways", "medium"),
    TaskSpec(37, "soccer-v3", "soccer-v3", "Kick a soccer into the goal", "medium"),
    TaskSpec(41, "push-wall-v3", "push-wall-v3", "Bypass a wall and push a puck to a goal", "medium"),
    TaskSpec(46, "sweep-into-goal-v3", "sweep-into-v3", "Sweep a puck into a hole", "medium"),
    TaskSpec(47, "sweep-v3", "sweep-v3", "Sweep a puck off the table", "medium"),
    TaskSpec(0, "nut-assembly-v3", "assembly-v3", "Pick up a nut and place it onto a peg", "hard"),
    TaskSpec(17, "hand-insert-v3", "hand-insert-v3", "Insert the gripper into a hole", "hard"),
    TaskSpec(29, "pick-out-of-hole-v3", "pick-out-of-hole-v3", "Pick up a puck from a hole", "hard"),
    TaskSpec(30, "pick-place-v3", "pick-place-v3", "Pick and place a puck to a goal", "hard"),
    TaskSpec(40, "push-v3", "push-v3", "Push the puck to a goal", "hard"),
    TaskSpec(42, "push-back-v3", "push-back-v3", "Push the puck back to a goal", "hard"),
    TaskSpec(12, "nut-disassemble-v3", "disassemble-v3", "Pick a nut out of a peg", "very_hard"),
    TaskSpec(
        28, "pick-place-wall-v3", "pick-place-wall-v3", "Pick a puck, bypass a wall and place the puck", "very_hard"
    ),
    TaskSpec(38, "stick-push-v3", "stick-push-v3", "Grasp a stick and push a box using the stick", "very_hard"),
    TaskSpec(39, "stick-pull-v3", "stick-pull-v3", "Grasp a stick and pull a box with the stick", "very_hard"),
    TaskSpec(45, "shelf-place-v3", "shelf-place-v3", "Pick and place a puck onto a shelf", "very_hard"),
)


@dataclasses.dataclass(frozen=True)
class Args:
    """CLI arguments for MT50 evaluation."""

    host: str = "127.0.0.1"
    port: int = 8000
    output_dir: pathlib.Path = pathlib.Path("data/metaworld/evaluation")
    episodes_per_task: int = 10
    max_steps: int = 400
    replan_steps: int = 5
    seed: int = 0
    tasks: tuple[str, ...] = ()
    camera_name: str = "corner2"
    render_size: int = 480
    image_size: int | None = 448
    rotate_180: bool = True
    center_crop_ratio: float = 2.0 / 3.0
    save_videos: int = 0
    video_fps: int = 10
    worker_index: int = 0
    num_workers: int = 1
    queue_path: pathlib.Path | None = None
    merge_workers: bool = False


class _EventLog:
    def __init__(self, path: pathlib.Path) -> None:
        self._file = path.open("x", encoding="utf-8")

    def append(self, record: dict[str, typing.Any]) -> None:
        self._file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


def _error_text(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"


def _configure_mujoco_backend() -> None:
    # MuJoCo chooses its backend while importing MetaWorld/MuJoCo. EGL is the
    # headless default on Linux; macOS keeps its supported configured backend.
    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")


def _default_benchmark_factory(seed: int) -> typing.Any:
    _configure_mujoco_backend()
    import metaworld

    return metaworld.MT50(seed=seed)


def _default_client_factory(host: str, port: int) -> typing.Any:
    from openpi_client import websocket_client_policy

    return websocket_client_policy.WebsocketClientPolicy(host, port)


def _default_env_factory(env_cls: typing.Any, args: Args) -> typing.Any:
    # MetaWorld 3's public Gymnasium API supports RGB rendering with these
    # dimensions. Backend selection has already happened before MetaWorld import.
    return env_cls(
        render_mode="rgb_array",
        camera_name=args.camera_name,
        width=args.render_size,
        height=args.render_size,
    )


def _write_video(path: pathlib.Path, frames: list[np.ndarray], fps: int) -> None:
    import imageio

    imageio.mimwrite(path, [np.asarray(frame) for frame in frames], fps=fps)


def _validate_args(args: Args) -> None:
    if args.episodes_per_task <= 0 or args.max_steps <= 0 or args.replan_steps <= 0:
        raise ValueError("episodes_per_task, max_steps, and replan_steps must be positive")
    if args.render_size <= 0 or (args.image_size is not None and args.image_size <= 0):
        raise ValueError("render_size and image_size must be positive")
    if not 0.0 < args.center_crop_ratio <= 1.0:
        raise ValueError("center_crop_ratio must be in (0, 1]")
    if args.save_videos < 0 or args.video_fps <= 0:
        raise ValueError("save_videos must be non-negative and video_fps must be positive")
    if args.num_workers <= 0 or not 0 <= args.worker_index < args.num_workers:
        raise ValueError("worker_index must be in [0, num_workers)")


def _resolve_task_specs(selected_names: tuple[str, ...]) -> tuple[TaskSpec, ...]:
    if not selected_names:
        return MT50_TASKS
    selected = set(selected_names)
    known = {name for spec in MT50_TASKS for name in (spec.slug, spec.env_name)}
    unknown = sorted(selected - known)
    if unknown:
        raise ValueError(f"Unknown MetaWorld task selection: {', '.join(unknown)}")
    return tuple(spec for spec in MT50_TASKS if spec.slug in selected or spec.env_name in selected)


def _preprocess_image(image: typing.Any, args: Args) -> np.ndarray:
    """Apply the Evo-1 camera transform while preserving RGB channel order."""

    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected rendered RGB image with shape (H, W, 3), got {rgb.shape}")
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = (255 * np.clip(rgb, 0.0, 1.0)).astype(np.uint8)
    else:
        rgb = rgb.astype(np.uint8, copy=False)
    if args.rotate_180:
        rgb = rgb[::-1, ::-1]
    if args.center_crop_ratio < 1.0:
        height, width = rgb.shape[:2]
        crop_height = max(1, round(height * args.center_crop_ratio))
        crop_width = max(1, round(width * args.center_crop_ratio))
        y0 = (height - crop_height) // 2
        x0 = (width - crop_width) // 2
        rgb = rgb[y0 : y0 + crop_height, x0 : x0 + crop_width]
    rgb = np.ascontiguousarray(rgb)
    if args.image_size is None:
        return rgb

    from PIL import Image

    return np.ascontiguousarray(
        np.asarray(Image.fromarray(rgb).resize((args.image_size, args.image_size), Image.Resampling.BILINEAR))
    )


def _state_first_four(observation: typing.Any) -> np.ndarray:
    state = np.asarray(observation, dtype=np.float32).reshape(-1)
    if state.size < 4:
        raise ValueError(f"MetaWorld observation must contain at least four state values, got {state.size}")
    return np.ascontiguousarray(state[:4])


def _action_chunk(policy: typing.Any, element: dict[str, typing.Any], replan_steps: int) -> np.ndarray:
    response = policy.infer(element)
    if not isinstance(response, dict) or "actions" not in response:
        raise ValueError("Policy response must be a dictionary containing 'actions'")
    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.ndim != 2 or actions.shape[0] < replan_steps or actions.shape[1] != 4:
        raise ValueError(f"Policy action chunk must have shape (N>={replan_steps}, 4), got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("Policy action chunk contains NaN or infinity")
    return actions


def _clip_action(action: np.ndarray, action_space: typing.Any) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.size != 4:
        raise ValueError(f"Policy action must have exactly four values, got {action.size}")
    low = np.asarray(action_space.low, dtype=np.float32).reshape(-1)
    high = np.asarray(action_space.high, dtype=np.float32).reshape(-1)
    if low.size != 4 or high.size != 4:
        raise ValueError(f"Expected a four-dimensional MetaWorld action space, got low={low.shape}, high={high.shape}")
    return np.clip(action[-4:], low, high)


def _rollout_episode(
    *,
    env: typing.Any,
    task: typing.Any,
    spec: TaskSpec,
    episode_seed: int,
    policy: typing.Any,
    args: Args,
    capture_video: bool,
    policy_seed: int,
    episode_index: int,
) -> tuple[dict[str, typing.Any], list[np.ndarray]]:
    frames: list[np.ndarray] = []
    # A rerun of the same case must start a new server-side random stream.
    episode_id = f"{uuid.uuid4().hex}/{spec.slug}/{episode_index}"
    started_at = time.monotonic()
    steps = 0
    try:
        reset = getattr(policy, "reset", None)
        if callable(reset):
            reset()
        env.set_task(task)
        observation, _ = env.reset(seed=episode_seed)
        pending_actions: list[np.ndarray] = []
        while steps < args.max_steps:
            image = None
            if capture_video:
                image = _preprocess_image(env.render(), args)
                frames.append(image)
            if not pending_actions:
                if image is None:
                    image = _preprocess_image(env.render(), args)
                element = {
                    "observation/image": image,
                    "observation/state": _state_first_four(observation),
                    "prompt": spec.prompt,
                    "_eval_episode_id": episode_id,
                    "_eval_policy_seed": policy_seed,
                }
                chunk = _action_chunk(policy, element, args.replan_steps)
                pending_actions = list(chunk[: args.replan_steps])

            action = _clip_action(pending_actions.pop(0), env.action_space)
            observation, _, terminated, truncated, info = env.step(action)
            steps += 1
            success = bool(info.get("success", False)) if isinstance(info, dict) else False
            if success:
                return _episode_result(
                    "success", success=True, steps=steps, started_at=started_at, error=None, termination="success"
                ), frames
            if terminated:
                return _episode_result(
                    "failure", success=False, steps=steps, started_at=started_at, error=None, termination="terminated"
                ), frames
            if truncated:
                return _episode_result(
                    "failure", success=False, steps=steps, started_at=started_at, error=None, termination="truncated"
                ), frames
        return _episode_result(
            "failure", success=False, steps=steps, started_at=started_at, error=None, termination="horizon"
        ), frames
    except Exception as error:
        return _episode_result(
            "error", success=False, steps=steps, started_at=started_at, error=_error_text(error), termination="error"
        ), frames


def _episode_result(
    status: str,
    *,
    success: bool,
    steps: int,
    started_at: float,
    error: str | None,
    termination: str,
) -> dict[str, typing.Any]:
    return {
        "status": status,
        "success": success,
        "steps": steps,
        "elapsed_s": round(time.monotonic() - started_at, 3),
        "error": error,
        "termination": termination,
    }


def _task_stats(records: list[dict[str, typing.Any]], spec: TaskSpec) -> dict[str, typing.Any]:
    task_records = [record for record in records if record["task"] == spec.slug]
    successes = sum(record["status"] == "success" for record in task_records)
    failures = sum(record["status"] == "failure" for record in task_records)
    errors = sum(record["status"] == "error" for record in task_records)
    total = len(task_records)
    return {
        "task": spec.slug,
        "metaworld_env": spec.env_name,
        "reference_index": spec.reference_index,
        "prompt": spec.prompt,
        "difficulty": spec.difficulty,
        "episodes": total,
        "successes": successes,
        "failures": failures,
        "errors": errors,
        # Errors stay in the denominator so an infrastructure error cannot
        # inflate a benchmark result.
        "success_rate": successes / total if total else 0.0,
    }


def _group_stats(per_task: list[dict[str, typing.Any]]) -> list[dict[str, typing.Any]]:
    groups: dict[str, list[dict[str, typing.Any]]] = {}
    for task in per_task:
        groups.setdefault(task["difficulty"], []).append(task)
    return [
        {
            "difficulty": difficulty,
            "episodes": sum(task["episodes"] for task in tasks),
            "successes": sum(task["successes"] for task in tasks),
            "failures": sum(task["failures"] for task in tasks),
            "errors": sum(task["errors"] for task in tasks),
            "success_rate": (
                sum(task["successes"] for task in tasks) / sum(task["episodes"] for task in tasks)
                if sum(task["episodes"] for task in tasks) else 0.0
            ),
        }
        for difficulty, tasks in groups.items()
    ]


def _mean_success_rate(
    args: Args,
    specs: tuple[TaskSpec, ...],
    records: list[dict[str, typing.Any]],
    per_difficulty: list[dict[str, typing.Any]],
) -> float | None:
    """Report the four-difficulty Mean only for a complete, error-free MT50 run."""
    if len(specs) != len(MT50_TASKS) or set(specs) != set(MT50_TASKS):
        return None
    expected = {(spec.reference_index, episode) for spec in specs for episode in range(args.episodes_per_task)}
    if not expected or len(records) != len(expected):
        return None
    by_id = {spec.reference_index: spec for spec in specs}
    seen = set()
    for record in records:
        task_id, episode = record.get("reference_index"), record.get("episode_index")
        if type(task_id) is not int or type(episode) is not int:
            return None
        case = (task_id, episode)
        if case not in expected or case in seen or record["task"] != by_id[task_id].slug:
            return None
        if (record["status"] not in ("success", "failure") or type(record.get("success")) is not bool
                or record["success"] != (record["status"] == "success") or record.get("error") is not None):
            return None
        seen.add(case)
    if {group["difficulty"] for group in per_difficulty} != {"easy", "medium", "hard", "very_hard"}:
        return None
    return sum(group["success_rate"] for group in per_difficulty) / 4


def _write_summary(output_dir: pathlib.Path, summary: dict[str, typing.Any]) -> None:
    with (output_dir / "summary.json").open("x", encoding="utf-8") as summary_file:
        json.dump(summary, summary_file, ensure_ascii=False, indent=2)
        summary_file.write("\n")


def _server_metadata(policy: typing.Any) -> typing.Any:
    getter = getattr(policy, "get_server_metadata", None)
    if not callable(getter):
        return None
    try:
        return getter()
    except Exception as error:
        return {"error": _error_text(error)}


def _runtime_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("metaworld", "mujoco", "gymnasium"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _summary(
    args: Args,
    specs: tuple[TaskSpec, ...],
    records: list[dict[str, typing.Any]],
    server_metadata: typing.Any,
    runtime_versions: dict[str, str | None],
) -> dict[str, typing.Any]:
    per_task = [_task_stats(records, spec) for spec in specs]
    per_difficulty = _group_stats(per_task)
    total = len(records)
    successes = sum(record["status"] == "success" for record in records)
    failures = sum(record["status"] == "failure" for record in records)
    errors = sum(record["status"] == "error" for record in records)
    return {
        "schema_version": 1,
        "created_at": _utc_now(),
        "runtime_versions": runtime_versions,
        "server_metadata": server_metadata,
        "configuration": {
            "seed": args.seed,
            "seed_scheme": SEED_SCHEME,
            "episodes_per_task": args.episodes_per_task,
            "max_steps": args.max_steps,
            "replan_steps": args.replan_steps,
            "camera_name": args.camera_name,
            "render_size": args.render_size,
            "image_size": args.image_size,
            "rotate_180": args.rotate_180,
            "center_crop_ratio": args.center_crop_ratio,
        },
        "episodes": total,
        "successes": successes,
        "failures": failures,
        "errors": errors,
        "success_rate": successes / total if total else 0.0,
        "mean_success_rate": _mean_success_rate(args, specs, records, per_difficulty),
        "per_task": per_task,
        "per_difficulty": per_difficulty,
    }


def merge_worker_results(args: Args) -> dict[str, typing.Any]:
    """Validate and combine disjoint episode shards from policy workers."""
    _validate_args(args)
    specs = _resolve_task_specs(args.tasks)
    expected = {
        (spec.reference_index, episode): (task_index * args.episodes_per_task + episode) % args.num_workers
        for task_index, spec in enumerate(specs)
        for episode in range(args.episodes_per_task)
    }
    records: dict[tuple[int, int], dict[str, typing.Any]] = {}
    metadata = []
    runtime_versions = None
    for worker in range(args.num_workers):
        worker_dir = args.output_dir / f"worker-{worker}"
        worker_summary = json.loads((worker_dir / "summary.json").read_text(encoding="utf-8"))
        if worker_summary.get("worker_index") != worker or worker_summary.get("num_workers") != args.num_workers:
            raise ValueError(f"Invalid worker identity: {worker_dir}")
        if worker_summary.get("scheduling", "static") != ("dynamic" if args.queue_path else "static"):
            raise ValueError(f"Mismatched worker scheduling: {worker_dir}")
        config = worker_summary["configuration"]
        for key in (
            "seed", "episodes_per_task", "max_steps", "replan_steps", "camera_name",
            "render_size", "image_size", "rotate_180", "center_crop_ratio",
        ):
            if config[key] != getattr(args, key):
                raise ValueError(f"Mismatched {key}: {worker_dir}")
        if config["seed_scheme"] != SEED_SCHEME:
            raise ValueError(f"Mismatched seed scheme: {worker_dir}")
        if runtime_versions is None:
            runtime_versions = worker_summary["runtime_versions"]
        metadata.append(worker_summary["server_metadata"])
        worker_records = [json.loads(line) for line in (worker_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
        if (len(worker_records) != worker_summary["episodes"] or worker_summary["errors"]
                or sum(row["status"] == "success" for row in worker_records) != worker_summary["successes"]):
            raise ValueError(f"Incomplete or errored worker: {worker_dir}")
        for record in worker_records:
            case = (record["reference_index"], record["episode_index"])
            if (case in records or case not in expected
                    or (args.queue_path is None and expected[case] != worker)):
                raise ValueError(f"Duplicate or incorrectly assigned episode: {case}")
            if (record["status"] not in ("success", "failure") or type(record["success"]) is not bool
                    or record["success"] != (record["status"] == "success")):
                raise ValueError(f"Invalid episode result: {case}")
            for stream, key in (("environment", "episode_seed"), ("policy", "policy_seed")):
                if record[key] != episode_seed(args.seed, "metaworld_mt50", case[0], case[1], stream):
                    raise ValueError(f"Mismatched episode seed: {case}")
            if record.get("video"):
                record["video"] = f"worker-{worker}/{record['video']}"
            records[case] = record
    if set(records) != set(expected):
        raise ValueError(f"Missing episodes: {len(records)}/{len(expected)}")
    ordered = [records[spec.reference_index, episode] for spec in specs for episode in range(args.episodes_per_task)]
    summary = _summary(args, specs, ordered, metadata, runtime_versions)
    summary["workers"] = args.num_workers
    with (args.output_dir / "episodes.jsonl").open("x", encoding="utf-8") as event_log:
        for record in ordered:
            event_log.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    _write_summary(args.output_dir, summary)
    return summary


def evaluate(
    args: Args,
    *,
    benchmark_factory: typing.Callable[[int], typing.Any] = _default_benchmark_factory,
    client_factory: typing.Callable[[str, int], typing.Any] = _default_client_factory,
    env_factory: typing.Callable[[typing.Any, Args], typing.Any] = _default_env_factory,
    video_writer: typing.Callable[[pathlib.Path, list[np.ndarray], int], None] = _write_video,
) -> dict[str, typing.Any]:
    """Evaluate the selected MT50 tasks and return the persisted summary."""

    _validate_args(args)
    specs = _resolve_task_specs(args.tasks)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    event_log = _EventLog(args.output_dir / "episodes.jsonl")
    records: list[dict[str, typing.Any]] = []
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        benchmark = benchmark_factory(args.seed)
        policy = client_factory(args.host, args.port)
        server_metadata = _server_metadata(policy)
        jobs = task_episodes(range(len(specs)), args.episodes_per_task,
                             args.worker_index, args.num_workers, args.queue_path)
        for task_index, episode_indices in jobs:
            spec = specs[task_index]
            if spec.env_name not in benchmark.train_classes:
                raise RuntimeError(f"MetaWorld MT50 does not expose expected environment {spec.env_name!r}")
            tasks = [task for task in benchmark.train_tasks if task.env_name == spec.env_name]
            if not tasks:
                raise RuntimeError(f"MetaWorld MT50 has no benchmark tasks for {spec.env_name!r}")
            for episode_index in episode_indices:
                env_seed = episode_seed(args.seed, "metaworld_mt50", spec.reference_index, episode_index, "environment")
                policy_seed = episode_seed(args.seed, "metaworld_mt50", spec.reference_index, episode_index, "policy")
                random.seed(env_seed)
                np.random.seed(env_seed)
                env = env_factory(benchmark.train_classes[spec.env_name], args)
                try:
                    result, frames = _rollout_episode(
                        env=env,
                        task=tasks[episode_index % len(tasks)],
                        spec=spec,
                        episode_seed=env_seed,
                        policy=policy,
                        args=args,
                        capture_video=episode_index < args.save_videos,
                        policy_seed=policy_seed,
                        episode_index=episode_index,
                    )
                    record = {
                        "timestamp": _utc_now(),
                        "task": spec.slug,
                        "metaworld_env": spec.env_name,
                        "reference_index": spec.reference_index,
                        "difficulty": spec.difficulty,
                        "prompt": spec.prompt,
                        "episode_index": episode_index,
                        "episode_seed": env_seed,
                        "policy_seed": policy_seed,
                        "video": None,
                        **result,
                    }
                    if frames and episode_index < args.save_videos and result["status"] != "error":
                        video_path = (
                            pathlib.Path("videos") / f"{spec.slug}_episode_{episode_index:03d}_{result['status']}.mp4"
                        )
                        try:
                            (args.output_dir / video_path).parent.mkdir(exist_ok=True)
                            video_writer(args.output_dir / video_path, frames, args.video_fps)
                            record["video"] = video_path.as_posix()
                        except Exception as error:
                            record["video_error"] = _error_text(error)
                    records.append(record)
                    event_log.append(record)
                finally:
                    env.close()
    finally:
        event_log.close()

    summary = _summary(args, specs, records, server_metadata, _runtime_versions())
    if args.num_workers > 1:
        summary.update(worker_index=args.worker_index, num_workers=args.num_workers)
        summary["scheduling"] = "dynamic" if args.queue_path else "static"
    _write_summary(args.output_dir, summary)
    return summary


def main() -> None:
    import tyro

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = tyro.cli(Args)
    try:
        summary = merge_worker_results(args) if args.merge_workers else evaluate(args)
    except Exception as error:
        logging.error("MetaWorld evaluation failed: %s", _error_text(error))
        raise SystemExit(2) from error
    mean = summary["mean_success_rate"]
    logging.info(
        "MetaWorld evaluation finished: Mean=%s micro_success_rate=%.3f successes=%s episodes=%s failures=%s errors=%s",
        f"{100 * mean:.2f}%" if mean is not None else "unavailable (requires complete, error-free MT50)",
        summary["success_rate"], summary["successes"], summary["episodes"], summary["failures"], summary["errors"],
    )
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
