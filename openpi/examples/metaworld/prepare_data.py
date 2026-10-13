"""Download and validate the Evo1 MetaWorld LeRobot v2.1 dataset.

By default this script only validates an existing dataset. ``--download`` fetches
metadata only, while ``--sample-episodes`` fetches a small, explicitly incomplete
probe. Downloading all videos requires ``--full-download``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable
import json
import os
from pathlib import Path
from typing import Any

REPO_ID = "MINT-SJTU/Evo1_MetaWorld_Dataset"
REVISION = "612c4e83d0e3b4e0be5227970bd8f5f07fc0c488"
CODEBASE_VERSION = "v2.1"
SPLIT = {"train": "0:2500"}
TOTAL_EPISODES = 2500
TOTAL_FRAMES = 188489
TOTAL_TASKS = 50
EPISODES_PER_TASK = 50
FPS = 30
CHUNK_SIZE = 1000
IMAGE_KEY = "observation.images.image"
DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
METADATA_PATTERNS = ["meta/*.json", "meta/*.jsonl"]
SOURCE_MARKER_FILENAME = ".morphact-source.json"
SOURCE_PROVENANCE = {"repo_id": REPO_ID, "revision": REVISION}


class DatasetValidationError(RuntimeError):
    """Validation failed, with a manifest that describes the local dataset."""

    def __init__(self, errors: list[str], manifest: dict[str, Any]):
        super().__init__("\n".join(errors))
        self.errors = errors
        self.manifest = manifest


def default_dataset_root() -> Path:
    """Return the LeRobot-compatible default location without importing lerobot."""
    lerobot_home = os.environ.get("HF_LEROBOT_HOME")
    if lerobot_home:
        return Path(lerobot_home).expanduser() / REPO_ID

    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    return hf_home / "lerobot" / REPO_ID


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise ValueError(f"{path} has an empty line at {line_number}")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{path} line {line_number} is not a JSON object")
            records.append(record)
    return records


def _add_equal_error(errors: list[str], label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        errors.append(f"{label}: expected {expected!r}, found {actual!r}")


def _feature_summary(feature: Any) -> dict[str, Any]:
    if not isinstance(feature, dict):
        return {"present": False}
    return {
        "present": True,
        "dtype": feature.get("dtype"),
        "shape": feature.get("shape"),
        "names_field_present": "names" in feature,
        "names": feature.get("names"),
    }


def episode_path(template: str, episode_index: int) -> str:
    """Format one published data or video path from an episode index."""
    return template.format(
        episode_chunk=episode_index // CHUNK_SIZE,
        episode_index=episode_index,
        video_key=IMAGE_KEY,
    )


def _files_with_suffix(root: Path, directory: str, suffix: str) -> set[str]:
    base = root / directory
    if not base.is_dir():
        return set()
    return {path.relative_to(root).as_posix() for path in base.rglob(f"*{suffix}") if path.is_file()}


def _collection_summary(expected: set[str], actual: set[str]) -> dict[str, Any]:
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    return {
        "expected": len(expected),
        "present": len(actual),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "missing_examples": missing[:10],
        "unexpected_examples": unexpected[:10],
    }


def _validate_info(info: Any, errors: list[str]) -> dict[str, Any]:
    if not isinstance(info, dict):
        errors.append("meta/info.json must contain a JSON object")
        return {}

    _add_equal_error(errors, "info.codebase_version", info.get("codebase_version"), CODEBASE_VERSION)
    _add_equal_error(errors, "info.fps", info.get("fps"), FPS)
    _add_equal_error(errors, "info.total_episodes", info.get("total_episodes"), TOTAL_EPISODES)
    _add_equal_error(errors, "info.total_frames", info.get("total_frames"), TOTAL_FRAMES)
    _add_equal_error(errors, "info.total_tasks", info.get("total_tasks"), TOTAL_TASKS)
    _add_equal_error(errors, "info.total_videos", info.get("total_videos"), TOTAL_EPISODES)
    _add_equal_error(errors, "info.total_chunks", info.get("total_chunks"), 3)
    _add_equal_error(errors, "info.chunks_size", info.get("chunks_size"), CHUNK_SIZE)
    _add_equal_error(errors, "info.splits", info.get("splits"), SPLIT)
    _add_equal_error(errors, "info.data_path", info.get("data_path"), DATA_PATH)
    _add_equal_error(errors, "info.video_path", info.get("video_path"), VIDEO_PATH)

    features = info.get("features")
    if not isinstance(features, dict):
        errors.append("info.features must contain a JSON object")
        return {}

    for key in ("action", "observation.state"):
        feature = features.get(key)
        if not isinstance(feature, dict):
            errors.append(f"info.features.{key} is missing")
            continue
        _add_equal_error(errors, f"info.features.{key}.dtype", feature.get("dtype"), "float32")
        _add_equal_error(errors, f"info.features.{key}.shape", feature.get("shape"), [4])
        if "names" in feature and feature["names"] is not None:
            errors.append(f"info.features.{key}.names must be absent or null for this dataset revision")

    image = features.get(IMAGE_KEY)
    if not isinstance(image, dict):
        errors.append(f"info.features.{IMAGE_KEY} is missing")
    else:
        _add_equal_error(errors, f"info.features.{IMAGE_KEY}.dtype", image.get("dtype"), "video")
        _add_equal_error(errors, f"info.features.{IMAGE_KEY}.shape", image.get("shape"), [480, 480, 3])
        video_info = image.get("info")
        if not isinstance(video_info, dict):
            errors.append(f"info.features.{IMAGE_KEY}.info is missing")
        else:
            _add_equal_error(
                errors,
                f"info.features.{IMAGE_KEY}.info.video.codec",
                video_info.get("video.codec"),
                "av1",
            )
            _add_equal_error(
                errors,
                f"info.features.{IMAGE_KEY}.info.video.fps",
                video_info.get("video.fps"),
                FPS,
            )

    return features


def _validate_tasks(tasks: Any, errors: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    if not isinstance(tasks, list):
        errors.append("meta/tasks.jsonl must contain JSON objects")
        return {}, {}

    task_index_to_prompt: dict[int, str] = {}
    for record in tasks:
        task_index = record.get("task_index")
        prompt = record.get("task")
        if not isinstance(task_index, int) or not isinstance(prompt, str) or not prompt:
            errors.append("every task record needs an integer task_index and non-empty task prompt")
            continue
        if task_index in task_index_to_prompt:
            errors.append(f"meta/tasks.jsonl repeats task_index {task_index}")
            continue
        task_index_to_prompt[task_index] = prompt

    _add_equal_error(errors, "tasks count", len(tasks), TOTAL_TASKS)
    _add_equal_error(errors, "task indices", sorted(task_index_to_prompt), list(range(TOTAL_TASKS)))
    prompts = list(task_index_to_prompt.values())
    if len(set(prompts)) != len(prompts):
        errors.append("meta/tasks.jsonl contains duplicate task prompts")

    # Prompts, rather than a second task-order file, are the stable join key.
    prompt_to_task_index = {prompt: index for index, prompt in task_index_to_prompt.items()}
    return prompt_to_task_index, task_index_to_prompt


def _validate_episodes(
    episodes: Any, prompt_to_task_index: dict[str, int], errors: list[str]
) -> tuple[Counter[str], int]:
    if not isinstance(episodes, list):
        errors.append("meta/episodes.jsonl must contain JSON objects")
        return Counter(), 0

    indices: list[int] = []
    lengths: list[int] = []
    prompt_counts: Counter[str] = Counter()
    for record in episodes:
        episode_index = record.get("episode_index")
        length = record.get("length")
        prompts = record.get("tasks")
        if not isinstance(episode_index, int):
            errors.append("every episode record needs an integer episode_index")
        else:
            indices.append(episode_index)
        if not isinstance(length, int) or length <= 0:
            errors.append("every episode record needs a positive integer length")
        else:
            lengths.append(length)
        if not isinstance(prompts, list) or len(prompts) != 1 or not isinstance(prompts[0], str):
            errors.append("every episode record needs exactly one task prompt")
            continue
        prompt = prompts[0]
        if prompt not in prompt_to_task_index:
            errors.append(f"episode {episode_index!r} references unknown task prompt {prompt!r}")
        else:
            prompt_counts[prompt] += 1

    _add_equal_error(errors, "episodes count", len(episodes), TOTAL_EPISODES)
    _add_equal_error(errors, "episode indices", sorted(indices), list(range(TOTAL_EPISODES)))
    length_sum = sum(lengths)
    _add_equal_error(errors, "sum of episode lengths", length_sum, TOTAL_FRAMES)
    for prompt in prompt_to_task_index:
        count = prompt_counts[prompt]
        if count != EPISODES_PER_TASK:
            errors.append(f"task prompt {prompt!r}: expected {EPISODES_PER_TASK} episodes, found {count}")
    return prompt_counts, length_sum


def _build_manifest(
    root: Path,
    info: dict[str, Any],
    features: dict[str, Any],
    task_index_to_prompt: dict[int, str],
    prompt_counts: Counter[str],
    length_sum: int,
    data_summary: dict[str, Any],
    video_summary: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    complete = (
        data_summary["missing"] == 0
        and data_summary["unexpected"] == 0
        and video_summary["missing"] == 0
        and video_summary["unexpected"] == 0
    )
    return {
        "check_type": "structural",
        "expected_source": dict(SOURCE_PROVENANCE),
        "local_root": str(root),
        "schema": {
            "codebase_version": info.get("codebase_version"),
            "fps": info.get("fps"),
            "image_feature": _feature_summary(features.get(IMAGE_KEY)),
            "action_feature": _feature_summary(features.get("action")),
            "state_feature": _feature_summary(features.get("observation.state")),
            "vector_names_note": "Published v2.1 metadata omits action/state names; this tool does not synthesize them.",
        },
        "split": info.get("splits"),
        "episodes": {
            "count": info.get("total_episodes"),
            "frames": info.get("total_frames"),
            "metadata_length_sum": length_sum,
        },
        "tasks": {
            "count": len(task_index_to_prompt),
            "episode_counts_by_prompt": dict(sorted(prompt_counts.items())),
            "task_index_to_prompt": {str(index): prompt for index, prompt in sorted(task_index_to_prompt.items())},
            "prompt_to_task_index": {prompt: index for index, prompt in sorted(task_index_to_prompt.items())},
        },
        "files": {"data": data_summary, "videos": video_summary},
        "complete": complete,
        "valid": not errors,
        "errors": errors,
    }


def validate_dataset(root: Path | str, *, allow_partial: bool = False) -> dict[str, Any]:
    """Validate local Evo1 files and return a JSON-serializable manifest.

    ``allow_partial`` accepts missing data/video files for metadata-only and sampled
    downloads, but it never accepts malformed metadata or unexpected data files.
    """
    root = Path(root).expanduser().resolve()
    errors: list[str] = []
    metadata_paths = {
        "info": root / "meta/info.json",
        "tasks": root / "meta/tasks.jsonl",
        "episodes": root / "meta/episodes.jsonl",
        "episodes_stats": root / "meta/episodes_stats.jsonl",
    }
    missing_metadata = [str(path.relative_to(root)) for path in metadata_paths.values() if not path.is_file()]
    if missing_metadata:
        errors.append(f"missing metadata files: {', '.join(missing_metadata)}")
        manifest = {
            "check_type": "structural",
            "expected_source": dict(SOURCE_PROVENANCE),
            "local_root": str(root),
            "complete": False,
            "valid": False,
            "errors": errors,
        }
        raise DatasetValidationError(errors, manifest)

    try:
        info = _read_json(metadata_paths["info"])
        tasks = _read_jsonl(metadata_paths["tasks"])
        episodes = _read_jsonl(metadata_paths["episodes"])
    except (OSError, ValueError) as exc:
        errors.append(f"failed to read metadata: {exc}")
        manifest = {
            "check_type": "structural",
            "expected_source": dict(SOURCE_PROVENANCE),
            "local_root": str(root),
            "complete": False,
            "valid": False,
            "errors": errors,
        }
        raise DatasetValidationError(errors, manifest) from exc

    features = _validate_info(info, errors)
    prompt_to_task_index, task_index_to_prompt = _validate_tasks(tasks, errors)
    prompt_counts, length_sum = _validate_episodes(episodes, prompt_to_task_index, errors)

    expected_data = {episode_path(DATA_PATH, index) for index in range(TOTAL_EPISODES)}
    expected_videos = {episode_path(VIDEO_PATH, index) for index in range(TOTAL_EPISODES)}
    actual_data = _files_with_suffix(root, "data", ".parquet")
    actual_videos = _files_with_suffix(root, "videos", ".mp4")
    data_summary = _collection_summary(expected_data, actual_data)
    video_summary = _collection_summary(expected_videos, actual_videos)

    for label, summary in (("data", data_summary), ("videos", video_summary)):
        if summary["unexpected"]:
            errors.append(f"{label} collection contains {summary['unexpected']} unexpected files")
        if summary["missing"] and not allow_partial:
            errors.append(f"{label} collection is incomplete: {summary['missing']} files are missing")

    manifest = _build_manifest(
        root,
        info if isinstance(info, dict) else {},
        features,
        task_index_to_prompt,
        prompt_counts,
        length_sum,
        data_summary,
        video_summary,
        errors,
    )
    if errors:
        raise DatasetValidationError(errors, manifest)
    return manifest


def _sample_patterns(sample_episodes: int) -> list[str]:
    return [
        *METADATA_PATTERNS,
        *(episode_path(DATA_PATH, index) for index in range(sample_episodes)),
        *(episode_path(VIDEO_PATH, index) for index in range(sample_episodes)),
    ]


def download_dataset(
    root: Path | str,
    *,
    sample_episodes: int | None = None,
    full_download: bool = False,
) -> Path:
    """Download Evo1 into a new directory or resume a marker-backed download."""
    root = Path(root).expanduser()
    if root.is_symlink():
        raise FileExistsError(f"refusing to download into symlinked path: {root}")
    root = root.resolve()
    if sample_episodes is not None and not 1 <= sample_episodes <= TOTAL_EPISODES:
        raise ValueError(f"sample_episodes must be in [1, {TOTAL_EPISODES}]")
    if full_download and sample_episodes is not None:
        raise ValueError("--full-download cannot be combined with --sample-episodes")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("--download requires huggingface_hub; install the OpenPI dependencies") from exc

    if full_download:
        allow_patterns: Iterable[str] | None = None
    elif sample_episodes is None:
        allow_patterns = METADATA_PATTERNS
    else:
        allow_patterns = _sample_patterns(sample_episodes)

    marker_path = root / SOURCE_MARKER_FILENAME
    if root.exists():
        try:
            marker = _read_json(marker_path)
        except (OSError, ValueError) as exc:
            raise FileExistsError(
                f"refusing to resume existing path without a readable matching source marker: {root}"
            ) from exc
        if marker != SOURCE_PROVENANCE:
            raise FileExistsError(f"refusing to resume path with a different source marker: {root}")
    else:
        root.parent.mkdir(parents=True, exist_ok=True)
        root.mkdir()
        with marker_path.open("x", encoding="utf-8") as file:
            json.dump(SOURCE_PROVENANCE, file, indent=2, sort_keys=True)
            file.write("\n")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        local_dir=root,
        allow_patterns=allow_patterns,
    )
    return root


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_dataset_root())
    parser.add_argument("--download", action="store_true", help="download metadata or the selected files")
    download_mode = parser.add_mutually_exclusive_group()
    download_mode.add_argument(
        "--metadata-only",
        action="store_true",
        help="explicitly select metadata-only download (the --download default)",
    )
    download_mode.add_argument(
        "--sample-episodes",
        type=int,
        metavar="N",
        help="download metadata plus episodes [0, N); this remains incomplete",
    )
    download_mode.add_argument(
        "--full-download",
        action="store_true",
        help="download all dataset files, including the full video collection",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="allow missing data/video files while still validating all metadata",
    )
    parser.add_argument("--manifest", type=Path, help="write the manifest to a new JSON file")
    args = parser.parse_args(argv)
    if (args.metadata_only or args.sample_episodes is not None or args.full_download) and not args.download:
        parser.error("download mode flags require --download")
    if args.sample_episodes is not None and not 1 <= args.sample_episodes <= TOTAL_EPISODES:
        parser.error(f"--sample-episodes must be in [1, {TOTAL_EPISODES}]")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.download:
        download_dataset(
            args.root,
            sample_episodes=args.sample_episodes,
            full_download=args.full_download,
        )

    allow_partial = args.allow_partial or (args.download and not args.full_download)
    try:
        manifest = validate_dataset(args.root, allow_partial=allow_partial)
        exit_code = 0
    except DatasetValidationError as exc:
        manifest = exc.manifest
        exit_code = 1

    if args.manifest:
        _write_manifest(args.manifest, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if exit_code:
        print("Validation failed:", *manifest["errors"], sep="\n", file=os.sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
