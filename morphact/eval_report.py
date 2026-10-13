"""Normalize completed simulator evaluations without importing model dependencies."""

import argparse
import json
from pathlib import Path, PurePosixPath
import sys

from morphact.eval_seed import SEED_SCHEME, episode_seed


LIBERO_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
MW_GROUPS = ("easy", "medium", "hard", "very_hard")
GROUP_LABELS = {
    "libero_spatial": "Spatial", "libero_object": "Object", "libero_goal": "Goal", "libero_10": "Long",
    "easy": "Easy", "medium": "Medium", "hard": "Hard", "very_hard": "Very Hard",
}


def _read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object: %s" % path)
    return value


def _integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError("%s must be a nonnegative integer" % name)
    return value


def _metadata(model, benchmark, output_dir):
    if model == "xvla":
        return _read_json(output_dir / "evaluation.json")
    if benchmark == "metaworld":
        return _read_json(output_dir / "summary.json")
    return {}


def _configuration(metadata, seed, episodes_per_task, suites):
    config = metadata.get("configuration", {})
    if not isinstance(config, dict):
        raise ValueError("configuration must be an object")
    config = dict(config)
    for key in ("seed", "episodes_per_task", "replan_steps", "max_steps"):
        if key in metadata and key not in config:
            config[key] = metadata[key]
    for key, expected in (("seed", seed), ("episodes_per_task", episodes_per_task)):
        if key in config and (type(config[key]) is not int or config[key] != expected):
            raise ValueError("Source configuration differs from requested %s" % key)
    for container in (metadata, config):
        for key in ("seed_scheme", "seed_protocol"):
            if key in container and container[key] != SEED_SCHEME:
                raise ValueError("Unknown source seed protocol")
    config.update(seed=seed, episodes_per_task=episodes_per_task, seed_scheme=SEED_SCHEME)
    if suites is not None:
        config["suites"] = list(suites)
    return config


def _task_specs(benchmark, metadata, suites):
    if benchmark == "libero":
        return {(suite, task): {"suite": suite, "task_id": task, "task": None, "group": suite}
                for suite in suites for task in range(10)}
    native = metadata.get("per_task")
    if not isinstance(native, list) or len(native) != 50:
        raise ValueError("Meta-World requires the native per_task configuration for all 50 tasks")
    specs = {}
    labels = set()
    for row in native:
        if not isinstance(row, dict):
            raise ValueError("Invalid Meta-World task configuration")
        task_id = _integer(row.get("reference_index", row.get("task_id")), "task_id")
        if "task_id" in row and row["task_id"] != task_id:
            raise ValueError("Inconsistent native task identity aliases")
        task = row.get("task")
        group = row.get("difficulty", row.get("group"))
        key = ("metaworld_mt50", task_id)
        if key in specs or not isinstance(task, str) or not task or task in labels or group not in MW_GROUPS:
            raise ValueError("Unknown or duplicate Meta-World task configuration")
        labels.add(task)
        specs[key] = {**row, "suite": key[0], "task_id": task_id, "task": task, "group": group,
                      "reference_index": task_id, "difficulty": group}
    if {row["group"] for row in specs.values()} != set(MW_GROUPS):
        raise ValueError("Meta-World task configuration must cover all four difficulty groups")
    if {row["task_id"] for row in specs.values()} != set(range(50)):
        raise ValueError("Meta-World task IDs must cover exactly 0..49")
    return specs


def _source_files(model, benchmark, output_dir, suites):
    if model == "xvla":
        return sorted(output_dir.glob("worker-*.jsonl"))
    if benchmark == "metaworld":
        return [output_dir / "episodes.jsonl"]
    paths = []
    for suite in suites:
        paths.extend(sorted((output_dir / suite).glob("episodes.jsonl")))
        paths.extend(sorted((output_dir / suite).glob("worker-*/episodes.jsonl")))
    return paths


def _records(paths):
    for path in paths:
        if not path.exists():
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as error:
                raise ValueError("Invalid JSON at %s:%s" % (path, line_number)) from error
            if not isinstance(row, dict):
                raise ValueError("Expected an episode object at %s:%s" % (path, line_number))
            yield path, row


def _normalize(row, path, output_dir, benchmark, specs, seed, episodes_per_task, source_protocol):
    suite = row.get("suite", "metaworld_mt50" if benchmark == "metaworld" else None)
    if not isinstance(suite, str):
        raise ValueError("Episode suite must be a string")
    task_id = _integer(row.get("task_id", row.get("reference_index")), "task_id")
    episode = _integer(row.get("episode_index"), "episode_index")
    key = (suite, task_id)
    if key not in specs or episode >= episodes_per_task:
        raise ValueError("Unknown evaluation case: %r" % ((suite, task_id, episode),))
    if row.get("benchmark", benchmark) != benchmark:
        raise ValueError("Episode benchmark differs from requested benchmark")
    prefix = path.parent.relative_to(output_dir)
    if benchmark == "libero" and prefix.parts and prefix.parts[0] in LIBERO_SUITES and prefix.parts[0] != suite:
        raise ValueError("Episode suite differs from its source directory")
    spec = specs[key]
    task = row.get("task", row.get("prompt"))
    if benchmark == "metaworld":
        task = row.get("task", spec["task"])
        if task != spec["task"] or any(row[name] != spec["group"] for name in ("difficulty", "group") if name in row):
            raise ValueError("Episode task or difficulty differs from the native task configuration")
        if "reference_index" in row and row["reference_index"] != task_id:
            raise ValueError("Inconsistent task identity aliases")
    elif task is not None:
        if not isinstance(task, str) or not task:
            raise ValueError("Episode task must be a nonempty string")
        if spec["task"] is not None and spec["task"] != task:
            raise ValueError("Task language differs between episodes")
        spec["task"] = task
    status, success, error = row.get("status"), row.get("success"), row.get("error")
    if status not in ("success", "failure", "error") or type(success) is not bool or success != (status == "success"):
        raise ValueError("Episode has an invalid or inconsistent status/success")
    if (status == "error" and (not isinstance(error, str) or not error)) or (status != "error" and error is not None):
        raise ValueError("Episode has an inconsistent error field")
    protocol = row.get("seed_protocol", row.get("seed_scheme", source_protocol))
    if protocol != SEED_SCHEME:
        raise ValueError("Unknown or missing episode seed protocol")
    for protocol_key in ("seed_protocol", "seed_scheme"):
        if protocol_key in row and row[protocol_key] != SEED_SCHEME:
            raise ValueError("Unknown episode seed protocol")
    env_seed = _integer(row.get("env_seed", row.get("episode_seed")), "env_seed")
    policy_seed = _integer(row.get("policy_seed"), "policy_seed")
    if "episode_seed" in row and row["episode_seed"] != env_seed:
        raise ValueError("Inconsistent environment seed aliases")
    if env_seed != episode_seed(seed, suite, task_id, episode, "environment") or policy_seed != episode_seed(
            seed, suite, task_id, episode, "policy"):
        raise ValueError("Episode seeds differ from the requested seed schedule")
    video = row.get("video")
    if video is not None:
        if not isinstance(video, str) or not video:
            raise ValueError("Episode video must be a relative path or null")
        video_path = PurePosixPath(video)
        if video_path.is_absolute() or ".." in video_path.parts:
            raise ValueError("Episode video must stay inside the evaluation directory")
        video = (PurePosixPath(prefix.as_posix()) / video_path).as_posix()
    return {**row, "benchmark": benchmark, "suite": suite, "task_id": task_id, "task": task,
            "group": spec["group"], "episode_index": episode, "env_seed": env_seed, "policy_seed": policy_seed,
            "seed_protocol": SEED_SCHEME, "status": status, "success": success, "error": error, "video": video}


def _counts(records):
    counts = {"episodes": len(records), "successes": 0, "failures": 0, "errors": 0}
    for record in records:
        counts[{"success": "successes", "failure": "failures", "error": "errors"}[record["status"]]] += 1
    counts["success_rate"] = counts["successes"] / counts["episodes"] if counts["episodes"] else 0.0
    return counts


def build_report(model, benchmark, output_dir, seed, episodes_per_task, suites=None):
    """Return (summary, normalized episodes), rejecting invalid case identities."""
    output_dir = Path(output_dir)
    if model not in ("pi0", "pi05", "xvla") or benchmark not in ("libero", "metaworld"):
        raise ValueError("Unknown model or benchmark")
    _integer(seed, "seed")
    _integer(episodes_per_task, "episodes_per_task")
    if seed >= 2**32 or episodes_per_task == 0:
        raise ValueError("seed must be uint32 and episodes_per_task must be positive")
    if benchmark == "metaworld" and suites is not None:
        raise ValueError("--suites applies only to LIBERO")
    if benchmark == "libero":
        suites = tuple(LIBERO_SUITES if suites is None else suites)
        if not suites or len(set(suites)) != len(suites) or any(suite not in LIBERO_SUITES for suite in suites):
            raise ValueError("Unknown, empty or duplicate LIBERO suite selection")
        suites = tuple(suite for suite in LIBERO_SUITES if suite in suites)
    metadata = _metadata(model, benchmark, output_dir)
    for key, expected_value in (("model", model), ("benchmark", benchmark)):
        if key in metadata and metadata[key] != expected_value:
            raise ValueError("Source metadata differs from requested %s" % key)
    config = _configuration(metadata, seed, episodes_per_task, suites)
    source_config = metadata.get("configuration", {})
    source_protocol = source_config.get("seed_scheme", source_config.get("seed_protocol",
                      metadata.get("seed_scheme", metadata.get("seed_protocol"))))
    specs = _task_specs(benchmark, metadata, suites)
    normalized, seen = [], set()
    for path, row in _records(_source_files(model, benchmark, output_dir, suites)):
        record = _normalize(row, path, output_dir, benchmark, specs, seed, episodes_per_task, source_protocol)
        case = (record["suite"], record["task_id"], record["episode_index"])
        if case in seen:
            raise ValueError("Duplicate evaluation case: %r" % (case,))
        seen.add(case)
        normalized.append(record)
    ordered_specs = sorted(specs.values(), key=lambda spec: (
        (LIBERO_SUITES if benchmark == "libero" else ("metaworld_mt50",)).index(spec["suite"]), spec["task_id"]))
    for spec in ordered_specs:
        if spec["task"] is None:
            spec["task"] = "Task %s" % spec["task_id"]
    for record in normalized:
        record["task"] = specs[record["suite"], record["task_id"]]["task"]
    order = {(spec["suite"], spec["task_id"]): index for index, spec in enumerate(ordered_specs)}
    normalized.sort(key=lambda record: (order[record["suite"], record["task_id"]], record["episode_index"]))
    per_task = [{**spec, **_counts([record for record in normalized
                                  if (record["suite"], record["task_id"]) == (spec["suite"], spec["task_id"])])}
                for spec in ordered_specs]
    groups = suites if benchmark == "libero" else MW_GROUPS
    per_group = [{"group": group, "tasks": sum(spec["group"] == group for spec in ordered_specs),
                  **_counts([record for record in normalized if record["group"] == group])} for group in groups]
    counts = _counts(normalized)
    requested = len(specs) * episodes_per_task
    expected = (40 if benchmark == "libero" else 50) * episodes_per_task
    coverage = len(seen) == requested
    complete = coverage and counts["errors"] == 0 and requested == expected
    summary = {**metadata, "schema_version": 2, "model": model, "benchmark": benchmark, "seed": seed,
               "episodes_per_task": episodes_per_task, "expected_episodes": expected,
               "requested_episodes": requested, "coverage_complete": coverage, **counts, "complete": complete,
               "mean_success_rate": sum(group["success_rate"] for group in per_group) / 4 if complete else None,
               "per_group": per_group, "per_task": per_task, "configuration": config}
    summary.pop("report_error", None)
    if benchmark == "metaworld":
        summary["per_difficulty"] = [{**group, "difficulty": group["group"]} for group in per_group]
    return summary, normalized


def _percent(rate):
    return "%.2f%%" % (100 * rate)


def _atomic_write(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _invalidate_existing(output_dir, error):
    """Do not leave a previously generated successful report after a failed rerun."""
    path = output_dir / "summary.json"
    if path.exists():
        try:
            previous = _read_json(path)
        except (ValueError, OSError):
            return
        if previous.get("schema_version") == 2:
            previous.update(complete=False, coverage_complete=False, mean_success_rate=None, report_error=str(error))
            _atomic_write(path, json.dumps(previous, ensure_ascii=False, indent=2) + "\n")


def generate_report(model, benchmark, output_dir, seed, episodes_per_task, suites=None):
    """Write episode records and the summary; preserve native worker files."""
    output_dir = Path(output_dir)
    try:
        summary, records = build_report(model, benchmark, output_dir, seed, episodes_per_task, suites)
    except (ValueError, OSError) as error:
        _invalidate_existing(output_dir, error)
        raise
    _atomic_write(output_dir / "episodes.jsonl", "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))
    _atomic_write(output_dir / "summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("pi0", "pi05", "xvla"), required=True)
    parser.add_argument("--benchmark", choices=("libero", "metaworld"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes-per-task", type=int, required=True)
    parser.add_argument("--suites", choices=LIBERO_SUITES, nargs="+")
    args = parser.parse_args(argv)
    try:
        summary = generate_report(args.model, args.benchmark, args.output_dir, args.seed, args.episodes_per_task, args.suites)
    except (ValueError, OSError) as error:
        print("Evaluation report failed: %s" % error, file=sys.stderr)
        return 2
    for group in summary["per_group"]:
        print("%s: successes=%s/%s (%s) errors=%s" % (
            GROUP_LABELS[group["group"]], group["successes"], group["episodes"],
            _percent(group["success_rate"]), group["errors"]))
    mean = summary["mean_success_rate"]
    print("%s %s: Mean=%s overall=%s successes=%s/%s errors=%s complete=%s; report=%s" % (
        args.model, args.benchmark, _percent(mean) if mean is not None else "unavailable",
        _percent(summary["success_rate"]), summary["successes"], summary["episodes"], summary["errors"],
        str(summary["complete"]).lower(), args.output_dir / "summary.json"))
    return 0 if summary["coverage_complete"] and summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
