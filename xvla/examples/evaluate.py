"""Evaluate X-VLA with one simulator worker per policy server."""

from __future__ import annotations

import argparse
import base64
import collections
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import random
import signal
import sys
import time
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from morphact.eval_seed import SEED_SCHEME, episode_seed

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def image_payload(image):
    """Encode the json-numpy wire format without a model-runtime dependency."""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    return json.dumps({"__numpy__": base64.b64encode(image.tobytes()).decode(),
                       "dtype": image.dtype.str, "shape": image.shape})


class HttpPolicy:
    def __init__(self, host, port, execution_id):
        self.url = f"http://{host}:{port}/act"
        self.execution_id = execution_id

    def act(self, payload):
        payload = {**payload, "episode_id": f"{self.execution_id}/{payload['episode_id']}"}
        request = urllib.request.Request(self.url, data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=180) as response:
            actions = np.asarray(json.load(response)["action"], dtype=np.float32)
        if actions.ndim != 2 or not np.isfinite(actions).all():
            raise ValueError("Policy returned invalid actions")
        return actions

    def infer(self, element):
        actions = self.act({"image": image_payload(element["observation/image"]),
                            "state": element["observation/state"].tolist(),
                            "language_instruction": element["prompt"],
                            "episode_id": element["_eval_episode_id"],
                            "seed": element["_eval_policy_seed"]})
        if actions.shape != (30, 4):
            raise ValueError(f"Expected MetaWorld actions (30, 4), got {actions.shape}")
        return {"actions": actions}


def load_metaworld_driver():
    path = ROOT / "openpi/examples/metaworld/evaluate.py"
    spec = importlib.util.spec_from_file_location("xvla_metaworld_environment", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def absolute_action(action):
    import robosuite.utils.transform_utils as transforms

    first, second = action[3:6], action[6:9]
    first = first / (np.linalg.norm(first) + 1e-6)
    second = second - np.dot(first, second) * first
    second = second / (np.linalg.norm(second) + 1e-6)
    rotation = np.stack((first, second, np.cross(first, second)), axis=-1)
    axis_angle = transforms.quat2axisangle(transforms.mat2quat(rotation))
    return np.concatenate((action[:3], axis_angle, [1.0 if action[9] > 0.5 else -1.0])).astype(np.float32)


def rollout_libero(args, suite, task_id, episode, env_seed, policy_seed, policy):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task = suite.get_task(task_id)
    states = suite.get_task_init_states(task_id)
    if episode >= len(states):
        raise ValueError("Requested episode exceeds the published LIBERO initial states")
    env = OffScreenRenderEnv(bddl_file_name=str(Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file),
                             camera_heights=256, camera_widths=256)
    frames, pending = [], collections.deque()
    try:
        random.seed(env_seed)
        np.random.seed(env_seed)
        env.seed(env_seed)
        env.reset()
        obs = env.set_init_state(states[episode])
        for _ in range(10):
            obs, _, _, _ = env.step([0., 0., 0., 0., 0., 0., -1.])
        for robot in env.env.robots:
            robot.controller.use_delta = False
        limit = args.max_steps or (900 if args.suite == "libero_10" else 800)
        success = False
        for step in range(limit):
            main_image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            if not args.no_video:
                frames.append(main_image)
            if not pending:
                # Match the published checkpoint's previous-controller-goal contract.
                controller = env.env.robots[0].controller
                orientation = controller.goal_ori
                proprio = np.concatenate((controller.goal_pos, orientation[:, 0], orientation[:, 1], [0.], np.zeros(10)))
                actions = policy.act({"image0": image_payload(main_image),
                                      "image1": image_payload(obs["robot0_eye_in_hand_image"]),
                                      "proprio": proprio.tolist(), "language_instruction": task.language,
                                      "domain_id": 3, "steps": 10,
                                      "episode_id": f"{args.suite}/{task_id}/{episode}", "seed": policy_seed})
                if actions.shape != (30, 20):
                    raise ValueError(f"Expected LIBERO actions (30, 20), got {actions.shape}")
                pending.extend(absolute_action(action) for action in actions[:args.replan_steps])
            obs, _, done, _ = env.step(pending.popleft())
            if done:
                success = True
                break
        return {"status": "success" if success else "failure", "success": success,
                "steps": step + 1, "prompt": task.language, "error": None}, frames
    finally:
        env.close()


def write_status(path, record):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=("libero", "metaworld"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--ports", type=int, nargs="+", help="One policy server port per worker")
    parser.add_argument("--gpus", type=int, nargs="+", help="Simulator GPU for each server, in port order")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Keep completed episodes and retry missing or errored cases")
    parser.add_argument("--episodes-per-task", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--replan-steps", type=int)
    parser.add_argument("--limit", type=int, help="Run only the first N assigned cases for an entrypoint check")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    args.ports = args.ports or [args.port]
    if len(set(args.ports)) != len(args.ports) or not 0 <= args.seed < 2**32:
        parser.error("Server ports must be unique and seed must be uint32")
    if args.gpus is not None and len(args.gpus) != len(args.ports):
        parser.error("Supply one simulator GPU per server port")
    if args.episodes_per_task is None:
        args.episodes_per_task = 50 if args.benchmark == "libero" else 20
    if args.replan_steps is None:
        args.replan_steps = 30 if args.benchmark == "libero" else 5
    if args.episodes_per_task < 1 or not 1 <= args.replan_steps <= 30:
        parser.error("Invalid episode count or execution horizon")
    return args


def take_jobs(case_queue, stop_event):
    while not stop_event.is_set():
        job = case_queue.get()
        if job is None:
            return
        yield job


def run_worker(args, case_queue, stop_event, completed_before):
    if args.gpus is not None:
        gpu = str(args.gpus[args.worker_id])
        os.environ.update(CUDA_VISIBLE_DEVICES=gpu, MUJOCO_EGL_DEVICE_ID=gpu)
    os.environ.setdefault("MUJOCO_GL", "egl")
    random.seed(args.seed)
    np.random.seed(args.seed)
    policy = HttpPolicy(args.host, args.port, args.execution_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "videos").mkdir(exist_ok=True)
    if args.benchmark == "libero":
        from libero.libero import benchmark

        suites = {name: benchmark.get_benchmark_dict()[name]() for name in SUITES}
    else:
        driver = load_metaworld_driver()
        benchmark = driver._default_benchmark_factory(args.seed)
        config = driver.Args(seed=args.seed, image_size=480, max_steps=args.max_steps or 500,
                             episodes_per_task=args.episodes_per_task, replan_steps=args.replan_steps)
        specs = {spec.reference_index: spec for spec in driver.MT50_TASKS}
    status_path = args.output_dir / f"worker-{args.worker_id}.json"
    result_path = args.output_dir / f"worker-{args.worker_id}.jsonl"
    completed = completed_before
    with result_path.open("a" if args.resume else "x") as results:
        for suite_name, task, episode in take_jobs(case_queue, stop_event):
            env_seed = episode_seed(args.seed, suite_name, task, episode, "environment")
            policy_seed = episode_seed(args.seed, suite_name, task, episode, "policy")
            record = {"benchmark": args.benchmark, "suite": suite_name, "task_id": task,
                      "episode_index": episode, "env_seed": env_seed, "policy_seed": policy_seed,
                      "seed_protocol": SEED_SCHEME, "worker": args.worker_id, "video": None}
            write_status(status_path, {**record, "phase": "running", "completed": completed,
                                      "scheduling": "dynamic"})
            started = time.monotonic()
            try:
                if args.benchmark == "libero":
                    args.suite = suite_name
                    result, frames = rollout_libero(args, suites[suite_name], task, episode, env_seed, policy_seed, policy)
                else:
                    spec = specs[task]
                    tasks = [item for item in benchmark.train_tasks if item.env_name == spec.env_name]
                    random.seed(env_seed)
                    np.random.seed(env_seed)
                    env = driver._default_env_factory(benchmark.train_classes[spec.env_name], config)
                    try:
                        result, frames = driver._rollout_episode(env=env, task=tasks[episode], spec=spec,
                            episode_seed=env_seed, policy_seed=policy_seed, episode_index=episode,
                            policy=policy, args=config, capture_video=not args.no_video)
                        result.update(prompt=spec.prompt, task=spec.slug, difficulty=spec.difficulty)
                    finally:
                        env.close()
                record.update(result)
                if frames:
                    import imageio

                    video = Path("videos") / f"{suite_name}_{task:02d}_{episode:03d}_{record['status']}.mp4"
                    imageio.mimwrite(args.output_dir / video, frames, fps=30 if args.benchmark == "libero" else 10,
                                    output_params=["-threads", "1"])
                    record["video"] = video.as_posix()
            except Exception as error:
                record.update(status="error", success=False, error=f"{type(error).__name__}: {error}")
            record["elapsed_s"] = round(time.monotonic() - started, 3)
            results.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.flush()
            completed += 1
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if record["status"] == "error":
                write_status(status_path, {**record, "phase": "failed", "completed": completed})
                raise SystemExit(1)
        write_status(status_path, {"phase": "paused" if stop_event.is_set() else "complete", "completed": completed})


def saved_results(args, cases, *, retry_errors=False):
    results = {}
    seen = set()
    rewrites = []
    for path in args.output_dir.glob("worker-*.jsonl"):
        if not args.resume:
            raise FileExistsError("Results already exist; use a new directory or --resume")
        kept, errors = [], []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            case = (row["suite"], row["task_id"], row["episode_index"])
            if case not in cases or case in seen or row["benchmark"] != args.benchmark:
                raise ValueError("Saved results contain an unknown or duplicate case")
            seen.add(case)
            if row["status"] not in ("success", "failure", "error"):
                raise ValueError("Saved result has an unknown status")
            if type(row.get("success")) is not bool or row["success"] != (row["status"] == "success"):
                raise ValueError("Saved result has an inconsistent success flag")
            for stream, key in (("environment", "env_seed"), ("policy", "policy_seed")):
                if row[key] != episode_seed(args.seed, *case, stream):
                    raise ValueError("Saved results use different seeds")
            if row["status"] == "error":
                if not retry_errors:
                    raise ValueError("Saved rollout errors require --resume to retry")
                errors.append(line + "\n")
                continue
            kept.append(line + "\n")
            results[case] = row
        if errors:
            rewrites.append((path, kept, errors))

    # Validate every file first; archive errors before removing them from active results.
    for path, kept, errors in rewrites:
        with (args.output_dir / "rollout-errors.jsonl").open("a", encoding="utf-8") as archive:
            archive.writelines(errors)
        temporary = path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(kept), encoding="utf-8")
        temporary.replace(path)
    return results


def evaluation_metrics(args, completed):
    records = list(completed.values())
    successes = sum(record["status"] == "success" for record in records)
    metrics = {"episodes": len(records), "successes": successes,
               "errors": sum(record["status"] == "error" for record in records),
               "success_rate": successes / len(records) if records else 0.0}
    if args.benchmark == "metaworld":
        driver = load_metaworld_driver()
        records = [{**record, "reference_index": record["task_id"]} for record in records]
        summary = driver._summary(driver.Args(episodes_per_task=args.episodes_per_task),
                                  driver.MT50_TASKS, records, None, {})
        metrics.update({key: summary[key] for key in ("per_task", "per_difficulty", "mean_success_rate")})
    return metrics


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configuration = {key: getattr(args, key) for key in
                     ("benchmark", "seed", "episodes_per_task", "replan_steps", "max_steps")}
    status_path = args.output_dir / "evaluation.json"
    if args.resume and status_path.exists():
        saved = json.loads(status_path.read_text())
        if any(saved.get(key) != value for key, value in configuration.items()):
            raise ValueError("Resume configuration differs from the saved evaluation")
    tasks = ([(suite, task) for suite in SUITES for task in range(10)] if args.benchmark == "libero"
             else [("metaworld_mt50", spec.reference_index) for spec in load_metaworld_driver().MT50_TASKS])
    jobs = [(suite, task, episode) for suite, task in tasks for episode in range(args.episodes_per_task)]
    previous = saved_results(args, set(jobs), retry_errors=args.resume)
    pending = [job for job in jobs if job not in previous]
    if args.limit is not None:
        pending = pending[:args.limit]
    context = multiprocessing.get_context("spawn")
    case_queue, stop_event = context.Queue(), context.Event()
    for job in pending:
        case_queue.put(job)
    for _ in args.ports:
        case_queue.put(None)
    case_queue.cancel_join_thread()
    args.execution_id = str(time.time_ns())  # Distinguishes a resumed execution; never enters the seed formula.
    processes = []
    status = {**configuration, "scheduling": "dynamic",
              "ports": args.ports, "gpus": args.gpus, "preserved_episodes": len(previous),
              "queued_episodes": len(pending), "expected_episodes": len(jobs), "pid": os.getpid()}
    signal.signal(signal.SIGTERM, lambda *_: stop_event.set())
    try:
        for worker, port in enumerate(args.ports):
            child_args = argparse.Namespace(**{**vars(args), "worker_id": worker, "port": port})
            completed = sum(row["worker"] == worker for row in previous.values())
            process = context.Process(target=run_worker, args=(child_args, case_queue, stop_event, completed))
            process.start()
            processes.append(process)
        write_status(args.output_dir / "evaluation.json", {**status, "phase": "running"})
        while any(process.is_alive() for process in processes):
            if any(process.exitcode not in (None, 0) for process in processes):
                raise RuntimeError("A worker failed; see its log and episode records")
            time.sleep(1)
        if any(process.exitcode != 0 for process in processes):
            raise RuntimeError("A worker failed")
        args.resume = True
        completed = saved_results(args, set(jobs))
        phase = "complete" if len(completed) == len(jobs) else "paused" if stop_event.is_set() else "partial"
        if phase == "partial" and args.limit is None:
            raise RuntimeError("The queue finished without completing every case")
        metrics = evaluation_metrics(args, completed)
        write_status(args.output_dir / "evaluation.json", {**status, "phase": phase, **metrics})
        if args.benchmark == "metaworld":
            mean = metrics["mean_success_rate"]
            mean_text = f"{100 * mean:.2f}%" if mean is not None else "unavailable (requires complete, error-free MT50)"
            print(f"MT50 evaluation {phase}: Mean={mean_text} micro_success_rate={metrics['success_rate']:.3f} "
                  f"successes={metrics['successes']} episodes={metrics['episodes']} errors={metrics['errors']}", flush=True)
    except BaseException as error:
        write_status(args.output_dir / "evaluation.json", {**status, "phase": "failed", "error": str(error)})
        raise
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()


if __name__ == "__main__":
    main()
