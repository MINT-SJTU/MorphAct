import collections
import dataclasses
import json
import logging
import math
import pathlib
import random
import typing
import uuid

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
from morphact.eval_seed import SEED_SCHEME, episode_seed
from morphact.eval_queue import task_episodes as claim_task_episodes

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    task_ids: typing.Tuple[int, ...] = ()  # Empty evaluates every task; set explicit IDs to shard a suite.
    worker_index: int = 0  # Episode shard served by this client.
    num_workers: int = 1
    queue_path: typing.Optional[pathlib.Path] = None

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    episode_out_path: typing.Optional[pathlib.Path] = None

    seed: int = 0  # Master seed for the shared evaluation schedule.


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    task_ids = _resolve_task_ids(args.task_ids, num_tasks_in_suite)
    total_jobs = len(task_ids) * args.num_trials_per_task
    if args.num_trials_per_task < 1 or not 1 <= args.num_workers <= total_jobs:
        raise ValueError("Need at least one trial per task and no more workers than episodes")
    if not 0 <= args.worker_index < args.num_workers:
        raise ValueError("worker_index must be in [0, num_workers)")
    logging.info(f"Task suite: {args.task_suite_name}")
    logging.info(f"Task IDs: {','.join(str(task_id) for task_id in task_ids)}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    if args.episode_out_path is not None:
        args.episode_out_path.parent.mkdir(parents=True, exist_ok=True)
        with args.episode_out_path.open("x", encoding="utf-8"):
            pass

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    jobs = claim_task_episodes(task_ids, args.num_trials_per_task, args.worker_index, args.num_workers, args.queue_path)
    for task_id, episode_indices in jobs:
        logging.info(f"Task ID: {task_id}")
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        first_env_seed = episode_seed(args.seed, args.task_suite_name, task_id, 0, "environment")
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, first_env_seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(episode_indices):
            # A rerun of the same case must start a new server-side random stream.
            episode_id = f"{uuid.uuid4().hex}/{args.task_suite_name}/{task_id}/{episode_idx}"
            env_seed = episode_seed(args.seed, args.task_suite_name, task_id, episode_idx, "environment")
            policy_seed = episode_seed(args.seed, args.task_suite_name, task_id, episode_idx, "policy")
            np.random.seed(env_seed)
            random.seed(env_seed)
            env.seed(env_seed)
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            episode_success = False
            episode_error = None

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                            "_eval_episode_id": episode_id,
                            "_eval_policy_seed": policy_seed,
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        episode_success = True
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    episode_error = f"{type(e).__name__}: {e}"
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if episode_success else "failure"
            task_segment = task_description.replace(" ", "_")
            video_name = f"rollout_task{task_id:02d}_episode{episode_idx:03d}_{task_segment}_{suffix}.mp4"
            record = {
                "benchmark": "libero", "suite": args.task_suite_name, "task_id": task_id,
                "task": str(task_description), "episode_index": episode_idx,
                "env_seed": env_seed, "policy_seed": policy_seed, "seed_protocol": SEED_SCHEME,
                "worker": args.worker_index,
                "status": "error" if episode_error else suffix,
                "success": episode_success and episode_error is None, "error": episode_error,
                "video": None, "replan_steps": args.replan_steps,
            }
            try:
                imageio.mimwrite(
                    pathlib.Path(args.video_out_path) / video_name,
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )
                if args.episode_out_path is not None:
                    record["video"] = (pathlib.Path(args.video_out_path) / video_name).relative_to(
                        args.episode_out_path.parent
                    ).as_posix()
            except Exception as error:
                record.update(status="error", success=False, error=f"{type(error).__name__}: {error}")
                raise
            finally:
                if args.episode_out_path is not None:
                    with args.episode_out_path.open("a", encoding="utf-8") as event_log:
                        event_log.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Log current results
            logging.info(f"Success: {episode_success}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {total_successes / total_episodes if total_episodes else 0.0}")
    logging.info(f"Total episodes: {total_episodes}")


def _resolve_task_ids(requested: typing.Tuple[int, ...], num_tasks: int) -> typing.Tuple[int, ...]:
    task_ids = requested or tuple(range(num_tasks))
    if len(set(task_ids)) != len(task_ids):
        raise ValueError(f"Task IDs must be unique, got {task_ids}")
    invalid = [task_id for task_id in task_ids if not 0 <= task_id < num_tasks]
    if invalid:
        raise ValueError(f"Task IDs out of range [0, {num_tasks}): {invalid}")
    return task_ids


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
