"""Evo-1 MT50 data contract for native X-VLA training and inference."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset

REPO_ID = "MINT-SJTU/Evo1_MetaWorld_Dataset"
REVISION = "612c4e83d0e3b4e0be5227970bd8f5f07fc0c488"


@dataclass(frozen=True)
class MetaWorldContract:
    dataset: str = REPO_ID
    revision: str = REVISION
    action_mode: str = "auto"
    real_action_dim: int = 4
    max_action_dim: int = 20
    num_actions: int = 30
    domain_id: int = 0
    boundary: str = "repeat_last_action"
    state: str = "observation.state_current_frame"
    action: str = "raw_dataset_action_no_clipping_no_delta_conversion"
    image: str = "stored_rgb_then_base_processor_no_extra_flip_or_crop"

    def native_kwargs(self):
        return {k: getattr(self, k) for k in ("action_mode", "real_action_dim", "max_action_dim", "num_actions")}

    def to_dict(self):
        return asdict(self)


CONTRACT = MetaWorldContract()


def pad_state(state):
    state = np.asarray(state, dtype=np.float32)
    if state.shape != (4,) or not np.isfinite(state).all():
        raise ValueError("MetaWorld state must be four finite current-frame values")
    return np.pad(state, (0, 16))


def action_chunk(actions, frame, horizon=30):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 4 or not np.isfinite(actions).all():
        raise ValueError("MetaWorld actions must be a finite [frames,4] array")
    if not 0 <= frame < len(actions) or horizon <= 0:
        raise ValueError("Invalid chunk start or length")
    # Repeat only this episode's last recorded action. No next-episode access,
    # no state/action shift, no coordinate conversion and no hidden clipping.
    indices = np.minimum(np.arange(frame, frame + horizon), len(actions) - 1)
    return np.pad(actions[indices], ((0, 0), (0, 16)))


def encode_observation(processor, image, state, normalizer):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("MetaWorld image must be uint8 HWC RGB")
    from PIL import Image

    encoded = processor.encode_image([Image.fromarray(image)])
    return {
        **{k: v[0] for k, v in encoded.items()},
        "proprio": torch.from_numpy(pad_state(normalizer.normalize(state, "state"))),
        "domain_id": torch.tensor(CONTRACT.domain_id, dtype=torch.long),
    }


def checkpoint_contract(processing):
    result = CONTRACT.to_dict()
    from .metaworld_processing import validate_processing

    validate_processing(processing)
    result.update(
        action="bounds_normalized_raw_action_v2",
        state="bounds_normalized_current_state_v2",
    )
    return result


def validate_checkpoint_contract(metadata):
    training_config = metadata.get("training_config")
    if not isinstance(training_config, dict):
        raise ValueError("Checkpoint lacks MetaWorld training configuration")
    contract = training_config.get("metaworld_contract")
    if not isinstance(contract, dict):
        raise ValueError("Checkpoint lacks MetaWorld data/action contract")
    contract = {key: value for key, value in contract.items() if key != "processing_sha256"}
    if contract != checkpoint_contract(training_config.get("metaworld_processing")):
        raise ValueError("Expected a MorphAct X-VLA MetaWorld checkpoint with the exact data/action contract")
    for key, value in CONTRACT.native_kwargs().items():
        if metadata["base_config"].get(key) != value:
            raise ValueError(f"Checkpoint native config disagrees with MetaWorld contract: {key}")


def make_index(root):
    root = Path(root).resolve()
    marker = json.loads((root / ".morphact-source.json").read_text())
    if marker != {"repo_id": REPO_ID, "revision": REVISION}:
        raise ValueError("Dataset provenance does not match the pinned Evo1 revision")
    info = json.loads((root / "meta/info.json").read_text())
    if info["fps"] != 30 or any(info["features"][k]["shape"] != [4] for k in ("action", "observation.state")):
        raise ValueError("Unexpected Evo1 data dimensions or frame rate")
    tasks = {x["task_index"]: x["task"] for x in map(json.loads, (root / "meta/tasks.jsonl").read_text().splitlines())}
    episodes = list(map(json.loads, (root / "meta/episodes.jsonl").read_text().splitlines()))
    if len(tasks) != 50 or len(episodes) != 2500 or len({e["episode_index"] for e in episodes}) != 2500:
        raise ValueError("Expected 50 tasks and 2500 distinct episodes")
    by_prompt = {v: k for k, v in tasks.items()}
    grouped = defaultdict(list)
    for episode in episodes:
        eid = episode["episode_index"]
        fields = dict(episode_chunk=eid // info["chunks_size"], episode_index=eid, video_key="observation.images.image")
        paths = [root / info[k].format(**fields) for k in ("data_path", "video_path")]
        if not all(p.is_file() for p in paths):
            raise FileNotFoundError(f"Incomplete MetaWorld training data: episode {eid}")
        prompts = episode["tasks"]
        if len(prompts) != 1 or prompts[0] not in by_prompt:
            raise ValueError(f"Unexpected task association in episode {eid}")
        grouped[by_prompt[prompts[0]]].append(
            {**episode, "data": str(paths[0]), "video": str(paths[1]), "task_index": by_prompt[prompts[0]]}
        )
    if len(grouped) != 50 or any(len(grouped[t]) != 50 for t in tasks):
        raise ValueError("Expected 50 available trajectories per task")
    return grouped


def read_episode(entry):
    import av
    import pyarrow.parquet as pq

    rows = pq.read_table(entry["data"]).to_pydict()
    actions = np.asarray(rows["action"], dtype=np.float32)
    states = np.asarray(rows["observation.state"], dtype=np.float32)
    n = len(actions)
    if actions.shape != (n, 4) or states.shape != (n, 4) or n != entry["length"]:
        raise ValueError("Episode shape/length mismatch")
    if not np.isfinite(actions).all() or not np.isfinite(states).all():
        raise ValueError("Non-finite data")
    np.testing.assert_array_equal(rows["episode_index"], np.full(n, entry["episode_index"]))
    np.testing.assert_array_equal(rows["frame_index"], np.arange(n))
    np.testing.assert_array_equal(rows["task_index"], np.full(n, entry["task_index"]))
    np.testing.assert_allclose(rows["timestamp"], np.arange(n) / 30, atol=1e-5, rtol=0)
    images, times = [], []
    with av.open(entry["video"]) as container:
        for frame in container.decode(video=0):
            images.append(frame.to_ndarray(format="rgb24"))
            times.append(float(frame.time))
    if len(images) != n:
        raise ValueError("Video/parquet length mismatch")
    np.testing.assert_allclose(times, rows["timestamp"], atol=1e-4, rtol=0)
    return images, states, actions


class MetaWorldDataset(IterableDataset):
    def __init__(self, groups, processor, *, augmentation="none", processing):
        self.groups, self.processor = groups, processor
        self.processing = processing
        from .metaworld_processing import BoundsNormalizer

        self.normalizer = BoundsNormalizer(processing)
        if augmentation not in ("none", "colorjitter"):
            raise ValueError("Unknown augmentation")
        self.augmentation = None
        if augmentation == "colorjitter":
            from torchvision.transforms import ColorJitter

            self.augmentation = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)

    def sample(self, entry, images, states, actions, frame):
        image = images[frame]
        if self.augmentation is not None:
            from PIL import Image

            image = np.asarray(self.augmentation(Image.fromarray(image)))
        return {
            **encode_observation(self.processor, image, states[frame], self.normalizer),
            "action": torch.from_numpy(action_chunk(self.normalizer.normalize(actions, "action"), frame)),
            "language_instruction": entry["tasks"][0],
        }

    def __iter__(self):
        # A pool covers distinct tasks; refresh a few decoded episodes each cycle.
        tasks = sorted(self.groups)
        size = min(self.processing["mixed_tasks"], len(tasks))

        def load(task):
            entry = random.choice(self.groups[task])
            return (task, entry, *read_episode(entry))

        pool = [load(task) for task in random.sample(tasks, size)]
        while True:
            for i in random.sample(range(size), size):
                task, entry, images, states, actions = pool[i]
                yield self.sample(entry, images, states, actions, random.randrange(len(actions)))
            for i in random.sample(range(size), min(size, self.processing["refresh_episodes"])):
                available = sorted(set(tasks) - {x[0] for j, x in enumerate(pool) if j != i})
                pool[i] = load(random.choice(available))


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)
