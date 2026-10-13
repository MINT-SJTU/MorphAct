"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


class LiberoStateActionOnly(transforms.DataTransformFn):
    """Keep the fields needed for LIBERO normalization without decoding images."""

    def __call__(self, x: dict) -> dict:
        return {"state": x["state"], "actions": x["actions"]}


class MetaWorldStateActionDataset:
    """Read Evo-1 state/action windows without opening any video files."""

    def __init__(self, dataset):
        # MT50's 4D vectors fit in memory; avoid rebuilding an HF dataset for every action window.
        rows = dataset.hf_dataset.select_columns(["observation.state", "action", "episode_index"]).with_format("numpy")
        self.states = np.asarray(rows["observation.state"])
        self.actions = np.asarray(rows["action"])
        self.episodes = np.asarray(rows["episode_index"])
        self.episode_starts = np.asarray(dataset.episode_data_index["from"])
        self.episode_ends = np.asarray(dataset.episode_data_index["to"])
        self.offsets = np.asarray(dataset.delta_indices["action"])

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        episode = int(self.episodes[index])
        indices = np.clip(index + self.offsets, self.episode_starts[episode], self.episode_ends[episode] - 1)
        return {"state": self.states[index], "actions": self.actions[indices]}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
    *,
    state_action_only: bool = False,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    if state_action_only:
        from openpi.policies.metaworld_policy import EVO1_DATASET_REPO_ID

        if data_config.repo_id not in {"physical-intelligence/libero", EVO1_DATASET_REPO_ID}:
            raise ValueError("state_action_only supports LIBERO and the Evo-1 MetaWorld dataset")
        while isinstance(dataset, _data_loader.TransformedDataset):
            dataset = dataset._dataset  # noqa: SLF001
        if data_config.repo_id == EVO1_DATASET_REPO_ID:
            dataset = MetaWorldStateActionDataset(dataset)
        else:
            dataset.hf_dataset = dataset.hf_dataset.select_columns(["state", "actions", "episode_index", "task_index"])
            dataset = _data_loader.TransformedDataset(
                dataset,
                [LiberoStateActionOnly(), *data_config.data_transforms.inputs[1:], RemoveStrings()],
            )
    else:
        dataset = _data_loader.TransformedDataset(
            dataset,
            [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
                RemoveStrings(),
            ],
        )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    if num_batches == 0:
        raise ValueError("No complete batches for normalization; reduce --batch-size or increase --max-frames")
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    max_frames: int | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    assets_base_dir: str | None = None,
    *,
    state_action_only: bool = False,
):
    config = _config.get_config(config_name)
    if assets_base_dir is not None:
        config = dataclasses.replace(config, assets_base_dir=assets_base_dir)
    data_config = config.data.create(config.assets_dirs, config.model)
    stats_batch_size = config.batch_size if batch_size is None else batch_size
    stats_num_workers = config.num_workers if num_workers is None else num_workers

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            stats_batch_size,
            config.model,
            stats_num_workers,
            max_frames,
            state_action_only=state_action_only,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
