from __future__ import annotations
from typing import Dict, Iterable, List
import io, json, random, numpy as np, torch
from torch.utils.data import IterableDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from mmengine import fileio
from .utils import action_slice
from .domain_config import DATA_DOMAIN_ID
from .domain_handler.registry import get_handler_cls

class InfiniteDataReader(IterableDataset):
    """
    Output sample:
      {
        'domain_id': LongTensor[],    # domain id
        'language_instruction': str,
        'image_input': FloatTensor[V, C, H, W],
        'image_mask': BoolTensor[V],
        'proprio': FloatTensor[dim_proprio],
        'action': FloatTensor[T, dim_action]
      }
    """
    def __init__(self,
                 metas_path: str,
                 num_actions: int = 10,
                 num_views: int = 3,
                 training: bool = True,
                 action_mode: str = "ee6d",
                 lang_aug: str = None,
                 ):
        self.num_views = num_views
        self.training = training
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.metas: Dict[str, dict] = {}
        print("use action mode:", action_mode)
        with io.BytesIO(fileio.get(metas_path)) as f:
            meta = json.load(f)
        if meta.get("dataset_name") != "libero" or not isinstance(meta.get("datalist"), list) or not meta["datalist"]:
            raise ValueError("Expected nonempty X-VLA LIBERO HDF5 metadata")
        self.metas["libero"] = meta

        self.image_aug = [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.) \
                if training else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
        ]
        self.image_aug = transforms.Compose(self.image_aug)

    def _iter_one_dataset(self, dataset_name: str) -> Iterable[dict]:
        meta = self.metas[dataset_name]
        traj_indices = list(range(len(meta["datalist"])))
        if self.training: random.shuffle(traj_indices)

        robot_type = dataset_name
        Handler = get_handler_cls(robot_type)
        handler = Handler(meta=meta, num_views=self.num_views)
        for traj_idx in traj_indices:
                for sample in handler.iter_episode(
                    traj_idx,
                    num_actions=self.num_actions,
                    training=self.training,
                    image_aug=self.image_aug,
                    lang_aug_map= meta["lang_aug_map"] if "lang_aug_map" in meta.keys() else None,
                    action_mode = self.action_mode
                ):
                    sample["domain_id"] = torch.tensor(DATA_DOMAIN_ID.get(robot_type, 0))
                    idx_for_delta = sample.pop("idx_for_delta", [])
                    idx_for_mask_proprio = sample.pop("idx_for_mask_proprio", [])
                    sample.update(action_slice(sample.pop("abs_trajectory", None), idx_for_delta, idx_for_mask_proprio))
                    yield sample
    def __iter__(self):
        if self.training:
            while True:
                yield from self._iter_one_dataset("libero")
        else:
            yield from self._iter_one_dataset("libero")
