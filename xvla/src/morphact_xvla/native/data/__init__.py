from __future__ import annotations

import random

import numpy as np
import torch


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
