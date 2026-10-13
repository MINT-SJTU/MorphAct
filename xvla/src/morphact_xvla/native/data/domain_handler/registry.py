"""Data handlers supported by the published X-VLA trainer."""

from .simulations import LiberoHandler


def get_handler_cls(dataset_name: str):
    if dataset_name != "libero":
        raise KeyError(f"Unsupported X-VLA training dataset: {dataset_name}")
    return LiberoHandler
