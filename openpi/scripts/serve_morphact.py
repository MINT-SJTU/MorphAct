"""Serve a MorphAct Pi0 or Pi0.5 checkpoint through the websocket API."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
import tyro


@dataclass(frozen=True, slots=True)
class Args:
    config: str
    morphact_checkpoint: str
    base_checkpoint: str
    default_prompt: str | None = None
    device: str | None = None
    cuda_memory_fraction: float | None = None
    port: int = 8000
    host: str = "127.0.0.1"
    denoise_steps: int = 10
    episode_seeded: bool = False
    inference_acceleration: bool = True


def main(args: Args) -> None:
    from openpi.policies.episode_seeded_policy import EpisodeSeededPolicy
    from openpi.policies.morphact_policy_config import create_morphact_policy
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    from openpi.training.config import get_config

    if args.denoise_steps <= 0:
        raise ValueError("denoise_steps must be positive")
    if args.cuda_memory_fraction is not None:
        if not 0 < args.cuda_memory_fraction <= 1:
            raise ValueError("cuda_memory_fraction must be in (0, 1]")
        memory_device = torch.device(args.device or "cuda")
        if memory_device.type != "cuda":
            raise ValueError("cuda_memory_fraction requires a CUDA device")
        device_index = memory_device.index if memory_device.index is not None else torch.cuda.current_device()
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device_index)
    loaded_policy = create_morphact_policy(
        get_config(args.config),
        args.morphact_checkpoint,
        base_checkpoint=args.base_checkpoint,
        default_prompt=args.default_prompt,
        device=args.device,
        sample_kwargs={"num_steps": args.denoise_steps},
        inference_acceleration=args.inference_acceleration,
    )
    if args.episode_seeded:
        loaded_policy = EpisodeSeededPolicy(loaded_policy)
    metadata = {
        **loaded_policy.metadata,
        "denoise_steps": args.denoise_steps,
    }
    WebsocketPolicyServer(
        policy=loaded_policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
