"""The episode seed protocol used by the existing Pi evaluations."""

import hashlib

SEED_SCHEME = "sha256-v1"


def episode_seed(
    master_seed: int,
    suite: str,
    task: int,
    episode: int,
    stream: str,
) -> int:
    """Keep the historical input string and byte order unchanged."""
    if not 0 <= master_seed < 2**32 or task < 0 or episode < 0:
        raise ValueError("master_seed must be uint32; task and episode must be nonnegative")
    if not suite or stream not in {"environment", "policy"}:
        raise ValueError("suite and stream must identify an evaluation case")
    identity = f"{master_seed}|{suite}|{task}|{episode}|{stream}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:4], "big")


class EpisodeSeed:
    """Reset Torch once per episode; one episode worker owns each policy server."""

    def __init__(self):
        self.episode = None
        self.seed = None

    def begin(self, episode: str, seed: int) -> None:
        if not isinstance(episode, str) or not episode:
            raise ValueError("episode_id must be a nonempty string")
        if not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be a uint32 integer")
        if episode != self.episode:
            import torch

            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            self.episode, self.seed = episode, seed
        elif seed != self.seed:
            raise ValueError("Seed changed inside one episode")
