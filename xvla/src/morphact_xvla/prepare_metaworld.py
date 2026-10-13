"""Download the pinned Evo-1 MT50 LeRobot v2.1 dataset outside the checkout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metaworld import REPO_ID, REVISION


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.dataset_dir.expanduser().resolve()
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=REPO_ID, repo_type="dataset", revision=REVISION, local_dir=root)
    if not (root / "meta/info.json").is_file():
        raise FileNotFoundError(root / "meta/info.json")
    (root / ".morphact-source.json").write_text(json.dumps({"repo_id": REPO_ID, "revision": REVISION}, indent=2) + "\n")


if __name__ == "__main__":
    main()
