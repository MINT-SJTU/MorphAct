"""Serve a MORPHACT-adapted native X-VLA through the native POST /act contract."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--inference-acceleration", action=argparse.BooleanOptionalAction, default=True,
        help="Use fused PG RMSNorm/RoPE, scripted FP32 low-rank operations and CUDA denoising graphs",
    )
    parser.add_argument("--episode-seeded", action="store_true", help="Require benchmark episode_id and seed fields")
    return parser


def create_app(model, processor, *, device, default_steps: int = 10, episode_seeded: bool = False):
    import json_numpy
    import numpy as np
    import torch
    from fastapi import FastAPI, HTTPException
    from PIL import Image
    from morphact.eval_seed import EpisodeSeed

    if default_steps <= 0:
        raise ValueError("steps must be positive")
    app = FastAPI(title="MorphAct X-VLA MORPHACT")
    lock = threading.Lock()
    episode_seed = EpisodeSeed()

    @app.post("/act")
    def act(payload: dict):
        try:
            if "image0" not in payload:
                raise ValueError("image0 must contain the primary view")
            images = []
            for key in ("image0", "image1", "image2"):
                if key in payload:
                    value = payload[key]
                    array = json_numpy.loads(value) if isinstance(value, str) else np.asarray(value)
                    if array.ndim != 3 or array.shape[-1] != 3 or array.dtype != np.uint8:
                        raise ValueError(f"{key} must contain a uint8 HWC RGB image")
                    images.append(Image.fromarray(array))
            if not images:
                raise ValueError("At least image0 must be supplied")
            inputs = processor(images, payload["language_instruction"])
            value = payload["proprio"]
            proprio = json_numpy.loads(value) if isinstance(value, str) else value
            inputs["proprio"] = torch.from_numpy(np.array(proprio, dtype=np.float32, copy=True)).unsqueeze(0)
            inputs["domain_id"] = torch.tensor([int(payload["domain_id"])], dtype=torch.long)
            inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
            steps = int(payload.get("steps", default_steps))
            if steps <= 0:
                raise ValueError("steps must be positive")
            with lock, torch.inference_mode():
                if episode_seeded:
                    episode_seed.begin(payload["episode_id"], payload["seed"])
                noise = torch.randn(
                    (1, model.base_model.num_actions, model.base_model.action_space.dim_action),
                    device=device,
                    dtype=inputs["proprio"].dtype,
                )
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                    actions = model.generate_actions(**inputs, steps=steps, noise=noise)
            return {"action": actions[0].float().cpu().tolist()}
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def main(argv: list[str] | None = None) -> None:
    args = get_parser().parse_args(argv)
    import torch
    import uvicorn

    from .checkpoint import base_pretrained_kwargs, load_checkpoint
    from .config import XVLAMorphActConfig
    from .model import XVLAMorphAct
    from .native.models.processing_xvla import XVLAProcessor

    metadata = json.loads((Path(args.checkpoint) / "metadata.json").read_text())
    pretrained_kwargs = base_pretrained_kwargs(args.base_model, metadata)
    model = XVLAMorphAct.from_pretrained(
        args.base_model,
        config=XVLAMorphActConfig.from_dict(metadata["morphact_config"]),
        torch_dtype=torch.bfloat16 if str(args.device).startswith("cuda") else torch.float32,
        **pretrained_kwargs,
    ).to(args.device)
    load_checkpoint(model, args.checkpoint, expected_base_model=args.base_model)
    model.eval()
    model.configure_inference(enabled=args.inference_acceleration)
    processor = XVLAProcessor.from_pretrained(args.base_model, **pretrained_kwargs)
    app = create_app(model, processor, device=args.device, default_steps=args.steps, episode_seeded=args.episode_seeded)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
