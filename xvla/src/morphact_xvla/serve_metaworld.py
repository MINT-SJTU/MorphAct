"""Serve MorphAct X-VLA MetaWorld checkpoints."""

import argparse
import json
from pathlib import Path
import threading


def create_app(model, processor, *, device="cpu", steps=10, processing, episode_seeded=False):
    import json_numpy
    import numpy as np
    import torch
    from fastapi import FastAPI, HTTPException
    from morphact.eval_seed import EpisodeSeed
    from .metaworld import CONTRACT, encode_observation, checkpoint_contract
    from .metaworld_processing import BoundsNormalizer

    normalizer = BoundsNormalizer(processing)
    if steps <= 0:
        raise ValueError("steps must be positive")
    app = FastAPI(title="MorphAct X-VLA MetaWorld")
    lock = threading.Lock()
    episode_seed = EpisodeSeed()

    @app.get("/health")
    def health():
        return {
            "ready": True,
            "benchmark": "metaworld",
            "contract": checkpoint_contract(processing),
            "processing": processing,
        }

    @app.post("/act")
    def act(payload: dict):
        try:
            # json_numpy preserves uint8 dtype; nested JSON arrays do not.
            image = payload["image"]
            image = json_numpy.loads(image) if isinstance(image, str) else np.asarray(image)
            encoded = encode_observation(processor, image, payload["state"], normalizer)
            inputs = {k: v.unsqueeze(0).to(device) for k, v in encoded.items()}
            prompt = payload["language_instruction"]
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("language_instruction must be nonempty text")
            inputs.update({k: v.to(device) for k, v in processor.encode_language(prompt).items()})
            with lock, torch.inference_mode():
                if episode_seeded:
                    episode_seed.begin(payload["episode_id"], payload["seed"])
                noise = torch.randn((1, CONTRACT.num_actions, CONTRACT.max_action_dim), device=device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
                    actions = model.generate_actions(**inputs, noise=noise, steps=steps)
                if actions.shape != (1, 30, 4) or not torch.isfinite(actions).all():
                    raise RuntimeError("Invalid MetaWorld policy output")
            actions = normalizer.denormalize_action(actions)
            return {"action": actions[0].float().cpu().tolist()}
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return app


def main(argv=None):
    import torch
    import uvicorn
    from .checkpoint import base_pretrained_kwargs, load_checkpoint
    from .config import XVLAMorphActConfig
    from .metaworld import CONTRACT, validate_checkpoint_contract
    from .native.models.processing_xvla import XVLAProcessor

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8014)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--inference-acceleration", action=argparse.BooleanOptionalAction, default=True,
        help="Use fused PG RMSNorm/RoPE, scripted FP32 low-rank operations and CUDA denoising graphs",
    )
    parser.add_argument("--episode-seeded", action="store_true", help="Require benchmark episode_id and seed fields")
    args = parser.parse_args(argv)
    meta = json.loads((Path(args.checkpoint) / "metadata.json").read_text())
    pretrained_kwargs = base_pretrained_kwargs(args.base_model, meta)
    validate_checkpoint_contract(meta)
    processing = meta["training_config"].get("metaworld_processing")
    from .metaworld_processing import MetaWorldPolicy

    if processing is None:
        raise ValueError("MetaWorld checkpoints require a processing configuration")
    model = MetaWorldPolicy.from_pretrained(
        args.base_model,
        config=XVLAMorphActConfig.from_dict(meta["morphact_config"]),
        torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32,
        **CONTRACT.native_kwargs(),
        **pretrained_kwargs,
    ).to(args.device)
    load_checkpoint(model, args.checkpoint, expected_base_model=args.base_model)
    model.eval()
    model.configure_inference(enabled=args.inference_acceleration)
    processor = XVLAProcessor.from_pretrained(args.base_model, **pretrained_kwargs)
    uvicorn.run(
        create_app(
            model, processor, device=args.device, steps=args.steps,
            processing=processing, episode_seeded=args.episode_seeded,
        ),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
