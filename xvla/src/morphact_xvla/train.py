"""Train native X-VLA with two MORPHACT generators and native visual/action interfaces."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True, help="Native X-VLA checkpoint directory or Hub model ID")
    parser.add_argument("--train-metas", required=True, help="Native X-VLA metadata JSON or directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--morphact-config", help="Optional XVLAMorphActConfig JSON")
    parser.add_argument("--resume", help="Complete MORPHACT checkpoint directory")
    parser.add_argument("--steps", type=int, default=100000, help="Total optimizer updates, including resumed updates")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-process microbatch")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.5e-5)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=1e-10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--mixed-precision", choices=("no", "bf16"), default="bf16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=2500)
    parser.add_argument("--log-interval", type=int, default=10)
    return parser


def learning_rate_at(step: int, *, peak: float, warmup: int, total: int, minimum_ratio: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    fraction = min(1.0, (step - warmup) / max(1, total - warmup - 1))
    return peak * (minimum_ratio + (1 - minimum_ratio) * 0.5 * (1 + math.cos(math.pi * fraction)))


def validate_args(args: argparse.Namespace) -> None:
    for name in ("steps", "batch_size", "gradient_accumulation_steps", "save_interval", "log_interval"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.num_workers < 0 or not 0 <= args.warmup_steps <= args.steps:
        raise ValueError("num_workers must be nonnegative and warmup_steps must be within the training budget")
    if not 0 <= args.min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must lie in [0, 1]")
    for name in ("learning_rate", "max_grad_norm"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")


def main(argv: list[str] | None = None) -> None:
    args = get_parser().parse_args(argv)
    validate_args(args)
    import torch
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, set_seed
    from torch.utils.data import DataLoader

    from .checkpoint import base_pretrained_kwargs, canonical_training_config, load_checkpoint, save_checkpoint
    from .config import XVLAMorphActConfig
    from .model import XVLAMorphAct
    from .native.data import worker_init_fn
    from .native.data.dataset import InfiniteDataReader
    from .native.models.processing_xvla import XVLAProcessor

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
    )
    set_seed(args.seed + accelerator.process_index)
    output_dir = Path(args.output_dir)
    if (output_dir / "run.json").exists() and not args.resume:
        raise FileExistsError("This output directory already contains a run; use --resume or a new directory")
    resume_meta = None
    if args.resume:
        resume_meta = json.loads((Path(args.resume) / "metadata.json").read_text())
    pretrained_kwargs = base_pretrained_kwargs(args.base_model, resume_meta)
    config_values = (
        json.loads(Path(args.morphact_config).read_text())
        if args.morphact_config
        else (resume_meta["morphact_config"] if resume_meta else {})
    )
    model = XVLAMorphAct.from_pretrained(
        args.base_model,
        config=XVLAMorphActConfig.from_dict(config_values),
        torch_dtype=torch.float32 if args.mixed_precision == "no" else torch.bfloat16,
        **pretrained_kwargs,
    ).to(accelerator.device)
    model.assert_trainable_state()
    parameters = list(model.trainable_parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay
    )
    training_config = canonical_training_config(
        {
            name: value
            for name, value in vars(args).items()
            if name
            not in {"resume", "output_dir", "base_model", "morphact_config", "save_interval", "log_interval", "num_workers"}
        }
    )
    training_config.update(world_size=accelerator.num_processes, betas=[0.9, 0.95])
    step = 0
    if resume_meta is not None:
        if canonical_training_config(resume_meta.get("training_config")) != training_config:
            raise ValueError("Resume requires the same optimizer schedule, global batch and model training settings")
        load_checkpoint(
            model, args.resume, optimizer=optimizer, expected_base_model=args.base_model, load_training_state=True
        )
        step = int(resume_meta["step"])
        accelerator.print(f"Restored model and optimizer at step {step}; training data: {args.train_metas}")
        if accelerator.num_processes > 1:
            # Data cursors and per-rank RNG streams are not checkpointed by this native iterable loader.
            set_seed(args.seed + accelerator.process_index + step)
    if step >= args.steps:
        accelerator.print(f"Checkpoint already reached {step} optimizer updates.")
        return
    processor = XVLAProcessor.from_pretrained(args.base_model, **pretrained_kwargs)
    dataset = InfiniteDataReader(
        args.train_metas,
        num_actions=model.base_model.num_actions,
        training=True,
        action_mode=model.base_model.action_mode,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=accelerator.device.type == "cuda",
        persistent_workers=args.num_workers > 0,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(args.seed + accelerator.process_index + step),
    )
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        if not args.resume:
            with (output_dir / "run.json").open("x") as handle:
                json.dump(
                    {
                        "training_config": training_config,
                        "morphact_config": model.morphact_config.to_dict(),
                        "trainable_parameters": sum(p.numel() for p in parameters),
                        "global_batch": args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes,
                    },
                    handle,
                    indent=2,
                )
    model, optimizer = accelerator.prepare(model, optimizer)
    # The native iterable samples independently on each rank; do not additionally shard its batches.
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = torch.zeros((), device=accelerator.device)
    microbatches = 0
    for batch in loader:
        language = processor.encode_language(batch.pop("language_instruction"))
        inputs = {key: value.to(accelerator.device, non_blocking=True) for key, value in {**batch, **language}.items()}
        for group in optimizer.param_groups:
            group["lr"] = learning_rate_at(
                step,
                peak=args.learning_rate,
                warmup=args.warmup_steps,
                total=args.steps,
                minimum_ratio=args.min_lr_ratio,
            )
        with accelerator.accumulate(model):
            loss_dict = model(**inputs)
            loss = sum(loss_dict.values())
            finite = torch.stack([torch.isfinite(value).all() for value in loss_dict.values()]).all().to(torch.int32)
            if int(accelerator.reduce(finite, reduction="sum")) != accelerator.num_processes:
                raise FloatingPointError("Non-finite X-VLA loss; optimizer update refused")
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(parameters, args.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("Non-finite gradient norm; optimizer update refused")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        loss_sum += loss.detach()
        microbatches += 1
        if not accelerator.sync_gradients:
            continue
        step += 1
        if step % args.log_interval == 0 or step == 1:
            mean_loss = accelerator.reduce(loss_sum / microbatches, reduction="mean")
            record = {
                "step": step,
                "loss": float(mean_loss),
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": float(grad_norm),
            }
            accelerator.print(json.dumps(record))
            if accelerator.is_main_process:
                with (output_dir / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
            loss_sum.zero_()
            microbatches = 0
        if step % args.save_interval == 0 or step == args.steps:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_checkpoint(
                    accelerator.unwrap_model(model),
                    output_dir / f"checkpoint-{step:08d}",
                    base_model=args.base_model,
                    step=step,
                    optimizer=optimizer,
                    training_config=training_config,
                )
            accelerator.wait_for_everyone()
        if step >= args.steps:
            break
    accelerator.end_training()


if __name__ == "__main__":
    main()
