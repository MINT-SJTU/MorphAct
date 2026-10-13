"""Configure inference operations without changing parameters or checkpoint keys."""

from __future__ import annotations

from .inference_ops import scripted_headwise_delta, scripted_lora_linear
from .inference_pg import GeneratorFusion


def configure_inference(model, generators, *, enabled: bool = True) -> None:
    executor = model.executor
    if not hasattr(executor, "_native_linear"):
        executor._native_linear = executor.linear
        if hasattr(executor, "delta"):
            executor._native_delta = executor.delta
    executor.linear = scripted_lora_linear if enabled else executor._native_linear
    if hasattr(executor, "delta"):
        executor.delta = scripted_headwise_delta if enabled else executor._native_delta

    # Each generator owns its bindings; disabling restores the original callables.
    for generator in generators:
        fusion = getattr(generator, "_inference_fusion", None)
        if enabled and fusion is None:
            generator._inference_fusion = GeneratorFusion(generator)
        elif not enabled and fusion is not None:
            fusion.restore()
            del generator._inference_fusion
    model.inference_acceleration = enabled
