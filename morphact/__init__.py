"""MorphAct: observation-conditioned generation of LoRA factors."""

__all__ = ["ParameterGenerator", "GeneratorConfig", "ProjectionSpec"]


def __getattr__(name):
    # Simulator clients need eval_seed without importing Torch or model code.
    if name in {"GeneratorConfig", "ProjectionSpec"}:
        from .config import GeneratorConfig, ProjectionSpec

        return {"GeneratorConfig": GeneratorConfig, "ProjectionSpec": ProjectionSpec}[name]
    if name == "ParameterGenerator":
        from .generator import ParameterGenerator

        return ParameterGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
