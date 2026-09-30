"""Compatibility entry point for OAT initialization."""
from ..integrations.oat.initialization import ZeroMathInitMixin


def __getattr__(name):
    if name not in {"build_maxent_controllers", "build_semantic_shannon_tracker"}:
        raise AttributeError(name)
    from ..experiments.oat import initialization
    return getattr(initialization, name)
