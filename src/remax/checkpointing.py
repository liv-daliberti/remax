"""Compatibility import for the framework-independent checkpoint protocol."""
from .core.checkpoints import *  # noqa: F401,F403
from .core import checkpoints as _impl

def __getattr__(name):
    return getattr(_impl, name)
