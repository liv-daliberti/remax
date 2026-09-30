"""Compatibility alias for the historical semantic_shannon implementation."""
import sys
from .experiments import semantic_shannon as _implementation
sys.modules[__name__] = _implementation
