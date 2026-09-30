"""Compatibility alias for the historical gapo implementation."""
import sys
from .experiments import gapo as _implementation
sys.modules[__name__] = _implementation
