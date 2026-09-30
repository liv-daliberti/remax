"""Compatibility alias for the historical seed_weights implementation."""
import sys
from .experiments import seed_weights as _implementation
sys.modules[__name__] = _implementation
