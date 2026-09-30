"""Compatibility alias for the historical ucpo implementation."""
import sys
from .experiments import ucpo as _implementation
sys.modules[__name__] = _implementation
