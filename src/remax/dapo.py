"""Compatibility alias for the historical dapo implementation."""
import sys
from .experiments import dapo as _implementation
sys.modules[__name__] = _implementation
