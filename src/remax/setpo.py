"""Compatibility alias for the historical setpo implementation."""
import sys
from .experiments import setpo as _implementation
sys.modules[__name__] = _implementation
