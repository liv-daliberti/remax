"""Compatibility alias for the historical rlep implementation."""
import sys
from .experiments import rlep as _implementation
sys.modules[__name__] = _implementation
