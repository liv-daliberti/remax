"""Compatibility alias for the historical outcome_collision implementation."""
import sys
from .experiments import outcome_collision as _implementation
sys.modules[__name__] = _implementation
