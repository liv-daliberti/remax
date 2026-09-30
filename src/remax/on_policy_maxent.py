"""Compatibility alias for the historical on_policy_maxent implementation."""
import sys
from .experiments import on_policy_maxent as _implementation
sys.modules[__name__] = _implementation
