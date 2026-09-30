"""Compatibility alias for the historical setpo_embedder implementation."""
import sys
from .experiments import setpo_embedder as _implementation
sys.modules[__name__] = _implementation
