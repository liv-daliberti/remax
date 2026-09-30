"""Compatibility alias for the historical xdr implementation."""
import sys
from .experiments import xdr as _implementation
sys.modules[__name__] = _implementation
