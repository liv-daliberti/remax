"""Compatibility alias for OAT method dispatch."""
import sys
from ..integrations.oat import dispatch as _implementation
sys.modules[__name__] = _implementation
