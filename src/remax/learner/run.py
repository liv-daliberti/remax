"""Compatibility alias for OAT run orchestration."""
import sys
from ..integrations.oat import runner as _implementation
sys.modules[__name__] = _implementation
