"""Compatibility entry point for the installed recipe launcher."""
import sys
from remax import launcher

if __name__ == "__main__":
    launcher.main()
else:
    sys.modules[__name__] = launcher
