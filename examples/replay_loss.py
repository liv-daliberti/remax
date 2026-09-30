"""The same CPU example exposed by the installed `remax demo` command."""
import json
from remax.cli import demo

print(json.dumps(demo()))
