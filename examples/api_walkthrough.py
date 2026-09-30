"""Run the same public API example as the installed `remax walkthrough` command."""

import json
from remax.walkthrough import walkthrough

print(json.dumps(walkthrough()))
