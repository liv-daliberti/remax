"""Check frozen training aggregates against their original verified-key archive."""
import json
from pathlib import Path
from build_mode_diversity_training import build

ROOT = Path(__file__).resolve().parents[1]

def reproduce():
    expected = json.loads((ROOT / 'evidence/mode_diversity_training.json').read_text())
    actual = build()
    if actual['archive']['sha256'] != expected['archive']['sha256']:
        raise AssertionError('frozen training archive hash differs')
    for field in ('schema', 'archive_manifest', 'definition', 'arms', 'seeds', 'coverage'):
        if actual[field] != expected[field]:
            raise AssertionError(f'training reproduction differs: {field}')
    return actual['coverage']

if __name__ == '__main__':
    print(json.dumps(reproduce()))
