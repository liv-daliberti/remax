"""Render a registered portable training command; --execute launches locally."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def environment(recipe, *, data_root, model, output, seed):
    env = {k:v for k,v in os.environ.items() if not k.startswith('OAT_ZERO_')}
    env.update({k:str(v) for k,v in recipe['environment'].items()})
    env.update({
        'OAT_ZERO_REPO_ROOT': str(ROOT), 'OAT_ZERO_PYTHON': sys.executable,
        'OAT_ZERO_SOURCE_ROOT': str(ROOT / 'src'),
        'OAT_ZERO_PRETRAIN': str(model),
        'OAT_ZERO_PROMPT_DATA': str(data_root / 'train'),
        'OAT_ZERO_EVAL_DATA': str(data_root / 'eval'),
        'OAT_ZERO_SEED': str(seed), 'SAVE_PATH': str(output),
        'OAT_ZERO_DRY_RUN': '1',
    })
    if env.get('OAT_ZERO_CANONICAL_ACTION_TASK', 'none') != 'none':
        env['VLLM_USE_V1'] = '0'
    return env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('recipe', type=Path)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True, help='local snapshot of the recipe model_revision')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    recipe = json.loads(args.recipe.read_text())
    env = environment(recipe, data_root=args.data_root.resolve(), model=args.model.resolve(), output=args.output.resolve(), seed=args.seed if args.seed is not None else recipe['seed'])
    if args.execute:
        for path in (args.data_root/'train/dataset_dict.json', args.data_root/'eval/dataset_dict.json', args.model/'config.json'):
            if not path.is_file():
                p.error(f'missing training input: {path}')
        env['OAT_ZERO_DRY_RUN'] = '0'
    subprocess.run(['bash', str(ROOT/'ops/train.sh')], env=env, check=True)

if __name__ == '__main__':
    main()
