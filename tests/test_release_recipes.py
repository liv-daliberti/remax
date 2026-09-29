"""Guard the scientific differences between the four portable method recipes."""
import json
from pathlib import Path
import shlex
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('recipe', sorted((ROOT/'configs').glob('*.json')), ids=lambda p:p.stem)
def test_rendered_recipe_preserves_objective_control_and_cadence(recipe, tmp_path):
    (tmp_path/'train').mkdir()
    (tmp_path/'eval').mkdir()
    output = subprocess.check_output([sys.executable,str(ROOT/'ops/run_recipe.py'),str(recipe),'--data-root',str(tmp_path),'--model',str(tmp_path/'model'),'--output',str(tmp_path/'output'),'--seed','47'],text=True)
    command = shlex.split(next(l for l in output.splitlines() if l.startswith('[train] command:')).split(':',1)[1])
    data = json.loads(recipe.read_text())
    method = data['method']
    assert ('--maxrl-task-objective' in command) == (method in {'maxrl','remax'})
    assert ('--online-canonical-replay-compute-only' in command) == (method in {'maxrl','drgrpo'})
    assert command[command.index('--seed')+1] == '47'
    assert command[command.index('--eval_steps')+1] == '192'
    assert command[command.index('--num_prompt_epoch')+1] == '8'
    assert command[command.index('--xdr-tau')+1] == 'inf'
    assert 'remax.train_zero_math' in command
