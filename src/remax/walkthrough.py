"""A CPU API example: grade saved responses, admit modes, score replay, restore a bank.

The tiny character-level bigram model illustrates the replay primitive only.
Actual Re:Dr/Re:Max training also needs the fresh RL objective and the registered
coefficient/accumulation scaling in the OAT adapter.
"""

from __future__ import annotations

import torch
from modebench.api import Task

from .benchmark import grade_task
from .core import (
    OnlineCanonicalBank,
    canonical_replay_uniform_verified_likelihood_loss,
    materialize_canonical_replay_batch,
)


def walkthrough() -> dict:
    task = Task(
        id="walkthrough-countdown",
        level=1,
        domain="countdown",
        problem="Use 1, 2, 3 exactly once to make 6.",
        answer={"verifier": "countdown", "numbers": [1, 2, 3], "target": 6},
    )
    responses = [r"\boxed{1+2+3}", r"\boxed{1*2*3}", r"\boxed{1+2-3}"]
    grades = [grade_task(task, response) for response in responses]
    # Token IDs here are ASCII characters, solely to make the example self-contained.
    prompt = tuple(map(ord, task.problem))
    bank = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        replay_capacity=16,
        global_replay_groups_per_step=1,
    )
    bank.score_and_update(
        prompt_token_ids=[prompt] * len(responses),
        outcome_keys=[grade["canonical_key"] for grade in grades],
        task_rewards=[float(grade["verified"]) for grade in grades],
        active_mask=[True] * len(responses),
        num_samples=len(responses),
        response_token_ids=[tuple(map(ord, response)) for response in responses],
    )
    groups = bank.scheduled_global_replay_groups(min_modes=1)
    batch = materialize_canonical_replay_batch(groups, pad_token_id=0, device="cpu")
    # A trainable bigram table: logits for the next character given the current one.
    logits_table = torch.nn.Parameter(torch.zeros(128, 128))
    optimizer = torch.optim.SGD([logits_table], lr=0.1)
    logits = logits_table[batch.input_ids[:, :-1]]
    token_scores = (
        logits.log_softmax(-1).gather(-1, batch.input_ids[:, 1:, None]).squeeze(-1)
    )
    mask = batch.response_masks
    scores = (token_scores * mask).sum(-1) / mask.sum(-1)
    loss = canonical_replay_uniform_verified_likelihood_loss(scores, batch.group_sizes)
    loss.loss.backward()
    optimizer.step()
    restored = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        replay_capacity=16,
        global_replay_groups_per_step=1,
    )
    restored.load_state_dict(bank.state_dict())
    return {
        "statuses": [grade["status"] for grade in grades],
        "retained_modes": sum(batch.group_sizes),
        "prompt_tokens_scored": int(mask[:, : len(prompt) - 1].sum()),
        "parameters_updated": bool(logits_table.detach().abs().sum() > 0),
        "bank_restored": restored.state_dict() == bank.state_dict(),
        "scope": "CPU replay API example; not a complete RL training run",
    }
