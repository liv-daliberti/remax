"""CPU adapter for the *unmodified* production learner, not a second learner.

Only external infrastructure is substituted: OAT's argument base and two
statistics helpers, its completion mask/reduction interfaces, CUDA device
selection, and DeepSpeed's accumulation/step boundary. No ReMax method, bank,
validator, scheduler, token scorer, objective or gradient is replaced.
"""

from contextlib import contextmanager
from dataclasses import asdict
import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import remax

from remax.online_canonical_bank import OnlineCanonicalBank


@contextmanager
def learner_types():
    # Scope import shims so neither test order nor an installed GPU stack changes
    # these CPU contracts. PPOArgs contributes no behavior exercised here.
    names = (
        "oat",
        "oat.utils",
        "oat.utils.ops",
        "oat.algorithms",
        "oat.algorithms.ppo",
    )
    shims = {name: ModuleType(name) for name in names}
    shims["oat.algorithms.ppo"].PPOArgs = type("PPOArgs", (), {})
    shims["oat.utils.ops"].masked_mean = lambda x, mask, axis=None: (
        (x * mask).sum(dim=axis) / mask.sum(dim=axis)
    )
    shims["oat.utils.ops"].entropy_from_logits = lambda x: (
        -(x.softmax(-1) * x.log_softmax(-1)).sum(-1)
    )
    imported = ("remax.args", "remax.learner.base", "remax.learner.grpo") + tuple(
        f"remax.{area}.oat.{path.stem}"
        for area in ("integrations", "experiments")
        for path in sorted(
            (Path(remax.__file__).parent / area / "oat").glob("*.py")
        )
        if path.stem != "__init__"
    )
    missing = object()
    # Restore only touched module entries/parent attributes, not all of
    # sys.modules: PyTorch may lazily import optimizer internals during a test.
    parents = [
        (importlib.import_module(name.rpartition(".")[0]), name.rpartition(".")[2])
        for name in imported
    ]
    attributes = [
        (parent, attr, getattr(parent, attr, missing)) for parent, attr in parents
    ]
    saved = {name: sys.modules.get(name, missing) for name in (*shims, *imported)}
    for name in imported:
        sys.modules.pop(name, None)
    sys.modules.update(shims)
    try:
        args_module = importlib.import_module("remax.args")
        base = importlib.import_module("remax.learner.base")
        grpo = importlib.import_module("remax.learner.grpo")

        class Learner(grpo.ZeroMathGrpoMixin, base.ZeroMathLearnerBaseMixin):
            @staticmethod
            def get_completion_mask(attention, prompt_lengths):
                positions = torch.arange(attention.shape[1])[None, :]
                return attention.bool() & (
                    positions >= torch.tensor(prompt_lengths)[:, None]
                )

        yield Learner, args_module.ZeroMathArgs
    finally:
        for name, value in saved.items():
            if value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value
        for parent, attr, value in attributes:
            if value is missing:
                if hasattr(parent, attr):
                    delattr(parent, attr)
            else:
                setattr(parent, attr, value)


class TinyPolicy(torch.nn.Module):
    """Eight-token causal bigram LM with all 64 trainable logits observable."""

    def __init__(self, initial):
        super().__init__()
        self.table = torch.nn.Parameter(torch.tensor(initial, dtype=torch.float32))
        self.config = SimpleNamespace(vocab_size=8)
        self.calls = []
        self.phase = "fresh"

    def forward(self, input_ids, attention_mask):
        self.calls.append(
            dict(
                phase=self.phase,
                grad=torch.is_grad_enabled(),
                training=self.training,
                rows=len(input_ids),
            )
        )
        return {"logits": self.table[input_ids]}


class AccumulatingSGD:
    """DeepSpeed contract double: divide *each* backward by accumulation width;
    step/clear only on each kth optimizer_step call, not kth backward call.
    PyTorch performs real autograd and SGD, with no mocked parameter updates.
    """

    def __init__(self, width):
        self.grad_acc_step = width
        self.micro_steps = 0
        self.backward_calls = []
        self.updates = []

    @staticmethod
    def is_rank_0():
        return False

    @staticmethod
    def get_gradient_norm(model):
        return float(model.table.grad.norm())

    def backward(self, loss, model, optimizer):
        before = (
            torch.zeros_like(model.table)
            if model.table.grad is None
            else model.table.grad.clone()
        )
        (loss / self.grad_acc_step).backward()
        self.backward_calls.append(
            dict(
                phase=model.phase,
                loss=float(loss.detach()),
                gradient=(model.table.grad - before).clone(),
            )
        )

    def optimizer_step(self, optimizer, model, scheduler):
        self.micro_steps += 1
        if self.micro_steps % self.grad_acc_step == 0:
            self.updates.append(model.table.grad.clone())
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


def make_learner(types, spec, method, microbatch=1, alpha=None):
    Learner, Args = types
    learner = Learner()
    args = Args()
    # Explicit maintained-method settings plus small fixture dimensions. Do not
    # make missing attributes silently return zero: new dependencies must fail.
    values = dict(
        num_samples=spec["num_samples"],
        train_batch_size=spec["num_samples"],
        train_batch_size_per_device=microbatch,
        num_ppo_epochs=1,
        critic_type="drgrpo",
        temperature=spec["temperature"],
        beta=0.0,
        reward_scale=1.0,
        kl_penalty_coef=0.0,
        reinforce_update=False,
        cliprange=0.2,
        generate_max_length=spec["max_response_length"],
        maxrl_task_objective=method in ("maxrl", "remax"),
        online_canonical_replay_compute_only=method in ("maxrl", "drgrpo"),
        online_canonical_replay=True,
        online_canonical_replay_alpha=spec["alpha"] if alpha is None else alpha,
        online_canonical_replay_objective="verified_likelihood_per_rollout",
        online_canonical_replay_key_weighting="uniform",
        online_canonical_key_mode="modebench_outcome",
        online_canonical_replay_global_groups_per_step=1,
        online_canonical_replay_global_bootstrap_steps=0,
        maxent_alpha=0.0,
        policy_entropy_coef=0.0,
        seed=17,
    )
    for key, value in values.items():
        setattr(args, key, value)
    learner.args = args
    surfaces = {tuple(row["tokens"]): row["text"] for row in spec["surfaces"]}
    learner.tokenizer = SimpleNamespace(
        pad_token_id=0,
        eos_token_id=7,
        vocab_size=8,
        decode=lambda tokens, **kw: surfaces[tuple(tokens)],
    )
    learner.model = TinyPolicy(spec["initial_logits"])
    learner.optimizer = torch.optim.SGD(
        learner.model.parameters(), lr=spec["learning_rate"]
    )
    learner.scheduler = None
    learner.ref_model = None
    learner.strategy = AccumulatingSGD(spec["num_samples"] // microbatch)
    learner._online_canonical_bank = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        replay_capacity=spec["capacity"],
        global_replay_groups_per_step=1,
    )
    learner._invalid_scoring_token_ids_warned_contexts = set()
    learner._invalid_logit_columns_warned_contexts = set()
    learner._baseline_grad_norm_logging_disabled_warned = False
    learner._canonical_action_token_ids = None
    learner.masked_aggregator = lambda x, mask, axis: (
        (x * mask).sum(dim=axis) / args.generate_max_length
    )
    learner.steps = 0
    learner.replay_batches = []
    learner.replay_scores = []
    # Observers call the original methods unchanged. They distinguish actual
    # replay work from telemetry which could incorrectly claim it happened.
    materialize = learner._materialize_canonical_replay
    score = learner._score_canonical_replay_rows

    def observed_materialize(groups, **kw):
        batch = materialize(groups, **kw)
        learner.replay_batches.append(
            dict(groups=[asdict(g) for g in groups], batch=batch)
        )
        return batch

    def observed_score(batch, **kw):
        learner.model.phase = "replay"
        try:
            result = score(batch, **kw)
            learner.replay_scores.append(
                dict(
                    start=kw["start"],
                    stop=kw["stop"],
                    grad=torch.is_grad_enabled(),
                    scores=result.detach().tolist(),
                )
            )
            return result
        finally:
            # Backward is called after score() returns, while model remains in
            # eval mode; strategy observer uses that to identify its phase.
            learner.model.phase = "fresh"

    learner._materialize_canonical_replay = observed_materialize
    learner._score_canonical_replay_rows = observed_score
    original_backward = learner.strategy.backward

    def observed_backward(loss, model, optimizer):
        model.phase = "fresh" if model.training else "replay"
        try:
            original_backward(loss, model, optimizer)
        finally:
            model.phase = "fresh"

    learner.strategy.backward = observed_backward
    return learner


def trajectory(spec, step):
    rows = [step["prompt"] + response for response in step["responses"]]
    width = max(map(len, rows)) + 2  # deliberate unused trailing padding
    return dict(
        input_ids=torch.tensor([row + [0] * (width - len(row)) for row in rows]),
        attention_mask=torch.tensor(
            [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        ),
        prompt_ids_lens=[len(step["prompt"])] * len(rows),
        rewards=[[r] for r in step["rewards"]],
        loss_masks=step["active"],
        references=[spec["reference"]] * len(rows),
    )


def run_step(learner, spec, step):
    learner.model.calls.clear()
    learner.strategy.backward_calls.clear()
    learner.replay_batches.clear()
    learner.replay_scores.clear()
    # Patch only device selection; all tensors and actual production methods run
    # on CPU. Preserve the caller's numpy RNG state.
    state = np.random.get_state()
    try:
        np.random.seed(17 + learner.steps)
        with patch("torch.cuda.current_device", return_value=torch.device("cpu")):
            infos = learner._grpo_learning_step_with_progress(trajectory(spec, step))
    finally:
        np.random.set_state(state)
    learner.steps += 1
    return infos


def snapshot(learner, infos):
    """Stable observables only; exclude wall-clock durations and RNG fingerprints."""
    replay = []
    for observed in learner.replay_batches:
        batch = observed["batch"]
        replay.append(
            dict(
                groups=observed["groups"],
                input_ids=batch.input_ids.tolist(),
                attention_mask=batch.attention_mask.tolist(),
                response_masks=batch.response_masks.tolist(),
            )
        )
    info_keys = (
        "canonical_replay_actuator_loss",
        "canonical_replay_raw_weighted_loss",
        "canonical_replay_weighted_loss",
        "canonical_replay_alpha_used",
        "canonical_replay_objective_scale",
        "canonical_replay_reward_estimator_scale",
        "canonical_replay_actuator_groups",
        "canonical_replay_actuator_modes",
        "canonical_replay_eligible_groups",
        "canonical_replay_retained_modes",
        "canonical_replay_normalized_model_entropy",
        "canonical_replay_applied_score_gradient_sum",
    )
    return dict(
        bank=learner._online_canonical_bank.state_dict(),
        replay=replay,
        scores=[
            v
            for chunk in learner.replay_scores
            if not chunk["grad"]
            for v in chunk["scores"]
        ],
        gradient=learner.strategy.updates[-1].tolist(),
        parameters=learner.model.table.detach().tolist(),
        infos={key: float(infos[key]) for key in info_keys if key in infos},
    )
