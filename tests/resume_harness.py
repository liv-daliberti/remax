"""Execute the production run loop + GRPO/replay updates with CPU infrastructure.

The tiny actor and DeepSpeed transport are replaced; AdamW, scheduler, bank,
checkpoint transaction, progress restoration, RNG handling and run loop are real.
"""

from collections import deque
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch._dynamo  # initialize optimizer imports before temporary infrastructure shims
from torch.utils.data import DataLoader, DistributedSampler

from boundary_runtime_harness import runtime_modules
from training_harness import AccumulatingSGD, TinyPolicy, make_learner, trajectory

SPEC = json.loads(
    (Path(__file__).parent / "fixtures/training_v1/inputs.json").read_text()
)


class Interrupted(Exception):
    pass


class Engine(TinyPolicy):
    @property
    def model(self):
        return self

    def save_checkpoint(self, directory, tag, client_state, **kwargs):
        folder = Path(directory) / tag
        folder.mkdir(exist_ok=True)
        learner = self.owner
        torch.save(
            {
                "model": self.state_dict(),
                "optimizer": learner.optimizer.state_dict(),
                "scheduler": learner.scheduler.state_dict(),
                "client": client_state,
            },
            folder / "state.pt",
        )
        return True


class Strategy(AccumulatingSGD):
    def __init__(self, learner):
        super().__init__(SPEC["num_samples"])
        self.learner = learner
        self.args = learner.args
        self.world_size = 1

    @staticmethod
    def is_rank_0():
        return True

    @staticmethod
    def print(*a):
        pass

    pprint = print

    @staticmethod
    def all_reduce(value, **kwargs):
        return value

    def save_model(self, *a, **kw):
        pass

    def optimizer_step(self, optimizer, model, scheduler):
        self.micro_steps = model.micro_steps
        super().optimizer_step(optimizer, model, scheduler)
        model.micro_steps = self.micro_steps
        if self.micro_steps % self.grad_acc_step == 0:
            scheduler.step()

    def load_ckpt(self, model, root, tag):
        state = torch.load(Path(root) / tag / "state.pt", weights_only=False)
        model.load_state_dict(state["model"])
        self.learner.optimizer.load_state_dict(state["optimizer"])
        self.learner.scheduler.load_state_dict(state["scheduler"])
        return str(Path(root) / tag), state["client"]


@contextmanager
def run_case(
    folder, method, *, resume=None, stop=None, identity=None, clear_every=float("inf")
):
    with runtime_modules() as (_, run):
        grpo = sys.modules["remax.learner.grpo"]
        base = sys.modules["remax.learner.base"]
        args_module = sys.modules["remax.args"]

        class Learner(
            run.ZeroMathRunMixin, grpo.ZeroMathGrpoMixin, base.ZeroMathLearnerBaseMixin
        ):
            @staticmethod
            def get_completion_mask(attention, prompt_lengths):
                return attention.bool() & (
                    torch.arange(attention.shape[1])[None, :]
                    >= torch.tensor(prompt_lengths)[:, None]
                )

        learner = make_learner((Learner, args_module.ZeroMathArgs), SPEC, method)
        args = learner.args
        for name, value in dict(
            num_prompt_epoch=3,
            rollout_batch_size=1,
            rollout_batch_size_per_device=1,
            resume_dir=str(resume.parent) if resume else None,
            resume_tag=resume.name if resume else None,
            debug=False,
            online_evaluation=True,
            eval_steps=2,
            save_ckpt=True,
            resume_steps=1,
            resume_from=1,
            max_resume_num=12,
            export_steps=-1,
            export_from=0,
            logging_steps=1,
            sync_params_every=1,
            buffer_clear_every=clear_every,
            dump_replay_every=0,
            dump_all_buffer=False,
            max_queries=10000,
            eval_only=False,
            enable_prefix_caching=True,
        ).items():
            setattr(args, name, value)
        args._remax_resume_identity = identity or {
            "protocol": "test-run-v1",
            "method": method,
            "horizon": 9,
        }
        learner.model = Engine(SPEC["initial_logits"])
        learner.model.owner = learner
        learner.model.micro_steps = 0
        learner.optimizer = torch.optim.AdamW(
            learner.model.parameters(), lr=0.015, weight_decay=0.02
        )
        learner.scheduler = torch.optim.lr_scheduler.LambdaLR(
            learner.optimizer, lambda step: 1.0 - 0.8 * step / 9
        )
        learner.strategy = Strategy(learner)
        learner.save_path = str(folder)
        folder.mkdir(parents=True, exist_ok=True)
        learner.actors = []
        learner.actor_info = {}
        learner.update_interval = 1
        learner.global_step = 0
        learner.policy_sgd_step = 0
        learner.query_step = 0
        learner.prompt_consumed = 0
        learner.prompt_epoch = 0
        learner._prompt_batches_consumed_total = 0
        learner._wandb = None
        learner._wandb_run_id = None
        learner._wandb_run_name = None
        learner.pi_buffer = deque(maxlen=SPEC["num_samples"])
        learner.all_buffer = deque()
        learner._same_actor_group = None
        dataset = [(f"prompt-{i}", str(i), "reference") for i in range(3)]
        sampler = DistributedSampler(
            dataset, num_replicas=1, rank=0, shuffle=True, seed=args.seed
        )
        learner.prompts_dataloader = DataLoader(
            dataset, batch_size=1, sampler=sampler, num_workers=0
        )
        learner.eval_prompts_dataloader = []
        learner.eval_history = []
        learner.decisions = []
        learner._init = lambda *a: None
        learner._init_local_actor_weight_sync = lambda: None
        learner._pre_learning = lambda: None
        learner._post_learning = lambda: None
        learner.sync_params_to_actors = lambda: None
        learner._init_wandb = lambda *a: None
        learner._finalize_successful_storage = lambda: None
        learner.get_misc_info = lambda: {}
        learner.get_current_query = lambda: learner.query_step
        learner._should_do = (
            lambda cadence: cadence > 0 and learner.steps % cadence == 0
        )

        actor_cache = {"warm": False}
        learner.cache_resets = []

        def reset_prefix_cache():
            actor_cache["warm"] = False
            learner.cache_resets.append(learner.steps)

        def actor_step(raw, processed, refs, *, sampling_seed):
            assert not actor_cache[
                "warm"
            ], "actor cache must not depend on pre-restart evaluation"
            actor_cache["warm"] = True
            row = copy.deepcopy(SPEC["steps"][int(raw[0])])
            # Seeded actor sampling is position-bound, independent of restarts.
            generator = torch.Generator().manual_seed(sampling_seed)
            permutation = torch.randperm(4, generator=generator).tolist()
            for name in ("responses", "rewards", "active"):
                row[name] = [row[name][i] for i in permutation]
            learner.current_row = row
            learner.current_decision = {
                "prompt": raw[0],
                "seed": sampling_seed,
                "samples": permutation,
                "rng_probe": [
                    random.random(),
                    float(np.random.rand()),
                    float(torch.rand(())),
                ],
            }
            return [
                SimpleNamespace(response_ids=response, rewards=[reward])
                for response, reward in zip(row["responses"], row["rewards"])
            ]

        learner.actors = [
            SimpleNamespace(step=actor_step, reset_prefix_cache=reset_prefix_cache)
        ]
        learner.collector = SimpleNamespace(
            ipc_client=SimpleNamespace(deserialize_ipc=lambda x: x),
            get_metrics=lambda *a: {},
        )

        def feedback(rows):
            learner.pi_buffer.extend(rows)
            learner.query_step += len(rows)

        learner.process_feedback_data = feedback

        def learn(step):
            start = len(learner.replay_batches)
            with patch("torch.cuda.current_device", return_value=torch.device("cpu")):
                infos = learner._grpo_learning_step_with_progress(
                    trajectory(SPEC, learner.current_row)
                )
            learner.global_step += 1
            learner.policy_sgd_step += 1
            learner.decisions.append(
                {
                    **learner.current_decision,
                    "step": learner.steps,
                    "bank": copy.deepcopy(learner._online_canonical_bank.state_dict()),
                    "replay": [r["groups"] for r in learner.replay_batches[start:]],
                    "lr": learner.scheduler.get_last_lr(),
                }
            )
            return {"train/" + k: v for k, v in infos.items()}

        learner.learn = learn

        def evaluate(*a):
            actor_cache["warm"] = True
            values = [random.random(), float(np.random.rand()), float(torch.rand(()))]
            learner.eval_history.append({"step": learner.steps, "rng_probe": values})
            return {"eval/reward": 0.5}

        learner.evaluate = evaluate
        original_save = learner._save_resume_checkpoint

        def save():
            original_save()
            if stop == learner.steps:
                raise Interrupted()

        learner._save_resume_checkpoint = save
        sys.modules["remax.integrations.oat.lifecycle"].tqdm = (
            lambda *a, **kw: SimpleNamespace(update=lambda: None)
        )
        with (
            patch.object(run.dist, "barrier"),
            patch.object(run.dist, "get_world_size", return_value=1),
            patch.object(run.dist, "is_initialized", return_value=False),
            patch.object(run.lp, "stop", create=True),
        ):
            yield learner
