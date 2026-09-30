"""Real Gloo/DDP transport around the unchanged production learner arithmetic."""

from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from tests.training_harness import learner_types, make_learner, run_step
from tests.training_oracle import reference_update


class DistributedPolicy(DistributedDataParallel):
    @property
    def table(self):
        return self.module.table

    @property
    def calls(self):
        return self.module.calls

    def forward(self, *args, **kwargs):
        self.module.phase = self.phase
        return super().forward(*args, **kwargs)


def worker(rank, world, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=90),
    )
    try:
        spec = json.loads(
            (Path(__file__).parent / "fixtures/training_v1/inputs.json").read_text()
        )
        spec["num_samples"] = 16
        for step in spec["steps"]:
            for key in ("responses", "rewards", "active"):
                step[key] *= 4
        cases = []
        with learner_types() as types:
            for method in ("drgrpo", "redr", "maxrl", "remax"):
                for microbatch in (1, 2, 4):
                    learner = make_learner(types, spec, method, microbatch)
                    learner.args.replicated_freeform_sampling = True
                    learner.strategy.grad_acc_step = 16 // world // microbatch
                    learner.model = DistributedPolicy(learner.model)
                    learner.model.phase = "fresh"
                    for step in spec["steps"]:
                        before = learner.model.table.detach().tolist()
                        run_step(learner, spec, step)
                        groups = learner.replay_batches[0]["groups"]
                        oracle = reference_update(
                            before, spec, step, groups, method=method
                        )
                        torch.testing.assert_close(
                            learner.strategy.updates[-1],
                            torch.tensor(oracle["gradient"]),
                            rtol=3e-6,
                            atol=3e-7,
                        )
                        torch.testing.assert_close(
                            learner.model.table.detach(),
                            torch.tensor(oracle["parameters"]),
                            rtol=3e-6,
                            atol=3e-7,
                        )
                        snapshots = [None] * world
                        dist.all_gather_object(
                            snapshots, learner._online_canonical_bank.state_dict()
                        )
                        assert all(s == snapshots[0] for s in snapshots)
                        calls = [
                            c
                            for c in learner.strategy.backward_calls
                            if c["phase"] == "replay"
                        ]
                        assert calls
                        if method in ("drgrpo", "maxrl"):
                            assert all(
                                torch.count_nonzero(c["gradient"]) == 0 for c in calls
                            )
                    cases.append(
                        {
                            "method": method,
                            "microbatch": microbatch,
                            "accumulation": learner.strategy.grad_acc_step,
                            "updates": len(learner.strategy.updates),
                        }
                    )
            # Every rank must reject before backward, including a bank mismatch.
            if world > 1:
                for fault in (
                    "coefficient",
                    "membership",
                    "empty_replay",
                    "unreplicated",
                ):
                    learner = make_learner(types, spec, "remax")
                    learner.args.replicated_freeform_sampling = fault != "unreplicated"
                    learner.strategy.grad_acc_step = 16 // world
                    if fault == "coefficient":
                        learner.args.online_canonical_replay_alpha += rank * 0.01
                    if fault in ("membership", "empty_replay"):
                        update = learner._baseline_update_with_precomputed_advantages

                        def different_bank(**kwargs):
                            if rank == 1:
                                groups = (
                                    []
                                    if fault == "empty_replay"
                                    else kwargs["canonical_replay_groups"]
                                )
                                kwargs["canonical_replay_groups"] = [
                                    replace(
                                        g,
                                        outcome_keys=tuple(
                                            k + ":changed" for k in g.outcome_keys
                                        ),
                                    )
                                    for g in groups
                                ]
                            return update(**kwargs)

                        learner._baseline_update_with_precomputed_advantages = (
                            different_bank
                        )
                    learner.model = DistributedPolicy(learner.model)
                    learner.model.phase = "fresh"
                    try:
                        run_step(learner, spec, spec["steps"][0])
                    except RuntimeError as error:
                        message = (
                            "require replicated"
                            if fault == "unreplicated"
                            else "differ across ranks"
                        )
                        assert message in str(error)
                    else:
                        raise AssertionError("rank mismatch accepted")
                    assert not learner.strategy.backward_calls
        if rank == 0:
            Path(output).write_text(json.dumps(cases))
    finally:
        dist.destroy_process_group()
