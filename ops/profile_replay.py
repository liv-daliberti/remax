"""Measure exact discovery storage and the maintained replay adapter (optional CUDA)."""

from __future__ import annotations
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import pickle
import time
import tracemalloc
from types import SimpleNamespace

from remax.core.bank import OnlineCanonicalBank


def bank_measurement(modes):
    gc.collect()
    tracemalloc.start()
    bank = OnlineCanonicalBank(
        entropy_alpha=0,
        retain_exemplars=True,
        replay_capacity=16,
        global_replay_groups_per_step=1,
    )
    start = time.perf_counter()
    for i in range(modes):
        bank.score_and_update(
            prompt_token_ids=[[1] * 64] * 2,
            outcome_keys=[f"mode:{i:08d}"] * 2,
            task_rewards=[1, 1],
            active_mask=[1, 1],
            num_samples=2,
            response_token_ids=[[2] * 32] * 2,
        )
    admission_seconds = time.perf_counter() - start
    gc.collect()
    live, peak = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    start = time.perf_counter()
    payload = pickle.dumps(bank.state_dict(), protocol=5)
    checkpoint_seconds = time.perf_counter() - start
    _, checkpoint_peak = tracemalloc.get_traced_memory()
    size = len(payload)
    del payload

    def repeat_known():
        bank.score_and_update(
            prompt_token_ids=[[1] * 64] * 2,
            outcome_keys=["mode:00000000"] * 2,
            task_rewards=[1, 1],
            active_mask=[1, 1],
            num_samples=2,
            response_token_ids=[[2] * 32] * 2,
        )
        bank.scheduled_global_replay_groups(min_modes=1)

    # Warm scheduling and Python caches before checking fixed-support growth.
    for _ in range(8):
        repeat_known()
    gc.collect()
    before = tracemalloc.get_traced_memory()[0]
    for _ in range(256):
        repeat_known()
    gc.collect()
    after = tracemalloc.get_traced_memory()[0]
    counts = bank.resource_counts()
    assert counts["discovered_modes"] == modes
    assert counts["retained_exemplars"] == min(modes, 16)
    result = {
        **counts,
        "admission_seconds": admission_seconds,
        "new_modes_per_second": modes / admission_seconds,
        "traced_live_bytes": live,
        "traced_admission_peak_bytes": peak,
        "bank_pickle_bytes": size,
        "checkpoint_serialization_seconds": checkpoint_seconds,
        "checkpoint_serialization_peak_bytes": checkpoint_peak,
        "repeat_256_live_growth_bytes": after - before,
    }
    tracemalloc.stop()
    return result


def gpu_measurements(model_path, repeats):
    import torch
    from transformers import AutoModelForCausalLM
    from remax.integrations.oat.replay import OatReplayMixin
    from remax.integrations.oat.scoring import OatScoringMixin
    from remax.integrations.oat.policy import ZeroMathLearnerBaseMixin
    from remax.core.bank_types import VerifiedCanonicalReplayGroup

    class Replay(OatReplayMixin, OatScoringMixin, ZeroMathLearnerBaseMixin):
        pass

    class Strategy:
        grad_acc_step = 4

        def backward(self, loss, model, optimizer):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            a.record()
            (loss / self.grad_acc_step).backward()
            b.record()
            self.events.append((a, b))

    torch.cuda.set_device(0)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).cuda()
    model.config.use_cache = False
    learner = Replay()
    learner.model = model
    learner.optimizer = None
    learner.tokenizer = SimpleNamespace(pad_token_id=0)
    learner.strategy = Strategy()
    learner._invalid_scoring_token_ids_warned_contexts = set()
    learner._invalid_logit_columns_warned_contexts = set()
    score = learner._score_canonical_replay_rows

    def measured_score(*args, **kwargs):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        a.record()
        value = score(*args, **kwargs)
        b.record()
        learner.score_events.append((a, b))
        return value

    learner._score_canonical_replay_rows = measured_score
    results = []
    cases = [
        (1, 1, 64, 16, False),
        (16, 1, 64, 16, False),
        (16, 4, 64, 16, False),
        (16, 4, 256, 64, False),
        (64, 4, 64, 16, False),
        (16, 4, 64, 16, True),
    ]
    for rows, micro, prompt_len, response_len, control in cases:
        groups = [
            VerifiedCanonicalReplayGroup(
                tuple([10] * prompt_len),
                tuple(f"mode:{i}" for i in range(min(16, rows))),
                tuple(tuple([11 + i] * response_len) for i in range(min(16, rows))),
            )
            for _ in range(max(1, rows // 16))
        ]
        learner.args = SimpleNamespace(
            train_batch_size_per_device=micro,
            online_canonical_replay_objective="verified_likelihood_per_rollout",
            online_canonical_replay_alpha=0.1,
            num_samples=16,
            online_canonical_replay_compute_only=control,
            temperature=1.0,
        )
        times = []
        peaks = []
        reserved = []
        live = []
        scores = []
        backward = []
        for iteration in range(repeats + 2):
            model.zero_grad(set_to_none=True)
            learner.strategy.events = []
            learner.score_events = []
            infos = {}
            stats = defaultdict(list)
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            learner._backward_verified_replay(
                args=learner.args,
                canonical_replay_groups=groups,
                fresh_advantage_fingerprint=0,
                infos=infos,
                input_ids=torch.ones((1, 1), dtype=torch.long, device="cuda"),
                policy_vocab_upper_bound=model.config.vocab_size,
                stats=stats,
            )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated() - baseline
            if iteration >= 2:
                times.append(elapsed)
                peaks.append(peak)
                reserved.append(torch.cuda.max_memory_reserved())
                scores.append(sum(a.elapsed_time(b) for a, b in learner.score_events))
                backward.append(
                    sum(a.elapsed_time(b) for a, b in learner.strategy.events)
                )
            if control:
                assert all(
                    p.grad is None or torch.count_nonzero(p.grad) == 0
                    for p in model.parameters()
                )
            del infos, stats
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.synchronize()
            if iteration >= 2:
                live.append(torch.cuda.memory_allocated())
        import statistics

        result = {
            "rows": rows,
            "microbatch": micro,
            "prompt_tokens": prompt_len,
            "response_tokens": response_len,
            "compute_only": control,
            "repeats": repeats,
            "median_seconds": statistics.median(times),
            "replay_updates_per_second": 1 / statistics.median(times),
            "scored_rows_per_second": 2 * rows / statistics.median(times),
            "median_scoring_ms": statistics.median(scores),
            "median_backward_ms": statistics.median(backward),
            "peak_incremental_allocated_bytes": max(peaks),
            "peak_reserved_bytes": max(reserved),
            "post_cleanup_live_bytes": live,
            "post_warmup_live_growth_bytes": max(live) - min(live),
            "score_passes": 2,
            "scoring_chunks_per_update": 2 * ((rows + micro - 1) // micro),
        }
        assert result["post_warmup_live_growth_bytes"] <= 1024 * 1024, result
        print(json.dumps(result), flush=True)
        results.append(result)
    return {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "model": str(model_path),
        "cases": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        help="local model snapshot; enables CUDA scoring measurements",
    )
    parser.add_argument("--repeats", type=int, default=6)
    args = parser.parse_args()
    if args.repeats < 2:
        parser.error("--repeats must be at least two")
    if args.output.exists():
        parser.error("output exists; choose a new report")
    report = {
        "schema": "remax-resource-audit-v1",
        "bank": [bank_measurement(n) for n in (16, 256, 4096)],
        "scope": "Synthetic token shapes, frozen weights, production replay scoring/backward; excludes fresh generation, verification, optimizer state and distributed communication. Bank pickle is not a complete model/optimizer checkpoint.",
    }
    if args.model:
        report["gpu"] = gpu_measurements(args.model, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
