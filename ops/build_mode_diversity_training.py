#!/usr/bin/env python3
"""Recompute pairwise correct-mode diversity for the training comparisons.

Reads the frozen verified-sample archive built for the registered
conditional-concentration analysis, which stores every evaluation response's
canonical key already gated by its verification reward. Nothing is resampled or
re-graded: this re-reads saved keys and reports pairwise correct-mode diversity (PCMD)
for each arm at step 0 and at the terminal step, so the manuscript's training
claims can be stated on a breadth axis that does not move with accuracy.

The archive is the same source the registered ``collision`` statistic uses, and
``PCMD = 1 - collision``, so this adds an orientation and an aggregation rather
than a new measurement.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if __package__:
    from .followup_metrics import atomic_new, file_sha
else:
    from followup_metrics import atomic_new, file_sha
from modebench.metrics import (  # noqa: E402
    DEFAULT_MIN_DEFINED_PROMPTS, effective_modes, mode_diversity,
)

SCHEMA = 'paper-mode-diversity-training-v1'
ARCHIVE = ROOT / 'evidence/verified_samples_completed_cohort.jsonl.gz'
OUT = ROOT / 'outputs/mode_diversity_training.json'
BEFORE, AFTER = '0', '3072'


def _checkpoint_summary(checkpoint: dict, min_defined: int) -> dict | None:
    """PCMD over a checkpoint's prompts, pooling each prompt's four draws."""
    if not checkpoint:
        return None
    diversities, pass8, distinct8 = [], [], []
    for entry in checkpoint['prompts'].values():
        pooled: Counter = Counter()
        for draw in entry['draws']:
            verified = [key for key in draw if key is not None]
            pooled.update(verified)
            pass8.append(1.0 if verified else 0.0)
            distinct8.append(float(len(set(verified))))
        value = mode_diversity(pooled)
        if value is not None:
            diversities.append(value)
    prompts = len(checkpoint['prompts'])
    return {
        'pmd': statistics.fmean(diversities) if diversities else None,
        'defined_prompts': len(diversities),
        'prompts': prompts,
        'support': len(diversities) / prompts if prompts else 0.0,
        'reportable': len(diversities) >= min_defined,
        'pass8': statistics.fmean(pass8) if pass8 else None,
        'distinct8': statistics.fmean(distinct8) if distinct8 else None,
    }


def _terminal_only(issues) -> bool:
    """True when every recorded problem is confined to step 0.

    The frozen snapshot admits each checkpoint separately, and a step-0 refusal
    says the untrained model could not be measured -- nothing about the trained
    arms being compared. Dropping the whole record for it discards a sound
    terminal measurement, which is the comparison the results actually make.

    A step-0 ``raw_sample_integrity_failure`` is the same kind of statement.
    ``load_paper_collision_samples`` checks each step on its own and nulls only
    the step it failed on, so a step-0 conflict leaves the terminal checkpoint
    assembled and verified. What conflicts there is the raw payload of origins
    the snapshot had already found *metric-equivalent* -- a pass-0 requeue
    artifact, in a field PCMD never reads, since PCMD is computed from canonical
    keys. Excluding it cost the Falcon3-1B Python arms every seed they had while
    an independent archive reported the same terminal contrast at five.

    An issue at any other step still removes the record, integrity failures
    included, and so does a withdrawal: those speak to the terminal comparison.
    """
    return bool(issues) and all(
        isinstance(item, dict)
        and item.get('kind') in {'not_admitted_in_frozen_snapshot',
                                 'raw_sample_integrity_failure'}
        and item.get('step') == 0
        for item in issues
    )


def build(archive: Path = ARCHIVE, min_defined: int = DEFAULT_MIN_DEFINED_PROMPTS) -> dict:
    seeds, manifest = [], None
    with gzip.open(archive, 'rt') as handle:
        for line in handle:
            record = json.loads(line)
            if record.get('record_kind') == 'manifest':
                manifest = record
                continue
            terminal_only = _terminal_only(record.get('sample_issues'))
            if record.get('sample_issues') and not terminal_only:
                continue
            if not record.get('before_after_available') and not terminal_only:
                continue
            before = (None if terminal_only else
                      _checkpoint_summary(record['checkpoints'].get(BEFORE), min_defined))
            after = _checkpoint_summary(record['checkpoints'].get(AFTER), min_defined)
            if after is None or (before is None and not terminal_only):
                continue
            seeds.append({
                'level': record['level'], 'scale': record['scale'], 'domain': record['domain'],
                'method': record['method'], 'seed': record['seed'],
                'in_terminal_paired_cohort': record['in_terminal_paired_cohort'],
                'before': before, 'after': after,
                'terminal_only': terminal_only,
                'delta_pmd': (after['pmd'] - before['pmd']
                              if before is not None and before['pmd'] is not None
                              and after['pmd'] is not None else None),
            })

    arms: dict[tuple, list] = defaultdict(list)
    for row in seeds:
        arms[(row['level'], row['scale'], row['domain'], row['method'])].append(row)

    summaries = []
    for (level, scale, domain, method), rows in sorted(arms.items()):
        # A terminal arm-vs-arm comparison needs support at the terminal step
        # only. Requiring it at step 0 as well would discard blocks merely
        # because the frozen model was too weak to measure, which says nothing
        # about the arms being compared. Change claims still need both.
        terminal = [r for r in rows if r['after']['reportable']]
        usable = [r for r in terminal
                  if r['delta_pmd'] is not None and r['before'] is not None
                  and r['before']['reportable']]
        before = [r['before']['pmd'] for r in usable]
        after_paired = [r['after']['pmd'] for r in usable]
        after = [r['after']['pmd'] for r in terminal]
        deltas = [r['delta_pmd'] for r in usable]
        summaries.append({
            'level': level, 'scale': scale, 'domain': domain, 'method': method,
            'seeds': len(rows),
            'terminal_only_seeds': sum(1 for r in rows if r['terminal_only']),
            'terminal_seeds': len(terminal), 'reportable_seeds': len(usable),
            'terminal_reportable': len(terminal) > 0,
            'reportable': len(usable) > 0,
            'pmd_before': statistics.fmean(before) if before else None,
            'pmd_after': statistics.fmean(after) if after else None,
            'pmd_after_paired': statistics.fmean(after_paired) if after_paired else None,
            'pmd_delta': statistics.fmean(deltas) if deltas else None,
            'pmd_delta_seed_range': [min(deltas), max(deltas)] if deltas else None,
            'seeds_improving': sum(1 for d in deltas if d > 0) if deltas else 0,
            'effective_modes_before': effective_modes(statistics.fmean(before)) if before else None,
            'effective_modes_after': effective_modes(statistics.fmean(after)) if after else None,
            'pass8_before': statistics.fmean(r['before']['pass8'] for r in usable) if usable else None,
            'pass8_after': statistics.fmean(r['after']['pass8'] for r in terminal) if terminal else None,
            'distinct8_before': statistics.fmean(r['before']['distinct8'] for r in usable) if usable else None,
            'distinct8_after': statistics.fmean(r['after']['distinct8'] for r in terminal) if terminal else None,
        })

    return {
        'schema': SCHEMA,
        'archive': {'path': str(archive.relative_to(ROOT)), 'sha256': file_sha(archive)},
        'archive_manifest': {k: manifest.get(k) for k in ('schema', 'created_at_utc', 'protocol')} if manifest else None,
        'builder': {'path': 'ops/build_mode_diversity_training.py',
                    'sha256': file_sha(Path(__file__).resolve())},
        'definition': {
            'metric': 'pairwise correct-mode diversity (PCMD)',
            'estimator': 'PCMD = 1 - sum_m n_m (n_m - 1) / (K (K - 1)) over a prompt\'s verified responses',
            'aggregation': 'pooled over a cell\'s four draws, unweighted mean over defined prompts, then over seeds',
            'equals': 'one minus the registered conditional-concentration collision U-statistic',
            'min_defined_prompts': min_defined,
        },
        'arms': summaries,
        'seeds': seeds,
        'coverage': {'cells': len(seeds), 'arms': len(summaries),
                     'reportable_arms': sum(1 for s in summaries if s['reportable']),
                     'terminal_reportable_arms': sum(1 for s in summaries if s['terminal_reportable'])},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ARCHIVE)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--min-defined', type=int, default=DEFAULT_MIN_DEFINED_PROMPTS)
    args = parser.parse_args()
    payload = build(args.archive, args.min_defined)
    atomic_new(args.output, payload)
    print(json.dumps({'event': 'built', 'output': str(args.output), **payload['coverage']}))


if __name__ == '__main__':
    main()
