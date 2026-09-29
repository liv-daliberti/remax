"""GAPO group-level frequency-aware rewards over enumerated valid supports.

This implements the frequency-aware reward of Anschel et al. (EMNLP 2025).
For a rollout group, a verifier-positive row carries

    Rtilde_i = 1 - (f_i - 1/L)

where ``f_i`` is the row's empirical frequency **among the group's valid rows**
and ``L`` is the size of the prompt's valid response set. Verifier-negative
rows carry ``-1``. Rows collapse onto one another through the task grader's
canonical answer key, so two different strings that certify the same mode are
one valid completion rather than two.

Two properties of this port are deliberate and are declared in the
preregistration rather than discovered later.

``L`` comes from ModeBench's enumerated support. GAPO assumes a known valid
set ``V``; ModeBench supplies exactly that, which is why this comparison can be
run faithfully here at all. The support index maps a prompt's frozen reference
string onto its certified mode count, so the learner never re-derives a support
it might get wrong.

The published reward spans ``[-1, 1]`` and rides GRPO, whose advantage divides
by the group standard deviation and is therefore scale-free. This campaign's
backbone is Dr.GRPO, which does not divide by that standard deviation, so the
raw span would change the effective step size relative to the paired control
and confound the objective with the learning rate. ``unit`` scaling applies the
affine map ``(Rtilde + 1) / 2``. Affine maps commute with group centering up to
the shared factor, so GAPO's relative structure is preserved exactly while the
reward span returns to the control's. ``paper`` keeps the published span for
anyone who wants it; it is not what the registered cells run.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

from .outcome_collision import INVALID_OUTCOME_KEY


#: Affine rescaling of the published reward onto the control's unit span.
SCALE_UNIT = "unit"
#: The published ``[-1, 1]`` span, unscaled.
SCALE_PAPER = "paper"
REWARD_SCALES = (SCALE_UNIT, SCALE_PAPER)


@dataclass(frozen=True)
class GAPODiagnostics:
    """Batch diagnostics for the frequency-aware group reward."""

    groups: int
    valid_rows: int
    invalid_rows: int
    distinct_valid_keys_mean: float
    support_size_mean: float
    support_size_min: int
    support_size_max: int
    #: Fraction of groups whose support exceeds the rollout group size. There
    #: the uniform target is unreachable by construction and the 1/L term is
    #: numerically inert, leaving a pure within-group duplicate penalty. This
    #: is a property of the domain, not a failure, and it is reported so the
    #: reading of a domain's effect can account for it.
    unreachable_support_group_fraction: float
    frequency_mean: float
    frequency_max: float
    reward_mean: float
    reward_min: float
    reward_max: float


def gapo_group_rewards(
    answer_keys: Sequence[str | None],
    task_rewards: Sequence[float],
    support_sizes: Sequence[int],
    *,
    num_samples: int,
    reward_scale: str = SCALE_UNIT,
) -> tuple[list[float], GAPODiagnostics]:
    """Return one GAPO reward per row, replacing the binary task reward.

    ``answer_keys`` are the grader's canonical keys, ``task_rewards`` the
    binary verifier outcome, and ``support_sizes`` the enumerated support ``L``
    repeated for every row of a prompt group. A row is valid when the verifier
    accepted it; an accepted row whose key is missing is refused rather than
    folded into the shared invalid key, because a certified-correct row with no
    identity would silently collide with other such rows and understate
    breadth.
    """

    if num_samples <= 1:
        raise ValueError("num_samples must be greater than one")
    if reward_scale not in REWARD_SCALES:
        raise ValueError(f"reward_scale must be one of {REWARD_SCALES}")
    rows = len(answer_keys)
    if rows == 0 or rows % num_samples:
        raise ValueError("answer_keys must contain complete candidate groups")
    if len(task_rewards) != rows or len(support_sizes) != rows:
        raise ValueError("GAPO inputs must contain the same number of rows")

    rewards: list[float] = []
    distinct_valid_counts: list[int] = []
    support_observations: list[int] = []
    unreachable_groups = 0
    frequencies: list[float] = []
    valid_rows = 0
    invalid_rows = 0

    for start in range(0, rows, num_samples):
        stop = start + num_samples
        group_keys = list(answer_keys[start:stop])
        group_valid = [float(value) > 0.0 for value in task_rewards[start:stop]]
        group_support = list(support_sizes[start:stop])

        support = int(group_support[0])
        if any(int(value) != support for value in group_support):
            raise ValueError("one prompt group must carry one support size")
        if support <= 0:
            raise ValueError("GAPO requires a positive enumerated support size")
        support_observations.append(support)
        if support > num_samples:
            unreachable_groups += 1

        valid_keys: list[str] = []
        for key, is_valid in zip(group_keys, group_valid):
            if not is_valid:
                continue
            if key is None or key == INVALID_OUTCOME_KEY:
                raise ValueError(
                    "GAPO refuses a verifier-positive row with no answer key"
                )
            valid_keys.append(str(key))
        counts = Counter(valid_keys)
        n_valid = len(valid_keys)
        distinct_valid_counts.append(len(counts))
        valid_rows += n_valid
        invalid_rows += num_samples - n_valid

        for key, is_valid in zip(group_keys, group_valid):
            if not is_valid:
                rewards.append(_scale(-1.0, reward_scale))
                continue
            frequency = counts[str(key)] / n_valid
            frequencies.append(frequency)
            rewards.append(
                _scale(1.0 - (frequency - 1.0 / support), reward_scale)
            )

    if not all(math.isfinite(value) for value in rewards):
        raise ValueError("GAPO produced a non-finite reward")

    groups = rows // num_samples
    diagnostics = GAPODiagnostics(
        groups=groups,
        valid_rows=valid_rows,
        invalid_rows=invalid_rows,
        distinct_valid_keys_mean=_mean(distinct_valid_counts),
        support_size_mean=_mean(support_observations),
        support_size_min=min(support_observations),
        support_size_max=max(support_observations),
        unreachable_support_group_fraction=unreachable_groups / groups,
        frequency_mean=_mean(frequencies),
        frequency_max=max(frequencies) if frequencies else 0.0,
        reward_mean=_mean(rewards),
        reward_min=min(rewards),
        reward_max=max(rewards),
    )
    return rewards, diagnostics


def _scale(reward: float, reward_scale: str) -> float:
    if reward_scale == SCALE_PAPER:
        return float(reward)
    return float((reward + 1.0) / 2.0)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


@dataclass(frozen=True)
class GAPOSupportIndex:
    """Frozen map from a prompt's reference string onto its support size.

    The index is built once, before submission, by
    ``ops/build_gapo_support_index.py`` and pinned by digest in the cohort
    ledger. The learner only ever reads it.

    The reference string is the join key rather than an instance id because it
    is what the learner already holds for every row, and because the support
    column's counterpart inside the reference payload is not uniform across
    domains --- PythonFactors carries a ``num_modes`` that is not the certified
    support. Joining on the reference avoids re-deriving a number the dataset
    already certifies.
    """

    digest: str
    sizes: dict[str, int]

    @classmethod
    def load(cls, path: str | Path) -> "GAPOSupportIndex":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("schema") != SUPPORT_INDEX_SCHEMA:
            raise ValueError(
                f"GAPO support index must carry schema {SUPPORT_INDEX_SCHEMA!r}"
            )
        sizes = {
            str(key): int(value)
            for key, value in dict(payload["support_sizes"]).items()
        }
        if not sizes:
            raise ValueError("GAPO support index is empty")
        if any(value <= 0 for value in sizes.values()):
            raise ValueError("GAPO support index carries a non-positive size")
        return cls(digest=file_digest(source), sizes=sizes)

    def lookup(self, references: Sequence[object]) -> list[int]:
        """Return one support size per row, refusing an unknown reference.

        A missing prompt is a mechanism failure, not a row to skip: silently
        defaulting a support size would change the objective on exactly the
        prompts whose provenance is least certain.
        """

        sizes: list[int] = []
        missing = 0
        for reference in references:
            if reference is None:
                missing += 1
                continue
            key = reference_key(reference)
            size = self.sizes.get(key)
            if size is None:
                missing += 1
                continue
            sizes.append(int(size))
        if missing:
            raise ValueError(
                f"GAPO support index does not cover {missing} of "
                f"{len(references)} prompt rows"
            )
        return sizes


SUPPORT_INDEX_SCHEMA = "gapo_support_index_v1"


def reference_key(reference: object) -> str:
    """Hash one reference payload onto its index key."""

    return hashlib.sha256(str(reference).encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
