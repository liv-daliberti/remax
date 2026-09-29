"""DIAYN-style answer-option mutual information rewards."""

from __future__ import annotations

import math
import hashlib
import json
from dataclasses import dataclass
from typing import Any


from modebench.templates import CHAT_SURFACES

ASSISTANT_MARKERS = tuple(
    assistant_marker for _, _, assistant_marker in CHAT_SURFACES.values()
) + ("\nAssistant: <think>",)


def format_answer_option_prompt(prompt: str, z: int, num_options: int) -> str:
    """Condition a formatted prompt on a latent answer option id."""

    option_text = (
        f"Answer-option latent: z={int(z)} of {int(num_options)}. "
        "Use this latent to choose which valid final answer mode to pursue. "
        "Do not mention the latent."
    )
    # Put the intervention inside the system message when using a chat
    # template.  Text between the system and user role markers is outside
    # either chat role and is not a valid place to carry the latent condition.
    # Both surfaces are handled so a Falcon-surface run receives the identical
    # intervention a Qwen-surface run does.
    for system_marker, user_marker, _ in CHAT_SURFACES.values():
        if not prompt.startswith(system_marker):
            continue
        system_end = prompt.find(user_marker, len(system_marker))
        if system_end >= 0:
            return (
                prompt[:system_end].rstrip() + "\n" + option_text + prompt[system_end:]
            )
    for marker in ASSISTANT_MARKERS:
        if prompt.endswith(marker):
            return prompt[: -len(marker)] + option_text + "\n" + marker
    return prompt + "\n" + option_text + "\n"


def option_id_for_sample(
    sample_index: int, num_options: int, samples_per_option: int
) -> int:
    """Map a contiguous candidate index to its option id."""

    if num_options <= 0 or samples_per_option <= 0:
        raise ValueError("num_options and samples_per_option must be positive")
    z = int(sample_index) // int(samples_per_option)
    if z < 0 or z >= int(num_options):
        raise ValueError("sample_index is outside the option-expanded group")
    return z


@dataclass
class AnswerOptionMIDiagnostics:
    bonus_mean: float = 0.0
    bonus_min: float = 0.0
    bonus_max: float = 0.0
    eligible_fraction: float = 0.0
    classifier_accuracy: float = 0.0
    distinct_answer_reprs: float = 0.0
    lower_bound_nats: float = 0.0
    leave_one_out_supported_fraction: float = 0.0


def _stable_context_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):
        return repr(value)


def conditional_answer_repr(context: Any, answer_repr: str | None) -> str | None:
    """Namespace a semantic answer key by its prompt/reference identity.

    The discriminator estimates conditional MI, ``I(Z; A | X)``.  Hashing X
    keeps the table compact and prevents identical-looking answers from two
    different problems from sharing discriminator counts.
    """

    if answer_repr is None:
        return None
    context_hash = hashlib.sha256(
        _stable_context_text(context).encode("utf-8")
    ).hexdigest()[:24]
    return f"{context_hash}|{answer_repr}"


class AnswerOptionMITracker:
    r"""EMA table for the variational DIAYN reward.

    The lower bound is

    ``I(Z; A) >= E[log q(z | answer_repr)] + H(Z)``.

    This tracker uses an EMA count table as the discriminator ``q``. It is
    intentionally nonparametric so the first version of the objective depends
    only on the answer representation and not on response style.
    """

    def __init__(
        self,
        *,
        num_options: int,
        ema_decay: float = 0.9,
        smoothing: float = 1.0,
        bonus_clip: float = 5.0,
        leave_one_out: bool = False,
    ) -> None:
        if int(num_options) <= 1:
            raise ValueError("num_options must be greater than one")
        if not math.isfinite(float(ema_decay)) or not 0.0 <= float(ema_decay) < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if not math.isfinite(float(smoothing)) or float(smoothing) <= 0.0:
            raise ValueError("smoothing must be finite and positive")
        if not math.isfinite(float(bonus_clip)) or float(bonus_clip) <= 0.0:
            raise ValueError("bonus_clip must be finite and positive")
        self.num_options = int(num_options)
        self.ema_decay = float(ema_decay)
        self.smoothing = float(smoothing)
        self.bonus_clip = float(bonus_clip)
        self.leave_one_out = bool(leave_one_out)
        self._counts: dict[str, list[float]] = {}

    def _row_counts(self, answer_repr: str) -> list[float]:
        counts = self._counts.get(answer_repr)
        if counts is None:
            counts = [self.smoothing] * self.num_options
            self._counts[answer_repr] = counts
        return counts

    def state_dict(self) -> dict[str, Any]:
        """Return the discriminator state required for exact training resume."""

        return {
            "schema": "answer_option_mi_tracker_v1",
            "num_options": self.num_options,
            "ema_decay": self.ema_decay,
            "smoothing": self.smoothing,
            "bonus_clip": self.bonus_clip,
            "leave_one_out": self.leave_one_out,
            "counts": {key: list(values) for key, values in self._counts.items()},
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a discriminator without silently changing its contract."""

        if not isinstance(state, dict) or state.get("schema") != (
            "answer_option_mi_tracker_v1"
        ):
            raise ValueError("invalid answer-option MI tracker state")
        expected = {
            "num_options": self.num_options,
            "ema_decay": self.ema_decay,
            "smoothing": self.smoothing,
            "bonus_clip": self.bonus_clip,
            "leave_one_out": self.leave_one_out,
        }
        for name, value in expected.items():
            saved = state.get(name)
            if name == "num_options":
                matches = int(saved) == int(value)
            elif name == "leave_one_out":
                # Older history-only checkpoints predate the explicit field.
                matches = bool(saved) == bool(value)
            else:
                matches = math.isclose(
                    float(saved), float(value), rel_tol=0.0, abs_tol=1e-12
                )
            if not matches:
                raise ValueError(
                    f"answer-option MI resume mismatch for {name}: "
                    f"saved={saved!r} configured={value!r}"
                )
        raw_counts = state.get("counts")
        if not isinstance(raw_counts, dict):
            raise ValueError("answer-option MI tracker counts are missing")
        restored: dict[str, list[float]] = {}
        for key, values in raw_counts.items():
            if not isinstance(key, str) or not isinstance(values, (list, tuple)):
                raise ValueError("answer-option MI tracker contains malformed counts")
            counts = [float(value) for value in values]
            if len(counts) != self.num_options or any(
                not math.isfinite(value) or value <= 0.0 for value in counts
            ):
                raise ValueError("answer-option MI tracker counts are invalid")
            restored[key] = counts
        self._counts = restored

    def update_and_score(
        self,
        *,
        answer_reprs: list[str | None],
        option_ids: list[int | None],
        correct: list[bool],
        loss_masks: list[float],
        beta: float,
        correct_only: bool,
    ) -> tuple[list[float], AnswerOptionMIDiagnostics]:
        if not (
            len(answer_reprs) == len(option_ids) == len(correct) == len(loss_masks)
        ):
            raise ValueError("MI rows must have matching lengths")
        coefficient = float(beta)
        if not math.isfinite(coefficient) or coefficient < 0.0:
            raise ValueError("beta must be finite and non-negative")

        eligible = []
        batch_counts: dict[str, list[float]] = {}
        for answer_repr, option_id, is_correct, loss_mask in zip(
            answer_reprs, option_ids, correct, loss_masks
        ):
            ok = (
                answer_repr is not None
                and option_id is not None
                and 0 <= int(option_id) < self.num_options
                and float(loss_mask) > 0.0
                and (bool(is_correct) or not bool(correct_only))
            )
            eligible.append(ok)
            if ok:
                row = batch_counts.setdefault(
                    str(answer_repr), [0.0] * self.num_options
                )
                row[int(option_id)] += 1.0

        log_prior = -math.log(float(self.num_options))
        bonuses: list[float] = []
        correct_predictions = 0
        predicted_rows = 0
        lower_bound_terms: list[float] = []
        leave_one_out_supported = 0
        for answer_repr, option_id, ok in zip(answer_reprs, option_ids, eligible):
            if not ok:
                bonuses.append(0.0)
                continue
            history_counts = self._row_counts(str(answer_repr))
            scoring_counts = list(history_counts)
            if self.leave_one_out:
                observed = batch_counts[str(answer_repr)]
                other_count = sum(observed) - 1.0
                leave_one_out_supported += int(other_count > 0.0)
                for index, value in enumerate(observed):
                    scoring_counts[index] += value - int(index == int(option_id))
            total = sum(scoring_counts)
            q = max(scoring_counts[int(option_id)] / total, 1e-12)
            lower_bound_term = math.log(q) - log_prior
            lower_bound_terms.append(lower_bound_term)
            clipped_term = max(-self.bonus_clip, min(self.bonus_clip, lower_bound_term))
            bonuses.append(coefficient * clipped_term)
            predicted_rows += 1
            if max(
                range(self.num_options),
                key=lambda index: scoring_counts[index],
            ) == int(option_id):
                correct_predictions += 1

        # Score against the discriminator state available before this batch,
        # then update it.  This one-step delay prevents a sample from boosting
        # its own q(z|x,a) and manufacturing a positive MI reward.
        for answer_repr, observed in batch_counts.items():
            counts = self._row_counts(answer_repr)
            for index, value in enumerate(observed):
                counts[index] = self.ema_decay * counts[index] + (
                    1.0 - self.ema_decay
                ) * (self.smoothing + value)

        if bonuses:
            bonus_mean = sum(bonuses) / len(bonuses)
            bonus_min = min(bonuses)
            bonus_max = max(bonuses)
        else:
            bonus_mean = bonus_min = bonus_max = 0.0
        diagnostics = AnswerOptionMIDiagnostics(
            bonus_mean=bonus_mean,
            bonus_min=bonus_min,
            bonus_max=bonus_max,
            eligible_fraction=(
                sum(1 for value in eligible if value) / len(eligible)
                if eligible
                else 0.0
            ),
            classifier_accuracy=(
                correct_predictions / predicted_rows if predicted_rows else 0.0
            ),
            distinct_answer_reprs=float(
                len({value for value in answer_reprs if value})
            ),
            lower_bound_nats=(
                sum(lower_bound_terms) / len(lower_bound_terms)
                if lower_bound_terms
                else 0.0
            ),
            leave_one_out_supported_fraction=(
                leave_one_out_supported / sum(eligible)
                if self.leave_one_out and any(eligible)
                else 0.0
            ),
        )
        return bonuses, diagnostics


def empirical_option_answer_mi(
    *,
    answer_reprs: list[str | None],
    option_ids: list[int | None],
    correct: list[bool],
    num_options: int,
) -> dict[str, float]:
    """Plug-in held-out diagnostics for semantic latent/answer binding.

    MI is computed among correct, parseable answers.  Accuracy is the Bayes
    classification accuracy of predicting z from the answer representation.
    Correct-rate range is reported separately so binding cannot hide a latent
    that simply makes some options fail more often.
    """

    if not (len(answer_reprs) == len(option_ids) == len(correct)):
        raise ValueError("option evaluation rows must have matching lengths")
    if int(num_options) <= 1:
        return {
            "option_answer_mi_nats": 0.0,
            "option_classifier_accuracy": 0.0,
            "option_eligible_fraction": 0.0,
            "option_correct_rate_range": 0.0,
        }

    joint: dict[tuple[int, str], int] = {}
    z_counts = [0] * int(num_options)
    answer_counts: dict[str, int] = {}
    option_total = [0] * int(num_options)
    option_correct = [0] * int(num_options)
    eligible = 0
    for answer_repr, option_id, is_correct in zip(answer_reprs, option_ids, correct):
        if option_id is None or not 0 <= int(option_id) < int(num_options):
            continue
        z = int(option_id)
        option_total[z] += 1
        option_correct[z] += int(bool(is_correct))
        if not bool(is_correct) or answer_repr is None:
            continue
        key = str(answer_repr)
        eligible += 1
        joint[(z, key)] = joint.get((z, key), 0) + 1
        z_counts[z] += 1
        answer_counts[key] = answer_counts.get(key, 0) + 1

    mi = 0.0
    classifier_hits = 0
    if eligible:
        for (z, key), count in joint.items():
            p_joint = count / eligible
            p_z = z_counts[z] / eligible
            p_answer = answer_counts[key] / eligible
            mi += p_joint * math.log(p_joint / (p_z * p_answer))
        for key in answer_counts:
            classifier_hits += max(
                joint.get((z, key), 0) for z in range(int(num_options))
            )
    correct_rates = [
        option_correct[z] / option_total[z]
        for z in range(int(num_options))
        if option_total[z]
    ]
    return {
        "option_answer_mi_nats": float(mi),
        "option_classifier_accuracy": (classifier_hits / eligible if eligible else 0.0),
        "option_eligible_fraction": eligible / len(answer_reprs)
        if answer_reprs
        else 0.0,
        "option_correct_rate_range": (
            max(correct_rates) - min(correct_rates) if correct_rates else 0.0
        ),
    }


def crossfit_option_answer_mi(
    *,
    answer_reprs_by_draw: list[list[str | None]],
    option_ids_by_draw: list[list[int | None]],
    correct_by_draw: list[list[bool]],
    num_options: int,
    smoothing: float = 1.0,
) -> list[dict[str, float]]:
    """Score a semantic MI lower bound without evaluating on fitted rows.

    Each fixed evaluation draw is held out in turn.  The categorical
    discriminator is fit on every other draw and evaluated on the held-out
    one.  An answer representation absent from the training folds therefore
    receives the uniform-prior score of zero rather than a memorization bonus.
    """

    draw_count = len(answer_reprs_by_draw)
    if not (draw_count == len(option_ids_by_draw) == len(correct_by_draw)):
        raise ValueError("cross-fit option draws must have matching lengths")
    if draw_count < 2:
        raise ValueError("cross-fit option MI requires at least two draws")
    if int(num_options) <= 1 or not math.isfinite(float(smoothing)) or smoothing <= 0:
        raise ValueError("cross-fit option MI requires options > 1 and smoothing > 0")
    row_count = len(answer_reprs_by_draw[0])
    for answers, options, correct in zip(
        answer_reprs_by_draw, option_ids_by_draw, correct_by_draw
    ):
        if not (len(answers) == len(options) == len(correct) == row_count):
            raise ValueError("cross-fit option draws must share one row grid")

    results: list[dict[str, float]] = []
    log_prior = -math.log(float(num_options))
    for held_out in range(draw_count):
        counts_by_answer: dict[str, list[float]] = {}
        for draw_index in range(draw_count):
            if draw_index == held_out:
                continue
            for answer_repr, option_id, is_correct in zip(
                answer_reprs_by_draw[draw_index],
                option_ids_by_draw[draw_index],
                correct_by_draw[draw_index],
            ):
                if (
                    not is_correct
                    or answer_repr is None
                    or option_id is None
                    or not 0 <= int(option_id) < int(num_options)
                ):
                    continue
                counts = counts_by_answer.setdefault(
                    str(answer_repr), [float(smoothing)] * int(num_options)
                )
                counts[int(option_id)] += 1.0

        terms: list[float] = []
        classifier_hits = 0
        option_total = [0] * int(num_options)
        option_correct = [0] * int(num_options)
        for answer_repr, option_id, is_correct in zip(
            answer_reprs_by_draw[held_out],
            option_ids_by_draw[held_out],
            correct_by_draw[held_out],
        ):
            if option_id is None or not 0 <= int(option_id) < int(num_options):
                continue
            z = int(option_id)
            option_total[z] += 1
            option_correct[z] += int(bool(is_correct))
            if not is_correct or answer_repr is None:
                continue
            counts = counts_by_answer.get(
                str(answer_repr), [float(smoothing)] * int(num_options)
            )
            q = counts[z] / sum(counts)
            terms.append(math.log(max(q, 1e-12)) - log_prior)
            classifier_hits += int(
                max(range(int(num_options)), key=lambda index: counts[index]) == z
            )

        correct_rates = [
            option_correct[z] / option_total[z]
            for z in range(int(num_options))
            if option_total[z]
        ]
        results.append(
            {
                "option_answer_mi_lower_bound_nats": (
                    sum(terms) / len(terms) if terms else 0.0
                ),
                "option_classifier_accuracy": (
                    classifier_hits / len(terms) if terms else 0.0
                ),
                "option_eligible_fraction": (
                    len(terms) / row_count if row_count else 0.0
                ),
                "option_correct_rate_range": (
                    max(correct_rates) - min(correct_rates) if correct_rates else 0.0
                ),
            }
        )
    return results


def coerce_option_id(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
