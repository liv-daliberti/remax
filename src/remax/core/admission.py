"""Admission for verified replay; retained state semantics are unchanged."""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from .bank_types import (
    OnlineCanonicalBankDiagnostics,
    VerifiedProposalAdmissionDiagnostics,
    _normalized_token_tuple,
    _prompt_key,
)


class BankAdmissionMixin:
    def admit_verified_proposals(
        self,
        *,
        prompt_token_ids: Sequence[Sequence[int]],
        outcome_keys: Sequence[str],
        response_token_ids: Sequence[Sequence[int]],
    ) -> VerifiedProposalAdmissionDiagnostics:
        """Atomically add novel validator-positive proposal outcomes.

        Proposal rows were sampled under a conditioned prompt, so they are not
        on-policy rows for the neutral prompt. This is a support-only update:
        every supplied distinct outcome enters with count one, irrespective of
        how often it appeared in the conditioned candidate group. The caller
        must supply only outcomes absent from the pre-proposal replay group.
        """

        if not self.retain_exemplars:
            raise RuntimeError("proposal admission requires replay exemplars")
        row_count = len(outcome_keys)
        if not (len(prompt_token_ids) == len(response_token_ids) == row_count):
            raise ValueError(
                "verified proposal admission inputs must have equal lengths"
            )
        if row_count == 0:
            return VerifiedProposalAdmissionDiagnostics(
                proposal_groups=0,
                proposal_rows=0,
                new_outcomes=0,
                stored_exemplars=0,
                tracked_prompts=sum(
                    bool(exemplars) for exemplars in self._exemplars.values()
                ),
                tracked_outcomes=sum(
                    len(exemplars) for exemplars in self._exemplars.values()
                ),
            )

        normalized_rows: list[tuple[str, tuple[int, ...], str, tuple[int, ...]]] = []
        seen_pairs: set[tuple[str, str]] = set()
        for raw_prompt, raw_key, raw_response in zip(
            prompt_token_ids,
            outcome_keys,
            response_token_ids,
        ):
            normalized_prompt = _normalized_token_tuple(
                raw_prompt,
                label="verified proposal prompt token ids",
            )
            if not isinstance(raw_key, str) or not raw_key:
                raise ValueError(
                    "verified proposal outcome keys must be non-empty strings"
                )
            normalized_response = _normalized_token_tuple(
                raw_response,
                label="verified proposal response token ids",
            )
            prompt_key = _prompt_key(normalized_prompt)
            pair = (prompt_key, raw_key)
            if pair in seen_pairs:
                raise ValueError(
                    "verified proposal admission requires one row per "
                    "prompt/outcome pair"
                )
            seen_pairs.add(pair)
            normalized_rows.append(
                (
                    prompt_key,
                    normalized_prompt,
                    raw_key,
                    normalized_response,
                )
            )

        # Stage all maps before committing so a hash collision, duplicate, or
        # malformed later row cannot partially mutate the live replay bank.
        staged_counts = {
            prompt_key: dict(counts) for prompt_key, counts in self._counts.items()
        }
        staged_prompts = dict(self._prompt_token_ids)
        staged_exemplars = {
            prompt_key: dict(exemplars)
            for prompt_key, exemplars in self._exemplars.items()
        }
        staged_proposal_only = {
            prompt_key: set(outcomes)
            for prompt_key, outcomes in self._proposal_only_outcomes.items()
        }
        staged_priority = {
            prompt_key: dict(outcomes)
            for prompt_key, outcomes in self._proposal_priority_remaining.items()
        }
        staged_priority_queue = list(self._proposal_priority_queue)
        staged_retention = (
            self._proposal_retention_tracker.clone()
            if self._proposal_retention_tracker is not None
            else None
        )
        new_outcomes = 0
        stored_exemplars = 0
        proposal_prompt_keys: set[str] = set()
        for prompt_key, prompt_tokens, outcome_key, response_tokens in normalized_rows:
            proposal_prompt_keys.add(prompt_key)
            stored_prompt = staged_prompts.get(prompt_key)
            if stored_prompt is not None and stored_prompt != prompt_tokens:
                raise ValueError("canonical replay prompt hash collision or mutation")
            counts = staged_counts.get(prompt_key, {})
            if not self.separate_proposal_objective_support:
                counts = staged_counts.setdefault(prompt_key, {})
            exemplars = staged_exemplars.setdefault(prompt_key, {})
            known_outcomes = (
                set(exemplars)
                if self.separate_proposal_objective_support
                else set(counts)
            )
            if outcome_key in known_outcomes:
                raise ValueError(
                    "verified proposal admission accepts only outcomes absent "
                    "from the current bank"
                )
            staged_prompts[prompt_key] = prompt_tokens
            if (
                self.separate_proposal_objective_support
                and len(exemplars) >= self.replay_capacity
            ):
                raise ValueError("verified proposal admission exceeds replay capacity")
            if not self.separate_proposal_objective_support:
                counts[outcome_key] = 1
            if len(exemplars) < self.replay_capacity:
                exemplars[outcome_key] = response_tokens
                if self.separate_proposal_objective_support:
                    staged_proposal_only.setdefault(
                        prompt_key,
                        set(),
                    ).add(outcome_key)
                if self.proposal_replay_priority_visits > 0:
                    staged_priority.setdefault(prompt_key, {})[outcome_key] = (
                        self.proposal_replay_priority_visits
                    )
                    if prompt_key not in staged_priority_queue:
                        staged_priority_queue.append(prompt_key)
                if staged_retention is not None:
                    staged_retention.admit(prompt_key, outcome_key)
                stored_exemplars += 1
            new_outcomes += 1

        self._counts = staged_counts
        self._prompt_token_ids = staged_prompts
        self._exemplars = staged_exemplars
        self._proposal_only_outcomes = staged_proposal_only
        self._proposal_priority_remaining = staged_priority
        self._proposal_priority_queue = staged_priority_queue
        self._proposal_retention_tracker = staged_retention
        proposal_groups = len(proposal_prompt_keys)
        self._proposal_groups += proposal_groups
        self._proposal_rows += row_count
        self._proposal_new_outcomes += new_outcomes
        return VerifiedProposalAdmissionDiagnostics(
            proposal_groups=proposal_groups,
            proposal_rows=row_count,
            new_outcomes=new_outcomes,
            stored_exemplars=stored_exemplars,
            tracked_prompts=sum(
                bool(exemplars) for exemplars in self._exemplars.values()
            ),
            tracked_outcomes=sum(
                len(exemplars) for exemplars in self._exemplars.values()
            ),
        )

    def score_and_update(
        self,
        *,
        prompt_token_ids: Sequence[Sequence[int]],
        outcome_keys: Sequence[str | None],
        task_rewards: Sequence[float],
        active_mask: Sequence[float | bool],
        num_samples: int,
        entropy_alpha_override: float | None = None,
        response_token_ids: Sequence[Sequence[int]] | None = None,
        update_bank: bool = True,
    ) -> tuple[list[float], OnlineCanonicalBankDiagnostics]:
        """Score complete groups against snapshots, then optionally commit keys.

        ``update_bank=False`` freezes both membership and fresh-observation
        counts. Replay may continue to read and score the immutable exemplar
        bank, which is the longitudinal-survival telemetry contract.
        """

        if not isinstance(update_bank, bool):
            raise ValueError("update_bank must be a boolean")

        entropy_alpha = (
            self.entropy_alpha
            if entropy_alpha_override is None
            else float(entropy_alpha_override)
        )
        if not math.isfinite(entropy_alpha) or entropy_alpha < 0:
            raise ValueError("entropy_alpha_override must be finite and non-negative")
        row_count = len(outcome_keys)
        if num_samples <= 1:
            raise ValueError("num_samples must be greater than one")
        if row_count == 0 or row_count % num_samples != 0:
            raise ValueError("outcome_keys must contain complete groups")
        if not (
            len(prompt_token_ids) == len(task_rewards) == len(active_mask) == row_count
        ):
            raise ValueError("online canonical bank inputs must have equal lengths")
        if self.retain_exemplars:
            if response_token_ids is None or len(response_token_ids) != row_count:
                raise ValueError(
                    "canonical replay requires one response token sequence per row"
                )

        prompt_keys = [_prompt_key(tokens) for tokens in prompt_token_ids]
        combined = [0.0] * row_count
        entropy_values: list[float] = []
        entropy_advantages = [0.0] * row_count
        eligible_rows = 0
        positive_rows = 0
        canonicalizable_positive_rows = 0
        new_outcome_count = 0
        new_outcome_rows = 0
        bank_sizes_before: list[int] = []
        bank_sizes_after: list[int] = []
        normalized_entropies: list[float] = []
        normalized_entropy_ratios: list[float] = []
        log_supports: list[float] = []
        group_count = row_count // num_samples

        for start in range(0, row_count, num_samples):
            stop = start + num_samples
            group_prompt_keys = prompt_keys[start:stop]
            if len(set(group_prompt_keys)) != 1:
                raise ValueError(
                    "each online canonical candidate group must share one prompt"
                )
            prompt_key = group_prompt_keys[0]
            historical = Counter(self._counts.get(prompt_key, {}))
            bank_sizes_before.append(len(historical))
            group_keys = list(outcome_keys[start:stop])
            group_rewards = [float(value) for value in task_rewards[start:stop]]
            group_active = [bool(value) for value in active_mask[start:stop]]
            for reward, active in zip(group_rewards, group_active):
                if active and reward > 0:
                    positive_rows += 1

            eligible = [
                active and reward > 0 and isinstance(key, str) and bool(key)
                for key, reward, active in zip(group_keys, group_rewards, group_active)
            ]
            canonicalizable_positive_rows += sum(eligible)
            eligible_rows += sum(eligible)
            current = Counter(
                str(key) for key, keep in zip(group_keys, eligible) if keep
            )
            if update_bank and self._proposal_retention_tracker is not None:
                retention_requests = self._proposal_retention_tracker.observe_rollout(
                    prompt_key=prompt_key,
                    verified_outcome_counts=current,
                    row_count=num_samples,
                )
                self._apply_retention_priority_requests(retention_requests)
            support = set(historical) | set(current)
            new_keys = set(current) - set(historical)
            new_outcome_count += len(new_keys)
            new_outcome_rows += sum(current[key] for key in new_keys)

            for offset, (key, keep) in enumerate(zip(group_keys, eligible)):
                if not keep:
                    continue
                key = str(key)
                loo = historical.copy()
                loo.update(current)
                loo[key] -= 1
                if loo[key] <= 0:
                    del loo[key]
                support_size = max(len(support), 1)
                denominator = sum(loo.values()) + self.pseudocount * support_size
                probabilities = {
                    candidate: (loo.get(candidate, 0) + self.pseudocount) / denominator
                    for candidate in support
                }
                probability = probabilities[key]
                clipped_surprisal = min(-math.log(probability), self.surprisal_clip)
                clipped_entropy = sum(
                    value * min(-math.log(value), self.surprisal_clip)
                    for value in probabilities.values()
                )
                entropy_values.append(clipped_entropy)
                entropy_advantage = entropy_alpha * (
                    clipped_surprisal - clipped_entropy
                )
                row_index = start + offset
                entropy_advantages[row_index] = entropy_advantage
                combined[row_index] = entropy_advantage

            updated = historical.copy()
            if update_bank:
                updated.update(current)
            if updated:
                self._counts[prompt_key] = dict(updated)
            if update_bank and self.retain_exemplars and new_keys:
                normalized_prompt = _normalized_token_tuple(
                    prompt_token_ids[start],
                    label="canonical replay prompt token ids",
                )
                stored_prompt = self._prompt_token_ids.get(prompt_key)
                if stored_prompt is not None and stored_prompt != normalized_prompt:
                    raise ValueError(
                        "canonical replay prompt hash collision or mutation"
                    )
                self._prompt_token_ids[prompt_key] = normalized_prompt
                prompt_exemplars = self._exemplars.setdefault(
                    prompt_key,
                    {},
                )
                proposal_only = self._proposal_only_outcomes.setdefault(
                    prompt_key,
                    set(),
                )
                assert response_token_ids is not None
                for new_key in sorted(new_keys):
                    if (
                        len(prompt_exemplars) >= self.replay_capacity
                        and new_key not in proposal_only
                    ):
                        break
                    candidates = [
                        _normalized_token_tuple(
                            response_token_ids[start + offset],
                            label="canonical replay response token ids",
                        )
                        for offset, (key, keep) in enumerate(zip(group_keys, eligible))
                        if keep and str(key) == new_key
                    ]
                    if not candidates:
                        raise RuntimeError(
                            "new canonical replay mode lacks an eligible exemplar"
                        )
                    prompt_exemplars[new_key] = min(candidates)
                    proposal_only.discard(new_key)
                if not proposal_only:
                    self._proposal_only_outcomes.pop(prompt_key, None)
            bank_sizes_after.append(len(updated))
            if len(updated) >= 2:
                support_size = len(updated)
                denominator = sum(updated.values()) + self.pseudocount * support_size
                probabilities = [
                    (count + self.pseudocount) / denominator
                    for count in updated.values()
                ]
                exact_entropy = -sum(
                    probability * math.log(probability) for probability in probabilities
                )
                log_support = math.log(support_size)
                normalized_entropies.append(exact_entropy)
                log_supports.append(log_support)
                normalized_entropy_ratios.append(
                    min(max(exact_entropy / log_support, 0.0), 1.0)
                )
            self._groups_scored += 1

        self._rows_scored += row_count

        def mean(values: Sequence[float]) -> float:
            return float(sum(values) / len(values)) if values else 0.0

        def rms(values: Sequence[float]) -> float:
            return (
                math.sqrt(sum(value * value for value in values) / len(values))
                if values
                else 0.0
            )

        diagnostics = OnlineCanonicalBankDiagnostics(
            entropy_estimate_mean=mean(entropy_values),
            normalized_entropy_mean=mean(normalized_entropies),
            normalized_entropy_ratio_mean=mean(normalized_entropy_ratios),
            normalized_entropy_ratio_eligible_fraction=(
                len(normalized_entropy_ratios) / group_count
            ),
            log_support_mean=mean(log_supports),
            entropy_alpha_used=entropy_alpha,
            entropy_advantage_mean=mean(entropy_advantages),
            entropy_advantage_rms=rms(entropy_advantages),
            combined_advantage_mean=mean(combined),
            combined_advantage_rms=rms(combined),
            eligible_fraction=eligible_rows / row_count,
            reward_positive_fraction=positive_rows / row_count,
            canonicalizable_correct_fraction=(
                canonicalizable_positive_rows / positive_rows if positive_rows else 0.0
            ),
            new_outcome_count=float(new_outcome_count),
            new_outcome_row_fraction=new_outcome_rows / row_count,
            bank_size_before_mean=mean(bank_sizes_before),
            bank_size_after_mean=mean(bank_sizes_after),
            tracked_prompts=float(self.tracked_prompt_count),
            tracked_outcomes=float(self.tracked_outcome_count),
            support_at_least_two_prompt_fraction=(
                self.support_at_least_two_prompt_fraction
            ),
        )
        return combined, diagnostics
