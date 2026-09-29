import copy
import math

import pytest

from remax.online_canonical_bank import (
    OnlineCanonicalBank,
)


def test_zero_coefficient_bank_passively_tracks_verified_discoveries():
    bank = OnlineCanonicalBank(entropy_alpha=0.0)

    advantages, diagnostics = bank.score_and_update(
        prompt_token_ids=[[1, 2]] * 4 + [[3, 4]] * 4,
        outcome_keys=["a", "a", "b", None, "c", "c", None, None],
        task_rewards=[1.0] * 8,
        active_mask=[1] * 8,
        num_samples=4,
    )

    assert bank.objective_active is False
    assert advantages == [0.0] * 8
    assert diagnostics.tracked_outcomes == 3.0
    assert diagnostics.tracked_prompts == 2.0
    assert bank.mean_support_per_prompt == pytest.approx(1.5)
    assert bank.support_at_least_two_prompt_fraction == pytest.approx(0.5)
    assert (
        diagnostics.support_at_least_two_prompt_fraction
        == pytest.approx(0.5)
    )
    restored = OnlineCanonicalBank(entropy_alpha=0.0)
    restored.load_state_dict(bank.state_dict())
    assert restored.tracked_outcome_count == 3
    assert restored.mean_support_per_prompt == pytest.approx(1.5)
    assert restored.support_at_least_two_prompt_fraction == pytest.approx(0.5)


def test_online_bank_scores_snapshot_entropy_and_tracks_new_classes():
    bank = OnlineCanonicalBank(
        entropy_alpha=0.1,
        pseudocount=1.0,
        surprisal_clip=5.0,
    )
    advantages, diagnostics = bank.score_and_update(
        prompt_token_ids=[[1, 2]] * 4,
        outcome_keys=["a", "a", "b", None],
        task_rewards=[1.0, 1.0, 1.0, 1.0],
        active_mask=[1, 1, 1, 1],
        num_samples=4,
    )
    # The rarer singleton receives more centered-surprisal pressure.
    assert advantages[2] > advantages[0]
    assert advantages[3] == 0.0
    assert diagnostics.new_outcome_count == 2.0
    assert diagnostics.bank_size_before_mean == 0.0
    assert diagnostics.bank_size_after_mean == 2.0
    assert diagnostics.canonicalizable_correct_fraction == 0.75
    expected_entropy = -(0.6 * math.log(0.6) + 0.4 * math.log(0.4))
    assert diagnostics.normalized_entropy_ratio_mean == pytest.approx(
        expected_entropy / math.log(2)
    )
    assert (
        diagnostics.normalized_entropy_ratio_eligible_fraction
        == pytest.approx(1.0)
    )
    assert diagnostics.normalized_entropy_mean == pytest.approx(
        expected_entropy
    )
    assert diagnostics.log_support_mean == pytest.approx(math.log(2))
    assert diagnostics.entropy_alpha_used == pytest.approx(0.1)

    second, second_diagnostics = bank.score_and_update(
        prompt_token_ids=[[1, 2]] * 4,
        outcome_keys=["a", "a", "b", "b"],
        task_rewards=[1.0] * 4,
        active_mask=[1] * 4,
        num_samples=4,
    )
    assert second_diagnostics.new_outcome_count == 0.0
    # Historical counts favor a, so the rarer b rows receive positive entropy
    # pressure even though no class is newly discovered.
    assert second[2] == pytest.approx(second[3])
    assert second[2] > second[0]


def test_wrong_inactive_and_unparseable_rows_never_enter_bank():
    bank = OnlineCanonicalBank(entropy_alpha=0.1)
    advantages, diagnostics = bank.score_and_update(
        prompt_token_ids=[[7]] * 4,
        outcome_keys=["wrong", "inactive", None, "valid"],
        task_rewards=[0.0, 1.0, 1.0, 1.0],
        active_mask=[1, 0, 1, 1],
        num_samples=4,
    )
    assert advantages[:3] == [0.0, 0.0, 0.0]
    assert advantages[3] == pytest.approx(0.0)
    assert diagnostics.tracked_outcomes == 1.0
    assert diagnostics.normalized_entropy_ratio_mean == 0.0
    assert diagnostics.normalized_entropy_ratio_eligible_fraction == 0.0


def test_online_bank_uses_override_without_mutating_frozen_configuration():
    fixed_bank = OnlineCanonicalBank(entropy_alpha=0.1)
    adaptive_bank = OnlineCanonicalBank(entropy_alpha=0.1)
    inputs = dict(
        prompt_token_ids=[[5]] * 4,
        outcome_keys=["a", "a", "a", "b"],
        task_rewards=[1.0] * 4,
        active_mask=[1] * 4,
        num_samples=4,
    )
    fixed, _ = fixed_bank.score_and_update(**inputs)
    adaptive, diagnostics = adaptive_bank.score_and_update(
        **inputs,
        entropy_alpha_override=0.2,
    )
    assert diagnostics.entropy_alpha_used == pytest.approx(0.2)
    assert adaptive[0] == pytest.approx(2 * fixed[0], abs=1e-12)
    assert adaptive_bank.entropy_alpha == pytest.approx(0.1)
    assert adaptive_bank.state_dict()["entropy_alpha"] == pytest.approx(0.1)


def test_bank_resume_is_exact_and_configuration_checked():
    bank = OnlineCanonicalBank(entropy_alpha=0.1)
    bank.score_and_update(
        prompt_token_ids=[[11]] * 2,
        outcome_keys=["a", "b"],
        task_rewards=[1.0, 1.0],
        active_mask=[1, 1],
        num_samples=2,
    )
    state = bank.state_dict()
    restored = OnlineCanonicalBank(entropy_alpha=0.1)
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    mismatch = OnlineCanonicalBank(entropy_alpha=0.2)
    with pytest.raises(ValueError, match="resume mismatch for entropy_alpha"):
        mismatch.load_state_dict(state)


def test_replay_bank_retains_one_deterministic_verified_exemplar_per_mode():
    bank = OnlineCanonicalBank(
        entropy_alpha=0.1,
        retain_exemplars=True,
    )
    bank.score_and_update(
        prompt_token_ids=[[11, 12]] * 4,
        outcome_keys=["b", "a", "a", None],
        task_rewards=[1.0, 1.0, 1.0, 0.0],
        active_mask=[1, 1, 1, 1],
        response_token_ids=[[8, 9], [5, 7], [5, 6], [99]],
        num_samples=4,
    )

    groups = bank.replay_groups([[11, 12], [11, 12]])

    assert len(groups) == 1
    assert groups[0].prompt_token_ids == (11, 12)
    assert groups[0].outcome_keys == ("a", "b")
    # The within-group lexicographic minimum makes duplicate-row order
    # irrelevant; the exemplar is immutable after first admission.
    assert groups[0].response_token_ids == ((5, 6), (8, 9))

    bank.score_and_update(
        prompt_token_ids=[[11, 12]] * 2,
        outcome_keys=["a", "b"],
        task_rewards=[1.0, 1.0],
        active_mask=[1, 1],
        response_token_ids=[[1], [2]],
        num_samples=2,
    )
    assert bank.replay_groups([[11, 12]])[0].response_token_ids == (
        (5, 6),
        (8, 9),
    )


def test_replay_bank_resume_is_exact_and_replay_configuration_bound():
    bank = OnlineCanonicalBank(
        entropy_alpha=0.1,
        retain_exemplars=True,
    )
    bank.score_and_update(
        prompt_token_ids=[[3]] * 2,
        outcome_keys=["a", "b"],
        task_rewards=[1.0, 1.0],
        active_mask=[1, 1],
        response_token_ids=[[4], [5]],
        num_samples=2,
    )
    state = bank.state_dict()
    restored = OnlineCanonicalBank(entropy_alpha=0.1)
    assert state["schema"] == (
        "online_growing_support_canonical_maxent_replay_fixed_v5"
    )

    restored = OnlineCanonicalBank(
        entropy_alpha=0.1,
        retain_exemplars=True,
    )
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    mismatch = OnlineCanonicalBank(
        entropy_alpha=0.2,
        retain_exemplars=True,
    )
    with pytest.raises(ValueError, match="resume mismatch for entropy_alpha"):
        mismatch.load_state_dict(state)

    non_replay = OnlineCanonicalBank(
        entropy_alpha=0.1,
    )
    with pytest.raises(ValueError, match="replay configuration mismatch"):
        non_replay.load_state_dict(state)


def test_replay_bank_requires_tokens_and_exposes_singleton_only_on_request():
    bank = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
    )
    inputs = dict(
        prompt_token_ids=[[1]] * 2,
        outcome_keys=["a", None],
        task_rewards=[1.0, 0.0],
        active_mask=[1, 1],
        num_samples=2,
    )
    with pytest.raises(ValueError, match="response token"):
        bank.score_and_update(**inputs)

    bank.score_and_update(
        **inputs,
        response_token_ids=[[2], [3]],
    )
    assert bank.replay_groups([[1]]) == []
    singleton = bank.replay_groups([[1]], min_modes=1)
    assert len(singleton) == 1
    assert singleton[0].outcome_keys == ("a",)
    assert singleton[0].response_token_ids == ((2,),)
    with pytest.raises(ValueError, match="min_modes"):
        bank.replay_groups([[1]], min_modes=0)


def test_replay_capacity_is_compute_bound_while_full_discovery_counts_continue():
    bank = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        replay_capacity=2,
    )
    bank.score_and_update(
        prompt_token_ids=[[1]] * 4,
        outcome_keys=["d", "c", "b", "a"],
        task_rewards=[1.0] * 4,
        active_mask=[1] * 4,
        response_token_ids=[[4], [3], [2], [1]],
        num_samples=4,
    )

    group = bank.replay_groups([[1]])[0]
    assert group.outcome_keys == ("a", "b")
    assert group.response_token_ids == ((1,), (2,))
    assert bank.tracked_outcome_count == 4
    assert bank.state_dict()["replay_capacity"] == 2

    mismatched = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        replay_capacity=3,
    )
    with pytest.raises(ValueError, match="replay_capacity"):
        mismatched.load_state_dict(bank.state_dict())


def test_replay_capacity_of_one_is_a_legal_single_slot_bank():
    """Capacity one keeps the first mode and admits no second one.

    This is the bank the E130 mode-agnostic ablation trains against: the replay
    loss still rehearses a verified success, but mode identity cannot enter it
    because there is never more than one key to weight.
    """

    bank = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        replay_capacity=1,
    )
    assert bank.replay_capacity == 1
    assert bank.state_dict()["replay_capacity"] == 1

    with pytest.raises(ValueError, match="replay_capacity"):
        OnlineCanonicalBank(
            entropy_alpha=0.0,
            retain_exemplars=True,
            replay_capacity=0,
        )


def test_global_replay_round_robins_model_discovered_prompt_banks_and_resumes():
    bank = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        global_replay_groups_per_step=1,
    )
    bank.score_and_update(
        prompt_token_ids=[[1]] * 2 + [[2]] * 2 + [[3]] * 2,
        outcome_keys=["a", None, "b", None, "c", None],
        task_rewards=[1.0, 0.0] * 3,
        active_mask=[1] * 6,
        response_token_ids=[[11], [99], [22], [99], [33], [99]],
        num_samples=2,
    )

    first = bank.scheduled_global_replay_groups()
    state = bank.state_dict()
    restored = OnlineCanonicalBank(entropy_alpha=0.1)
    second = bank.scheduled_global_replay_groups()
    third = bank.scheduled_global_replay_groups()
    fourth = bank.scheduled_global_replay_groups()

    cycle = [
        first[0].prompt_token_ids,
        second[0].prompt_token_ids,
        third[0].prompt_token_ids,
    ]
    assert set(cycle) == {(1,), (2,), (3,)}
    assert fourth[0].prompt_token_ids == first[0].prompt_token_ids
    assert state["global_replay_groups_per_step"] == 1
    assert state["global_replay_cursor"] == 1

    restored = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        global_replay_groups_per_step=1,
    )
    restored.load_state_dict(state)
    assert (
        restored.scheduled_global_replay_groups()[0].prompt_token_ids
        == second[0].prompt_token_ids
    )


def test_global_replay_is_default_off_and_configuration_bound():
    local = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
    )
    local.score_and_update(
        prompt_token_ids=[[1]] * 2,
        outcome_keys=["a", None],
        task_rewards=[1.0, 0.0],
        active_mask=[1, 1],
        response_token_ids=[[2], [3]],
        num_samples=2,
    )
    assert local.scheduled_global_replay_groups() == []

    global_bank = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        global_replay_groups_per_step=1,
    )
    with pytest.raises(
        ValueError,
        match="global_replay_groups_per_step",
    ):
        global_bank.load_state_dict(local.state_dict())

    with pytest.raises(
        ValueError,
        match="global_replay_groups_per_step",
    ):
        OnlineCanonicalBank(
            entropy_alpha=0.0,
            retain_exemplars=True,
            global_replay_groups_per_step=-1,
        )


def test_global_replay_round_robins_over_verified_banks_and_resumes_exactly():
    bank = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        global_replay_groups_per_step=1,
    )
    for prompt, outcome, response in (
        (11, "a", 101),
        (22, "b", 202),
        (33, "c", 303),
    ):
        bank.score_and_update(
            prompt_token_ids=[[prompt]] * 2,
            outcome_keys=[outcome, None],
            task_rewards=[1.0, 0.0],
            active_mask=[1, 1],
            response_token_ids=[[response], [999]],
            num_samples=2,
        )

    first_cycle = [
        bank.scheduled_global_replay_groups()[0]
        for _ in range(3)
    ]
    assert len({group.prompt_token_ids for group in first_cycle}) == 3
    assert {
        (group.outcome_keys, group.response_token_ids)
        for group in first_cycle
    } == {
        (("a",), ((101,),)),
        (("b",), ((202,),)),
        (("c",), ((303,),)),
    }
    assert (
        bank.scheduled_global_replay_groups()[0].prompt_token_ids
        == first_cycle[0].prompt_token_ids
    )

    state = bank.state_dict()
    restored = OnlineCanonicalBank(entropy_alpha=0.1)
    restored = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        global_replay_groups_per_step=1,
    )
    restored.load_state_dict(state)
    assert (
        restored.scheduled_global_replay_groups()
        == bank.scheduled_global_replay_groups()
    )


def test_global_replay_budget_and_mode_eligibility_are_target_free():
    bank = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        global_replay_groups_per_step=2,
    )
    bank.score_and_update(
        prompt_token_ids=[[1]] * 2 + [[2]] * 2 + [[3]] * 2,
        outcome_keys=["a", None, "b", "c", "d", None],
        task_rewards=[1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        active_mask=[1] * 6,
        response_token_ids=[[10], [90], [20], [21], [30], [91]],
        num_samples=2,
    )

    singleton_eligible = bank.scheduled_global_replay_groups(min_modes=1)
    assert len(singleton_eligible) == 2
    balance_eligible = bank.scheduled_global_replay_groups(min_modes=2)
    assert len(balance_eligible) == 1
    assert balance_eligible[0].prompt_token_ids == (2,)
    assert balance_eligible[0].outcome_keys == ("b", "c")
    state = bank.state_dict()
    assert state["global_replay_groups_per_step"] == 2
    assert "gold" not in " ".join(state)
    assert "target" not in " ".join(state)

    disabled = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
    )
    assert disabled.scheduled_global_replay_groups() == []
    with pytest.raises(ValueError, match="global_replay_groups_per_step"):
        disabled.load_state_dict(state)


def test_finite_global_bootstrap_retires_exactly_and_resumes_checkpoint_exactly():
    bank = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        global_replay_groups_per_step=1,
        global_replay_bootstrap_steps=2,
    )
    bank.score_and_update(
        prompt_token_ids=[[1]] * 2 + [[2]] * 2,
        outcome_keys=["a", None, "b", None],
        task_rewards=[1.0, 0.0, 1.0, 0.0],
        active_mask=[1] * 4,
        response_token_ids=[[11], [99], [22], [99]],
        num_samples=2,
    )

    first = bank.scheduled_global_replay_groups()
    assert first
    assert bank.global_replay_updates == 1
    assert bank.global_replay_bootstrap_active
    state = bank.state_dict()
    restored = OnlineCanonicalBank(entropy_alpha=0.1)
    assert state["global_replay_bootstrap_steps"] == 2
    assert state["global_replay_updates"] == 1

    restored = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        global_replay_groups_per_step=1,
        global_replay_bootstrap_steps=2,
    )
    restored.load_state_dict(state)
    second = restored.scheduled_global_replay_groups()
    assert second
    assert second[0].prompt_token_ids != first[0].prompt_token_ids
    assert restored.global_replay_updates == 2
    assert not restored.global_replay_bootstrap_active
    assert restored.scheduled_global_replay_groups() == []

    # The caller can now use the ordinary prompt-local selector. This depends
    # only on the current prompt and validator-positive model discoveries.
    local = restored.replay_groups([[1]], min_modes=1)
    assert len(local) == 1
    assert local[0].prompt_token_ids == (1,)


def test_finite_global_bootstrap_configuration_is_strict_and_target_free():
    with pytest.raises(ValueError, match="non-negative integer"):
        OnlineCanonicalBank(
            entropy_alpha=0.0,
            retain_exemplars=True,
            global_replay_groups_per_step=1,
            global_replay_bootstrap_steps=-1,
        )
    with pytest.raises(ValueError, match="requires positive"):
        OnlineCanonicalBank(
            entropy_alpha=0.0,
            retain_exemplars=True,
            global_replay_bootstrap_steps=1,
        )

    bank = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        global_replay_groups_per_step=1,
        global_replay_bootstrap_steps=1,
    )
    state_text = " ".join(bank.state_dict()).lower()
    assert "gold" not in state_text
    assert "desired" not in state_text
    mismatched = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        global_replay_groups_per_step=1,
        global_replay_bootstrap_steps=2,
    )
    with pytest.raises(ValueError, match="global_replay_bootstrap_steps"):
        mismatched.load_state_dict(bank.state_dict())


def test_verified_proposal_admission_is_atomic_support_only_and_resumable():
    bank = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        replay_capacity=3,
    )
    bank.score_and_update(
        prompt_token_ids=[[7, 8]] * 2,
        outcome_keys=["a", None],
        task_rewards=[1.0, 0.0],
        active_mask=[1, 1],
        response_token_ids=[[11], [99]],
        num_samples=2,
    )

    diagnostics = bank.admit_verified_proposals(
        prompt_token_ids=[[7, 8], [7, 8]],
        outcome_keys=["b", "c"],
        response_token_ids=[[12], [13]],
    )

    assert diagnostics.proposal_groups == 1
    assert diagnostics.proposal_rows == 2
    assert diagnostics.new_outcomes == 2
    assert diagnostics.stored_exemplars == 2
    assert bank.tracked_outcome_count == 3
    assert bank.proposal_groups == 1
    assert bank.proposal_rows == 2
    assert bank.proposal_new_outcomes == 2
    group = bank.replay_groups([[7, 8]], min_modes=1)[0]
    assert group.outcome_keys == ("a", "b", "c")
    assert group.response_token_ids == ((11,), (12,), (13,))

    state_before_failure = bank.state_dict()
    with pytest.raises(ValueError, match="only outcomes absent"):
        bank.admit_verified_proposals(
            prompt_token_ids=[[7, 8], [7, 8]],
            outcome_keys=["d", "a"],
            response_token_ids=[[14], [15]],
        )
    assert bank.state_dict() == state_before_failure

    restored = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        replay_capacity=3,
    )
    restored.load_state_dict(bank.state_dict())
    assert restored.state_dict() == bank.state_dict()


def test_separated_proposals_change_replay_support_not_on_policy_counts():
    bank = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        replay_capacity=3,
        separate_proposal_objective_support=True,
    )
    first_advantages, _ = bank.score_and_update(
        prompt_token_ids=[[7, 8]] * 2,
        outcome_keys=["a", None],
        task_rewards=[1.0, 0.0],
        active_mask=[1, 1],
        response_token_ids=[[11], [99]],
        num_samples=2,
    )
    assert first_advantages == pytest.approx([0.0, 0.0])

    before = bank.tracked_outcome_count
    admission = bank.admit_verified_proposals(
        prompt_token_ids=[[7, 8]],
        outcome_keys=["b"],
        response_token_ids=[[12]],
    )
    assert admission.tracked_outcomes == 2
    assert bank.tracked_outcome_count == before == 1
    assert bank.mean_support_per_prompt == pytest.approx(1.0)
    assert bank.replay_mean_support_per_prompt == pytest.approx(2.0)
    group = bank.replay_groups([[7, 8]], min_modes=2)[0]
    assert group.outcome_keys == ("a", "b")
    assert group.response_token_ids == ((11,), (12,))

    separated_state = bank.state_dict()
    assert (
        separated_state["schema"]
        == "online_growing_support_canonical_maxent_replay_separated_proposal_fixed_v6"
    )
    assert list(separated_state["counts"].values()) == [{"a": 1}]
    assert list(separated_state["proposal_only_outcomes"].values()) == [["b"]]

    restored = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        replay_capacity=3,
        separate_proposal_objective_support=True,
    )
    restored.load_state_dict(separated_state)
    assert restored.state_dict() == separated_state

    second_advantages, _ = restored.score_and_update(
        prompt_token_ids=[[7, 8]] * 2,
        outcome_keys=["b", None],
        task_rewards=[1.0, 0.0],
        active_mask=[1, 1],
        response_token_ids=[[10], [98]],
        num_samples=2,
    )
    assert second_advantages == pytest.approx([0.0, 0.0])
    assert restored.tracked_outcome_count == 2
    assert restored.state_dict()["proposal_only_outcomes"] == {}
    graduated = restored.replay_groups([[7, 8]], min_modes=2)[0]
    assert graduated.response_token_ids == ((11,), (10,))


def test_proposal_priority_is_fifo_fixed_budget_and_checkpoint_exact():
    bank = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        replay_capacity=3,
        global_replay_groups_per_step=1,
        separate_proposal_objective_support=True,
        proposal_replay_priority_visits=2,
        proposal_replay_priority_multiplier=4.0,
    )
    bank.score_and_update(
        prompt_token_ids=[[7, 8]] * 2 + [[9, 10]] * 2,
        outcome_keys=["a", None, "z", None],
        task_rewards=[1.0, 0.0, 1.0, 0.0],
        active_mask=[1, 1, 1, 1],
        response_token_ids=[[11], [99], [21], [98]],
        num_samples=2,
    )
    bank.admit_verified_proposals(
        prompt_token_ids=[[7, 8]],
        outcome_keys=["b"],
        response_token_ids=[[12]],
    )

    first = bank.scheduled_global_replay_groups()[0]
    assert first.prompt_token_ids == (7, 8)
    assert first.outcome_keys == ("a", "b")
    assert first.mass_weights == pytest.approx((0.4, 1.6))
    assert first.priority_modes == 1
    assert bank.proposal_priority_remaining_visits == 1
    assert bank.proposal_priority_replay_groups == 1

    state = bank.state_dict()
    assert state["schema"].endswith("proposal_priority_fixed_v7")
    restored = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        replay_capacity=3,
        global_replay_groups_per_step=1,
        separate_proposal_objective_support=True,
        proposal_replay_priority_visits=2,
        proposal_replay_priority_multiplier=4.0,
    )
    restored.load_state_dict(state)
    assert restored.state_dict() == state

    second = restored.scheduled_global_replay_groups()[0]
    assert second.prompt_token_ids == (7, 8)
    assert second.mass_weights == pytest.approx((0.4, 1.6))
    assert second.priority_modes == 1
    assert restored.proposal_priority_remaining_visits == 0
    assert restored.proposal_priority_replay_groups == 2

    ordinary = restored.replay_groups([[7, 8]], min_modes=2)[0]
    assert ordinary.mass_weights == pytest.approx((1.0, 1.0))
    assert ordinary.priority_modes == 0


def _adaptive_retention_bank() -> OnlineCanonicalBank:
    return OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        replay_capacity=3,
        global_replay_groups_per_step=1,
        separate_proposal_objective_support=True,
        proposal_replay_priority_visits=2,
        proposal_replay_priority_multiplier=4.0,
        proposal_retention_tracking=True,
        proposal_adaptive_retention_priority=True,
        proposal_retention_max_missed_rollout_opportunities=2,
        proposal_retention_max_mean_logprob_drop=0.5,
        proposal_retention_refresh_visits=2,
        proposal_retention_score_cooldown_observations=2,
    )


def _seed_and_admit_retention_pair(bank: OnlineCanonicalBank) -> None:
    bank.score_and_update(
        prompt_token_ids=[[7, 8]] * 2,
        outcome_keys=["a", None],
        task_rewards=[1.0, 0.0],
        active_mask=[1, 1],
        response_token_ids=[[11], [99]],
        num_samples=2,
    )
    bank.admit_verified_proposals(
        prompt_token_ids=[[7, 8]],
        outcome_keys=["b"],
        response_token_ids=[[12]],
    )


def test_admission_retention_rollout_absence_refreshes_only_bounded_mass_priority():
    bank = _adaptive_retention_bank()
    _seed_and_admit_retention_pair(bank)
    assert bank.state_dict()["schema"].endswith("priority_retention_v8")

    # Consume the fixed initial two-visit admission priority first.
    bank.scheduled_global_replay_groups()
    bank.scheduled_global_replay_groups()
    assert bank.proposal_priority_remaining_visits == 0

    neutral_a_only = dict(
        prompt_token_ids=[[7, 8]] * 2,
        outcome_keys=["a", None],
        task_rewards=[1.0, 0.0],
        active_mask=[1, 1],
        response_token_ids=[[11], [99]],
        num_samples=2,
    )
    # The first group was sampled before proposal admission and is excluded;
    # the next two neutral groups are genuine post-admission misses.
    bank.score_and_update(**neutral_a_only)
    bank.score_and_update(**neutral_a_only)
    assert bank.proposal_priority_remaining_visits == 0
    bank.score_and_update(**neutral_a_only)
    assert bank.proposal_priority_remaining_visits == 2

    prioritized = bank.scheduled_global_replay_groups()[0]
    assert prioritized.outcome_keys == ("a", "b")
    assert prioritized.mass_weights == pytest.approx((0.4, 1.6))
    diagnostics = bank.proposal_retention_diagnostics()
    assert diagnostics["rollout_conversion_fraction"] == 0.0
    assert diagnostics["rollout_refresh_requests_cumulative"] == 1.0
    assert diagnostics["gold_support_feedback"] == 0.0
    assert diagnostics["eval_feedback"] == 0.0


def test_admission_retention_likelihood_drop_refreshes_priority_and_resumes():
    bank = _adaptive_retention_bank()
    _seed_and_admit_retention_pair(bank)
    bank.scheduled_global_replay_groups()
    bank.scheduled_global_replay_groups()
    replay_group = bank.replay_groups([[7, 8]], min_modes=2)[0]

    baseline = bank.observe_replay_retention_scores(
        groups=[replay_group],
        mean_logprobs=[-0.5, -1.0],
        sequence_logprobs=[-0.5, -2.0],
    )
    assert baseline["score_followup_admissions"] == 0.0
    followup = bank.observe_replay_retention_scores(
        groups=[replay_group],
        mean_logprobs=[-0.5, -1.6],
        sequence_logprobs=[-0.5, -3.2],
    )
    assert followup["score_retained_fraction"] == 0.0
    assert followup["mean_logprob_drop_mean"] == pytest.approx(0.6)
    assert followup["sequence_logprob_drop_mean"] == pytest.approx(1.2)
    assert followup["score_refresh_requests_cumulative"] == 1.0
    assert bank.proposal_priority_remaining_visits == 2

    state = bank.state_dict()
    restored = _adaptive_retention_bank()
    restored.load_state_dict(state)
    assert restored.state_dict() == state
    assert restored.proposal_retention_diagnostics() == followup

    legacy = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        replay_capacity=3,
        global_replay_groups_per_step=1,
        separate_proposal_objective_support=True,
        proposal_replay_priority_visits=2,
        proposal_replay_priority_multiplier=4.0,
    )
    with pytest.raises(ValueError, match="replay configuration mismatch"):
        legacy.load_state_dict(state)

    corrupted = copy.deepcopy(state)
    tracked_records = next(
        iter(corrupted["proposal_admission_retention"]["records"].values())
    )
    tracked_records["a"] = tracked_records.pop("b")
    with pytest.raises(ValueError, match="inconsistent admission lifecycle"):
        _adaptive_retention_bank().load_state_dict(corrupted)


def test_admission_retention_resume_accepts_on_policy_converted_proposal():
    bank = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        replay_capacity=3,
        separate_proposal_objective_support=True,
        proposal_retention_tracking=True,
    )
    neutral = dict(
        prompt_token_ids=[[7, 8]] * 2,
        outcome_keys=["a", None],
        task_rewards=[1.0, 0.0],
        active_mask=[1, 1],
        response_token_ids=[[11], [99]],
        num_samples=2,
    )
    bank.score_and_update(**neutral)
    bank.admit_verified_proposals(
        prompt_token_ids=[[7, 8]],
        outcome_keys=["b"],
        response_token_ids=[[12]],
    )

    # Consume the neutral group sampled before admission, then observe the
    # proposal in a genuinely post-admission neutral group.
    bank.score_and_update(**neutral)
    bank.score_and_update(
        prompt_token_ids=[[7, 8]] * 2,
        outcome_keys=["b", None],
        task_rewards=[1.0, 0.0],
        active_mask=[1, 1],
        response_token_ids=[[13], [98]],
        num_samples=2,
    )
    state = bank.state_dict()
    prompt_key = next(iter(state["counts"]))
    assert state["counts"][prompt_key] == {"a": 2, "b": 1}
    assert state["proposal_only_outcomes"] == {}
    assert state["proposal_admission_retention"]["records"][prompt_key]["b"][
        "converted_on_policy"
    ] is True

    restored = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        replay_capacity=3,
        separate_proposal_objective_support=True,
        proposal_retention_tracking=True,
    )
    restored.load_state_dict(state)
    assert restored.state_dict() == state


def test_separated_proposal_admission_is_atomic_at_replay_capacity():
    bank = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        replay_capacity=2,
        separate_proposal_objective_support=True,
    )
    bank.score_and_update(
        prompt_token_ids=[[1]] * 2,
        outcome_keys=["a", None],
        task_rewards=[1.0, 0.0],
        active_mask=[1, 1],
        response_token_ids=[[10], [99]],
        num_samples=2,
    )
    state = bank.state_dict()
    with pytest.raises(ValueError, match="exceeds replay capacity"):
        bank.admit_verified_proposals(
            prompt_token_ids=[[1], [1]],
            outcome_keys=["b", "c"],
            response_token_ids=[[11], [12]],
        )
    assert bank.state_dict() == state


def test_replay_resume_handles_more_discoveries_than_exemplar_capacity():
    bank = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        replay_capacity=2,
    )
    bank.score_and_update(
        prompt_token_ids=[[1]] * 4,
        outcome_keys=["a", "b", "c", "d"],
        task_rewards=[1.0] * 4,
        active_mask=[1] * 4,
        response_token_ids=[[1], [2], [3], [4]],
        num_samples=4,
    )
    assert bank.tracked_outcome_count == 4
    assert len(bank.replay_groups([[1]], min_modes=1)[0].outcome_keys) == 2

    restored = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        replay_capacity=2,
    )
    restored.load_state_dict(bank.state_dict())
    assert restored.tracked_outcome_count == 4
    assert restored.state_dict() == bank.state_dict()



def test_verified_replay_support_includes_proposals_but_not_their_counts():
    bank = OnlineCanonicalBank(
        entropy_alpha=0.0,
        retain_exemplars=True,
        replay_capacity=3,
        separate_proposal_objective_support=True,
    )
    bank.score_and_update(
        prompt_token_ids=[[7, 8]] * 2,
        outcome_keys=["common", None],
        task_rewards=[1.0, 0.0],
        active_mask=[1, 1],
        response_token_ids=[[11], [99]],
        num_samples=2,
    )
    bank.admit_verified_proposals(
        prompt_token_ids=[[7, 8]],
        outcome_keys=["proposal_rare"],
        response_token_ids=[[12]],
    )

    assert bank.verified_replay_support([7, 8]) == (
        "common",
        "proposal_rare",
    )
    assert bank.verified_replay_support([9, 10]) == ()
    replay_group = bank.replay_groups([[7, 8]], min_modes=1)[0]
    assert replay_group.outcome_keys == ("common", "proposal_rare")
    assert replay_group.fresh_observation_counts == (1, 0)
    counts = next(iter(bank.state_dict()["counts"].values()))
    assert counts == {"common": 1}
    assert "proposal_rare" not in counts
