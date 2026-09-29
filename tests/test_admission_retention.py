import copy

import pytest

from remax.admission_retention import (
    AdmissionRetentionTracker,
    RetentionPriorityRequest,
)


def _tracker(*, adaptive: bool = True) -> AdmissionRetentionTracker:
    return AdmissionRetentionTracker(
        adaptive_priority=adaptive,
        max_missed_rollout_opportunities=2,
        max_mean_logprob_drop=0.5,
        refresh_visits=4,
        score_cooldown_observations=2,
    )


def test_rollout_conversion_excludes_pre_admission_group_and_tracks_hits():
    tracker = _tracker()
    tracker.admit("prompt", "new-mode")

    assert (
        tracker.observe_rollout(
            prompt_key="prompt",
            verified_outcome_counts={"new-mode": 1},
            row_count=8,
        )
        == ()
    )
    diagnostics = tracker.diagnostics()
    assert diagnostics["rollout_eligible_admissions"] == 0.0

    assert (
        tracker.observe_rollout(
            prompt_key="prompt",
            verified_outcome_counts={"old-mode": 2},
            row_count=8,
        )
        == ()
    )
    requests = tracker.observe_rollout(
        prompt_key="prompt",
        verified_outcome_counts={"new-mode": 2},
        row_count=8,
    )
    assert requests == ()
    diagnostics = tracker.diagnostics()
    assert diagnostics["rollout_conversion_fraction"] == 1.0
    assert diagnostics["rollout_row_frequency"] == pytest.approx(2.0 / 16.0)


def test_repeated_post_admission_absence_requests_bounded_priority_refresh():
    tracker = _tracker()
    tracker.admit("prompt", "new-mode")
    for _ in range(2):
        requests = tracker.observe_rollout(
            prompt_key="prompt",
            verified_outcome_counts={},
            row_count=8,
        )
        assert requests == ()

    requests = tracker.observe_rollout(
        prompt_key="prompt",
        verified_outcome_counts={},
        row_count=8,
    )
    assert requests == (
        RetentionPriorityRequest(
            prompt_key="prompt",
            outcome_key="new-mode",
            reason="rollout_absence",
            visits=4,
        ),
    )
    tracker.record_priority_application(requests[0], visits_added=3)
    diagnostics = tracker.diagnostics()
    assert diagnostics["rollout_refresh_requests_cumulative"] == 1.0
    assert diagnostics["priority_visits_added_cumulative"] == 3.0


def test_score_drop_uses_first_observation_as_own_baseline_and_cooldown():
    tracker = _tracker()
    tracker.admit("prompt", "new-mode")
    assert tracker.observe_scores([("prompt", "new-mode", -1.0, -4.0)]) == ()
    requests = tracker.observe_scores([("prompt", "new-mode", -1.6, -6.4)])
    assert requests == (
        RetentionPriorityRequest(
            prompt_key="prompt",
            outcome_key="new-mode",
            reason="score_drop",
            visits=4,
        ),
    )
    assert tracker.observe_scores([("prompt", "new-mode", -1.7, -6.8)]) == ()
    diagnostics = tracker.diagnostics()
    assert diagnostics["score_retained_fraction"] == 0.0
    assert diagnostics["mean_logprob_drop_mean"] == pytest.approx(0.7)
    assert diagnostics["sequence_logprob_drop_mean"] == pytest.approx(2.8)


def test_passive_tracker_measures_without_emitting_priority_requests():
    tracker = _tracker(adaptive=False)
    tracker.admit("prompt", "new-mode")
    tracker.observe_rollout(
        prompt_key="prompt", verified_outcome_counts={}, row_count=8
    )
    for _ in range(4):
        assert (
            tracker.observe_rollout(
                prompt_key="prompt", verified_outcome_counts={}, row_count=8
            )
            == ()
        )
    tracker.observe_scores([("prompt", "new-mode", -1.0, -2.0)])
    assert tracker.observe_scores([("prompt", "new-mode", -3.0, -6.0)]) == ()
    diagnostics = tracker.diagnostics()
    assert diagnostics["tracking_enabled"] == 1.0
    assert diagnostics["adaptive_priority_enabled"] == 0.0
    assert diagnostics["refresh_requests_cumulative"] == 0.0


def test_state_round_trip_and_configuration_mismatch_fail_closed():
    tracker = _tracker()
    tracker.admit("prompt", "new-mode")
    tracker.observe_rollout(
        prompt_key="prompt", verified_outcome_counts={}, row_count=8
    )
    tracker.observe_scores([("prompt", "new-mode", -1.0, -4.0)])
    state = tracker.state_dict()

    restored = _tracker()
    restored.load_state_dict(state)
    assert restored.state_dict() == state

    mismatched = AdmissionRetentionTracker(
        adaptive_priority=True,
        max_missed_rollout_opportunities=3,
        max_mean_logprob_drop=0.5,
        refresh_visits=4,
        score_cooldown_observations=2,
    )
    with pytest.raises(ValueError, match="resume mismatch"):
        mismatched.load_state_dict(state)


def test_state_counter_and_boolean_corruption_fail_closed():
    tracker = _tracker()
    tracker.admit("prompt", "new-mode")
    state = tracker.state_dict()
    bad_counter = copy.deepcopy(state)
    bad_counter["rollout_refresh_requests"] = 1
    with pytest.raises(ValueError, match="request count mismatch"):
        _tracker().load_state_dict(bad_counter)

    bad_flag = copy.deepcopy(state)
    bad_flag["records"]["prompt"]["new-mode"]["converted_on_policy"] = 1
    with pytest.raises(ValueError, match="boolean flags"):
        _tracker().load_state_dict(bad_flag)
