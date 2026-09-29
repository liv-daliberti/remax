import math

import pytest
import torch

from remax.canonical_replay import (
    canonical_replay_key_target_weights,
    canonical_replay_split_mass_balance_loss,
    canonical_replay_uniform_loss,
    canonical_replay_uniform_verified_likelihood_loss,
    materialize_canonical_replay_batch,
)
from remax.online_canonical_bank import (
    VerifiedCanonicalReplayGroup,
)

def test_uniform_switch_is_byte_identical_to_historical_loss_and_gradient():
    base_scores = torch.tensor([-1.25, -3.5, -0.75], dtype=torch.float32)
    historical_scores = base_scores.clone().requires_grad_(True)
    switched_scores = base_scores.clone().requires_grad_(True)

    historical = canonical_replay_uniform_verified_likelihood_loss(
        historical_scores, [3]
    )
    uniform_weights = canonical_replay_key_target_weights(
        torch.tensor([20, 2, 1]), [3], weighting="uniform"
    )
    switched = canonical_replay_uniform_verified_likelihood_loss(
        switched_scores, [3], uniform_weights
    )
    historical.loss.backward()
    switched.loss.backward()

    assert torch.equal(historical.loss, switched.loss)
    assert torch.equal(historical.score_gradients, switched.score_gradients)
    assert historical_scores.grad is not None
    assert switched_scores.grad is not None
    assert torch.equal(historical_scores.grad, switched_scores.grad)


def test_uniform_target_weight_switch_is_literal_ones():
    counts = torch.tensor([9, 1, 4, 2], dtype=torch.int64)

    weights = canonical_replay_key_target_weights(
        counts,
        [2, 2],
        weighting="uniform",
    )

    assert torch.equal(weights, torch.ones(4, dtype=torch.float32))


def test_fresh_frequency_target_weights_only_change_within_bank_vector():
    counts = torch.tensor([9, 1, 4, 2], dtype=torch.int64)
    weights = canonical_replay_key_target_weights(
        counts,
        [2, 2],
        weighting="fresh_frequency",
    )
    scores = torch.tensor([-1.0, -3.0, -2.0, -4.0], requires_grad=True)

    result = canonical_replay_uniform_verified_likelihood_loss(
        scores,
        [2, 2],
        weights,
    )
    result.loss.backward()

    assert weights.tolist() == pytest.approx([1.8, 0.2, 4 / 3, 2 / 3])
    assert weights[:2].sum().item() == pytest.approx(2.0)
    assert weights[2:].sum().item() == pytest.approx(2.0)
    assert result.loss.item() == pytest.approx((1.2 + 8 / 3) / 2)
    assert scores.grad is not None
    assert scores.grad.tolist() == pytest.approx([-0.45, -0.05, -1 / 3, -1 / 6])


def test_frequency_target_rejects_nonfresh_bank_keys():
    with pytest.raises(ValueError, match="positive fresh observation"):
        canonical_replay_key_target_weights(
            torch.tensor([3, 0]), [2], weighting="fresh_frequency"
        )


def test_uniform_replay_loss_is_zero_only_at_balanced_model_scores():
    balanced = torch.tensor([0.0, 0.0], requires_grad=True)
    result = canonical_replay_uniform_loss(balanced, [2])

    assert result.loss.item() == pytest.approx(0.0, abs=1e-7)
    assert result.normalized_entropy.item() == pytest.approx(1.0)
    assert result.eligible_groups == 1
    assert result.retained_modes == 2


def test_replay_loss_restores_a_near_missing_observed_mode():
    scores = torch.tensor([4.0, -4.0], requires_grad=True)
    result = canonical_replay_uniform_loss(scores, [2])
    result.loss.backward()

    assert result.loss.item() > 3
    assert 0 < result.normalized_entropy.item() < 0.01
    assert scores.grad is not None
    assert torch.allclose(scores.grad, result.score_gradients)
    # Gradient descent lowers the dominant score and raises the missing one.
    assert scores.grad[0].item() > 0
    assert scores.grad[1].item() < 0


def test_replay_loss_averages_prompt_local_banks_without_gold_support():
    scores = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    result = canonical_replay_uniform_loss(scores, [2, 3])

    assert result.eligible_groups == 2
    assert result.retained_modes == 5
    assert torch.isfinite(result.loss)
    assert 0 < result.normalized_entropy.item() <= 1


def test_uniform_verified_likelihood_has_common_mass_gradient():
    scores = torch.tensor([-1.0, -1.0], requires_grad=True)
    result = canonical_replay_uniform_verified_likelihood_loss(scores, [2])
    result.loss.backward()

    assert result.loss.item() == pytest.approx(1.0)
    assert result.cross_entropy_excess.item() == pytest.approx(0.0, abs=1e-7)
    assert result.normalized_entropy.item() == pytest.approx(1.0)
    assert scores.grad is not None
    assert torch.allclose(scores.grad, torch.tensor([-0.5, -0.5]))
    assert torch.allclose(scores.grad, result.score_gradients)
    assert scores.grad.sum().item() == pytest.approx(-1.0)


def test_uniform_verified_likelihood_weights_prompts_then_modes_equally():
    scores = torch.tensor(
        [-1.0, -3.0, -2.0, -4.0, -6.0],
        requires_grad=True,
    )
    result = canonical_replay_uniform_verified_likelihood_loss(
        scores,
        [2, 3],
    )
    result.loss.backward()

    assert result.loss.item() == pytest.approx(
        ((1.0 + 3.0) / 2.0 + (2.0 + 4.0 + 6.0) / 3.0) / 2.0
    )
    assert scores.grad is not None
    assert torch.allclose(
        scores.grad,
        torch.tensor([-0.25, -0.25, -1 / 6, -1 / 6, -1 / 6]),
    )
    assert scores.grad[:2].sum().item() == pytest.approx(-0.5)
    assert scores.grad[2:].sum().item() == pytest.approx(-0.5)


def test_uniform_verified_likelihood_anchors_singleton_without_fake_entropy():
    scores = torch.tensor([-2.0], requires_grad=True)
    result = canonical_replay_uniform_verified_likelihood_loss(scores, [1])
    result.loss.backward()

    assert result.loss.item() == pytest.approx(2.0)
    assert result.cross_entropy_excess.item() == pytest.approx(0.0)
    assert result.normalized_entropy.item() == pytest.approx(1.0)
    assert result.eligible_groups == 0
    assert result.retained_modes == 0
    assert result.actuator_groups == 1
    assert result.actuator_modes == 1
    assert scores.grad is not None
    assert scores.grad.item() == pytest.approx(-1.0)


def test_uniform_verified_likelihood_senses_only_multimode_groups():
    scores = torch.tensor([-2.0, -1.0, -3.0])
    result = canonical_replay_uniform_verified_likelihood_loss(
        scores,
        [1, 2],
    )

    assert result.eligible_groups == 1
    assert result.retained_modes == 2
    assert result.actuator_groups == 2
    assert result.actuator_modes == 3
    assert result.score_gradients.tolist() == pytest.approx(
        [-0.5, -0.25, -0.25]
    )


def test_split_replay_keeps_mass_and_balance_gradients_independent():
    scores = torch.tensor([-2.0, 3.0, -3.0], requires_grad=True)
    result = canonical_replay_split_mass_balance_loss(scores, [1, 2])

    assert result.mass_score_gradients.tolist() == pytest.approx(
        [-0.5, -0.25, -0.25]
    )
    assert result.mass_score_gradients.sum().item() == pytest.approx(-1.0)
    assert result.balance_score_gradients[0].item() == pytest.approx(0.0)
    assert result.balance_score_gradients.sum().item() == pytest.approx(
        0.0,
        abs=1e-7,
    )
    assert result.balance_score_gradients[1].item() > 0
    assert result.balance_score_gradients[2].item() < 0
    assert result.balance_eligible_groups == 1
    assert result.balance_retained_modes == 2


def test_fresh_mode_priority_reweights_mass_not_whole_bank_balance():
    scores = torch.tensor([2.0, -2.0])
    baseline = canonical_replay_split_mass_balance_loss(scores, [2])
    prioritized = canonical_replay_split_mass_balance_loss(
        scores,
        [2],
        torch.tensor([0.4, 1.6]),
    )

    assert prioritized.mass_score_gradients.tolist() == pytest.approx(
        [-0.2, -0.8]
    )
    assert prioritized.mass_score_gradients.sum().item() == pytest.approx(-1.0)
    assert torch.allclose(
        prioritized.balance_score_gradients,
        baseline.balance_score_gradients,
    )
    assert prioritized.balance_loss.item() == pytest.approx(
        baseline.balance_loss.item()
    )


def test_split_replay_singleton_has_mass_but_no_fake_balance():
    scores = torch.tensor([-2.0])
    result = canonical_replay_split_mass_balance_loss(scores, [1])

    assert result.mass_loss.item() == pytest.approx(2.0)
    assert result.mass_score_gradients.item() == pytest.approx(-1.0)
    assert result.balance_loss.item() == pytest.approx(0.0)
    assert result.balance_score_gradients.item() == pytest.approx(0.0)
    assert result.balance_eligible_groups == 0


def test_replay_batch_marks_exact_response_labels_after_each_prompt():
    groups = [
        VerifiedCanonicalReplayGroup(
            prompt_token_ids=(10, 11, 12),
            outcome_keys=("a", "b"),
            response_token_ids=((20, 21), (30,)),
        ),
        VerifiedCanonicalReplayGroup(
            prompt_token_ids=(40, 41),
            outcome_keys=("c", "d"),
            response_token_ids=((50,), (60, 61, 62)),
        ),
    ]

    batch = materialize_canonical_replay_batch(
        groups,
        pad_token_id=0,
        device="cpu",
    )

    assert batch.group_sizes == (2, 2)
    assert batch.mass_weights.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])
    assert batch.priority_modes == 0
    assert batch.fresh_observation_counts.tolist() == [0, 0, 0, 0]
    assert batch.input_ids.tolist() == [
        [10, 11, 12, 20, 21],
        [10, 11, 12, 30, 0],
        [40, 41, 50, 0, 0],
        [40, 41, 60, 61, 62],
    ]
    assert batch.attention_mask.tolist() == [
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1],
    ]
    assert batch.response_masks.tolist() == [
        [False, False, True, True],
        [False, False, True, False],
        [False, True, False, False],
        [False, True, True, True],
    ]


def test_replay_batch_preserves_normalized_priority_weights():
    group = VerifiedCanonicalReplayGroup(
        prompt_token_ids=(10, 11),
        outcome_keys=("old", "fresh"),
        response_token_ids=((20,), (30,)),
        mass_weights=(0.4, 1.6),
        priority_modes=1,
    )

    batch = materialize_canonical_replay_batch(
        [group],
        pad_token_id=0,
        device="cpu",
    )

    assert batch.mass_weights.tolist() == pytest.approx([0.4, 1.6])
    assert batch.priority_modes == 1


def test_replay_entropy_sensor_survives_float32_probability_underflow():
    scores = torch.tensor([0.0, -100.0], requires_grad=True)

    result = canonical_replay_uniform_loss(scores, [2])

    assert result.normalized_entropy.item() > 0.0


@pytest.mark.parametrize(
    ("scores", "sizes", "message"),
    (
        (torch.zeros(2, 1), [2], "one-dimensional"),
        (torch.zeros(2), [1, 1], "at least two"),
        (torch.zeros(3), [2], "do not partition"),
        (torch.tensor([0.0, math.inf]), [2], "finite"),
    ),
)
def test_replay_loss_fails_closed_on_malformed_banks(scores, sizes, message):
    with pytest.raises(ValueError, match=message):
        canonical_replay_uniform_loss(scores, sizes)
