from __future__ import annotations

import itertools
import math

import pytest
import torch

from remax.on_policy_maxent import (
    prefix_ratio_expected_length_surrogate,
    prefix_ratio_maxent_surrogate,
    standard_maxent_length_penalty_loss,
    standard_maxent_loss,
)


def test_standard_maxent_loss_has_only_the_shared_outer_tmax_scale():
    raw_entropy = torch.tensor(12.0, requires_grad=True)

    loss = standard_maxent_loss(
        raw_entropy,
        alpha=0.05,
        reward_estimator_scale=15 / 16,
        update_normalizer=192,
    )
    loss.backward()

    expected = -0.05 * (15 / 16) / 192
    assert loss.item() == pytest.approx(expected * 12.0)
    assert raw_entropy.grad.item() == pytest.approx(expected)


def test_zero_maxent_coefficient_is_an_exact_loss_and_gradient_no_op():
    raw_entropy = torch.tensor(2.0, requires_grad=True)

    loss = standard_maxent_loss(
        raw_entropy,
        alpha=0.0,
        reward_estimator_scale=15 / 16,
        update_normalizer=192,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert raw_entropy.grad.item() == 0.0


def test_standard_maxent_length_penalty_has_positive_shared_outer_scale():
    raw_length = torch.tensor(12.0, requires_grad=True)

    loss = standard_maxent_length_penalty_loss(
        raw_length,
        length_lambda=0.002,
        reward_estimator_scale=15 / 16,
        update_normalizer=192,
    )
    loss.backward()

    expected = 0.002 * (15 / 16) / 192
    assert loss.item() == pytest.approx(expected * 12.0)
    assert raw_length.grad.item() == pytest.approx(expected)


def test_zero_length_multiplier_is_an_exact_actor_loss_no_op():
    raw_length = torch.tensor(37.0, requires_grad=True)

    loss = standard_maxent_length_penalty_loss(
        raw_length,
        length_lambda=0.0,
        reward_estimator_scale=15 / 16,
        update_normalizer=192,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert raw_length.grad.item() == 0.0


def _bernoulli_entropy(logit: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logit)
    return -(
        probability * torch.log(probability)
        + (1.0 - probability) * torch.log(1.0 - probability)
    )


def _bernoulli_log_prob(logit: torch.Tensor, action: int) -> torch.Tensor:
    probability = torch.sigmoid(logit)
    return torch.log(probability if action else 1.0 - probability)


def test_prefix_ratio_is_exclusive_masked_and_ppo_clipped():
    token_entropy = torch.tensor(
        [[1.0, 2.0, 99.0], [0.5, 99.0, 99.0]], requires_grad=True
    )
    old_logps = torch.tensor([[-1.0, -1.0, 0.0], [-0.3, 0.0, 0.0]])
    new_logps = old_logps.clone()
    new_logps[0, 0] += math.log(1.5)
    new_logps.requires_grad_()
    masks = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.float32)

    surrogate, observed, sampled, ratio_mean, ratio_max, clipfrac = (
        prefix_ratio_maxent_surrogate(
            token_entropy,
            new_logps,
            old_logps,
            masks,
            normalization_constant=4.0,
            cliprange=0.2,
        )
    )

    # The first response token has the empty-prefix ratio one. The first
    # action's 1.5 ratio affects only the second token and clips to 1.2 there.
    assert torch.allclose(surrogate, torch.tensor([0.85, 0.125]))
    assert torch.allclose(observed, torch.tensor([1.0, 0.125]))
    assert torch.allclose(sampled, torch.tensor([0.75, 0.125]))
    assert torch.allclose(ratio_mean, torch.tensor([1.25, 1.0]))
    assert torch.allclose(ratio_max, torch.tensor([1.5, 1.0]))
    assert torch.allclose(clipfrac, torch.tensor([0.5, 0.0]))
    assert surrogate.requires_grad
    assert not observed.requires_grad


def test_full_categorical_gradient_survives_identical_sampled_rows():
    logits = torch.ones(4, 2, requires_grad=True)
    token_entropy = _bernoulli_entropy(logits)
    selected_new_logps = torch.log(torch.sigmoid(logits))
    selected_old_logps = selected_new_logps.detach().clone()
    masks = torch.ones_like(logits)

    surrogate, *_ = prefix_ratio_maxent_surrogate(
        token_entropy,
        selected_new_logps,
        selected_old_logps,
        masks,
        normalization_constant=2.0,
        cliprange=0.2,
    )
    surrogate.mean().backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() > 0


def test_unclipped_prefix_ratio_matches_off_policy_entropy_value_and_gradient():
    """Enumeration checks distinct old/new policies, not only their tangent."""

    old_root = torch.tensor(-0.3)
    old_child_zero = torch.tensor(0.6)
    old_child_one = torch.tensor(-0.5)
    new_root = torch.tensor(0.4, requires_grad=True)
    new_child_zero = torch.tensor(-0.7, requires_grad=True)
    new_child_one = torch.tensor(1.1, requires_grad=True)
    parameters = (new_root, new_child_zero, new_child_one)

    row_entropies = []
    new_row_logps = []
    old_row_logps = []
    old_trajectory_probabilities = []
    for first, second in itertools.product((0, 1), repeat=2):
        new_child = new_child_one if first else new_child_zero
        old_child = old_child_one if first else old_child_zero
        row_entropies.append(
            torch.stack((_bernoulli_entropy(new_root), _bernoulli_entropy(new_child)))
        )
        new_row_logps.append(
            torch.stack(
                (
                    _bernoulli_log_prob(new_root, first),
                    _bernoulli_log_prob(new_child, second),
                )
            )
        )
        old_row_logps.append(
            torch.stack(
                (
                    _bernoulli_log_prob(old_root, first),
                    _bernoulli_log_prob(old_child, second),
                )
            )
        )
        old_trajectory_probabilities.append(torch.exp(old_row_logps[-1].sum()))

    surrogate, *_ = prefix_ratio_maxent_surrogate(
        torch.stack(row_entropies),
        torch.stack(new_row_logps),
        torch.stack(old_row_logps),
        torch.ones(4, 2),
        normalization_constant=1.0,
        cliprange=None,
    )
    importance_sampled_entropy = (
        surrogate * torch.stack(old_trajectory_probabilities)
    ).sum()

    new_root_probability = torch.sigmoid(new_root)
    exact_entropy = (
        _bernoulli_entropy(new_root)
        + (1.0 - new_root_probability) * _bernoulli_entropy(new_child_zero)
        + new_root_probability * _bernoulli_entropy(new_child_one)
    )
    assert importance_sampled_entropy.item() == pytest.approx(
        exact_entropy.item(), abs=1e-6
    )

    estimated_gradient = torch.autograd.grad(
        importance_sampled_entropy, parameters, retain_graph=True
    )
    exact_gradient = torch.autograd.grad(exact_entropy, parameters)
    for estimated, exact in zip(estimated_gradient, exact_gradient):
        assert estimated.item() == pytest.approx(exact.item(), abs=1e-6)


def test_unclipped_prefix_ratio_matches_variable_length_value_and_gradient():
    """A stop/continue tree verifies exclusive-prefix survival exactly."""

    old_root = torch.tensor(-0.4)
    old_child = torch.tensor(0.7)
    new_root = torch.tensor(0.6, requires_grad=True)
    new_child = torch.tensor(-0.8, requires_grad=True)

    new_rows = []
    old_rows = []
    masks = []
    old_probabilities = []
    # Root action zero stops after one token. Root action one reaches a second
    # token, whose sampled action does not affect the two-token horizon.
    for first, second in ((0, 0), (1, 0), (1, 1)):
        mask = torch.tensor([1.0, float(first)])
        new_rows.append(
            torch.stack(
                (
                    _bernoulli_log_prob(new_root, first),
                    _bernoulli_log_prob(new_child, second)
                    if first
                    else torch.zeros(()),
                )
            )
        )
        old_rows.append(
            torch.stack(
                (
                    _bernoulli_log_prob(old_root, first),
                    _bernoulli_log_prob(old_child, second)
                    if first
                    else torch.zeros(()),
                )
            )
        )
        masks.append(mask)
        old_probabilities.append(torch.exp((old_rows[-1] * mask).sum()))

    surrogate, *_ = prefix_ratio_expected_length_surrogate(
        torch.stack(new_rows),
        torch.stack(old_rows),
        torch.stack(masks),
        cliprange=None,
    )
    estimated_length = (surrogate * torch.stack(old_probabilities)).sum()
    exact_length = 1.0 + torch.sigmoid(new_root)

    assert estimated_length.item() == pytest.approx(exact_length.item(), abs=1e-6)
    estimated_gradient = torch.autograd.grad(
        estimated_length,
        (new_root, new_child),
        allow_unused=True,
    )
    exact_root_gradient = torch.autograd.grad(exact_length, new_root)[0]
    assert estimated_gradient[0].item() == pytest.approx(
        exact_root_gradient.item(), abs=1e-6
    )
    assert estimated_gradient[1] is not None
    assert estimated_gradient[1].item() == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize(
    ("prefix_ratio", "expected_surrogate", "expected_first_logp_gradient"),
    ((0.5, 1.8, 0.0), (1.5, 2.5, 1.5)),
)
def test_expected_length_uses_conservative_cost_clipping(
    prefix_ratio, expected_surrogate, expected_first_logp_gradient
):
    old_logps = torch.zeros(1, 2)
    new_logps = torch.tensor(
        [[math.log(prefix_ratio), math.log(prefix_ratio)]], requires_grad=True
    )
    masks = torch.ones(1, 2)

    surrogate, observed, sampled, *_ = prefix_ratio_expected_length_surrogate(
        new_logps,
        old_logps,
        masks,
        cliprange=0.2,
    )

    # Empty prefix contributes one. The first action affects only token two.
    assert surrogate.item() == pytest.approx(expected_surrogate)
    assert observed.item() == pytest.approx(1.0 + prefix_ratio)
    assert sampled.item() == pytest.approx(2.0)
    gradient = torch.autograd.grad(surrogate, new_logps)[0]
    assert gradient[0, 0].item() == pytest.approx(expected_first_logp_gradient)
    assert gradient[0, 1].item() == pytest.approx(0.0)
