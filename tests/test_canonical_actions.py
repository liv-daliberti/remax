from __future__ import annotations

import math

import pytest
import torch

from remax.canonical_actions import (
    _sample_canonical_actions_from_logits,
    canonical_behavior_overlap_diagnostics,
    materialize_canonical_behavior_policy,
    resolve_graph_color_action_token_ids,
    resolve_single_token_actions,
    restricted_action_log_probs_and_entropy,
    restricted_action_log_probs_entropy_and_distribution,
)
from remax.on_policy_maxent import (
    prefix_ratio_maxent_surrogate,
    standard_maxent_loss,
)


class _DigitTokenizer:
    _ids = {"1": 16, "2": 17, "3": 18}
    _tokens = {value: key for key, value in _ids.items()}

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [self._ids[text]]

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens,
        clean_up_tokenization_spaces,
    ):
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        return "".join(self._tokens[token_id] for token_id in token_ids)


def _log_probability_logits(probabilities):
    return torch.tensor(probabilities, dtype=torch.float32).log()


def test_canonical_sampler_uses_inverse_cdf_and_returns_exact_behavior_rows():
    # Disallowed vocabulary entries are intentionally much larger than the
    # canonical logits.  Sampling must still be from the ordered finite support.
    logits = torch.full((3, 8), 1_000.0)
    allowed = (1, 3, 6)
    logits[:, list(allowed)] = _log_probability_logits([0.2, 0.3, 0.5])

    token_ids, selected, full = _sample_canonical_actions_from_logits(
        logits,
        allowed_token_ids=allowed,
        uniforms=torch.tensor([0.10, 0.35, 0.80]),
    )

    assert token_ids.dtype == torch.long
    assert token_ids.tolist() == [1, 3, 6]
    expected_full = _log_probability_logits([0.2, 0.3, 0.5]).expand(3, -1)
    torch.testing.assert_close(full, expected_full)
    torch.testing.assert_close(
        selected,
        torch.tensor([math.log(0.2), math.log(0.3), math.log(0.5)]),
    )
    selected_indices = torch.tensor([0, 1, 2])
    torch.testing.assert_close(
        selected, full.gather(1, selected_indices[:, None]).squeeze(1)
    )
    torch.testing.assert_close(
        torch.logsumexp(full, dim=-1), torch.zeros(3), atol=1e-6, rtol=0
    )


def test_three_autoregressive_sampler_calls_build_consistent_behavior_traces():
    """Later distributions are selected from sampled prefixes, not precomputed rows."""

    allowed = (16, 17, 18)
    uniforms = torch.tensor(
        [
            [0.05, 0.30, 0.50],
            [0.85, 0.60, 0.40],
        ]
    )
    def vocab_logits(action_logits):
        logits = torch.full((19,), -1_000.0)
        logits[list(allowed)] = action_logits
        return logits

    root = vocab_logits(_log_probability_logits([0.2, 0.3, 0.5])).expand(2, -1)
    second_by_first = {
        16: vocab_logits(_log_probability_logits([0.1, 0.8, 0.1])),
        18: vocab_logits(_log_probability_logits([0.7, 0.2, 0.1])),
    }
    third_by_prefix = {
        (16, 17): vocab_logits(_log_probability_logits([0.1, 0.1, 0.8])),
        (18, 16): vocab_logits(_log_probability_logits([0.2, 0.6, 0.2])),
    }

    action_columns = []
    selected_columns = []
    behavior_columns = []
    next_logits = root
    prefixes = [tuple(), tuple()]
    for depth in range(3):
        action_ids, selected, full = _sample_canonical_actions_from_logits(
            next_logits,
            allowed_token_ids=allowed,
            uniforms=uniforms[:, depth],
        )
        action_columns.append(action_ids)
        selected_columns.append(selected)
        behavior_columns.append(full)
        prefixes = [
            (*prefix, int(action_id))
            for prefix, action_id in zip(prefixes, action_ids.tolist())
        ]
        if depth == 0:
            next_logits = torch.stack(
                [second_by_first[prefix[0]] for prefix in prefixes]
            )
        elif depth == 1:
            next_logits = torch.stack([third_by_prefix[prefix] for prefix in prefixes])

    action_trace = torch.stack(action_columns, dim=1)
    selected_trace = torch.stack(selected_columns, dim=1)
    behavior_trace = torch.stack(behavior_columns, dim=1)
    assert action_trace.tolist() == [[16, 17, 18], [18, 16, 17]]
    support_indices = torch.empty_like(action_trace)
    for support_index, token_id in enumerate(allowed):
        support_indices[action_trace.eq(token_id)] = support_index
    torch.testing.assert_close(
        selected_trace,
        behavior_trace.gather(2, support_indices.unsqueeze(-1)).squeeze(-1),
    )
    torch.testing.assert_close(
        torch.logsumexp(behavior_trace, dim=-1),
        torch.zeros(2, 3),
        atol=1e-6,
        rtol=0,
    )


def test_canonical_sampler_is_invariant_to_every_disallowed_logit():
    allowed = (1, 4, 7)
    base = torch.randn(4, 9)
    base[:, list(allowed)] = torch.tensor(
        [[-1.0, 0.0, 1.0], [1.5, -0.5, 0.0], [0.1, 0.2, 0.3], [3.0, 2.0, 1.0]]
    )
    changed = base.clone()
    disallowed = [index for index in range(9) if index not in allowed]
    changed[:, disallowed] = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), 1e30, -1e30, 17.0]
    )
    uniforms = torch.tensor([0.01, 0.31, 0.57, 0.99])

    base_result = _sample_canonical_actions_from_logits(
        base, allowed_token_ids=allowed, uniforms=uniforms
    )
    changed_result = _sample_canonical_actions_from_logits(
        changed, allowed_token_ids=allowed, uniforms=uniforms
    )

    for observed, expected in zip(changed_result, base_result):
        torch.testing.assert_close(observed, expected)


def test_canonical_sampler_preserves_support_order_and_detaches_float32_trace():
    logits = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    # The order (5, 1, 3), rather than token-id order, defines both the CDF and
    # the columns transported as the behavior policy.
    allowed = (5, 1, 3)
    token_ids, selected, full = _sample_canonical_actions_from_logits(
        logits,
        allowed_token_ids=allowed,
        uniforms=torch.tensor([0.05]),
    )

    expected = torch.log_softmax(logits.detach().float()[:, [5, 1, 3]], dim=-1)
    assert token_ids.tolist() == [5]
    assert selected.dtype == torch.float32
    assert full.dtype == torch.float32
    assert not selected.requires_grad
    assert not full.requires_grad
    torch.testing.assert_close(full, expected)
    torch.testing.assert_close(selected, expected[:, 0])


@pytest.mark.parametrize(
    ("logits", "uniforms", "message"),
    [
        (torch.zeros(2, 1, 8), torch.tensor([0.1, 0.2]), "shape"),
        (torch.zeros(2, 8), torch.tensor([[0.1], [0.2]]), "shape"),
        (torch.zeros(0, 8), torch.empty(0), "batch"),
    ],
)
def test_canonical_sampler_rejects_malformed_batch_shapes(logits, uniforms, message):
    with pytest.raises(ValueError, match=message):
        _sample_canonical_actions_from_logits(
            logits, allowed_token_ids=(1, 2, 3), uniforms=uniforms
        )


@pytest.mark.parametrize(
    ("allowed", "message"),
    [
        ((), "nonempty"),
        ((1, 1, 3), "unique"),
        ((1, 2, 8), "vocabulary"),
        ((-1, 1, 2), "vocabulary"),
    ],
)
def test_canonical_sampler_rejects_invalid_support(allowed, message):
    with pytest.raises(ValueError, match=message):
        _sample_canonical_actions_from_logits(
            torch.zeros(2, 8),
            allowed_token_ids=allowed,
            uniforms=torch.tensor([0.1, 0.2]),
        )


@pytest.mark.parametrize(
    "uniforms",
    [
        torch.tensor([-1e-6, 0.5]),
        torch.tensor([1.0, 0.5]),
        torch.tensor([float("nan"), 0.5]),
        torch.tensor([float("inf"), 0.5]),
    ],
)
def test_canonical_sampler_rejects_uniforms_outside_half_open_unit_interval(uniforms):
    with pytest.raises(ValueError, match="uniform"):
        _sample_canonical_actions_from_logits(
            torch.zeros(2, 8),
            allowed_token_ids=(1, 2, 3),
            uniforms=uniforms,
        )


def test_canonical_sampler_rejects_nonfinite_allowed_logits():
    logits = torch.zeros(2, 8)
    logits[1, 2] = float("nan")
    with pytest.raises(FloatingPointError, match="nonfinite"):
        _sample_canonical_actions_from_logits(
            logits,
            allowed_token_ids=(1, 2, 3),
            uniforms=torch.tensor([0.1, 0.2]),
        )


def test_graph_actions_resolve_to_three_distinct_round_trip_tokens():
    assert resolve_graph_color_action_token_ids(_DigitTokenizer()) == (16, 17, 18)


def test_action_resolution_rejects_a_multi_token_serialization():
    class BadTokenizer(_DigitTokenizer):
        def encode(self, text, *, add_special_tokens):
            return [16, 17]

    with pytest.raises(ValueError, match="exactly one token"):
        resolve_single_token_actions(BadTokenizer(), ("1",))


def test_restricted_policy_matches_a_three_way_categorical_distribution():
    logits = torch.zeros(1, 4, 7)
    logits[0, 0, 1:4] = torch.tensor([0.0, 1.0, 2.0])
    logits[0, 1, 1:4] = torch.tensor([2.0, 1.0, 0.0])
    labels = torch.tensor([[5, 3, 1, 0]])
    masks = torch.tensor([[1, 1, 0]], dtype=torch.float32)

    selected, entropy = restricted_action_log_probs_and_entropy(
        logits,
        labels,
        masks,
        allowed_token_ids=(1, 2, 3),
    )

    first = torch.log_softmax(torch.tensor([0.0, 1.0, 2.0]), dim=0)[2]
    second = torch.log_softmax(torch.tensor([2.0, 1.0, 0.0]), dim=0)[0]
    expected_entropy = -(
        torch.softmax(torch.tensor([0.0, 1.0, 2.0]), dim=0)
        * torch.log_softmax(torch.tensor([0.0, 1.0, 2.0]), dim=0)
    ).sum()
    torch.testing.assert_close(selected, torch.tensor([[first, second, 0.0]]))
    torch.testing.assert_close(
        entropy, torch.tensor([[expected_entropy, expected_entropy, 0.0]])
    )


def test_restricted_policy_exposes_full_ordered_action_distribution():
    logits = torch.zeros(1, 3, 8)
    logits[0, 0, [1, 2, 3]] = torch.tensor([0.0, 1.0, 2.0])
    logits[0, 1, [1, 2, 3]] = torch.tensor([2.0, 0.0, 1.0])
    labels = torch.tensor([[7, 3, 1]])
    masks = torch.tensor([[1.0, 0.0]])

    selected, entropy, full = restricted_action_log_probs_entropy_and_distribution(
        logits, labels, masks, allowed_token_ids=(1, 2, 3)
    )

    expected = torch.log_softmax(torch.tensor([0.0, 1.0, 2.0]), dim=0)
    torch.testing.assert_close(full[0, 0], expected)
    torch.testing.assert_close(full[0, 1], torch.zeros(3))
    torch.testing.assert_close(selected[0, 0], expected[2])
    assert entropy[0, 0] > 0


def test_materialize_behavior_policy_aligns_full_rows_to_response_positions():
    shifted_labels = torch.tensor([[9, 16, 18, 17, 0]])
    masks = torch.tensor([[0.0, 1.0, 1.0, 1.0, 0.0]])
    full_rows = [
        [
            [math.log(0.5), math.log(0.3), math.log(0.2)],
            [math.log(0.1), math.log(0.2), math.log(0.7)],
            [math.log(0.2), math.log(0.6), math.log(0.2)],
        ]
    ]
    selected_rows = [[full_rows[0][0][0], full_rows[0][1][2], full_rows[0][2][1]]]

    selected, full, norm_error, echo_diff = materialize_canonical_behavior_policy(
        shifted_labels,
        masks,
        action_ids=[[16, 18, 17]],
        selected_log_probs=selected_rows,
        full_log_probs=full_rows,
        behavior_action_token_ids=[[16, 17, 18]],
        allowed_token_ids=(16, 17, 18),
    )

    torch.testing.assert_close(
        selected[0, 1:4], torch.tensor(selected_rows[0], dtype=torch.float32)
    )
    torch.testing.assert_close(
        full[0, 1:4], torch.tensor(full_rows[0], dtype=torch.float32)
    )
    torch.testing.assert_close(selected[0, [0, 4]], torch.zeros(2))
    assert norm_error <= 1e-6
    assert echo_diff <= 1e-7


def test_materialize_behavior_policy_rejects_support_and_selected_drift():
    kwargs = dict(
        shifted_labels=torch.tensor([[16]]),
        response_masks=torch.ones(1, 1),
        action_ids=[[16]],
        selected_log_probs=[[math.log(0.5)]],
        full_log_probs=[[[math.log(0.5), math.log(0.3), math.log(0.2)]]],
        behavior_action_token_ids=[[16, 17, 18]],
        allowed_token_ids=(16, 17, 18),
    )
    bad_support = dict(kwargs)
    bad_support["behavior_action_token_ids"] = [[18, 17, 16]]
    with pytest.raises(RuntimeError, match="support order"):
        materialize_canonical_behavior_policy(**bad_support)

    bad_selected = dict(kwargs)
    bad_selected["selected_log_probs"] = [[math.log(0.4)]]
    with pytest.raises(RuntimeError, match="does not match"):
        materialize_canonical_behavior_policy(**bad_selected)


def test_behavior_overlap_diagnostics_match_analytic_distributions_and_ess():
    behavior_probabilities = torch.tensor(
        [
            [[0.5, 0.3, 0.2], [0.2, 0.5, 0.3], [0.3, 0.3, 0.4]],
            [[0.4, 0.4, 0.2], [0.2, 0.3, 0.5], [0.6, 0.2, 0.2]],
        ]
    )
    learner_probabilities = torch.tensor(
        [
            [[0.45, 0.35, 0.2], [0.25, 0.45, 0.3], [0.3, 0.35, 0.35]],
            [[0.35, 0.45, 0.2], [0.2, 0.35, 0.45], [0.55, 0.25, 0.2]],
        ]
    )
    action_indices = torch.tensor([[0, 1, 2], [1, 2, 0]])
    behavior_log = behavior_probabilities.log()
    learner_log = learner_probabilities.log()
    selected_behavior = torch.gather(
        behavior_log, -1, action_indices.unsqueeze(-1)
    ).squeeze(-1)
    selected_learner = torch.gather(
        learner_log, -1, action_indices.unsqueeze(-1)
    ).squeeze(-1)

    diagnostics = canonical_behavior_overlap_diagnostics(
        selected_behavior,
        behavior_log,
        selected_learner,
        learner_log,
        torch.ones(2, 3),
    )

    expected_tv = 0.5 * torch.abs(learner_probabilities - behavior_probabilities).sum(
        -1
    )
    torch.testing.assert_close(
        diagnostics["canonical_behavior_tv_mean"], expected_tv.double().mean()
    )
    torch.testing.assert_close(
        diagnostics["canonical_behavior_tv_max"], expected_tv.double().max()
    )
    assert 0 < diagnostics["canonical_behavior_sequence_ess_fraction"] <= 1
    assert 0 < diagnostics["canonical_behavior_prefix_ess_fraction_min"] <= 1
    assert diagnostics["canonical_behavior_ratio_min"] < 1
    assert diagnostics["canonical_behavior_ratio_max"] > 1


def test_disallowed_logits_receive_exactly_zero_gradient():
    logits = torch.randn(1, 4, 8, requires_grad=True)
    labels = torch.tensor([[7, 1, 2, 3]])
    masks = torch.ones(1, 3)

    selected, entropy = restricted_action_log_probs_and_entropy(
        logits,
        labels,
        masks,
        allowed_token_ids=(1, 2, 3),
    )
    (selected.sum() + entropy.sum()).backward()

    assert logits.grad is not None
    disallowed = torch.tensor([0, 4, 5, 6, 7])
    torch.testing.assert_close(
        logits.grad.index_select(-1, disallowed),
        torch.zeros(1, 4, len(disallowed)),
    )


def test_restricted_policy_rejects_an_illegal_active_action():
    with pytest.raises(ValueError, match="outside its action support"):
        restricted_action_log_probs_and_entropy(
            torch.zeros(1, 3, 8),
            torch.tensor([[7, 4, 2]]),
            torch.ones(1, 2),
            allowed_token_ids=(1, 2, 3),
        )


def test_restricted_policy_rejects_nonfinite_active_support_logits():
    logits = torch.zeros(1, 3, 8)
    logits[0, 0, 1] = torch.nan

    with pytest.raises(FloatingPointError, match="nonfinite"):
        restricted_action_log_probs_and_entropy(
            logits,
            torch.tensor([[7, 1, 2]]),
            torch.ones(1, 2),
            allowed_token_ids=(1, 2, 3),
        )


def test_three_way_restricted_policy_and_prefix_ratio_match_exact_tree_entropy():
    old_root = torch.tensor([-0.4, 0.3, 0.8])
    old_second = torch.tensor([[0.5, -0.2, 0.1], [-0.7, 0.4, 0.2], [0.1, 0.6, -0.3]])
    old_third = torch.tensor(
        [
            [[0.3, -0.4, 0.5], [-0.2, 0.8, 0.1], [0.7, 0.0, -0.5]],
            [[-0.6, 0.2, 0.9], [0.1, 0.4, -0.3], [0.5, -0.7, 0.2]],
            [[0.8, -0.1, 0.0], [-0.4, 0.6, 0.3], [0.2, 0.5, -0.8]],
        ]
    )
    new_root = torch.tensor([0.2, -0.5, 0.9], requires_grad=True)
    new_second = torch.tensor(
        [[-0.1, 0.7, 0.3], [0.8, -0.6, 0.2], [0.4, 0.1, -0.8]],
        requires_grad=True,
    )
    new_third = torch.tensor(
        [
            [[-0.3, 0.2, 0.8], [0.6, -0.5, 0.1], [0.0, 0.9, -0.4]],
            [[0.4, 0.7, -0.2], [-0.8, 0.3, 0.5], [0.2, -0.1, 0.6]],
            [[-0.5, 0.8, 0.2], [0.3, -0.7, 0.9], [0.6, 0.1, -0.4]],
        ],
        requires_grad=True,
    )

    new_rows = []
    old_rows = []
    labels = []
    for first in range(3):
        for second in range(3):
            for third in range(3):
                # A fourth shifted position is inactive padding, exercising
                # the response mask as well as the full 27-leaf E14 tree.
                new_rows.append(
                    torch.stack(
                        (
                            new_root,
                            new_second[first],
                            new_third[first, second],
                            torch.zeros(3),
                            torch.zeros(3),
                        )
                    )
                )
                old_rows.append(
                    torch.stack(
                        (
                            old_root,
                            old_second[first],
                            old_third[first, second],
                            torch.zeros(3),
                            torch.zeros(3),
                        )
                    )
                )
                labels.append([0, first, second, third, 0])
    masks = torch.tensor([[1.0, 1.0, 1.0, 0.0]]).expand(27, -1)
    label_tensor = torch.tensor(labels)
    new_logps, local_entropy = restricted_action_log_probs_and_entropy(
        torch.stack(new_rows),
        label_tensor,
        masks,
        allowed_token_ids=(0, 1, 2),
    )
    old_logps, _ = restricted_action_log_probs_and_entropy(
        torch.stack(old_rows),
        label_tensor,
        masks,
        allowed_token_ids=(0, 1, 2),
    )
    surrogate, *_ = prefix_ratio_maxent_surrogate(
        local_entropy,
        new_logps,
        old_logps,
        masks,
        normalization_constant=1.0,
        cliprange=None,
    )
    old_trajectory_probability = torch.exp(old_logps.sum(dim=1))
    estimated_entropy = (old_trajectory_probability * surrogate).sum()
    torch.testing.assert_close(local_entropy[:, -1], torch.zeros(27))

    new_root_probability = torch.softmax(new_root, dim=0)
    root_entropy = -(new_root_probability * torch.log(new_root_probability)).sum()
    second_probabilities = torch.softmax(new_second, dim=1)
    second_entropies = -(second_probabilities * torch.log(second_probabilities)).sum(
        dim=1
    )
    third_probabilities = torch.softmax(new_third, dim=2)
    third_entropies = -(third_probabilities * torch.log(third_probabilities)).sum(dim=2)
    exact_entropy = (
        root_entropy
        + (new_root_probability * second_entropies).sum()
        + (new_root_probability[:, None] * second_probabilities * third_entropies).sum()
    )

    torch.testing.assert_close(estimated_entropy, exact_entropy, atol=1e-6, rtol=0)
    estimated_gradient = torch.autograd.grad(
        estimated_entropy, (new_root, new_second, new_third), retain_graph=True
    )
    exact_gradient = torch.autograd.grad(
        exact_entropy, (new_root, new_second, new_third)
    )
    for estimated, exact in zip(estimated_gradient, exact_gradient):
        torch.testing.assert_close(estimated, exact, atol=1e-6, rtol=0)


def test_alpha_zero_preserves_the_integrated_constrained_policy_loss_and_gradient():
    base_logits = torch.tensor(
        [[[0.2, -0.3, 0.7], [0.8, 0.1, -0.4], [-0.2, 0.6, 0.4], [0.0, 0.0, 0.0]]],
        requires_grad=True,
    )
    treatment_logits = base_logits.detach().clone().requires_grad_(True)
    labels = torch.tensor([[0, 2, 0, 1]])
    masks = torch.ones(1, 3)
    advantages = torch.tensor([[0.4, 0.4, 0.4]])

    base_logps, _ = restricted_action_log_probs_and_entropy(
        base_logits, labels, masks, allowed_token_ids=(0, 1, 2)
    )
    treatment_logps, treatment_entropy = restricted_action_log_probs_and_entropy(
        treatment_logits, labels, masks, allowed_token_ids=(0, 1, 2)
    )
    reward_loss = -(base_logps * advantages * masks).sum() / 192
    treatment_reward_loss = -(treatment_logps * advantages * masks).sum() / 192
    entropy_surrogate, *_ = prefix_ratio_maxent_surrogate(
        treatment_entropy,
        treatment_logps,
        treatment_logps.detach(),
        masks,
        normalization_constant=1.0,
        cliprange=None,
    )
    treatment_loss = treatment_reward_loss + standard_maxent_loss(
        entropy_surrogate.mean(),
        alpha=0.0,
        reward_estimator_scale=15 / 16,
        update_normalizer=192,
    )

    base_gradient = torch.autograd.grad(reward_loss, base_logits)[0]
    treatment_gradient = torch.autograd.grad(treatment_loss, treatment_logits)[0]
    torch.testing.assert_close(treatment_loss, reward_loss.detach())
    torch.testing.assert_close(treatment_gradient, base_gradient)
