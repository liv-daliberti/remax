import pytest
import torch

from remax.seed_weights import compute_seed_row_weights


def _weights(logps, keys, *, g=4, alpha=1.0, masks=None):
    return compute_seed_row_weights(
        torch.tensor(logps, dtype=torch.float32),
        keys,
        num_samples=g,
        alpha=alpha,
        loss_masks=None if masks is None else torch.tensor(masks),
    )


def test_single_cluster_prompt_gets_scale_one():
    # All completions share one answer: zero semantic entropy, s_x = 1.
    w = _weights([-1.0, -2.0, -0.5, -1.5], ["a", "a", "a", "a"])
    assert torch.allclose(w, torch.ones(4))


def test_uniform_clusters_give_max_entropy_scale():
    # Four equally likely singleton clusters: H = log G, so with
    # alpha_eff = alpha/log G the scale is exactly 1/(1 + alpha).
    w = _weights([-1.0, -1.0, -1.0, -1.0], ["a", "b", "c", "d"], alpha=1.0)
    assert torch.allclose(w, torch.full((4,), 0.5))


def test_scale_is_per_prompt_uniform_within_group():
    w = _weights(
        [-1.0, -1.0, -1.0, -1.0, -1.0, -2.0, -0.5, -1.5],
        ["a", "b", "c", "d", "x", "x", "x", "x"],
        g=4,
    )
    assert torch.allclose(w[:4], w[0].expand(4))
    assert torch.allclose(w[4:], torch.ones(4))
    assert w[0] < 1.0


def test_none_keys_form_singleton_clusters():
    w_none = _weights([-1.0, -1.0, -1.0, -1.0], [None, None, None, None])
    w_distinct = _weights([-1.0, -1.0, -1.0, -1.0], ["a", "b", "c", "d"])
    assert torch.allclose(w_none, w_distinct)


def test_likelihood_mass_drives_entropy():
    # One dominant cluster (much higher likelihood) has lower entropy than
    # evenly split clusters, hence a scale closer to 1.
    w_skew = _weights([5.0, -5.0, -5.0, -5.0], ["a", "b", "b", "b"])
    w_even = _weights([-1.0, -1.0, -1.0, -1.0], ["a", "b", "b", "b"])
    assert w_skew[0] > w_even[0]


def test_alpha_zero_and_g1_disable():
    assert torch.equal(
        _weights([-1.0, -1.0, -1.0, -1.0], ["a", "b", "c", "d"], alpha=0.0),
        torch.ones(4),
    )
    assert torch.equal(
        compute_seed_row_weights(
            torch.tensor([-1.0]), ["a"], num_samples=1, alpha=1.0
        ),
        torch.ones(1),
    )


def test_masked_rows_excluded_from_cluster_mass():
    # Masking the only row of cluster "b" leaves a single-cluster prompt.
    w = _weights(
        [-1.0, -1.0, -1.0, -1.0],
        ["a", "a", "a", "b"],
        masks=[1.0, 1.0, 1.0, 0.0],
    )
    assert torch.allclose(w, torch.ones(4))


def test_detached_and_validated():
    logps = torch.tensor([-1.0, -1.0, -1.0, -1.0], requires_grad=True)
    w = compute_seed_row_weights(logps, ["a", "b", "c", "d"], num_samples=4, alpha=1.0)
    assert not w.requires_grad
    with pytest.raises(ValueError):
        compute_seed_row_weights(
            torch.tensor([-1.0, -1.0]), ["a"], num_samples=2, alpha=1.0
        )
    with pytest.raises(ValueError):
        compute_seed_row_weights(
            torch.tensor([-1.0, -1.0]), ["a", "b"], num_samples=2, alpha=-1.0
        )
