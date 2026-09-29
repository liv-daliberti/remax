from __future__ import annotations

import pytest
import torch

from remax.maxrl import binary_maxrl_advantages


def test_binary_maxrl_matches_success_normalization_and_centering():
    rewards = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    observed = binary_maxrl_advantages(rewards)
    expected = torch.tensor(
        [
            [3.0, -1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    torch.testing.assert_close(observed, expected)


def test_binary_maxrl_rejects_continuous_or_nonfinite_rewards():
    with pytest.raises(ValueError, match="rewards in"):
        binary_maxrl_advantages(torch.tensor([[0.0, 0.5]]))
    with pytest.raises(ValueError, match="finite"):
        binary_maxrl_advantages(torch.tensor([[0.0, float("nan")]]))
