"""CPU smoke example for binary MaxRL advantages and verified replay gradients.

The inputs are illustrative scores, not outputs of a trained language model.
"""
import json
import torch
from remax.maxrl import binary_maxrl_advantages
from remax.canonical_replay import canonical_replay_uniform_verified_likelihood_loss

rewards = torch.tensor([[1.0, 0.0]])
advantages = binary_maxrl_advantages(rewards)
# Two retained modes for prompt A and one for prompt B.
scores = torch.tensor([-1.0, -3.0, -2.0], requires_grad=True)
result = canonical_replay_uniform_verified_likelihood_loss(scores, [2, 1])
result.loss.backward()
assert torch.allclose(advantages, torch.tensor([[1.0, -1.0]]))
assert scores.grad is not None
assert torch.allclose(scores.grad, torch.tensor([-0.25, -0.25, -0.5]))
print(json.dumps({
    'maxrl_advantages': advantages.tolist(),
    'replay_loss': result.loss.item(),
    'score_gradients': scores.grad.tolist(),
}))
