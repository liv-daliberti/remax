"""Small tensor helpers shared by the Dr.GRPO learner."""

from __future__ import annotations

import torch


def cap_last_valid_token_pos_for_zero_advantage(
    *, prompt_len: int, last_valid_token_pos: int, response_token_budget: int
) -> int:
    """Trim a zero-signal row without changing optimizer cadence."""

    safe_last = max(int(last_valid_token_pos), 0)
    if safe_last <= 0:
        return 0
    minimum_last = max(int(prompt_len), 0) + 1
    if safe_last <= minimum_last:
        return safe_last
    return min(safe_last, max(int(prompt_len), 0) + max(int(response_token_budget), 1))


def gather_selected_logps_chunked(
    logits: torch.Tensor,
    labels: torch.Tensor,
    response_masks: torch.Tensor,
    *,
    token_chunk_size: int,
) -> torch.Tensor:
    """Return selected-token log probabilities without a full log-softmax."""

    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    if logits.shape[:-1] != labels.shape:
        raise ValueError("logits and labels must agree on batch/sequence shape")

    shifted_labels = labels[:, 1:].clone()
    shifted_logits = logits[:, :-1, :]
    safe_chunk = min(token_chunk_size, max(int(shifted_logits.size(1)), 1))
    selected_logps = []
    for start in range(0, int(shifted_logits.size(1)), safe_chunk):
        stop = min(start + safe_chunk, int(shifted_logits.size(1)))
        chunk_logits = shifted_logits[:, start:stop, :]
        chunk_labels = shifted_labels[:, start:stop]
        chunk_masks = response_masks[:, start:stop].to(torch.bool)
        chunk_labels = chunk_labels.masked_fill(~chunk_masks, 0)
        chunk_logits = chunk_logits.float()
        chunk_logps = torch.gather(
            chunk_logits, dim=2, index=chunk_labels.unsqueeze(2)
        ).squeeze(2) - torch.logsumexp(chunk_logits, dim=-1)
        selected_logps.append(
            torch.where(chunk_masks, chunk_logps, torch.zeros_like(chunk_logps))
        )
    return torch.cat(selected_logps, dim=1)
