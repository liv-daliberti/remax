"""Per-prompt semantic-entropy scaling for SEED-Dr.GRPO.

SEED-GRPO (Chen et al., 2025, arXiv:2505.12346) measures prompt uncertainty
with the semantic entropy of the G sampled completions — clusters of
equivalent final answers, each cluster weighted by the policy's sequence
likelihood — and scales the whole prompt's update by a decreasing function of
that entropy. The paper's variant implemented here (SEED-Dr.GRPO) makes three
deliberate refinements, specified in the paper appendix:

  * sequence log-likelihoods are length-normalized before the cluster-mass
    softmax (a raw sum reintroduces exactly the length bias Dr.GRPO removes);
  * the proper Shannon entropy H = -sum_c p_c log p_c over cluster masses is
    used instead of the original Monte-Carlo estimator -1/K sum_k log p_k;
  * the scaling is s_x = (1 + alpha_eff * H)^{-1} with alpha_eff = alpha/log G,
    so s_x in (0, 1], s_x = 1 for a zero-entropy (single-cluster) prompt.

The scaling is per-prompt and uniform within the group: every row of prompt x
receives loss weight s_x, so alpha = 0 (off) reproduces Dr.GRPO exactly.
"""

from __future__ import annotations

import math

import torch


def compute_seed_row_weights(
    seq_logps_normalized: torch.Tensor,
    answer_keys: list[str | None],
    *,
    num_samples: int,
    alpha: float,
    loss_masks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-row weights s_x = (1 + (alpha/log G) * H_sem(x))^{-1}.

    Args:
        seq_logps_normalized: [N] length-normalized sequence log-probs under
            the current policy, prompt-major (G consecutive rows per prompt).
        answer_keys: length-N list of canonical final-answer keys; ``None``
            (unparseable output) forms its own singleton cluster, matching
            SEED-GRPO's treatment of distinct meanings.
        num_samples: group size G.
        alpha: non-negative sensitivity; 0 disables (returns all ones).
        loss_masks: optional [N] 0/1 mask; masked rows are excluded from the
            cluster-mass softmax (their returned weight is still s_x, but they
            are zeroed by the loss mask downstream).

    Returns:
        Detached float tensor [N] of per-row weights in (0, 1].
    """
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}")
    logps = seq_logps_normalized.detach().reshape(-1).float()
    n = logps.numel()
    if len(answer_keys) != n:
        raise ValueError(
            f"answer_keys ({len(answer_keys)}) must match rows ({n})"
        )
    if n % num_samples != 0:
        raise ValueError(
            f"batch of {n} rows is not divisible by group size {num_samples}"
        )
    if alpha == 0 or num_samples < 2:
        return torch.ones(n, dtype=torch.float32, device=logps.device)
    mask = None
    if loss_masks is not None:
        mask = loss_masks.detach().reshape(-1).float()
        if mask.numel() != n:
            raise ValueError("loss_masks must match rows")

    alpha_eff = float(alpha) / math.log(num_samples)
    weights = torch.ones(n, dtype=torch.float32, device=logps.device)
    for start in range(0, n, num_samples):
        rows = slice(start, start + num_samples)
        group_logps = logps[rows]
        if mask is not None:
            valid = mask[rows] > 0
        else:
            valid = torch.ones(
                num_samples, dtype=torch.bool, device=logps.device
            )
        if int(valid.sum()) == 0:
            continue
        probs = torch.softmax(
            group_logps[valid], dim=0
        )  # row masses over valid rows
        cluster_mass: dict[object, float] = {}
        valid_indices = torch.nonzero(valid, as_tuple=False).reshape(-1)
        for pos, row_index in enumerate(valid_indices.tolist()):
            key = answer_keys[start + row_index]
            cluster_id: object = key if key is not None else (
                "__singleton__",
                start + row_index,
            )
            cluster_mass[cluster_id] = cluster_mass.get(cluster_id, 0.0) + float(
                probs[pos]
            )
        entropy = 0.0
        for mass in cluster_mass.values():
            if mass > 0:
                entropy -= mass * math.log(mass)
        weights[rows] = 1.0 / (1.0 + alpha_eff * entropy)
    return weights
