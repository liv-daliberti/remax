"""SetPO set-level diversity shaping over kernelized trajectory similarity.

This implements the plug-in advantage term of Li et al. (arXiv:2602.01062).
For a rollout group ``Omega`` the set-level functional is

    D(Omega) = (1/|Omega|) sum_i g( (1/|Omega|) sum_j k(y_i, y_j) )

with ``g(x) = -log(1 + x)`` and ``k`` a cosine similarity between embedded
trajectories. Each row's leave-one-out marginal contribution is

    s_i = D(Omega) - D(Omega \\ {o_i})

and the shaped advantage is ``Ahat_i = Abar_i + lambda * s_i``.

Three details of this port are declared rather than inferred.

The paper specifies "a pretrained embedding model" without naming one. These
cells use MiniLM through the training environment's existing ``transformers``
install, mean-pooled over the attention mask --- the same family DQO reports,
so the two embedding-kernel methods in the comparison remain commensurable.
The choice is frozen per cohort and pinned by digest.

Cosine similarity is clamped onto ``[0, 1]``. ``g`` is undefined at ``x <= -1``
and a negative kernel mass would make a maximally dissimilar pair score as more
than empty, which is not what a similarity kernel is for.

The marginal term is added without group centering, exactly as published. That
leaves a non-zero group mean in the shaped advantage, so the mean and spread
are reported per batch rather than silently absorbed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class SetPODiagnostics:
    """Batch diagnostics for set-level diversity shaping."""

    groups: int
    rows: int
    kernel_mean: float
    kernel_min: float
    kernel_max: float
    set_diversity_mean: float
    marginal_mean: float
    marginal_min: float
    marginal_max: float
    marginal_abs_mean: float
    #: Group-mean of the added term. The published rule does not centre, so
    #: this is the bias the shaped advantage carries relative to the control.
    group_mean_shift_abs_max: float
    base_advantage_rms: float
    shaped_advantage_rms: float


def setpo_marginal_contributions(
    embeddings: torch.Tensor,
    *,
    num_samples: int,
) -> torch.Tensor:
    """Return the leave-one-out marginal diversity contribution of every row.

    ``embeddings`` are one vector per rollout row, ordered so that each
    consecutive block of ``num_samples`` rows is one prompt group.
    """

    if num_samples <= 1:
        raise ValueError("num_samples must be greater than one")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [rows, features]")
    rows = int(embeddings.size(0))
    if rows == 0 or rows % num_samples:
        raise ValueError("embeddings must contain complete rollout groups")
    if not bool(torch.isfinite(embeddings).all()):
        raise ValueError("embeddings must be finite")

    work = embeddings.detach().to(torch.float32)
    normalized = torch.nn.functional.normalize(work, dim=1, eps=1e-8)
    grouped = normalized.reshape(-1, num_samples, normalized.size(1))
    # [groups, G, G] cosine similarity, clamped onto a similarity kernel.
    kernel = torch.clamp(torch.bmm(grouped, grouped.transpose(1, 2)), 0.0, 1.0)

    size = float(num_samples)
    row_sums = kernel.sum(dim=2)
    full_mass = row_sums / size
    full_diversity = _g(full_mass).mean(dim=1, keepdim=True)

    # Removing column r drops K[i, r] from row i's mass and renormalizes over
    # the G-1 rows that remain; row r itself leaves the outer average.
    reduced_sums = row_sums.unsqueeze(2) - kernel
    reduced_mass = reduced_sums / (size - 1.0)
    reduced_terms = _g(reduced_mass)
    # Zero the removed row's own term before averaging over the G-1 survivors.
    eye = torch.eye(num_samples, device=kernel.device, dtype=kernel.dtype)
    survivors = reduced_terms * (1.0 - eye).unsqueeze(0)
    reduced_diversity = survivors.sum(dim=1) / (size - 1.0)

    marginals = full_diversity - reduced_diversity
    if not bool(torch.isfinite(marginals).all()):
        raise ValueError("SetPO produced a non-finite marginal contribution")
    return marginals.reshape(-1)


def shape_setpo_advantages(
    advantages: torch.Tensor,
    embeddings: torch.Tensor,
    *,
    num_samples: int,
    coefficient: float,
) -> tuple[torch.Tensor, SetPODiagnostics]:
    """Return ``Abar + lambda * s`` and the audit diagnostics."""

    coefficient = float(coefficient)
    if not math.isfinite(coefficient) or coefficient < 0.0:
        raise ValueError("coefficient must be finite and non-negative")

    original_shape = advantages.shape
    base = advantages.detach().reshape(-1)
    marginals = setpo_marginal_contributions(
        embeddings, num_samples=num_samples
    )
    if marginals.numel() != base.numel():
        raise ValueError("SetPO embeddings and advantages must agree in rows")

    term = coefficient * marginals.to(dtype=base.dtype, device=base.device)
    shaped = base + term

    grouped_term = term.reshape(-1, num_samples)
    normalized = torch.nn.functional.normalize(
        embeddings.detach().to(torch.float32), dim=1, eps=1e-8
    )
    grouped = normalized.reshape(-1, num_samples, normalized.size(1))
    kernel = torch.clamp(torch.bmm(grouped, grouped.transpose(1, 2)), 0.0, 1.0)
    mass = kernel.sum(dim=2) / float(num_samples)

    diagnostics = SetPODiagnostics(
        groups=int(base.numel() // num_samples),
        rows=int(base.numel()),
        kernel_mean=float(kernel.mean()),
        kernel_min=float(kernel.min()),
        kernel_max=float(kernel.max()),
        set_diversity_mean=float(_g(mass).mean()),
        marginal_mean=float(marginals.mean()),
        marginal_min=float(marginals.min()),
        marginal_max=float(marginals.max()),
        marginal_abs_mean=float(marginals.abs().mean()),
        group_mean_shift_abs_max=float(grouped_term.mean(dim=1).abs().max()),
        base_advantage_rms=float(torch.sqrt(base.float().square().mean())),
        shaped_advantage_rms=float(torch.sqrt(shaped.float().square().mean())),
    )
    return shaped.reshape(original_shape), diagnostics


def _g(mass: torch.Tensor) -> torch.Tensor:
    """The paper's concave diversity transform ``g(x) = -log(1 + x)``."""

    return -torch.log1p(mass)
