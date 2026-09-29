"""Haarnoja-style entropy-dual control for xDr's aggregation temperature.

This is not Soft Actor-Critic and it does not add token entropy to the policy
loss.  It borrows SAC's automatic-temperature *controller*: learn a positive
dual variable by minimizing the signed target-entropy objective in log space.

xDr's aggregation temperature acts in the opposite direction from SAC's
entropy coefficient: smaller ``tau`` makes the answer-level credit assignment
more selective.  We therefore define a dimensionless dual strength

``alpha_xdr = base_tau / tau``

and update ``log(alpha_xdr)`` with Adam.  When observed entropy is below the
target, the learned dual strength rises and the next xDr temperature falls;
when entropy is above target, the update reverses.  This signed, accumulated
dual update is deliberately distinct from the maintained one-sided
proportional controller in :mod:`xdr_tau_controller`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass
class XdrSacDualController:
    """Bounded scalar Adam optimizer for xDr inverse temperature.

    The first ``warmup_steps`` observations run at ``base_tau`` and calibrate
    ``target_entropy = target_ratio * mean(warmup_entropy)``.  Thereafter the
    controller minimizes the SAC dual objective

    ``J(alpha) = alpha * (observed_entropy - target_entropy)``

    with respect to ``log(alpha)``.  The resulting temperature is
    ``base_tau / alpha``, projected into ``[min_tau, max_tau]``.  Bounds keep
    this exploratory transfer of SAC's controller numerically finite; unlike
    the proportional controller, ``max_tau > base_tau`` permits signed
    relaxation when entropy is above target.
    """

    base_tau: float
    min_tau: float
    max_tau: float
    target_ratio: float
    warmup_steps: int
    alpha_lr: float
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8
    current_tau: float | None = None
    target_entropy: float | None = None
    observation_count: int = 0
    log_alpha: float = 0.0
    _warmup_entropy_sum: float = 0.0
    _adam_m: float = 0.0
    _adam_v: float = 0.0
    _adam_step: int = 0

    observation_metric_key = "xdr_sac_dual_observed_entropy"

    def __post_init__(self) -> None:
        for name, value in (
            ("base_tau", self.base_tau),
            ("min_tau", self.min_tau),
            ("max_tau", self.max_tau),
            ("target_ratio", self.target_ratio),
            ("alpha_lr", self.alpha_lr),
            ("beta1", self.beta1),
            ("beta2", self.beta2),
            ("adam_eps", self.adam_eps),
            ("log_alpha", self.log_alpha),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.base_tau <= 0:
            raise ValueError("base_tau must be positive")
        if self.min_tau <= 0 or self.min_tau > self.base_tau:
            raise ValueError("min_tau must be in (0, base_tau]")
        if self.max_tau < self.base_tau:
            raise ValueError("max_tau must be at least base_tau")
        if not 0 < self.target_ratio <= 1:
            raise ValueError("target_ratio must be in (0, 1]")
        if int(self.warmup_steps) <= 0:
            raise ValueError("warmup_steps must be positive")
        if self.alpha_lr <= 0:
            raise ValueError("alpha_lr must be positive")
        if not 0 <= self.beta1 < 1:
            raise ValueError("beta1 must be in [0, 1)")
        if not 0 <= self.beta2 < 1:
            raise ValueError("beta2 must be in [0, 1)")
        if self.adam_eps <= 0:
            raise ValueError("adam_eps must be positive")
        self._project_log_alpha()
        if self.current_tau is None:
            self.current_tau = float(self.base_tau)

    @property
    def alpha(self) -> float:
        return math.exp(self.log_alpha)

    def _project_log_alpha(self) -> None:
        min_log_alpha = math.log(self.base_tau / self.max_tau)
        max_log_alpha = math.log(self.base_tau / self.min_tau)
        self.log_alpha = min(max_log_alpha, max(min_log_alpha, self.log_alpha))

    def observe(self, entropy: float) -> dict[str, float]:
        """Take one stochastic dual step from a global entropy observation."""

        value = float(entropy)
        if not math.isfinite(value) or value < 0:
            raise ValueError("entropy observation must be finite and non-negative")

        self.observation_count += 1
        if self.observation_count <= self.warmup_steps:
            self._warmup_entropy_sum += value
            if self.observation_count == self.warmup_steps:
                warmup_mean = self._warmup_entropy_sum / float(self.warmup_steps)
                self.target_entropy = self.target_ratio * warmup_mean

        entropy_error = 0.0
        alpha_loss = 0.0
        alpha_gradient = 0.0
        if self.target_entropy is not None:
            entropy_error = value - self.target_entropy
            alpha_before = self.alpha
            # Haarnoja et al.'s J(alpha)=alpha*(H-H_target), optimized through
            # alpha=exp(log_alpha), gives this log-space gradient.
            alpha_loss = alpha_before * entropy_error
            alpha_gradient = alpha_before * entropy_error
            self._adam_step += 1
            self._adam_m = (
                self.beta1 * self._adam_m + (1.0 - self.beta1) * alpha_gradient
            )
            self._adam_v = (
                self.beta2 * self._adam_v
                + (1.0 - self.beta2) * alpha_gradient * alpha_gradient
            )
            m_hat = self._adam_m / (1.0 - self.beta1**self._adam_step)
            v_hat = self._adam_v / (1.0 - self.beta2**self._adam_step)
            self.log_alpha -= self.alpha_lr * m_hat / (
                math.sqrt(v_hat) + self.adam_eps
            )
            self._project_log_alpha()
            self.current_tau = min(
                self.max_tau,
                max(self.min_tau, self.base_tau / self.alpha),
            )
        else:
            self.log_alpha = 0.0
            self.current_tau = float(self.base_tau)

        diagnostics = {
            "xdr_sac_dual_observed_entropy": value,
            "xdr_sac_dual_entropy_error": float(entropy_error),
            "xdr_sac_dual_alpha_loss": float(alpha_loss),
            "xdr_sac_dual_alpha_gradient": float(alpha_gradient),
            "xdr_sac_dual_alpha": float(self.alpha),
            "xdr_sac_dual_log_alpha": float(self.log_alpha),
            "xdr_sac_dual_next_tau": float(self.current_tau),
            "xdr_sac_dual_observations": float(self.observation_count),
            "xdr_sac_dual_optimizer_steps": float(self._adam_step),
        }
        if self.target_entropy is not None:
            diagnostics["xdr_sac_dual_target_entropy"] = float(
                self.target_entropy
            )
        return diagnostics

    def state_dict(self) -> dict[str, Any]:
        return {
            "controller_kind": "xdr_sac_dual",
            "current_tau": float(self.current_tau),
            "target_entropy": self.target_entropy,
            "observation_count": int(self.observation_count),
            "log_alpha": float(self.log_alpha),
            "warmup_entropy_sum": float(self._warmup_entropy_sum),
            "adam_m": float(self._adam_m),
            "adam_v": float(self._adam_v),
            "adam_step": int(self._adam_step),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("controller state must be a dictionary")
        controller_kind = state.get("controller_kind")
        if controller_kind not in (None, "xdr_sac_dual"):
            raise ValueError("checkpoint contains a different tau controller")
        target_entropy = state.get("target_entropy")
        self.target_entropy = (
            None if target_entropy is None else float(target_entropy)
        )
        self.observation_count = max(int(state.get("observation_count", 0)), 0)
        self.log_alpha = float(state.get("log_alpha", 0.0))
        self._warmup_entropy_sum = float(state.get("warmup_entropy_sum", 0.0))
        self._adam_m = float(state.get("adam_m", 0.0))
        self._adam_v = max(float(state.get("adam_v", 0.0)), 0.0)
        self._adam_step = max(int(state.get("adam_step", 0)), 0)
        self._project_log_alpha()
        self.current_tau = min(
            self.max_tau,
            max(self.min_tau, self.base_tau / self.alpha),
        )
