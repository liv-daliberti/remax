"""Adaptive entropy control for an online verified canonical-outcome bank.

Two deliberately separate controller contracts live here:

* the historical Haarnoja dual observes the exact, post-update entropy of the
  empirical distribution over each prompt's validated canonical outcomes,
  normalized by its current maximum:

    H(q_x) / log |B_x^+|.

  Singleton banks are not observations: their normalized entropy is undefined,
  and treating it as zero would spuriously drive alpha upward before the
  learner has discovered a second valid outcome;
* the policy-entropy controller observes the model's own masked-mean token
  entropy. It keeps the canonical coefficient at its registered reference dose
  during a per-run calibration window, then scales it in inverse proportion to
  the entropy EMA relative to that run's warmup mean. Thus falling model
  entropy raises canonical pressure. It has no optimizer and no alpha
  projection.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass
class OnlineCanonicalPolicyEntropyController:
    """Projection-free canonical alpha from the model's token uncertainty."""

    base_alpha: float
    warmup_steps: int
    ema_decay: float = 0.9
    current_alpha: float | None = None
    entropy_ema: float | None = None
    reference_entropy: float | None = None
    observation_count: int = 0
    _warmup_entropy_sum: float = 0.0

    entropy_units: str = "policy_token_entropy_nats_masked_mean_v1"
    observation_metric_key: str = "entropy"

    def __post_init__(self) -> None:
        for name, value in (
            ("base_alpha", self.base_alpha),
            ("ema_decay", self.ema_decay),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.base_alpha <= 0:
            raise ValueError("base_alpha must be positive")
        if int(self.warmup_steps) <= 0:
            raise ValueError("warmup_steps must be positive")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.current_alpha is None:
            self.current_alpha = float(self.base_alpha)
        self._validate_alpha(self.current_alpha)
        if self.entropy_ema is not None:
            self.entropy_ema = self._validate_entropy(
                self.entropy_ema, name="entropy_ema"
            )
        if self.reference_entropy is not None:
            self.reference_entropy = self._validate_entropy(
                self.reference_entropy,
                name="reference_entropy",
                positive=True,
            )

    def _validate_alpha(self, value: float) -> float:
        alpha = float(value)
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("current_alpha must be finite and non-negative")
        return alpha

    @staticmethod
    def _validate_entropy(
        value: float, *, name: str, positive: bool = False
    ) -> float:
        entropy = float(value)
        if (
            not math.isfinite(entropy)
            or entropy < 0
            or (positive and entropy <= 0)
        ):
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(f"{name} must be finite and {qualifier}")
        return entropy

    def observe(self, entropy: float) -> dict[str, float]:
        """Consume a global policy-entropy score for the next rollout round."""

        value = self._validate_entropy(entropy, name="entropy observation")
        alpha_before = float(self.current_alpha)
        self.observation_count += 1
        if self.entropy_ema is None:
            self.entropy_ema = value
        else:
            self.entropy_ema = (
                self.ema_decay * self.entropy_ema
                + (1.0 - self.ema_decay) * value
            )

        if self.observation_count <= self.warmup_steps:
            self._warmup_entropy_sum += value
            self.current_alpha = float(self.base_alpha)
            if self.observation_count == self.warmup_steps:
                self.reference_entropy = (
                    self._warmup_entropy_sum / float(self.warmup_steps)
                )
                if self.reference_entropy <= 0:
                    raise ValueError(
                        "policy-entropy warmup mean must be positive"
                    )
            normalized_score = 1.0
        else:
            if self.reference_entropy is None or self.reference_entropy <= 0:
                raise RuntimeError(
                    "policy-entropy adaptation lacks a positive warmup reference"
                )
            if self.entropy_ema <= 0:
                raise ValueError(
                    "policy-entropy EMA must remain positive for unbounded "
                    "inverse adaptation"
                )
            normalized_score = self.reference_entropy / self.entropy_ema
            self.current_alpha = self.base_alpha * normalized_score
            self._validate_alpha(self.current_alpha)

        diagnostics = {
            "online_canonical_policy_entropy_observed": value,
            "online_canonical_policy_entropy_ema": float(self.entropy_ema),
            "online_canonical_policy_entropy_normalized_score": float(
                normalized_score
            ),
            "online_canonical_policy_entropy_alpha_before": alpha_before,
            "online_canonical_policy_entropy_next_alpha": float(
                self.current_alpha
            ),
            "online_canonical_policy_entropy_observations": float(
                self.observation_count
            ),
            "online_canonical_policy_entropy_warmup_complete": float(
                self.observation_count >= self.warmup_steps
            ),
        }
        if self.reference_entropy is not None:
            diagnostics[
                "online_canonical_policy_entropy_reference"
            ] = float(self.reference_entropy)
        return diagnostics

    def state_dict(self) -> dict[str, Any]:
        return {
            "controller_kind": "online_canonical_policy_entropy",
            "controller_rule": (
                "unprojected_warmup_inverse_policy_entropy_alpha_v2"
            ),
            "entropy_units": self.entropy_units,
            "base_alpha": float(self.base_alpha),
            "warmup_steps": int(self.warmup_steps),
            "ema_decay": float(self.ema_decay),
            "current_alpha": float(self.current_alpha),
            "entropy_ema": self.entropy_ema,
            "reference_entropy": self.reference_entropy,
            "observation_count": int(self.observation_count),
            "warmup_entropy_sum": float(self._warmup_entropy_sum),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError(
                "online canonical policy-entropy state must be a dictionary"
            )
        if state.get("controller_kind") != (
            "online_canonical_policy_entropy"
        ):
            raise ValueError(
                "checkpoint contains a different online canonical controller"
            )
        if state.get("controller_rule") != (
            "unprojected_warmup_inverse_policy_entropy_alpha_v2"
        ):
            raise ValueError(
                "checkpoint uses an incompatible policy-entropy rule"
            )
        if state.get("entropy_units") != self.entropy_units:
            raise ValueError(
                "checkpoint uses incompatible policy-entropy units"
            )
        for name in ("base_alpha", "ema_decay"):
            saved = float(state.get(name, float("nan")))
            configured = float(getattr(self, name))
            if not math.isclose(
                saved, configured, rel_tol=1e-9, abs_tol=1e-12
            ):
                raise ValueError(f"resume mismatch for {name}")
        if int(state.get("warmup_steps", -1)) != int(self.warmup_steps):
            raise ValueError("resume mismatch for warmup_steps")

        self.current_alpha = self._validate_alpha(
            state.get("current_alpha")
        )
        entropy_ema = state.get("entropy_ema")
        self.entropy_ema = (
            None
            if entropy_ema is None
            else self._validate_entropy(entropy_ema, name="entropy_ema")
        )
        reference_entropy = state.get("reference_entropy")
        self.reference_entropy = (
            None
            if reference_entropy is None
            else self._validate_entropy(
                reference_entropy,
                name="reference_entropy",
                positive=True,
            )
        )
        self.observation_count = int(state.get("observation_count", -1))
        self._warmup_entropy_sum = float(
            state.get("warmup_entropy_sum", float("nan"))
        )
        if (
            self.observation_count < 0
            or not math.isfinite(self._warmup_entropy_sum)
            or self._warmup_entropy_sum < 0
        ):
            raise ValueError(
                "checkpoint contains invalid policy-entropy progress"
            )
        if (
            self.observation_count >= self.warmup_steps
            and self.reference_entropy is None
        ):
            raise ValueError(
                "completed policy-entropy warmup lacks its reference"
            )


@dataclass
class OnlineCanonicalDualController:
    """Haarnoja-style log-alpha dual descent on normalized bank entropy."""

    base_alpha: float
    min_alpha: float
    max_alpha: float
    target_ratio: float
    alpha_lr: float
    ema_decay: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8
    entropy_ema: float | None = None
    observation_count: int = 0
    log_alpha: float | None = None
    _adam_m: float = 0.0
    _adam_v: float = 0.0
    _adam_step: int = 0

    entropy_units: str = "verified_bank_entropy_over_log_support_v1"
    observation_metric_key: str = (
        "online_canonical_normalized_entropy_ratio_mean"
    )
    eligibility_metric_key: str = (
        "online_canonical_normalized_entropy_ratio_eligible_fraction"
    )

    def __post_init__(self) -> None:
        for name, value in (
            ("base_alpha", self.base_alpha),
            ("min_alpha", self.min_alpha),
            ("target_ratio", self.target_ratio),
            ("alpha_lr", self.alpha_lr),
            ("ema_decay", self.ema_decay),
            ("beta1", self.beta1),
            ("beta2", self.beta2),
            ("adam_eps", self.adam_eps),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if math.isnan(float(self.max_alpha)) or self.max_alpha <= 0:
            raise ValueError("max_alpha must be positive or +inf")
        if self.base_alpha <= 0:
            raise ValueError("base_alpha must be positive")
        if self.min_alpha <= 0 or self.min_alpha > self.base_alpha:
            raise ValueError("min_alpha must be in (0, base_alpha]")
        if self.max_alpha < self.base_alpha:
            raise ValueError("max_alpha must be at least base_alpha")
        if not 0 < self.target_ratio <= 1:
            raise ValueError("target_ratio must be in (0, 1]")
        if self.alpha_lr <= 0:
            raise ValueError("alpha_lr must be positive")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam betas must be in [0, 1)")
        if self.adam_eps <= 0:
            raise ValueError("adam_eps must be positive")
        if self.entropy_ema is not None:
            self.entropy_ema = self._validated_ratio(
                self.entropy_ema, name="entropy_ema"
            )
        if self.log_alpha is None:
            self.log_alpha = math.log(self.base_alpha)
        self._project_log_alpha()

    @staticmethod
    def _validated_ratio(value: float, *, name: str) -> float:
        ratio = float(value)
        if not math.isfinite(ratio) or ratio < 0 or ratio > 1 + 1e-9:
            raise ValueError(f"{name} must be finite and in [0, 1]")
        return min(ratio, 1.0)

    @property
    def current_alpha(self) -> float:
        return math.exp(float(self.log_alpha))

    def _project_log_alpha(self) -> None:
        self.log_alpha = min(
            math.log(self.max_alpha),
            max(math.log(self.min_alpha), float(self.log_alpha)),
        )

    def observe(self, normalized_entropy_ratio: float) -> dict[str, float]:
        """Consume one eligible global ratio and return next-batch diagnostics."""

        value = self._validated_ratio(
            normalized_entropy_ratio,
            name="normalized_entropy_ratio",
        )
        self.observation_count += 1
        if self.entropy_ema is None:
            self.entropy_ema = value
        else:
            self.entropy_ema = (
                self.ema_decay * self.entropy_ema
                + (1.0 - self.ema_decay) * value
            )

        entropy_error = self.entropy_ema - self.target_ratio
        alpha_before = self.current_alpha
        alpha_loss = alpha_before * entropy_error
        alpha_gradient = alpha_before * entropy_error
        self._adam_step += 1
        self._adam_m = (
            self.beta1 * self._adam_m
            + (1.0 - self.beta1) * alpha_gradient
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
        return {
            "online_canonical_dual_observed_normalized_entropy": value,
            "online_canonical_dual_normalized_entropy_ema": float(
                self.entropy_ema
            ),
            "online_canonical_dual_target_ratio": float(self.target_ratio),
            "online_canonical_dual_entropy_error": float(entropy_error),
            "online_canonical_dual_alpha_loss": float(alpha_loss),
            "online_canonical_dual_alpha_gradient": float(alpha_gradient),
            "online_canonical_dual_alpha_before": float(alpha_before),
            "online_canonical_dual_next_alpha": float(self.current_alpha),
            "online_canonical_dual_log_alpha": float(self.log_alpha),
            "online_canonical_dual_observations": float(
                self.observation_count
            ),
            "online_canonical_dual_optimizer_steps": float(self._adam_step),
        }

    def idle_diagnostics(self) -> dict[str, float]:
        """Report state without inventing an observation for singleton banks."""

        diagnostics = {
            "online_canonical_dual_target_ratio": float(self.target_ratio),
            "online_canonical_dual_next_alpha": float(self.current_alpha),
            "online_canonical_dual_log_alpha": float(self.log_alpha),
            "online_canonical_dual_observations": float(
                self.observation_count
            ),
            "online_canonical_dual_optimizer_steps": float(self._adam_step),
            "online_canonical_dual_observation_skipped": 1.0,
        }
        if self.entropy_ema is not None:
            diagnostics[
                "online_canonical_dual_normalized_entropy_ema"
            ] = float(self.entropy_ema)
        return diagnostics

    def state_dict(self) -> dict[str, Any]:
        return {
            "controller_kind": "online_canonical_dual",
            "controller_rule": (
                "log_alpha_adam_postupdate_normalized_bank_entropy_v1"
            ),
            "entropy_units": self.entropy_units,
            "base_alpha": float(self.base_alpha),
            "min_alpha": float(self.min_alpha),
            "max_alpha": float(self.max_alpha),
            "target_ratio": float(self.target_ratio),
            "alpha_lr": float(self.alpha_lr),
            "ema_decay": float(self.ema_decay),
            "beta1": float(self.beta1),
            "beta2": float(self.beta2),
            "adam_eps": float(self.adam_eps),
            "entropy_ema": self.entropy_ema,
            "observation_count": int(self.observation_count),
            "log_alpha": float(self.log_alpha),
            "adam_m": float(self._adam_m),
            "adam_v": float(self._adam_v),
            "adam_step": int(self._adam_step),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("online canonical dual state must be a dictionary")
        if state.get("controller_kind") != "online_canonical_dual":
            raise ValueError(
                "checkpoint contains a different online canonical controller"
            )
        if state.get("controller_rule") != (
            "log_alpha_adam_postupdate_normalized_bank_entropy_v1"
        ):
            raise ValueError(
                "checkpoint uses an incompatible online canonical dual rule"
            )
        if state.get("entropy_units") != self.entropy_units:
            raise ValueError(
                "checkpoint uses incompatible online canonical entropy units"
            )
        for name in (
            "base_alpha",
            "min_alpha",
            "max_alpha",
            "target_ratio",
            "alpha_lr",
            "ema_decay",
            "beta1",
            "beta2",
            "adam_eps",
        ):
            try:
                saved = float(state[name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"online canonical dual state is missing valid {name}"
                ) from exc
            configured = float(getattr(self, name))
            finite_match = (
                math.isfinite(saved)
                and math.isfinite(configured)
                and math.isclose(
                    saved,
                    configured,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
            unbounded_match = saved == configured == math.inf
            if not (finite_match or unbounded_match):
                raise ValueError(
                    f"online canonical dual resume mismatch for {name}: "
                    f"saved={saved!r} configured={configured!r}"
                )

        raw_ema = state.get("entropy_ema")
        self.entropy_ema = (
            None
            if raw_ema is None
            else self._validated_ratio(raw_ema, name="entropy_ema")
        )
        self.observation_count = max(
            int(state.get("observation_count", 0)), 0
        )
        self.log_alpha = float(
            state.get("log_alpha", math.log(self.base_alpha))
        )
        self._adam_m = float(state.get("adam_m", 0.0))
        self._adam_v = max(float(state.get("adam_v", 0.0)), 0.0)
        self._adam_step = max(int(state.get("adam_step", 0)), 0)
        for name, value in (
            ("log_alpha", self.log_alpha),
            ("adam_m", self._adam_m),
            ("adam_v", self._adam_v),
        ):
            if not math.isfinite(value):
                raise ValueError(
                    f"checkpoint contains invalid online canonical {name}"
                )
        self._project_log_alpha()
