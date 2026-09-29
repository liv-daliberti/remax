"""Controllers for direct on-policy completion-policy entropy.

Unlike xDr's aggregation-temperature controllers, these controllers act on
the coefficient that directly multiplies raw sequence entropy.  For free-form
policies their observation is an exclusive-prefix importance estimate.  A
canonical finite-action policy instead uses exact post-update enumeration of
the current prompt's action tree.  Checkpoint units distinguish these sensor
contracts so an approximate or legacy sensor cannot be resumed silently.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass
class MaxEntInverseController:
    """Projection-free inverse control of a direct entropy coefficient.

    The controller calibrates only from the policy entropy that its actor
    objective differentiates.  After warmup it applies the memoryless rule

    ``alpha = base_alpha * reference_entropy / entropy_ema``.

    There is no alpha optimizer, lower projection, upper projection, target
    derived from evaluation labels, or numerical epsilon.  A nonpositive
    post-warmup EMA fails closed because the requested inverse is undefined.
    """

    base_alpha: float
    warmup_steps: int
    ema_decay: float
    entropy_units: str = "sequence_nats_v1"
    observation_metric_key: str = "maxent_sequence_entropy"
    current_alpha: float | None = None
    entropy_ema: float | None = None
    reference_entropy: float | None = None
    observation_count: int = 0
    _warmup_entropy_sum: float = 0.0

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
        expected_metric = {
            # The masked-mean token entropy the manuscript reports as
            # `train/entropy`. A dual controller whose target was measured from
            # that telemetry has to observe the same estimator: the
            # conditional-content-token mean below is a different one and runs
            # an order of magnitude higher, so regulating on it against such a
            # target pins the coefficient at a bound and the policy never
            # reaches the entropy it is supposed to hold.
            "masked_mean_token_nats_v1": "entropy",
            "sequence_nats_v1": "maxent_sequence_entropy",
            "conditional_content_token_nats_mean_v1": (
                "maxent_conditional_token_entropy"
            ),
            "canonical_action_nats_exact_v1": (
                "canonical_exact_sequence_entropy"
            ),
        }.get(self.entropy_units)
        if expected_metric is None or self.observation_metric_key != expected_metric:
            raise ValueError("MaxEnt entropy units and observation metric are incompatible")
        if self.current_alpha is None:
            self.current_alpha = float(self.base_alpha)
        self.current_alpha = self._validate_alpha(self.current_alpha)
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

    @staticmethod
    def _validate_alpha(value: float) -> float:
        alpha = float(value)
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("current_alpha must be finite and positive")
        return alpha

    def observe(self, entropy: float) -> dict[str, float]:
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
                        "MaxEnt inverse warmup mean must be positive"
                    )
            multiplier = 1.0
        else:
            if self.reference_entropy is None or self.reference_entropy <= 0:
                raise RuntimeError(
                    "MaxEnt inverse adaptation lacks a positive warmup reference"
                )
            if self.entropy_ema <= 0:
                raise ValueError(
                    "MaxEnt inverse entropy EMA must remain positive"
                )
            multiplier = self.reference_entropy / self.entropy_ema
            self.current_alpha = self._validate_alpha(
                self.base_alpha * multiplier
            )

        diagnostics = {
            "maxent_inverse_observed_entropy": value,
            "maxent_inverse_entropy_ema": float(self.entropy_ema),
            "maxent_inverse_multiplier": float(multiplier),
            "maxent_inverse_alpha_before": alpha_before,
            "maxent_inverse_next_alpha": float(self.current_alpha),
            "maxent_inverse_observations": float(self.observation_count),
            "maxent_inverse_warmup_complete": float(
                self.observation_count >= self.warmup_steps
            ),
            "maxent_inverse_projection_active": 0.0,
        }
        if self.reference_entropy is not None:
            diagnostics["maxent_inverse_reference_entropy"] = float(
                self.reference_entropy
            )
        return diagnostics

    def state_dict(self) -> dict[str, Any]:
        return {
            "controller_kind": "maxent_inverse",
            "controller_rule": "unprojected_warmup_inverse_direct_entropy_v1",
            "entropy_units": self.entropy_units,
            "observation_metric_key": self.observation_metric_key,
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
            raise ValueError("controller state must be a dictionary")
        if state.get("controller_kind") != "maxent_inverse":
            raise ValueError("checkpoint contains a different MaxEnt controller")
        if state.get("controller_rule") != (
            "unprojected_warmup_inverse_direct_entropy_v1"
        ):
            raise ValueError("checkpoint uses an incompatible MaxEnt inverse rule")
        if state.get("entropy_units") != self.entropy_units:
            raise ValueError("checkpoint uses incompatible MaxEnt entropy units")
        if state.get("observation_metric_key") != self.observation_metric_key:
            raise ValueError("checkpoint uses an incompatible entropy metric")
        for name in ("base_alpha", "ema_decay"):
            saved = float(state.get(name, float("nan")))
            configured = float(getattr(self, name))
            if not math.isclose(
                saved,
                configured,
                rel_tol=1e-9,
                abs_tol=1e-12,
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
            raise ValueError("checkpoint contains invalid inverse-controller state")


@dataclass
class MaxEntProportionalController:
    """One-sided proportional control of the direct entropy coefficient."""

    base_alpha: float
    max_alpha: float
    target_ratio: float
    warmup_steps: int
    ema_decay: float
    gain: float
    configured_target_entropy: float = 0.0
    entropy_units: str = "sequence_nats_v1"
    observation_metric_key: str = "maxent_sequence_entropy"
    current_alpha: float | None = None
    entropy_ema: float | None = None
    target_entropy: float | None = None
    observation_count: int = 0
    _warmup_entropy_sum: float = 0.0

    def __post_init__(self) -> None:
        for name, value in (
            ("base_alpha", self.base_alpha),
            ("max_alpha", self.max_alpha),
            ("target_ratio", self.target_ratio),
            ("ema_decay", self.ema_decay),
            ("gain", self.gain),
            ("configured_target_entropy", self.configured_target_entropy),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.base_alpha <= 0:
            raise ValueError("base_alpha must be positive")
        if self.max_alpha < self.base_alpha:
            raise ValueError("max_alpha must be at least base_alpha")
        if not 0 < self.target_ratio <= 1:
            raise ValueError("target_ratio must be in (0, 1]")
        if int(self.warmup_steps) <= 0:
            raise ValueError("warmup_steps must be positive")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.gain <= 0:
            raise ValueError("gain must be positive")
        if self.configured_target_entropy < 0:
            raise ValueError("configured_target_entropy must be non-negative")
        expected_metric = {
            # The masked-mean token entropy the manuscript reports as
            # `train/entropy`. A dual controller whose target was measured from
            # that telemetry has to observe the same estimator: the
            # conditional-content-token mean below is a different one and runs
            # an order of magnitude higher, so regulating on it against such a
            # target pins the coefficient at a bound and the policy never
            # reaches the entropy it is supposed to hold.
            "masked_mean_token_nats_v1": "entropy",
            "sequence_nats_v1": "maxent_sequence_entropy",
            "conditional_content_token_nats_mean_v1": (
                "maxent_conditional_token_entropy"
            ),
            "canonical_action_nats_exact_v1": (
                "canonical_exact_sequence_entropy"
            ),
        }.get(self.entropy_units)
        if expected_metric is None or self.observation_metric_key != expected_metric:
            raise ValueError("MaxEnt entropy units and observation metric are incompatible")
        if self.current_alpha is None:
            self.current_alpha = float(self.base_alpha)
        if self.configured_target_entropy > 0:
            self.target_entropy = float(self.configured_target_entropy)

    def observe(self, entropy: float) -> dict[str, float]:
        value = float(entropy)
        if not math.isfinite(value) or value < 0:
            raise ValueError("entropy observation must be finite and non-negative")
        self.observation_count += 1
        if self.entropy_ema is None:
            self.entropy_ema = value
        else:
            self.entropy_ema = (
                self.ema_decay * self.entropy_ema
                + (1.0 - self.ema_decay) * value
            )
        if (
            self.configured_target_entropy == 0
            and self.observation_count <= self.warmup_steps
        ):
            self._warmup_entropy_sum += value
            if self.observation_count == self.warmup_steps:
                self.target_entropy = self.target_ratio * (
                    self._warmup_entropy_sum / float(self.warmup_steps)
                )

        deficit = 0.0
        if (
            self.target_entropy is not None
            and self.target_entropy > 0
            and (
                self.configured_target_entropy > 0
                or self.observation_count > self.warmup_steps
            )
        ):
            deficit = max(self.target_entropy - self.entropy_ema, 0.0)
            relative_deficit = deficit / self.target_entropy
            log_alpha_span = math.log(self.max_alpha / self.base_alpha)
            proposed = self.base_alpha * math.exp(
                self.gain * log_alpha_span * relative_deficit
            )
            self.current_alpha = min(
                self.max_alpha, max(self.base_alpha, proposed)
            )
        else:
            relative_deficit = 0.0
            self.current_alpha = float(self.base_alpha)
        diagnostics = {
            "maxent_control_entropy_ema": float(self.entropy_ema),
            "maxent_control_deficit": float(deficit),
            "maxent_control_relative_deficit": float(relative_deficit),
            "maxent_control_next_alpha": float(self.current_alpha),
            "maxent_control_observations": float(self.observation_count),
        }
        if self.target_entropy is not None:
            diagnostics["maxent_control_target_entropy"] = float(
                self.target_entropy
            )
        return diagnostics

    def state_dict(self) -> dict[str, Any]:
        return {
            "controller_kind": "maxent_proportional",
            "controller_rule": "relative_deficit_log_span_v1",
            "entropy_units": self.entropy_units,
            "configured_target_entropy": float(self.configured_target_entropy),
            "current_alpha": float(self.current_alpha),
            "entropy_ema": self.entropy_ema,
            "target_entropy": self.target_entropy,
            "observation_count": int(self.observation_count),
            "warmup_entropy_sum": float(self._warmup_entropy_sum),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not isinstance(state, dict):
            raise ValueError("controller state must be a dictionary")
        if state.get("controller_kind") not in (None, "maxent_proportional"):
            raise ValueError("checkpoint contains a different MaxEnt controller")
        if state.get("entropy_units") != self.entropy_units:
            raise ValueError("checkpoint uses incompatible MaxEnt entropy units")
        if state.get("controller_rule") != "relative_deficit_log_span_v1":
            raise ValueError("checkpoint uses an incompatible proportional rule")
        saved_configured_target = float(
            state.get("configured_target_entropy", 0.0)
        )
        if not math.isclose(
            saved_configured_target,
            self.configured_target_entropy,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("checkpoint uses a different configured entropy target")
        current = float(state.get("current_alpha", self.base_alpha))
        self.current_alpha = min(self.max_alpha, max(self.base_alpha, current))
        entropy_ema = state.get("entropy_ema")
        target_entropy = state.get("target_entropy")
        self.entropy_ema = None if entropy_ema is None else float(entropy_ema)
        self.target_entropy = (
            None if target_entropy is None else float(target_entropy)
        )
        self.observation_count = max(int(state.get("observation_count", 0)), 0)
        self._warmup_entropy_sum = float(state.get("warmup_entropy_sum", 0.0))


@dataclass
class MaxEntDualController:
    """Haarnoja-style dual descent on the direct entropy coefficient."""

    base_alpha: float
    min_alpha: float
    max_alpha: float
    target_ratio: float
    warmup_steps: int
    alpha_lr: float
    ema_decay: float = 0.7
    configured_target_entropy: float = 0.0
    entropy_units: str = "sequence_nats_v1"
    observation_metric_key: str = "maxent_sequence_entropy"
    beta1: float = 0.9
    beta2: float = 0.999
    adam_eps: float = 1e-8
    current_alpha: float | None = None
    target_entropy: float | None = None
    entropy_ema: float | None = None
    observation_count: int = 0
    log_alpha: float | None = None
    _warmup_entropy_sum: float = 0.0
    _adam_m: float = 0.0
    _adam_v: float = 0.0
    _adam_step: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("base_alpha", self.base_alpha),
            ("min_alpha", self.min_alpha),
            ("max_alpha", self.max_alpha),
            ("target_ratio", self.target_ratio),
            ("alpha_lr", self.alpha_lr),
            ("ema_decay", self.ema_decay),
            ("beta1", self.beta1),
            ("beta2", self.beta2),
            ("adam_eps", self.adam_eps),
            ("configured_target_entropy", self.configured_target_entropy),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.base_alpha <= 0:
            raise ValueError("base_alpha must be positive")
        if self.min_alpha <= 0 or self.min_alpha > self.base_alpha:
            raise ValueError("min_alpha must be in (0, base_alpha]")
        if self.max_alpha < self.base_alpha:
            raise ValueError("max_alpha must be at least base_alpha")
        if not 0 < self.target_ratio <= 1:
            raise ValueError("target_ratio must be in (0, 1]")
        if int(self.warmup_steps) <= 0:
            raise ValueError("warmup_steps must be positive")
        if self.alpha_lr <= 0:
            raise ValueError("alpha_lr must be positive")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("Adam betas must be in [0, 1)")
        if self.adam_eps <= 0:
            raise ValueError("adam_eps must be positive")
        if self.configured_target_entropy < 0:
            raise ValueError("configured_target_entropy must be non-negative")
        expected_metric = {
            # The masked-mean token entropy the manuscript reports as
            # `train/entropy`. A dual controller whose target was measured from
            # that telemetry has to observe the same estimator: the
            # conditional-content-token mean below is a different one and runs
            # an order of magnitude higher, so regulating on it against such a
            # target pins the coefficient at a bound and the policy never
            # reaches the entropy it is supposed to hold.
            "masked_mean_token_nats_v1": "entropy",
            "sequence_nats_v1": "maxent_sequence_entropy",
            "conditional_content_token_nats_mean_v1": (
                "maxent_conditional_token_entropy"
            ),
            "canonical_action_nats_exact_v1": (
                "canonical_exact_sequence_entropy"
            ),
        }.get(self.entropy_units)
        if expected_metric is None or self.observation_metric_key != expected_metric:
            raise ValueError("MaxEnt entropy units and observation metric are incompatible")
        if self.configured_target_entropy > 0:
            self.target_entropy = float(self.configured_target_entropy)
        if self.entropy_ema is not None:
            self.entropy_ema = float(self.entropy_ema)
            if not math.isfinite(self.entropy_ema) or self.entropy_ema < 0:
                raise ValueError("entropy_ema must be finite and non-negative")
        if self.log_alpha is None:
            self.log_alpha = math.log(self.base_alpha)
        self._project_log_alpha()
        if self.current_alpha is None:
            self.current_alpha = self.alpha

    @property
    def alpha(self) -> float:
        return math.exp(float(self.log_alpha))

    def _project_log_alpha(self) -> None:
        self.log_alpha = min(
            math.log(self.max_alpha),
            max(math.log(self.min_alpha), float(self.log_alpha)),
        )

    def observe(self, entropy: float) -> dict[str, float]:
        value = float(entropy)
        if not math.isfinite(value) or value < 0:
            raise ValueError("entropy observation must be finite and non-negative")
        self.observation_count += 1
        if self.entropy_ema is None:
            self.entropy_ema = value
        else:
            self.entropy_ema = (
                self.ema_decay * self.entropy_ema
                + (1.0 - self.ema_decay) * value
            )
        if (
            self.configured_target_entropy == 0
            and self.observation_count <= self.warmup_steps
        ):
            self._warmup_entropy_sum += value
            if self.observation_count == self.warmup_steps:
                self.target_entropy = self.target_ratio * (
                    self._warmup_entropy_sum / float(self.warmup_steps)
                )

        entropy_error = 0.0
        alpha_loss = 0.0
        alpha_gradient = 0.0
        if (
            self.target_entropy is not None
            and self.target_entropy > 0
            and (
                self.configured_target_entropy > 0
                or self.observation_count > self.warmup_steps
            )
        ):
            entropy_error = self.entropy_ema - self.target_entropy
            alpha_before = self.alpha
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
            self.current_alpha = self.alpha
        else:
            self.log_alpha = math.log(self.base_alpha)
            self.current_alpha = float(self.base_alpha)

        diagnostics = {
            "maxent_dual_observed_entropy": value,
            "maxent_dual_entropy_ema": float(self.entropy_ema),
            "maxent_dual_ema_decay": float(self.ema_decay),
            "maxent_dual_entropy_error": float(entropy_error),
            "maxent_dual_alpha_loss": float(alpha_loss),
            "maxent_dual_alpha_gradient": float(alpha_gradient),
            "maxent_dual_alpha": float(self.alpha),
            "maxent_dual_log_alpha": float(self.log_alpha),
            "maxent_dual_next_alpha": float(self.current_alpha),
            "maxent_dual_observations": float(self.observation_count),
            "maxent_dual_optimizer_steps": float(self._adam_step),
        }
        if self.target_entropy is not None:
            diagnostics["maxent_dual_target_entropy"] = float(
                self.target_entropy
            )
        return diagnostics

    def state_dict(self) -> dict[str, Any]:
        return {
            "controller_kind": "maxent_dual",
            "controller_rule": "log_alpha_adam_entropy_ema_v2",
            "entropy_units": self.entropy_units,
            "ema_decay": float(self.ema_decay),
            "configured_target_entropy": float(self.configured_target_entropy),
            "current_alpha": float(self.current_alpha),
            "target_entropy": self.target_entropy,
            "entropy_ema": self.entropy_ema,
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
        if state.get("controller_kind") not in (None, "maxent_dual"):
            raise ValueError("checkpoint contains a different MaxEnt controller")
        if state.get("entropy_units") != self.entropy_units:
            raise ValueError("checkpoint uses incompatible MaxEnt entropy units")
        if state.get("controller_rule") != "log_alpha_adam_entropy_ema_v2":
            raise ValueError("checkpoint uses an incompatible MaxEnt dual rule")
        saved_ema_decay = float(state.get("ema_decay", -1.0))
        if not math.isclose(
            saved_ema_decay,
            self.ema_decay,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("checkpoint uses a different MaxEnt dual EMA decay")
        saved_configured_target = float(
            state.get("configured_target_entropy", 0.0)
        )
        if not math.isclose(
            saved_configured_target,
            self.configured_target_entropy,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("checkpoint uses a different configured entropy target")
        self.target_entropy = (
            None
            if state.get("target_entropy") is None
            else float(state["target_entropy"])
        )
        entropy_ema = state.get("entropy_ema")
        self.entropy_ema = None if entropy_ema is None else float(entropy_ema)
        if self.entropy_ema is not None and (
            not math.isfinite(self.entropy_ema) or self.entropy_ema < 0
        ):
            raise ValueError("checkpoint contains an invalid entropy EMA")
        self.observation_count = max(int(state.get("observation_count", 0)), 0)
        self.log_alpha = float(state.get("log_alpha", math.log(self.base_alpha)))
        self._warmup_entropy_sum = float(state.get("warmup_entropy_sum", 0.0))
        self._adam_m = float(state.get("adam_m", 0.0))
        self._adam_v = max(float(state.get("adam_v", 0.0)), 0.0)
        self._adam_step = max(int(state.get("adam_step", 0)), 0)
        self._project_log_alpha()
        self.current_alpha = self.alpha
