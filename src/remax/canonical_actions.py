"""Fixed-support canonical action policies for exact-answer experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import itertools
import json
import math
from typing import Any

import torch

from modebench.pantry_support_action import (
    PANTRY_SUPPORT_MASK_TASK,
    PANTRY_SUPPORT_MASK_WIDTH,
    decode_pantry_support_mask,
    pantry_support_mask_from_allocation,
)


GRAPH_COLOR_ACTIONS = ("1", "2", "3")
COUNTDOWN_ACTIONS_BY_POSITION = (
    ("1", "2", "3", "4", "5", "6"),
    ("1", "2", "3"),
    ("1", "2", "3", "4", "5", "6"),
)


@dataclass(frozen=True)
class CanonicalActionSpace:
    """A finite three-position action grammar resolved into tokenizer IDs."""

    task: str
    action_strings_by_position: tuple[tuple[str, ...], ...]
    token_ids_by_position: tuple[tuple[int, ...], ...]

    @property
    def horizon(self) -> int:
        return len(self.token_ids_by_position)

    @property
    def union_token_ids(self) -> tuple[int, ...]:
        return tuple(
            dict.fromkeys(
                token_id
                for support in self.token_ids_by_position
                for token_id in support
            )
        )

    @property
    def sequence_count(self) -> int:
        return math.prod(len(support) for support in self.token_ids_by_position)

    @property
    def max_sequence_entropy(self) -> float:
        return math.log(self.sequence_count)


def canonical_action_strings_by_position(
    task: str,
) -> tuple[tuple[str, ...], ...]:
    """Return the audited action grammar for a canonical benchmark task."""

    if task == "graph_coloring":
        return (GRAPH_COLOR_ACTIONS,) * 3
    if task == "countdown":
        return COUNTDOWN_ACTIONS_BY_POSITION
    if task == PANTRY_SUPPORT_MASK_TASK:
        return (("0", "1"),) * PANTRY_SUPPORT_MASK_WIDTH
    raise ValueError(f"unsupported canonical action task: {task!r}")


def resolve_canonical_action_space(tokenizer: Any, task: str) -> CanonicalActionSpace:
    """Resolve every grammar position while preserving its support order."""

    action_strings = canonical_action_strings_by_position(task)
    union_actions = tuple(
        dict.fromkeys(action for support in action_strings for action in support)
    )
    union_ids = resolve_single_token_actions(tokenizer, union_actions)
    action_to_id = dict(zip(union_actions, union_ids))
    return CanonicalActionSpace(
        task=task,
        action_strings_by_position=action_strings,
        token_ids_by_position=tuple(
            tuple(action_to_id[action] for action in support)
            for support in action_strings
        ),
    )


def enumerate_countdown_action_codes() -> tuple[str, ...]:
    """Enumerate the 6 x 3 x 6 Countdown codes in lexical digit order."""

    return tuple(
        "".join(digits)
        for digits in itertools.product(
            COUNTDOWN_ACTIONS_BY_POSITION[0],
            COUNTDOWN_ACTIONS_BY_POSITION[1],
            COUNTDOWN_ACTIONS_BY_POSITION[2],
        )
    )


def decode_countdown_action_code(code: str, reference: str | dict[str, Any]) -> str:
    """Decode one audited three-digit code into a full Countdown expression.

    The singleton digit selects one operand in dataset order. The inner digit
    combines the other two operands (also in dataset order), and the root
    digit combines that pair with the singleton. No answer or target is used.
    """

    text = str(code).strip()
    if (
        len(text) != 3
        or text[0] not in COUNTDOWN_ACTIONS_BY_POSITION[0]
        or text[1] not in COUNTDOWN_ACTIONS_BY_POSITION[1]
        or text[2] not in COUNTDOWN_ACTIONS_BY_POSITION[2]
    ):
        raise ValueError(f"invalid canonical Countdown action code: {code!r}")
    try:
        spec = json.loads(reference) if isinstance(reference, str) else reference
        if not isinstance(spec, dict) or spec.get("verifier") != "countdown":
            raise ValueError
        numbers = tuple(int(value) for value in spec["numbers"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("canonical Countdown reference is malformed") from exc
    if len(numbers) != 3 or len(set(numbers)) != 3:
        raise ValueError(
            "canonical Countdown requires exactly three distinct operands"
        )

    root_action, singleton_action, inner_action = (int(value) for value in text)
    singleton_index = singleton_action - 1
    singleton = numbers[singleton_index]
    a, b = (
        value for index, value in enumerate(numbers) if index != singleton_index
    )
    pair = {
        1: f"({a} + {b})",
        2: f"({a} * {b})",
        3: f"({a} - {b})",
        4: f"({b} - {a})",
        5: f"({a} / {b})",
        6: f"({b} / {a})",
    }[inner_action]
    return {
        1: f"({pair} + {singleton})",
        2: f"({pair} * {singleton})",
        3: f"({pair} - {singleton})",
        4: f"({singleton} - {pair})",
        5: f"({pair} / {singleton})",
        6: f"({singleton} / {pair})",
    }[root_action]


def decode_canonical_action_response(
    task: str, code: str, reference: str | dict[str, Any]
) -> str:
    """Map policy action tokens to the benchmark response seen by the grader."""

    text = str(code).strip()
    if task == "graph_coloring":
        if len(text) != 3 or any(action not in GRAPH_COLOR_ACTIONS for action in text):
            raise ValueError(f"invalid canonical graph action code: {code!r}")
        return text
    if task == "countdown":
        return decode_countdown_action_code(text, reference)
    if task == PANTRY_SUPPORT_MASK_TASK:
        try:
            spec = json.loads(reference) if isinstance(reference, str) else reference
        except json.JSONDecodeError:
            spec = {}
        if not isinstance(spec, dict):
            spec = {}
        return decode_pantry_support_mask(text, spec)
    raise ValueError(f"unsupported canonical action task: {task!r}")


def canonical_action_code_from_verified_response(
    task: str,
    response: str,
    reference: str | dict[str, Any],
) -> str:
    """Represent a verified replay witness on its registered policy surface."""

    text = str(response).strip()
    if task == "graph_coloring":
        decode_canonical_action_response(task, text, reference)
        return text
    if task == PANTRY_SUPPORT_MASK_TASK:
        try:
            spec = json.loads(reference) if isinstance(reference, str) else reference
        except json.JSONDecodeError as exc:
            raise ValueError("canonical Pantry reference is malformed") from exc
        if not isinstance(spec, dict):
            raise ValueError("canonical Pantry reference is malformed")
        return pantry_support_mask_from_allocation(text, spec)
    raise ValueError(
        f"verified replay projection is unsupported for canonical task {task!r}"
    )


def canonical_action_code_token_ids(
    action_space: CanonicalActionSpace,
    code: str,
) -> tuple[int, ...]:
    """Encode one canonical code as exactly one registered token per action."""

    text = str(code).strip()
    if len(text) != action_space.horizon:
        raise ValueError(
            f"canonical action code has length {len(text)}, expected "
            f"{action_space.horizon}"
        )
    token_ids: list[int] = []
    for position, action in enumerate(text):
        actions = action_space.action_strings_by_position[position]
        try:
            action_index = actions.index(action)
        except ValueError as exc:
            raise ValueError(
                f"canonical action {action!r} is outside position {position} support"
            ) from exc
        token_ids.append(action_space.token_ids_by_position[position][action_index])
    return tuple(token_ids)


def canonical_action_code_from_token_ids(
    action_space: CanonicalActionSpace,
    token_ids: Sequence[int],
) -> str:
    """Read one canonical code back from its registered action tokens.

    This is the inverse of :func:`canonical_action_code_token_ids`. A learner
    row on the canonical surface carries exactly one registered token per
    position, so the code is recovered by position-wise lookup rather than by
    decoding through the tokenizer, whose text for these placeholder tokens is
    not the benchmark response.
    """

    ids = tuple(int(value) for value in token_ids)
    if len(ids) != action_space.horizon:
        raise ValueError(
            f"canonical action row has {len(ids)} tokens, expected "
            f"{action_space.horizon}"
        )
    code: list[str] = []
    for position, token_id in enumerate(ids):
        support = action_space.token_ids_by_position[position]
        try:
            action_index = support.index(token_id)
        except ValueError as exc:
            raise ValueError(
                f"token id {token_id} is outside position {position} support"
            ) from exc
        code.append(action_space.action_strings_by_position[position][action_index])
    return "".join(code)


def _sample_canonical_actions_from_logits(
    next_token_logits: torch.Tensor,
    *,
    allowed_token_ids: Sequence[int],
    uniforms: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one categorical action per row from a fixed token support.

    Explicit uniforms make the sampling stream reproducible independently of
    CUDA generator implementation.  The caller remains responsible for the
    autoregressive loop: this helper samples exactly one next action.
    """

    if next_token_logits.ndim != 2:
        raise ValueError("next_token_logits must have shape [batch, vocab]")
    if next_token_logits.shape[0] <= 0:
        raise ValueError("next_token_logits batch must be nonempty")
    if uniforms.shape != (next_token_logits.shape[0],):
        raise ValueError("uniforms must have shape [batch]")
    ids = tuple(int(token_id) for token_id in allowed_token_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("allowed_token_ids must be nonempty and unique")
    if min(ids) < 0 or max(ids) >= int(next_token_logits.shape[-1]):
        raise ValueError("allowed_token_ids must lie inside the model vocabulary")
    if not bool(torch.isfinite(uniforms).all()) or not bool(
        ((uniforms >= 0) & (uniforms < 1)).all()
    ):
        raise ValueError("uniforms must be finite values in [0, 1)")

    allowed = torch.tensor(ids, dtype=torch.long, device=next_token_logits.device)
    action_logits = next_token_logits.index_select(-1, allowed).float()
    if not bool(torch.isfinite(action_logits).all()):
        raise FloatingPointError("canonical next-action logits are nonfinite")
    full_log_probs = torch.log_softmax(action_logits, dim=-1)
    probabilities = torch.exp(full_log_probs)
    cumulative = torch.cumsum(probabilities, dim=-1)
    uniform_rows = uniforms.to(
        device=next_token_logits.device, dtype=probabilities.dtype
    )
    sampled_indices = (uniform_rows.unsqueeze(-1) >= cumulative).sum(dim=-1)
    sampled_indices = sampled_indices.clamp_max(len(ids) - 1)
    token_ids = allowed[sampled_indices]
    selected_log_probs = torch.gather(
        full_log_probs, -1, sampled_indices.unsqueeze(-1)
    ).squeeze(-1)
    return (
        token_ids.detach(),
        selected_log_probs.detach(),
        full_log_probs.detach(),
    )


def resolve_single_token_actions(
    tokenizer: Any,
    actions: Sequence[str],
) -> tuple[int, ...]:
    """Resolve a one-token, one-to-one serialization for canonical actions."""

    if not actions:
        raise ValueError("canonical action support cannot be empty")
    token_ids: list[int] = []
    for action in actions:
        encoded = list(tokenizer.encode(action, add_special_tokens=False))
        if len(encoded) != 1:
            raise ValueError(
                f"canonical action {action!r} must encode to exactly one token; "
                f"got {encoded}"
            )
        token_id = int(encoded[0])
        decoded = str(
            tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
        if decoded != action:
            raise ValueError(
                f"canonical action {action!r} is not one-to-one: token "
                f"{token_id} decodes as {decoded!r}"
            )
        token_ids.append(token_id)
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("canonical actions must resolve to distinct token ids")
    return tuple(token_ids)


def resolve_graph_color_action_token_ids(tokenizer: Any) -> tuple[int, ...]:
    """Return the exact token ids for graph-color actions 1, 2, and 3."""

    return resolve_single_token_actions(tokenizer, GRAPH_COLOR_ACTIONS)


def restricted_action_log_probs_entropy_and_distribution(
    logits: torch.Tensor,
    labels: torch.Tensor,
    response_masks: torch.Tensor,
    *,
    allowed_token_ids: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score a fixed-support autoregressive action policy.

    ``logits`` and ``labels`` have the ordinary causal-LM sequence shapes.
    Only active response labels must belong to ``allowed_token_ids``. Returned
    tensors have the shifted ``[batch, sequence - 1]`` shape and are zero away
    from response actions.
    """

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocab]")
    if labels.shape != logits.shape[:-1]:
        raise ValueError("labels must match logits batch/sequence dimensions")
    if response_masks.shape != (logits.shape[0], logits.shape[1] - 1):
        raise ValueError("response_masks must have shape [batch, sequence - 1]")
    ids = tuple(int(token_id) for token_id in allowed_token_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("allowed_token_ids must be nonempty and unique")
    if min(ids) < 0 or max(ids) >= int(logits.shape[-1]):
        raise ValueError("allowed_token_ids must lie inside the model vocabulary")

    shifted_labels = labels[:, 1:]
    active = response_masks.to(torch.bool)
    allowed = torch.tensor(ids, dtype=torch.long, device=logits.device)
    matches = shifted_labels.unsqueeze(-1).eq(allowed)
    match_count = matches.sum(dim=-1)
    invalid_active = active & match_count.ne(1)
    if bool(invalid_active.any()):
        invalid_ids = shifted_labels[invalid_active].detach().cpu().unique().tolist()
        raise ValueError(
            "canonical response contains token ids outside its action support: "
            f"{invalid_ids}"
        )

    action_logits = logits[:, :-1, :].index_select(-1, allowed).float()
    if not bool(torch.isfinite(action_logits[active]).all()):
        raise FloatingPointError(
            "canonical policy has nonfinite allowed-support logits on active actions"
        )
    action_log_probs = torch.log_softmax(action_logits, dim=-1)
    selected_indices = matches.to(torch.int64).argmax(dim=-1)
    selected = torch.gather(
        action_log_probs, dim=-1, index=selected_indices.unsqueeze(-1)
    ).squeeze(-1)
    probabilities = torch.softmax(action_logits, dim=-1)
    entropy = -(probabilities * action_log_probs).sum(dim=-1)
    if not bool(torch.isfinite(selected[active]).all()):
        raise FloatingPointError(
            "canonical policy has nonfinite selected log probabilities"
        )
    if not bool(torch.isfinite(entropy[active]).all()):
        raise FloatingPointError("canonical policy has nonfinite action entropy")
    zeros = torch.zeros_like(selected)
    masked_distribution = torch.where(
        active.unsqueeze(-1), action_log_probs, torch.zeros_like(action_log_probs)
    )
    return (
        torch.where(active, selected, zeros),
        torch.where(active, entropy, zeros),
        masked_distribution,
    )


def restricted_action_log_probs_and_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    response_masks: torch.Tensor,
    *,
    allowed_token_ids: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-compatible two-output canonical-policy scorer."""

    selected, entropy, _ = restricted_action_log_probs_entropy_and_distribution(
        logits,
        labels,
        response_masks,
        allowed_token_ids=allowed_token_ids,
    )
    return selected, entropy


def restricted_position_action_log_probs_entropy_and_distribution(
    logits: torch.Tensor,
    labels: torch.Tensor,
    response_masks: torch.Tensor,
    *,
    allowed_token_ids_by_position: Sequence[Sequence[int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score a fixed-horizon policy with position-specific token supports.

    The full distribution uses the ordered union vocabulary. Disallowed union
    entries are ``-inf`` on active rows and accompanied by an explicit support
    mask, so callers never treat a masked action as probability mass.
    """

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocab]")
    if labels.shape != logits.shape[:-1]:
        raise ValueError("labels must match logits batch/sequence dimensions")
    if response_masks.shape != (logits.shape[0], logits.shape[1] - 1):
        raise ValueError("response_masks must have shape [batch, sequence - 1]")
    supports = tuple(
        tuple(int(token_id) for token_id in support)
        for support in allowed_token_ids_by_position
    )
    if not supports or any(
        not support or len(support) != len(set(support)) for support in supports
    ):
        raise ValueError("every positional action support must be nonempty and unique")
    union_ids = tuple(
        dict.fromkeys(token_id for support in supports for token_id in support)
    )
    if min(union_ids) < 0 or max(union_ids) >= int(logits.shape[-1]):
        raise ValueError("allowed_token_ids must lie inside the model vocabulary")

    active = response_masks.to(torch.bool)
    row_counts = active.sum(dim=-1)
    if not bool(row_counts.eq(len(supports)).all()):
        raise ValueError(
            "canonical rows must contain exactly one action per positional support"
        )
    ordinals = active.to(torch.int64).cumsum(dim=-1) - 1
    safe_ordinals = ordinals.clamp(min=0, max=len(supports) - 1)
    union_allowed = torch.tensor(
        [
            [token_id in set(support) for token_id in union_ids]
            for support in supports
        ],
        dtype=torch.bool,
        device=logits.device,
    )
    support_mask = union_allowed[safe_ordinals] & active.unsqueeze(-1)
    union = torch.tensor(union_ids, dtype=torch.long, device=logits.device)
    action_logits = logits[:, :-1, :].index_select(-1, union).float()
    if not bool(torch.isfinite(action_logits[support_mask]).all()):
        raise FloatingPointError(
            "canonical policy has nonfinite positional-support logits"
        )
    masked_logits = action_logits.masked_fill(~support_mask, -torch.inf)
    # Inactive prompt/padding positions have no support by construction. Give
    # those rows a harmless finite distribution before log-softmax; they are
    # zeroed in every returned tensor below. This avoids all--inf softmax NaNs
    # (and their gradients) outside the response mask.
    masked_logits = torch.where(
        active.unsqueeze(-1), masked_logits, torch.zeros_like(masked_logits)
    )
    action_log_probs = torch.log_softmax(masked_logits, dim=-1)

    shifted_labels = labels[:, 1:]
    matches = shifted_labels.unsqueeze(-1).eq(union)
    valid_selected = (matches & support_mask).sum(dim=-1).eq(1)
    if bool((active & ~valid_selected).any()):
        invalid_ids = shifted_labels[active & ~valid_selected].unique().tolist()
        raise ValueError(
            "canonical response contains token ids outside its positional support: "
            f"{invalid_ids}"
        )
    selected_indices = matches.to(torch.int64).argmax(dim=-1)
    selected = torch.gather(
        action_log_probs, dim=-1, index=selected_indices.unsqueeze(-1)
    ).squeeze(-1)
    probabilities = torch.exp(action_log_probs)
    finite_log_probs = action_log_probs.masked_fill(~support_mask, 0.0)
    entropy_terms = probabilities * finite_log_probs
    entropy = -entropy_terms.sum(dim=-1)
    if not bool(torch.isfinite(selected[active]).all()) or not bool(
        torch.isfinite(entropy[active]).all()
    ):
        raise FloatingPointError("canonical positional policy scoring is nonfinite")
    zeros = torch.zeros_like(selected)
    return (
        torch.where(active, selected, zeros),
        torch.where(active, entropy, zeros),
        torch.where(
            active.unsqueeze(-1),
            action_log_probs,
            torch.zeros_like(action_log_probs),
        ),
        support_mask,
    )


def materialize_position_canonical_behavior_policy(
    shifted_labels: torch.Tensor,
    response_masks: torch.Tensor,
    *,
    action_ids: Sequence[Sequence[int]],
    selected_log_probs: Sequence[Sequence[float]],
    full_log_probs: Sequence[Sequence[Sequence[float]]],
    behavior_action_token_ids_by_position: Sequence[Sequence[Sequence[int]]],
    allowed_token_ids_by_position: Sequence[Sequence[int]],
    normalizer_atol: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Align a ragged positional behavior trace into an ordered union tensor."""

    if shifted_labels.shape != response_masks.shape:
        raise ValueError("shifted_labels and response_masks must have equal shape")
    if normalizer_atol < 0:
        raise ValueError("normalizer_atol must be nonnegative")
    supports = tuple(
        tuple(int(token_id) for token_id in support)
        for support in allowed_token_ids_by_position
    )
    union_ids = tuple(
        dict.fromkeys(token_id for support in supports for token_id in support)
    )
    union_index = {token_id: index for index, token_id in enumerate(union_ids)}
    batch_size, shifted_length = shifted_labels.shape
    fields = {
        "action_ids": action_ids,
        "selected_log_probs": selected_log_probs,
        "full_log_probs": full_log_probs,
        "behavior_action_token_ids_by_position": (
            behavior_action_token_ids_by_position
        ),
    }
    for name, rows in fields.items():
        if len(rows) != batch_size:
            raise RuntimeError(
                f"canonical behavior {name} has {len(rows)} rows; "
                f"expected {batch_size}"
            )

    dtype = torch.float32
    device = shifted_labels.device
    selected = torch.zeros((batch_size, shifted_length), dtype=dtype, device=device)
    full = torch.full(
        (batch_size, shifted_length, len(union_ids)),
        -torch.inf,
        dtype=dtype,
        device=device,
    )
    support_mask = torch.zeros_like(full, dtype=torch.bool)
    active = response_masks.to(torch.bool)
    normalization_error_max = 0.0
    selected_echo_diff_max = 0.0

    for row_index in range(batch_size):
        positions = torch.where(active[row_index])[0]
        row_actions = [int(value) for value in action_ids[row_index]]
        row_selected = [float(value) for value in selected_log_probs[row_index]]
        row_full = list(full_log_probs[row_index])
        row_supports = tuple(
            tuple(int(value) for value in support)
            for support in behavior_action_token_ids_by_position[row_index]
        )
        if row_supports != supports:
            raise RuntimeError(
                "canonical behavior positional support mismatch at row "
                f"{row_index}: expected={supports} observed={row_supports}"
            )
        if not (
            len(positions)
            == len(row_actions)
            == len(row_selected)
            == len(row_full)
            == len(supports)
        ):
            raise RuntimeError(
                f"canonical behavior trace length mismatch at row {row_index}"
            )
        active_labels = shifted_labels[row_index, positions].detach().cpu().tolist()
        if row_actions != [int(value) for value in active_labels]:
            raise RuntimeError(
                "canonical behavior action IDs do not match shifted learner labels "
                f"at row {row_index}"
            )

        for action_position, (position, action_id, support) in enumerate(
            zip(positions.tolist(), row_actions, supports)
        ):
            if action_id not in support:
                raise RuntimeError(
                    f"canonical behavior action {action_id} is outside support {support}"
                )
            values = torch.as_tensor(
                row_full[action_position], dtype=dtype, device=device
            )
            selected_value = torch.as_tensor(
                row_selected[action_position], dtype=dtype, device=device
            )
            if values.shape != (len(support),) or not bool(
                torch.isfinite(values).all()
            ) or not bool(torch.isfinite(selected_value)):
                raise RuntimeError("canonical behavior trace is malformed or nonfinite")
            log_normalizer = torch.logsumexp(values, dim=-1)
            normalization_error = float(torch.abs(log_normalizer).cpu().item())
            if normalization_error > normalizer_atol:
                raise RuntimeError(
                    "canonical behavior probabilities are not normalized at "
                    f"row={row_index} position={action_position}: "
                    f"abs(logsumexp)={normalization_error:.6g}"
                )
            support_action_index = support.index(action_id)
            echo_diff = float(
                torch.abs(selected_value - values[support_action_index]).cpu().item()
            )
            if echo_diff > 1e-7:
                raise RuntimeError(
                    "canonical selected behavior log probability does not match "
                    f"its positional row at row={row_index} "
                    f"position={action_position}: difference={echo_diff:.6g}"
                )
            normalized = values - log_normalizer
            selected[row_index, position] = normalized[support_action_index]
            union_indices = torch.tensor(
                [union_index[token_id] for token_id in support],
                dtype=torch.long,
                device=device,
            )
            full[row_index, position, union_indices] = normalized
            support_mask[row_index, position, union_indices] = True
            normalization_error_max = max(
                normalization_error_max, normalization_error
            )
            selected_echo_diff_max = max(selected_echo_diff_max, echo_diff)

    return (
        selected,
        full,
        support_mask,
        normalization_error_max,
        selected_echo_diff_max,
    )


def materialize_canonical_behavior_policy(
    shifted_labels: torch.Tensor,
    response_masks: torch.Tensor,
    *,
    action_ids: Sequence[Sequence[int]],
    selected_log_probs: Sequence[Sequence[float]],
    full_log_probs: Sequence[Sequence[Sequence[float]]],
    behavior_action_token_ids: Sequence[Sequence[int]],
    allowed_token_ids: Sequence[int],
    normalizer_atol: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    """Align the exact actor behavior trace with shifted learner positions.

    Returns selected behavior log probabilities, ordered full-support behavior
    log probabilities, maximum normalization error, and maximum discrepancy
    between the separately transported selected value and its full-row entry.
    """

    if shifted_labels.shape != response_masks.shape:
        raise ValueError("shifted_labels and response_masks must have equal shape")
    if normalizer_atol < 0:
        raise ValueError("normalizer_atol must be nonnegative")
    batch_size, shifted_length = shifted_labels.shape
    ids = tuple(int(token_id) for token_id in allowed_token_ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("allowed_token_ids must be nonempty and unique")
    fields = {
        "action_ids": action_ids,
        "selected_log_probs": selected_log_probs,
        "full_log_probs": full_log_probs,
        "behavior_action_token_ids": behavior_action_token_ids,
    }
    for name, rows in fields.items():
        if len(rows) != batch_size:
            raise RuntimeError(
                f"canonical behavior {name} has {len(rows)} rows; "
                f"expected {batch_size}"
            )

    dtype = torch.float32
    device = shifted_labels.device
    selected = torch.zeros((batch_size, shifted_length), dtype=dtype, device=device)
    full = torch.zeros(
        (batch_size, shifted_length, len(ids)), dtype=dtype, device=device
    )
    id_to_index = {token_id: index for index, token_id in enumerate(ids)}
    normalization_error_max = 0.0
    selected_echo_diff_max = 0.0
    active = response_masks.to(torch.bool)

    for row_index in range(batch_size):
        positions = torch.where(active[row_index])[0]
        expected_count = int(positions.numel())
        row_actions = [int(value) for value in action_ids[row_index]]
        row_selected = [float(value) for value in selected_log_probs[row_index]]
        row_full = list(full_log_probs[row_index])
        row_support = tuple(int(value) for value in behavior_action_token_ids[row_index])
        if row_support != ids:
            raise RuntimeError(
                "canonical behavior support order mismatch at row "
                f"{row_index}: expected={ids} observed={row_support}"
            )
        if not (
            len(row_actions)
            == len(row_selected)
            == len(row_full)
            == expected_count
        ):
            raise RuntimeError(
                "canonical behavior trace length mismatch at row "
                f"{row_index}: actions={len(row_actions)} "
                f"selected={len(row_selected)} full={len(row_full)} "
                f"active={expected_count}"
            )
        active_labels = shifted_labels[row_index, positions].detach().cpu().tolist()
        if row_actions != [int(value) for value in active_labels]:
            raise RuntimeError(
                "canonical behavior action IDs do not match shifted learner labels "
                f"at row {row_index}"
            )

        for trace_index, (position, action_id) in enumerate(
            zip(positions.tolist(), row_actions)
        ):
            if action_id not in id_to_index:
                raise RuntimeError(
                    f"canonical behavior action {action_id} is outside support {ids}"
                )
            values = torch.as_tensor(
                row_full[trace_index], dtype=dtype, device=device
            )
            if values.shape != (len(ids),):
                raise RuntimeError(
                    "canonical behavior full row has wrong shape at "
                    f"row={row_index} position={trace_index}: {tuple(values.shape)}"
                )
            selected_value = torch.as_tensor(
                row_selected[trace_index], dtype=dtype, device=device
            )
            if not bool(torch.isfinite(values).all()) or not bool(
                torch.isfinite(selected_value)
            ):
                raise RuntimeError("canonical behavior trace contains nonfinite values")
            log_normalizer = torch.logsumexp(values, dim=-1)
            normalization_error = float(
                torch.abs(log_normalizer).detach().cpu().item()
            )
            if normalization_error > normalizer_atol:
                raise RuntimeError(
                    "canonical behavior probabilities are not normalized at "
                    f"row={row_index} position={trace_index}: "
                    f"abs(logsumexp)={normalization_error:.6g}"
                )
            selected_from_full_raw = values[id_to_index[action_id]]
            echo_diff = float(
                torch.abs(selected_value - selected_from_full_raw)
                .detach()
                .cpu()
                .item()
            )
            if echo_diff > 1e-7:
                raise RuntimeError(
                    "canonical selected behavior log probability does not match "
                    f"its full row at row={row_index} position={trace_index}: "
                    f"difference={echo_diff:.6g}"
                )
            normalized_values = values - log_normalizer
            selected[row_index, position] = normalized_values[id_to_index[action_id]]
            full[row_index, position] = normalized_values
            normalization_error_max = max(
                normalization_error_max, normalization_error
            )
            selected_echo_diff_max = max(selected_echo_diff_max, echo_diff)

    return selected, full, normalization_error_max, selected_echo_diff_max


def canonical_behavior_overlap_diagnostics(
    behavior_selected_log_probs: torch.Tensor,
    behavior_full_log_probs: torch.Tensor,
    learner_selected_log_probs: torch.Tensor,
    learner_full_log_probs: torch.Tensor,
    response_masks: torch.Tensor,
    support_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compute full-distribution overlap and horizon-three IS diagnostics."""

    if behavior_selected_log_probs.shape != response_masks.shape:
        raise ValueError("behavior selected log-probabilities have the wrong shape")
    if learner_selected_log_probs.shape != response_masks.shape:
        raise ValueError("learner selected log-probabilities have the wrong shape")
    expected_full_shape = (*response_masks.shape, behavior_full_log_probs.shape[-1])
    if behavior_full_log_probs.shape != expected_full_shape:
        raise ValueError("behavior full log-probabilities have the wrong shape")
    if learner_full_log_probs.shape != expected_full_shape:
        raise ValueError("learner full log-probabilities have the wrong shape")
    if support_mask is not None and support_mask.shape != expected_full_shape:
        raise ValueError("canonical support mask has the wrong shape")

    active = response_masks.to(torch.bool)
    if not bool(active.any()):
        raise ValueError("canonical behavior diagnostics require active actions")
    behavior_rows = behavior_full_log_probs[active].double()
    learner_rows = learner_full_log_probs[active].double()
    active_support = (
        torch.ones_like(behavior_rows, dtype=torch.bool)
        if support_mask is None
        else support_mask[active].to(torch.bool)
    )
    if not bool(active_support.any(dim=-1).all()):
        raise ValueError("canonical behavior diagnostics found an empty support row")
    if not bool(torch.isfinite(behavior_rows[active_support]).all()) or not bool(
        torch.isfinite(learner_rows[active_support]).all()
    ):
        raise FloatingPointError("canonical behavior diagnostics are nonfinite")
    behavior_rows = behavior_rows - torch.logsumexp(
        behavior_rows.masked_fill(~active_support, -torch.inf),
        dim=-1,
        keepdim=True,
    )
    learner_rows = learner_rows - torch.logsumexp(
        learner_rows.masked_fill(~active_support, -torch.inf),
        dim=-1,
        keepdim=True,
    )

    log_ratio_rows = torch.where(
        active_support,
        learner_rows - behavior_rows,
        torch.zeros_like(learner_rows),
    )
    density_ratios = torch.exp(log_ratio_rows[active_support])
    behavior_probabilities = torch.where(
        active_support, torch.exp(behavior_rows), torch.zeros_like(behavior_rows)
    )
    learner_probabilities = torch.where(
        active_support, torch.exp(learner_rows), torch.zeros_like(learner_rows)
    )
    tv = 0.5 * torch.abs(
        learner_probabilities - behavior_probabilities
    ).sum(dim=-1)
    kl_actor_learner = (
        torch.where(
            active_support,
            behavior_probabilities * (behavior_rows - learner_rows),
            torch.zeros_like(behavior_probabilities),
        )
    ).sum(dim=-1)
    kl_learner_actor = (
        torch.where(
            active_support,
            learner_probabilities * (learner_rows - behavior_rows),
            torch.zeros_like(learner_probabilities),
        )
    ).sum(dim=-1)

    selected_log_ratio = (
        learner_selected_log_probs.double() - behavior_selected_log_probs.double()
    )
    row_counts = active.sum(dim=-1)
    if not bool(row_counts.eq(row_counts[0]).all()):
        raise RuntimeError("canonical behavior rows have unequal horizons")
    horizon = int(row_counts[0].item())
    compact_log_ratio = torch.stack(
        [selected_log_ratio[row][active[row]] for row in range(active.shape[0])]
    )
    inclusive_prefix_log_ratio = torch.cumsum(compact_log_ratio, dim=-1)
    exclusive_prefix_log_ratio = torch.cat(
        (
            torch.zeros_like(inclusive_prefix_log_ratio[:, :1]),
            inclusive_prefix_log_ratio[:, :-1],
        ),
        dim=-1,
    )

    def _ess_fraction(weights: torch.Tensor) -> torch.Tensor:
        count = torch.tensor(
            float(weights.numel()), dtype=weights.dtype, device=weights.device
        )
        return weights.sum().square() / weights.square().sum().clamp_min(1e-12) / count

    prefix_ess = []
    for depth in range(horizon):
        prefix_ess.append(
            _ess_fraction(torch.exp(exclusive_prefix_log_ratio[:, depth]))
        )
    sequence_weights = torch.exp(inclusive_prefix_log_ratio[:, horizon - 1])

    return {
        "canonical_behavior_ratio_min": density_ratios.min(),
        "canonical_behavior_ratio_max": density_ratios.max(),
        "canonical_behavior_tv_mean": tv.mean(),
        "canonical_behavior_tv_max": tv.max(),
        "canonical_behavior_kl_actor_learner_mean": kl_actor_learner.mean(),
        "canonical_behavior_kl_actor_learner_max": kl_actor_learner.max(),
        "canonical_behavior_kl_learner_actor_mean": kl_learner_actor.mean(),
        "canonical_behavior_kl_learner_actor_max": kl_learner_actor.max(),
        "canonical_behavior_sequence_ess_fraction": _ess_fraction(sequence_weights),
        "canonical_behavior_prefix_ess_fraction_min": torch.stack(prefix_ess).min(),
    }
