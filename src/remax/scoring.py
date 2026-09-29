"""Scoring-time token and vocabulary hygiene helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class SanitizedTokenIdsResult:
    """Best-effort sanitized token IDs plus replacement metadata."""

    token_ids: torch.Tensor
    invalid_count: int = 0
    replacement_id: int | None = None
    min_invalid: int | None = None
    max_invalid: int | None = None


def coerce_optional_int(value: object | None) -> Optional[int]:
    """Return ``value`` as an int when possible."""

    try:
        numeric = int(value)
    except (TypeError, ValueError):
        return None
    return numeric


def _get_config_value(config: Any, name: str, default: Any = None) -> Any:
    """Read ``name`` from config-like objects without assuming a concrete type."""

    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _get_embedding_vocab_size(model: Any, config: Any = None) -> Optional[int]:
    """Return the model embedding vocab size when exposed."""

    get_embeddings = getattr(model, "get_input_embeddings", None)
    embedding_module = None
    if callable(get_embeddings):
        try:
            embedding_module = get_embeddings()
        except Exception:
            embedding_module = None
    num_embeddings = coerce_optional_int(
        getattr(embedding_module, "num_embeddings", None)
    )
    if isinstance(num_embeddings, int) and num_embeddings > 0:
        return num_embeddings
    weight = getattr(embedding_module, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape is not None and len(shape) >= 1:
        size_value = coerce_optional_int(shape[0])
        if isinstance(size_value, int) and size_value > 0:
            return size_value
    config_vocab_size = coerce_optional_int(_get_config_value(config, "vocab_size"))
    if isinstance(config_vocab_size, int) and config_vocab_size > 0:
        return config_vocab_size
    return None


def resolve_model_vocab_limit(model: Any) -> Optional[int]:
    """Return the largest positive vocab-size limit exposed by the model."""

    config = getattr(model, "config", None)
    candidates = [
        value
        for value in (
            _get_embedding_vocab_size(model, config),
            coerce_optional_int(_get_config_value(config, "vocab_size")),
            coerce_optional_int(getattr(model, "vocab_size", None)),
        )
        if isinstance(value, int) and value > 0
    ]
    if not candidates:
        return None
    return max(candidates)


def resolve_tokenizer_vocab_limit(tokenizer: Any) -> Optional[int]:
    """Return the full addressable tokenizer range, including added tokens."""

    candidates = []
    vocab_size = coerce_optional_int(getattr(tokenizer, "vocab_size", None))
    if isinstance(vocab_size, int) and vocab_size > 0:
        candidates.append(vocab_size)
    try:
        tokenizer_len = coerce_optional_int(len(tokenizer))
    except Exception:
        tokenizer_len = None
    if isinstance(tokenizer_len, int) and tokenizer_len > 0:
        candidates.append(tokenizer_len)
    if not candidates:
        return None
    return max(candidates)


def resolve_token_id_upper_bound(model: Any, tokenizer: Any = None) -> Optional[int]:
    """Return a conservative upper bound for valid token IDs."""

    candidates = []
    model_limit = resolve_model_vocab_limit(model)
    if isinstance(model_limit, int) and model_limit > 0:
        candidates.append(model_limit)
    tokenizer_limit = resolve_tokenizer_vocab_limit(tokenizer)
    if isinstance(tokenizer_limit, int) and tokenizer_limit > 0:
        candidates.append(tokenizer_limit)
    if not candidates:
        return None
    return min(candidates)


def mask_invalid_logit_columns(
    logits: torch.Tensor,
    *,
    valid_vocab_size: Optional[int],
) -> torch.Tensor:
    """Mask logits that correspond to tokenizer-inaccessible token IDs."""

    if not isinstance(valid_vocab_size, int) or valid_vocab_size <= 0:
        return logits
    if logits.ndim < 1:
        return logits
    if int(logits.size(-1)) <= valid_vocab_size:
        return logits
    masked = logits.clone()
    masked[..., valid_vocab_size:] = torch.finfo(masked.dtype).min
    return masked


def sanitize_scoring_token_ids(
    token_ids: torch.Tensor,
    *,
    upper_bound: Optional[int],
    tokenizer: Any = None,
) -> SanitizedTokenIdsResult:
    """Clamp scorer token IDs into range before model/gather indexing."""

    if not isinstance(token_ids, torch.Tensor):
        raise TypeError("token_ids must be a torch.Tensor")
    if token_ids.dtype.is_floating_point or token_ids.dtype == torch.bool:
        return SanitizedTokenIdsResult(token_ids=token_ids)
    if not isinstance(upper_bound, int) or upper_bound <= 0:
        return SanitizedTokenIdsResult(token_ids=token_ids)

    replacement_id = coerce_optional_int(getattr(tokenizer, "pad_token_id", None))
    if replacement_id is None or replacement_id < 0 or replacement_id >= upper_bound:
        replacement_id = coerce_optional_int(getattr(tokenizer, "eos_token_id", None))
    if replacement_id is None or replacement_id < 0 or replacement_id >= upper_bound:
        replacement_id = max(upper_bound - 1, 0)

    invalid_mask = (token_ids < 0) | (token_ids >= upper_bound)
    invalid_count = int(invalid_mask.to(torch.long).sum().item())
    if invalid_count <= 0:
        return SanitizedTokenIdsResult(token_ids=token_ids)

    invalid_vals = token_ids[invalid_mask]
    min_invalid = int(invalid_vals.min().item()) if invalid_vals.numel() > 0 else None
    max_invalid = int(invalid_vals.max().item()) if invalid_vals.numel() > 0 else None
    sanitized = token_ids.clone()
    sanitized[invalid_mask] = int(replacement_id)
    return SanitizedTokenIdsResult(
        token_ids=sanitized,
        invalid_count=invalid_count,
        replacement_id=int(replacement_id),
        min_invalid=min_invalid,
        max_invalid=max_invalid,
    )
