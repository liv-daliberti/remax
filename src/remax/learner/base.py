"""Shared scoring helpers for the Dr.GRPO learner."""

from __future__ import annotations

import logging

import torch
from oat.utils.ops import entropy_from_logits

from ..canonical_actions import (
    restricted_action_log_probs_and_entropy,
    restricted_position_action_log_probs_entropy_and_distribution,
)
from ..scoring import (
    mask_invalid_logit_columns,
    resolve_token_id_upper_bound,
    sanitize_scoring_token_ids,
)
from ..tensor_utils import gather_selected_logps_chunked


class ZeroMathLearnerBaseMixin:
    """Numerically safe token scoring used by every retained arm."""

    def _unwrap_scoring_model(self, model: torch.nn.Module) -> torch.nn.Module:
        base_model = model
        visited = set()
        while hasattr(base_model, "module"):
            next_model = getattr(base_model, "module")
            if not isinstance(next_model, torch.nn.Module) or id(next_model) in visited:
                break
            visited.add(id(next_model))
            base_model = next_model
        return base_model

    def _resolve_scoring_vocab_upper_bound(self, model: torch.nn.Module) -> int | None:
        return resolve_token_id_upper_bound(
            self._unwrap_scoring_model(model), getattr(self, "tokenizer", None)
        )

    def _sanitize_scoring_token_ids(
        self,
        token_ids: torch.Tensor,
        *,
        upper_bound: int | None,
        context: str,
    ) -> torch.Tensor:
        sanitized = sanitize_scoring_token_ids(
            token_ids,
            upper_bound=upper_bound,
            tokenizer=getattr(self, "tokenizer", None),
        )
        if sanitized.invalid_count > 0:
            warned = self._invalid_scoring_token_ids_warned_contexts
            if context not in warned:
                logging.warning(
                    "Sanitized %d token ids for %s outside upper_bound=%d",
                    sanitized.invalid_count,
                    context,
                    upper_bound,
                )
                warned.add(context)
        return sanitized.token_ids

    def _mask_invalid_scoring_logit_columns(
        self,
        logits: torch.Tensor,
        *,
        valid_vocab_size: int | None,
        context: str,
    ) -> torch.Tensor:
        if not isinstance(valid_vocab_size, int) or valid_vocab_size <= 0:
            return logits
        if int(logits.size(-1)) <= valid_vocab_size:
            return logits
        warned = self._invalid_logit_columns_warned_contexts
        if context not in warned:
            logging.warning(
                "Masking %d tokenizer-inaccessible logit columns for %s",
                int(logits.size(-1)) - valid_vocab_size,
                context,
            )
            warned.add(context)
        return mask_invalid_logit_columns(logits, valid_vocab_size=valid_vocab_size)

    def _gather_selected_logps(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        response_masks: torch.Tensor,
    ) -> torch.Tensor:
        safe_labels = self._sanitize_scoring_token_ids(
            labels,
            upper_bound=int(logits.size(-1)),
            context="token_select",
        )
        return gather_selected_logps_chunked(
            logits, safe_labels, response_masks, token_chunk_size=64
        )

    def _policy_logps_and_optional_entropy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        response_masks: torch.Tensor,
        *,
        need_entropy: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Score the configured rollout policy, including canonical support."""

        allowed = getattr(self, "_canonical_action_token_ids", None)
        if allowed is not None:
            action_space = getattr(self, "_canonical_action_space", None)
            if action_space is not None and action_space.task == "countdown":
                logps, entropy, _, _ = (
                    restricted_position_action_log_probs_entropy_and_distribution(
                        logits,
                        labels,
                        response_masks,
                        allowed_token_ids_by_position=(
                            action_space.token_ids_by_position
                        ),
                    )
                )
            else:
                logps, entropy = restricted_action_log_probs_and_entropy(
                    logits,
                    labels,
                    response_masks,
                    allowed_token_ids=allowed,
                )
            return logps, entropy if need_entropy else None
        logps = self._gather_selected_logps(logits, labels, response_masks)
        entropy = self._chunked_entropy_from_logits(logits) if need_entropy else None
        return logps, entropy

    def _chunked_entropy_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        shifted_logits = logits[:, :-1, :]
        token_chunk_size = min(64, max(int(shifted_logits.size(1)), 1))
        batch_chunk_size = min(8, max(int(shifted_logits.size(0)), 1))
        batch_chunks = []
        for batch_start in range(0, int(shifted_logits.size(0)), batch_chunk_size):
            batch_stop = min(
                batch_start + batch_chunk_size, int(shifted_logits.size(0))
            )
            token_chunks = []
            for start in range(0, int(shifted_logits.size(1)), token_chunk_size):
                stop = min(start + token_chunk_size, int(shifted_logits.size(1)))
                token_chunks.append(
                    entropy_from_logits(
                        shifted_logits[batch_start:batch_stop, start:stop, :].float()
                    )
                )
            batch_chunks.append(torch.cat(token_chunks, dim=1))
        return torch.cat(batch_chunks, dim=0)

    def _chunked_conditional_content_entropy_from_logits(
        self,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """Entropy over next-token content conditional on not emitting EOS.

        Removing EOS before the softmax makes this quantity exactly
        independent of the EOS logit. It therefore cannot directly reward
        continuing a response rather than terminating it.
        """

        eos_token_id = getattr(getattr(self, "tokenizer", None), "eos_token_id", None)
        if not isinstance(eos_token_id, int):
            raise RuntimeError(
                "conditional-token MaxEnt requires an integer tokenizer EOS id"
            )
        if eos_token_id < 0 or eos_token_id >= int(logits.size(-1)):
            raise RuntimeError(
                "conditional-token MaxEnt tokenizer EOS id is outside model logits"
            )
        shifted_logits = logits[:, :-1, :]
        token_chunk_size = min(64, max(int(shifted_logits.size(1)), 1))
        batch_chunk_size = min(8, max(int(shifted_logits.size(0)), 1))
        batch_chunks = []
        for batch_start in range(0, int(shifted_logits.size(0)), batch_chunk_size):
            batch_stop = min(
                batch_start + batch_chunk_size, int(shifted_logits.size(0))
            )
            token_chunks = []
            for start in range(0, int(shifted_logits.size(1)), token_chunk_size):
                stop = min(start + token_chunk_size, int(shifted_logits.size(1)))
                logit_chunk = shifted_logits[
                    batch_start:batch_stop, start:stop, :
                ].float()
                # Remove the EOS column rather than filling it with -inf:
                # OAT's entropy helper multiplies probabilities by logits,
                # for which 0 * -inf would otherwise be NaN.
                content_logits = torch.cat(
                    (
                        logit_chunk[..., :eos_token_id],
                        logit_chunk[..., eos_token_id + 1 :],
                    ),
                    dim=-1,
                )
                token_chunks.append(entropy_from_logits(content_logits))
            batch_chunks.append(torch.cat(token_chunks, dim=1))
        return torch.cat(batch_chunks, dim=0)
