"""Frozen sentence embedder backing SetPO's similarity kernel.

SetPO's kernel needs sentence embeddings, and the paper does not name a model.
These cells load MiniLM from a local snapshot through the training
environment's existing ``transformers`` install, mean-pooled over the attention
mask. Two consequences are intended.

No package is added to the shared training environment. Every frozen cohort in
this campaign runs out of the same interpreter, so installing
``sentence-transformers`` to reach one comparator would perturb an environment
other admissions depend on. Mean pooling over ``AutoModel`` hidden states is
what that package does for this model family anyway.

The snapshot path is explicit and local. Compute nodes are not assumed to reach
the network, the weights are staged before submission, and the resolved path is
pinned in the cohort ledger so the kernel that trained a cell can be identified
afterwards.

The embedder is inference-only: weights are frozen, gradients never flow
through it, and it contributes to the advantage as a detached quantity. SetPO
shapes the advantage; it does not train the kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch


#: Longest embedded prefix. The campaign's completions are capped well below
#: this, so the truncation is a guard rather than a routine operation.
MAX_EMBED_TOKENS = 256


@dataclass
class SetPOEmbedder:
    """Lazily loaded, frozen sentence embedder."""

    snapshot: Path
    device: torch.device
    batch_size: int = 64
    _tokenizer: object | None = None
    _model: object | None = None

    @classmethod
    def from_path(
        cls,
        snapshot: str | Path,
        *,
        device: torch.device | str,
        batch_size: int = 64,
    ) -> "SetPOEmbedder":
        resolved = Path(snapshot).resolve()
        if not resolved.is_dir():
            raise ValueError(f"SetPO embedder snapshot is absent: {resolved}")
        return cls(
            snapshot=resolved,
            device=torch.device(device),
            batch_size=int(batch_size),
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.snapshot), local_files_only=True
        )
        model = AutoModel.from_pretrained(
            str(self.snapshot), local_files_only=True
        )
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._model = model.to(self.device)

    @torch.no_grad()
    def embed(self, texts: Sequence[str]) -> torch.Tensor:
        """Return one mean-pooled embedding per text, detached."""

        if not texts:
            raise ValueError("SetPO embedder requires at least one text")
        self._ensure_loaded()
        chunks: list[torch.Tensor] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [str(value) for value in texts[start : start + self.batch_size]]
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_EMBED_TOKENS,
                return_tensors="pt",
            ).to(self.device)
            hidden = self._model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            chunks.append(pooled.detach().float())
        embeddings = torch.cat(chunks, dim=0)
        if not bool(torch.isfinite(embeddings).all()):
            raise ValueError("SetPO embedder produced a non-finite embedding")
        return embeddings
