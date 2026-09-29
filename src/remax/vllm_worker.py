"""vLLM worker extensions for lossless level-2 sleep/wake cycles."""

from __future__ import annotations

from typing import Any

import torch
from oat.utils.distributed import WorkerWrap


def _named_model_buffers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return every model buffer, including non-persistent runtime caches."""

    # Shared buffers need one backup per storage; restoring that tensor repairs
    # every alias while avoiding one CPU copy per transformer layer.
    return dict(model.named_buffers())


class Level2SafeWorkerWrap(WorkerWrap):
    """Preserve model buffers that vLLM level-2 sleep otherwise discards.

    vLLM allocates parameters and non-parameter buffers in the same ``weights``
    memory pool. Level-2 sleep discards that entire pool. OAT subsequently
    reloads parameters, but buffers such as Qwen's rotary ``cos_sin_cache`` are
    not part of ``named_parameters()`` and would remain uninitialized. Keep a
    small CPU copy of every buffer across the discard window and verify the
    restoration before generation can resume.
    """

    _oat_zero_buffer_backup: dict[str, torch.Tensor] | None = None

    def backup_model_buffers(self) -> dict[str, int]:
        if self._oat_zero_buffer_backup is not None:
            raise RuntimeError("model-buffer backup already exists")
        buffers = _named_model_buffers(self.model_runner.model)
        self._oat_zero_buffer_backup = {
            name: value.detach().cpu().clone() for name, value in buffers.items()
        }
        return {
            "buffer_count": len(buffers),
            "buffer_bytes": sum(
                value.numel() * value.element_size() for value in buffers.values()
            ),
        }

    def restore_model_buffers(self) -> dict[str, int]:
        backup = self._oat_zero_buffer_backup
        if backup is None:
            raise RuntimeError("model-buffer restore requested without a backup")
        buffers = _named_model_buffers(self.model_runner.model)
        if buffers.keys() != backup.keys():
            missing = sorted(backup.keys() - buffers.keys())
            added = sorted(buffers.keys() - backup.keys())
            raise RuntimeError(
                f"model buffers changed across level-2 sleep: missing={missing} added={added}"
            )
        restored_bytes = 0
        with torch.no_grad():
            for name, saved in backup.items():
                target = buffers[name]
                if target.shape != saved.shape or target.dtype != saved.dtype:
                    raise RuntimeError(
                        "model buffer metadata changed across level-2 sleep: "
                        f"{name} target={target.shape}/{target.dtype} "
                        f"backup={saved.shape}/{saved.dtype}"
                    )
                target.copy_(saved, non_blocking=False)
                if not torch.equal(target.detach().cpu(), saved):
                    raise RuntimeError(f"model buffer restoration failed: {name}")
                restored_bytes += saved.numel() * saved.element_size()
        self._oat_zero_buffer_backup = None
        return {"buffer_count": len(backup), "buffer_bytes": restored_bytes}


class PinnedWorkerExtensionArgs(dict[str, Any]):
    """Keep OAT's actor initializer from replacing our safe worker extension."""

    worker_extension = "remax.vllm_worker.Level2SafeWorkerWrap"

    def update(self, *args: Any, **kwargs: Any) -> None:
        super().update(*args, **kwargs)
        super().__setitem__("worker_extension_cls", self.worker_extension)
