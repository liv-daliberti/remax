"""Execution-layout checks; scientific coefficients do not depend on rank count."""

from dataclasses import asdict
import hashlib
import json
import torch


def validate_update_partition(
    *, local_rows, microbatch, accumulation, global_batch=None, world_size=1
):
    for name, value in (
        ("local_rows", local_rows),
        ("microbatch", microbatch),
        ("accumulation", accumulation),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if global_batch is not None:
        if any(
            isinstance(v, bool) or not isinstance(v, int) or v < 1
            for v in (global_batch, world_size)
        ):
            raise ValueError("global batch and world size must be positive integers")
        if global_batch != microbatch * accumulation * world_size:
            raise ValueError(
                "accumulation must preserve the configured global batch across ranks"
            )
    if local_rows % (microbatch * accumulation):
        raise ValueError(
            "fresh rows must fill complete, equal microbatches and accumulation windows"
        )


def replay_contract_digest(*, groups, settings, tensors):
    """Hash rank-replicated inputs, including order, without serializing gradients."""
    h = hashlib.sha256(
        json.dumps(
            {"groups": [asdict(g) for g in groups], "settings": settings},
            sort_keys=True,
            allow_nan=False,
        ).encode()
    )
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
