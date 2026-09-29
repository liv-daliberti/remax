"""Swap DeepSpeed's JIT-compiled FusedAdam for torch's prebuilt fused AdamW.

DeepSpeed builds its FusedAdam CUDA kernel at first use via nvcc. After the
2026-07 cluster maintenance (CUDA 13-only system toolkit) that JIT step cannot
complete on compute nodes, so no job can construct its optimizer. torch ships
an equivalent fused multi-tensor AdamW kernel precompiled inside the wheel
(torch.optim.AdamW(fused=True)); this module rebinds the FusedAdam name to a
factory returning that optimizer, so the same AdamW update runs with no
compiler involved.

install() must run before oat.utils.deepspeed is imported (it binds FusedAdam
via `from deepspeed.ops.adam import FusedAdam`); if that module is already
loaded, its binding is re-patched too. Set OAT_ZERO_DISABLE_FUSED_ADAM_SHIM=1
to keep DeepSpeed's own kernel.
"""

import os
import sys

import torch


def torch_fused_adamw(
    params,
    lr=1e-3,
    bias_correction=True,
    betas=(0.9, 0.999),
    eps=1e-8,
    adam_w_mode=True,
    weight_decay=0.0,
    amsgrad=False,
    set_grad_none=True,
):
    """Factory matching deepspeed.ops.adam.FusedAdam's signature.

    Returns a plain torch.optim.AdamW so DeepSpeed's ZeRO supported-optimizer
    check (which matches on exact type) still passes.
    """
    if not adam_w_mode:
        raise ValueError(
            "fused_adam_shim only supports adam_w_mode=True (AdamW update)"
        )
    if not bias_correction:
        raise ValueError("fused_adam_shim only supports bias_correction=True")
    return torch.optim.AdamW(
        params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
        fused=True,
    )


def install() -> bool:
    if os.environ.get("OAT_ZERO_DISABLE_FUSED_ADAM_SHIM") == "1":
        return False
    import deepspeed.ops.adam as _ds_adam

    _ds_adam.FusedAdam = torch_fused_adamw
    _oat_ds = sys.modules.get("oat.utils.deepspeed")
    if _oat_ds is not None:
        _oat_ds.FusedAdam = torch_fused_adamw
    return True
