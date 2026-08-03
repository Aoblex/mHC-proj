from collections.abc import Callable
from dataclasses import dataclass

import torch
from mhc.layer import FusedMHC
from mhc.tilelang import (
    birkhoff_proj_n4_backward,
    birkhoff_proj_n4_forward,
    birkhoff_proj_n8_backward,
    birkhoff_proj_n8_forward,
)
from mhc.tilelang.tileexamples import (
    sinkhorn_knopp_tileexamples_n4_backward,
    sinkhorn_knopp_tileexamples_n4_forward,
    sinkhorn_knopp_tileexamples_n8_backward,
    sinkhorn_knopp_tileexamples_n8_forward,
)
from mhc.tilelang.tilekernels import (
    sinkhorn_knopp_tilekernels_n4_backward,
    sinkhorn_knopp_tilekernels_n4_forward,
    sinkhorn_knopp_tilekernels_n8_backward,
    sinkhorn_knopp_tilekernels_n8_forward,
)

import mhc_proj

Forward = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class Backend:
    name: str
    build: Callable[[int], Forward]


def _pytorch_sinkhorn(_: int) -> Forward:
    def forward(logits: torch.Tensor) -> torch.Tensor:
        output = logits.exp()
        for _ in range(20):
            output = output / (output.sum(dim=-1, keepdim=True) + 1e-6)
            output = output / (output.sum(dim=-2, keepdim=True) + 1e-6)
        return output

    return torch.compile(forward, dynamic=True)


def _triton_sinkhorn(_: int) -> Forward:
    return FusedMHC(mhc_iters=20)


def _cuda_sinkhorn(n: int) -> Forward:
    module = mhc_proj.MHCSinkhornN4 if n == 4 else mhc_proj.MHCSinkhornN8
    return module(max_iter=20)


def _cuda_projection(n: int) -> Forward:
    module = mhc_proj.MHCProjectionN4 if n == 4 else mhc_proj.MHCProjectionN8
    return module(tol=1e-6)


def _custom_autograd(
    raw_forward: Callable[[torch.Tensor], dict[str, torch.Tensor]],
    raw_backward: Callable[[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]],
    *,
    save_output: bool,
) -> Forward:
    class Function(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits: torch.Tensor) -> torch.Tensor:
            output = raw_forward(logits)["T"]
            ctx.save_for_backward(output if save_output else logits)
            return output

        @staticmethod
        def backward(ctx, gradient: torch.Tensor) -> torch.Tensor:
            (saved,) = ctx.saved_tensors
            return raw_backward(gradient, saved)["D"]

    return Function.apply


def _tilekernels_sinkhorn(n: int) -> Forward:
    if n == 4:
        raw_forward = sinkhorn_knopp_tilekernels_n4_forward
        raw_backward = sinkhorn_knopp_tilekernels_n4_backward
    else:
        raw_forward = sinkhorn_knopp_tilekernels_n8_forward
        raw_backward = sinkhorn_knopp_tilekernels_n8_backward
    return _custom_autograd(raw_forward, raw_backward, save_output=False)


def _tileexamples_sinkhorn(n: int) -> Forward:
    if n == 4:
        raw_forward = sinkhorn_knopp_tileexamples_n4_forward
        raw_backward = sinkhorn_knopp_tileexamples_n4_backward
    else:
        raw_forward = sinkhorn_knopp_tileexamples_n8_forward
        raw_backward = sinkhorn_knopp_tileexamples_n8_backward
    return _custom_autograd(raw_forward, raw_backward, save_output=True)


def _tilelang_projection(n: int) -> Forward:
    if n == 4:
        raw_forward = birkhoff_proj_n4_forward
        raw_backward = birkhoff_proj_n4_backward
    else:
        raw_forward = birkhoff_proj_n8_forward
        raw_backward = birkhoff_proj_n8_backward
    return _custom_autograd(raw_forward, raw_backward, save_output=True)


BACKENDS = {
    backend.name: backend
    for backend in (
        Backend("pytorch", _pytorch_sinkhorn),
        Backend("triton", _triton_sinkhorn),
        Backend("tilekernels", _tilekernels_sinkhorn),
        Backend("tileexamples", _tileexamples_sinkhorn),
        Backend("tilelang-projection", _tilelang_projection),
        Backend("cuda-sinkhorn", _cuda_sinkhorn),
        Backend("cuda-projection", _cuda_projection),
    )
}
