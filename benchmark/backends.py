from collections.abc import Callable
from dataclasses import dataclass

import torch
from mhc.layer import FusedMHC

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


BACKENDS = {
    backend.name: backend
    for backend in (
        Backend("pytorch", _pytorch_sinkhorn),
        Backend("triton", _triton_sinkhorn),
        Backend("cuda-sinkhorn", _cuda_sinkhorn),
        Backend("cuda-projection", _cuda_projection),
    )
}
