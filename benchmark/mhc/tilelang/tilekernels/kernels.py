import torch

from .sinkhorn_kernel import _mhc_sinkhorn_bwd, _mhc_sinkhorn_fwd

_N4 = 4
_N8 = 8


def _check_square_tensor(name: str, tensor: torch.Tensor, n: int) -> None:
    if tensor.ndim != 3 or tensor.shape[-2:] != (n, n):
        raise ValueError(f"{name} must be a tensor of size B x {n} x {n}")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")


def _check_max_iter(max_iter: int) -> None:
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")


def _cuda_float_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_cuda:
        tensor = tensor.to("cuda")
    return tensor.to(torch.float32).contiguous()


def _sinkhorn_knopp_tilekernels_forward(
    R: torch.Tensor,
    n: int,
    max_iter: int = 20,
    eps: float = 1e-6,
    token_block_size: int = 16,
) -> dict[str, torch.Tensor]:
    _check_square_tensor("R", R, n)
    _check_max_iter(max_iter)
    src_options = {"device": R.device, "dtype": R.dtype}
    comb_res_mix = _cuda_float_contiguous(R)
    comb_res_mix_out = torch.empty_like(comb_res_mix)

    kernel = _mhc_sinkhorn_fwd(
        hidden_size=n,
        token_block_size=token_block_size,
        repeat=max_iter,
        eps=eps,
    )
    if R.shape[0] > 0:
        kernel(comb_res_mix, comb_res_mix_out)

    return {"T": comb_res_mix_out.to(**src_options)}


def _sinkhorn_knopp_tilekernels_backward(
    G: torch.Tensor,
    R: torch.Tensor,
    n: int,
    max_iter: int = 20,
    eps: float = 1e-6,
    token_block_size: int = 16,
) -> dict[str, torch.Tensor]:
    _check_square_tensor("G", G, n)
    _check_square_tensor("R", R, n)
    _check_max_iter(max_iter)
    if G.shape != R.shape:
        raise ValueError("G and R must have the same shape")
    if G.device != R.device:
        raise ValueError("G and R must be on the same device")

    src_options = {"device": G.device, "dtype": G.dtype}
    grad_output = _cuda_float_contiguous(G)
    x = _cuda_float_contiguous(R)
    grad_input = torch.empty_like(grad_output)

    kernel = _mhc_sinkhorn_bwd(
        hidden_size=n,
        token_block_size=token_block_size,
        repeat=max_iter,
        eps=eps,
    )
    if G.shape[0] > 0:
        kernel(grad_output, x, grad_input)

    return {"D": grad_input.to(**src_options)}


def sinkhorn_knopp_tilekernels_n4_forward(
    R: torch.Tensor,
    max_iter: int = 20,
    eps: float = 1e-6,
    token_block_size: int = 16,
) -> dict[str, torch.Tensor]:
    return _sinkhorn_knopp_tilekernels_forward(R, _N4, max_iter, eps, token_block_size)


def sinkhorn_knopp_tilekernels_n4_backward(
    G: torch.Tensor,
    R: torch.Tensor,
    max_iter: int = 20,
    eps: float = 1e-6,
    token_block_size: int = 16,
) -> dict[str, torch.Tensor]:
    return _sinkhorn_knopp_tilekernels_backward(
        G, R, _N4, max_iter, eps, token_block_size
    )


def sinkhorn_knopp_tilekernels_n8_forward(
    R: torch.Tensor,
    max_iter: int = 20,
    eps: float = 1e-6,
    token_block_size: int = 8,
) -> dict[str, torch.Tensor]:
    return _sinkhorn_knopp_tilekernels_forward(R, _N8, max_iter, eps, token_block_size)


def sinkhorn_knopp_tilekernels_n8_backward(
    G: torch.Tensor,
    R: torch.Tensor,
    max_iter: int = 20,
    eps: float = 1e-6,
    token_block_size: int = 8,
) -> dict[str, torch.Tensor]:
    return _sinkhorn_knopp_tilekernels_backward(
        G, R, _N8, max_iter, eps, token_block_size
    )
