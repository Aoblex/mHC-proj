import math

import torch
from torch import nn

from . import _internal


def _check_matrix_input(x, n):
    if x.ndim < 2 or x.shape[-2:] != (n, n):
        raise ValueError(f"x must end with dimensions {n} x {n}")
    if not x.is_floating_point():
        raise TypeError("x must be a floating-point tensor")


def _check_tolerance(tol):
    if not isinstance(tol, (int, float)) or not math.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be a positive finite number")


def _check_max_iter(max_iter):
    if not isinstance(max_iter, int) or not 1 <= max_iter <= 20:
        raise ValueError("max_iter must be an integer between 1 and 20")


class MHCProjectionN4Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, R, tol=1e-6):
        in_shape = R.shape
        R = R.reshape(-1, 4, 4)
        res = _internal.torch.birkhoff_proj_n4(R, tol)
        T = res["T"]

        ctx.save_for_backward(T)
        ctx.set_materialize_grads(False)

        return T.reshape(in_shape)

    @staticmethod
    def backward(ctx, grad_output):
        # Early exit if R does not require gradient
        if not ctx.needs_input_grad[0]:
            return None, None

        in_shape = grad_output.shape
        G = grad_output.reshape(-1, 4, 4)
        T = ctx.saved_tensors[0]
        res = _internal.torch.birkhoff_proj_n4_backward(G, T)
        D = res["D"]

        return D.reshape(in_shape), None


class MHCProjectionN4(nn.Module):
    def __init__(self, tol=1e-6):
        super().__init__()
        _check_tolerance(tol)
        self.tol = tol

    def forward(self, x):
        _check_matrix_input(x, 4)
        return MHCProjectionN4Function.apply(x, self.tol)


class MHCProjectionN8Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, R, tol=1e-6):
        in_shape = R.shape
        R = R.reshape(-1, 8, 8)
        res = _internal.torch.birkhoff_proj_n8(R, tol)
        T = res["T"]

        ctx.save_for_backward(T)
        ctx.set_materialize_grads(False)

        return T.reshape(in_shape)

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return None, None

        in_shape = grad_output.shape
        G = grad_output.reshape(-1, 8, 8)
        T = ctx.saved_tensors[0]
        res = _internal.torch.birkhoff_proj_n8_backward(G, T)
        D = res["D"]

        return D.reshape(in_shape), None


class MHCProjectionN8(nn.Module):
    def __init__(self, tol=1e-6):
        super().__init__()
        _check_tolerance(tol)
        self.tol = tol

    def forward(self, x):
        _check_matrix_input(x, 8)
        return MHCProjectionN8Function.apply(x, self.tol)


class MHCSinkhornN4Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, R, max_iter=20):
        in_shape = R.shape
        expon = torch.exp(R.reshape(-1, 4, 4))
        res = _internal.torch.sinkhorn_knopp_n4(expon, max_iter)
        T = res["T"]

        ctx.max_iter = max_iter
        ctx.save_for_backward(expon)
        ctx.set_materialize_grads(False)

        return T.reshape(in_shape)

    @staticmethod
    def backward(ctx, grad_output):
        # Early exit if R does not require gradient
        if not ctx.needs_input_grad[0]:
            return None, None

        in_shape = grad_output.shape
        G = grad_output.reshape(-1, 4, 4)
        expon = ctx.saved_tensors[0]
        res = _internal.torch.sinkhorn_knopp_n4_backward(G, expon, ctx.max_iter)
        D = res["D"] * expon

        return D.reshape(in_shape), None


class MHCSinkhornN4(nn.Module):
    def __init__(self, max_iter=20):
        super().__init__()
        _check_max_iter(max_iter)
        self.max_iter = max_iter

    def forward(self, x):
        _check_matrix_input(x, 4)
        return MHCSinkhornN4Function.apply(x, self.max_iter)


class MHCSinkhornN8Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, R, max_iter=20):
        in_shape = R.shape
        expon = torch.exp(R.reshape(-1, 8, 8))
        res = _internal.torch.sinkhorn_knopp_n8(expon, max_iter)
        T = res["T"]

        ctx.max_iter = max_iter
        ctx.save_for_backward(expon)
        ctx.set_materialize_grads(False)

        return T.reshape(in_shape)

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return None, None

        in_shape = grad_output.shape
        G = grad_output.reshape(-1, 8, 8)
        expon = ctx.saved_tensors[0]
        res = _internal.torch.sinkhorn_knopp_n8_backward(G, expon, ctx.max_iter)
        D = res["D"] * expon

        return D.reshape(in_shape), None


class MHCSinkhornN8(nn.Module):
    def __init__(self, max_iter=20):
        super().__init__()
        _check_max_iter(max_iter)
        self.max_iter = max_iter

    def forward(self, x):
        _check_matrix_input(x, 8)
        return MHCSinkhornN8Function.apply(x, self.max_iter)
