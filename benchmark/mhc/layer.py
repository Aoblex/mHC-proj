import torch
import triton
from torch import nn

from .kernels import _mhc_sinkhorn_bwd_kernel, _mhc_sinkhorn_fwd_kernel


class MHCSinkhornFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, num_iters=5):
        W = W.contiguous()
        B, n, _ = W.shape
        M = torch.empty_like(W)

        NN = n * n
        BLOCK_SIZE = triton.next_power_of_2(NN)

        save_history = ctx.needs_input_grad[0]
        if save_history:
            # Exact backward needs both normalization states from every iter.
            H = torch.empty(
                (B, num_iters * 2 * NN),
                device=W.device,
                dtype=torch.float32,
            )
            hist_stride = H.stride(0)
        else:
            # SAVE_HISTORY=False removes all H accesses at compile time.
            H = M
            hist_stride = 0

        _mhc_sinkhorn_fwd_kernel[(B,)](
            W,
            M,
            H,
            W.stride(0),
            hist_stride,
            N_LANES=n,
            ITERS=num_iters,
            BLOCK_SIZE=BLOCK_SIZE,
            SAVE_HISTORY=save_history,
            num_warps=1 if BLOCK_SIZE <= 64 else 4,
        )

        if save_history:
            ctx.save_for_backward(W, H)
        ctx.n_lanes = n
        ctx.num_iters = num_iters
        return M

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        W, H = ctx.saved_tensors

        grad_W = torch.empty_like(W)
        B, n, iters = W.shape[0], ctx.n_lanes, ctx.num_iters

        NN = n * n
        BLOCK_SIZE = triton.next_power_of_2(NN)

        _mhc_sinkhorn_bwd_kernel[(B,)](
            grad_output,
            W,
            H,
            grad_W,
            W.stride(0),
            H.stride(0),
            N_LANES=n,
            ITERS=iters,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=1 if BLOCK_SIZE <= 64 else 4,
        )

        return grad_W, None


class FusedMHC(nn.Module):
    def __init__(self, mhc_iters=5):
        super().__init__()
        if not isinstance(mhc_iters, int) or mhc_iters < 1:
            raise ValueError("mhc_iters must be a positive integer")
        self.mhc_iters = mhc_iters

    @torch.amp.autocast("cuda", enabled=False)
    def forward(self, x):
        if x.ndim < 2 or x.shape[-2] != x.shape[-1]:
            raise ValueError("x must end with two equal matrix dimensions")
        if not x.is_floating_point():
            raise TypeError("x must be a floating-point tensor")
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")

        input_dtype = x.dtype
        input_shape = x.shape
        n_lanes = input_shape[-1]

        x_flat = x.contiguous().view(-1, n_lanes, n_lanes).float()
        out_flat = MHCSinkhornFunction.apply(x_flat, self.mhc_iters)
        return out_flat.view(input_shape).to(input_dtype)


def mhc_warmup(n_lanes=4, batch_size=32, device="cuda"):
    if not torch.cuda.is_available():
        return

    print(f"🔥 Warming up MHC kernels for {n_lanes} lanes...")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        dummy = torch.randn(
            batch_size, n_lanes, n_lanes, device=device, requires_grad=True
        )
        layer = FusedMHC(mhc_iters=5).to(device)
        out = layer(dummy)
        out.sum().backward()

    torch.cuda.current_stream().wait_stream(stream)
    print("✅ MHC Warmup Complete.")
