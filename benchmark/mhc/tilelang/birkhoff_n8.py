import math

import tilelang
import tilelang.language as T
import torch

_N8 = 8
_REDUCED_SIZE = 7
_EPS = 1e-8
_NEWTON_MAX_ITERS = 20
_LINE_SEARCH_MAX_ITERS = 5
# All shared-memory communication is warp-local and synchronized explicitly.
_WARP_LOCAL_SYNC = {"tl.disable_thread_storage_sync": True}


def _check_n8_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 3 or tensor.shape[-2:] != (_N8, _N8):
        raise ValueError(f"{name} must be a tensor of size B x 8 x 8")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")


def _float32_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.float32).contiguous()


@T.macro
def _warp_reduce_sum_row(value):
    acc = T.alloc_var(T.float32, value)
    acc = acc + T.shfl_xor(acc, 1, 8)
    acc = acc + T.shfl_xor(acc, 2, 8)
    acc = acc + T.shfl_xor(acc, 4, 8)
    return acc


@T.macro
def _warp_reduce_max_row(value):
    acc = T.alloc_var(T.float32, value)
    acc = T.max(acc, T.shfl_xor(acc, 1, 8))
    acc = T.max(acc, T.shfl_xor(acc, 2, 8))
    acc = T.max(acc, T.shfl_xor(acc, 4, 8))
    return acc


@T.macro
def _warp_reduce_sum_col(value0, value1):
    acc = T.alloc_var(T.float32, value0 + value1)
    acc = acc + T.shfl_xor(acc, 8)
    acc = acc + T.shfl_xor(acc, 16)
    return acc


@T.macro
def _warp_reduce_max_col(value0, value1):
    acc = T.alloc_var(T.float32, T.max(value0, value1))
    acc = T.max(acc, T.shfl_xor(acc, 8))
    acc = T.max(acc, T.shfl_xor(acc, 16))
    return acc


@T.macro
def _abs(value):
    return T.if_then_else(value < 0.0, -value, value)


@T.macro
def _compute_f_gradient(beta, R0, R1, col):
    u0 = R0 + beta
    u1 = R1 + beta
    row_max0 = _warp_reduce_max_row(u0)
    row_max1 = _warp_reduce_max_row(u1)
    exp0 = T.exp(u0 - row_max0)
    exp1 = T.exp(u1 - row_max1)
    row_sum0 = _warp_reduce_sum_row(exp0)
    row_sum1 = _warp_reduce_sum_row(exp1)

    T0 = exp0 / row_sum0
    T1 = exp1 / row_sum1
    alpha0 = -row_max0 - T.log(row_sum0)
    alpha1 = -row_max1 - T.log(row_sum1)
    c = _warp_reduce_sum_col(T0, T1)
    g = T.if_then_else(col < _REDUCED_SIZE, c - 1.0, 0.0)
    objective = -_warp_reduce_sum_col(alpha0, alpha1) - _warp_reduce_sum_row(beta)
    gnorm = _warp_reduce_sum_row(_abs(g))
    return c, T0, T1, objective, gnorm


@T.macro
def _initialize_beta_alpha0(R0, R1):
    col_max = _warp_reduce_max_col(R0, R1)
    col_sum = _warp_reduce_sum_col(
        T.exp(R0 - col_max),
        T.exp(R1 - col_max),
    )
    beta = -col_max - T.log(col_sum)
    objective = -_warp_reduce_sum_row(beta)
    beta_last = T.shfl_sync(beta, 7, 8)
    return beta - beta_last, objective


@T.macro
def _sinkhorn_iteration(beta, R0, R1):
    u0 = R0 + beta
    u1 = R1 + beta
    row_max0 = _warp_reduce_max_row(u0)
    row_max1 = _warp_reduce_max_row(u1)
    alpha0 = -row_max0 - T.log(_warp_reduce_sum_row(T.exp(u0 - row_max0)))
    alpha1 = -row_max1 - T.log(_warp_reduce_sum_row(T.exp(u1 - row_max1)))

    v0 = R0 + alpha0
    v1 = R1 + alpha1
    col_max = _warp_reduce_max_col(v0, v1)
    col_sum = _warp_reduce_sum_col(
        T.exp(v0 - col_max),
        T.exp(v1 - col_max),
    )
    next_beta = -col_max - T.log(col_sum)
    beta_last = T.shfl_sync(next_beta, 7, 8)
    return next_beta - beta_last


@T.macro
def _lower_triangle_row(pair_index):
    return T.if_then_else(
        pair_index < 1,
        0,
        T.if_then_else(
            pair_index < 3,
            1,
            T.if_then_else(
                pair_index < 6,
                2,
                T.if_then_else(
                    pair_index < 10,
                    3,
                    T.if_then_else(
                        pair_index < 15,
                        4,
                        T.if_then_else(pair_index < 21, 5, 6),
                    ),
                ),
            ),
        ),
    )


@T.macro
def _build_hessian(T0, T1, diagonal, damping, lane_id, col, H):
    for round_id in T.serial(4):
        pair_index = round_id * _N8 + col
        valid = pair_index < 28
        safe_pair_index = T.if_then_else(valid, pair_index, 27)
        j = _lower_triangle_row(safe_pair_index)
        k = safe_pair_index - j * (j + 1) // 2

        tj0 = T.shfl_sync(T0, j, 8)
        tk0 = T.shfl_sync(T0, k, 8)
        tj1 = T.shfl_sync(T1, j, 8)
        tk1 = T.shfl_sync(T1, k, 8)

        dot = T.alloc_var(T.float32, tj0 * tk0 + tj1 * tk1)
        dot = dot + T.shfl_xor(dot, 8)
        dot = dot + T.shfl_xor(dot, 16)

        diagonal_j = T.shfl_sync(diagonal, j, 8)
        h = T.alloc_var(T.float32, -dot)
        if j == k:
            h = h + diagonal_j + damping

        if T.And(lane_id < _N8, valid):
            H[j, k] = h


@T.macro
def _cholesky_solve(H, rhs, x):
    success = T.alloc_var(T.int32, 1)

    for i in T.serial(_REDUCED_SIZE):
        for j in T.serial(_REDUCED_SIZE):
            if j <= i:
                value = T.alloc_var(T.float32, H[i, j])
                for k in T.serial(_REDUCED_SIZE):
                    if k < j:
                        value = value - H[i, k] * H[j, k]

                if i == j:
                    if T.Or(value <= _EPS, T.Not(T.isfinite(value))):
                        success = 0
                        value = _EPS
                    H[i, i] = T.sqrt(value)
                else:
                    H[i, j] = value / H[j, j]

    if success == 0:
        for i in T.serial(_REDUCED_SIZE):
            x[i] = rhs[i]
    else:
        for i in T.serial(_REDUCED_SIZE):
            value = T.alloc_var(T.float32, rhs[i])
            for k in T.serial(_REDUCED_SIZE):
                if k < i:
                    value = value - H[i, k] * x[k]
            x[i] = value / H[i, i]

        for reverse_i in T.serial(_REDUCED_SIZE):
            i = _REDUCED_SIZE - 1 - reverse_i
            value = T.alloc_var(T.float32, x[i])
            for k in T.serial(_REDUCED_SIZE):
                if k > i:
                    value = value - H[k, i] * x[k]
            x[i] = value / H[i, i]


@T.macro
def _line_search_gamma(k):
    return T.if_then_else(
        k == 0,
        1.0,
        T.if_then_else(
            k == 1,
            0.5,
            T.if_then_else(
                k == 2,
                0.1,
                T.if_then_else(k == 3, 0.05, 0.01),
            ),
        ),
    )


@tilelang.jit(pass_configs=_WARP_LOCAL_SYNC)
def _birkhoff_proj_n8_forward_kernel(R, T_out, tol):
    batch_size = T.dynamic("batch_size")
    dtype = T.float32

    R: T.Tensor((batch_size, _N8, _N8), dtype)  # type: ignore
    T_out: T.Tensor((batch_size, _N8, _N8), dtype)  # type: ignore

    with T.Kernel(batch_size, threads=32) as batch:
        H = T.alloc_shared((_REDUCED_SIZE, _REDUCED_SIZE), dtype)
        rhs = T.alloc_shared((_REDUCED_SIZE,), dtype)
        x = T.alloc_shared((_REDUCED_SIZE,), dtype)

        lane_id = T.get_thread_binding()
        col = lane_id & 7
        row0 = lane_id >> 3
        row1 = row0 + 4
        R0 = R[batch, row0, col]
        R1 = R[batch, row1, col]

        beta = T.alloc_var(T.float32, 0.0)
        init_c, init_T0, init_T1, init_objective, init_gnorm = _compute_f_gradient(
            beta, R0, R1, col
        )
        c = T.alloc_var(T.float32, init_c)
        T0 = T.alloc_var(T.float32, init_T0)
        T1 = T.alloc_var(T.float32, init_T1)
        objective = T.alloc_var(T.float32, init_objective)
        gnorm = T.alloc_var(T.float32, init_gnorm)

        beta_alpha0, objective_alpha0 = _initialize_beta_alpha0(R0, R1)
        if objective_alpha0 < objective:
            beta = beta_alpha0
            c, T0, T1, objective, gnorm = _compute_f_gradient(beta, R0, R1, col)

        if gnorm >= tol:
            for _ in T.serial(_NEWTON_MAX_ITERS):
                damping = T.min(gnorm * gnorm, 1e-3)
                _build_hessian(T0, T1, c, damping, lane_id, col, H)

                g = T.if_then_else(col < _REDUCED_SIZE, c - 1.0, 0.0)
                if lane_id < _REDUCED_SIZE:
                    rhs[lane_id] = -g
                T.sync_warp()

                if lane_id == 0:
                    _cholesky_solve(H, rhs, x)
                T.sync_warp()

                direction = T.if_then_else(col < _REDUCED_SIZE, x[col], 0.0)
                accepted = T.alloc_var(T.int32, 0)

                for k in T.serial(_LINE_SEARCH_MAX_ITERS):
                    candidate_beta = beta + _line_search_gamma(k) * direction
                    (
                        candidate_c,
                        candidate_T0,
                        candidate_T1,
                        candidate_objective,
                        candidate_gnorm,
                    ) = _compute_f_gradient(candidate_beta, R0, R1, col)

                    if T.And(
                        candidate_objective < objective,
                        candidate_gnorm < gnorm,
                    ):
                        beta = candidate_beta
                        c = candidate_c
                        T0 = candidate_T0
                        T1 = candidate_T1
                        objective = candidate_objective
                        gnorm = candidate_gnorm
                        accepted = 1
                        T.loop_break()

                if accepted == 0:
                    beta = _sinkhorn_iteration(beta, R0, R1)
                    c, T0, T1, objective, gnorm = _compute_f_gradient(beta, R0, R1, col)

                if gnorm < tol:
                    T.loop_break()

        T_out[batch, row0, col] = T0
        T_out[batch, row1, col] = T1


@tilelang.jit(pass_configs=_WARP_LOCAL_SYNC)
def _birkhoff_proj_n8_backward_kernel(G, T_out, D):
    batch_size = T.dynamic("batch_size")
    dtype = T.float32

    G: T.Tensor((batch_size, _N8, _N8), dtype)  # type: ignore
    T_out: T.Tensor((batch_size, _N8, _N8), dtype)  # type: ignore
    D: T.Tensor((batch_size, _N8, _N8), dtype)  # type: ignore

    with T.Kernel(batch_size, threads=32) as batch:
        H = T.alloc_shared((_REDUCED_SIZE, _REDUCED_SIZE), dtype)
        rhs = T.alloc_shared((_REDUCED_SIZE,), dtype)
        x = T.alloc_shared((_REDUCED_SIZE,), dtype)

        lane_id = T.get_thread_binding()
        col = lane_id & 7
        row0 = lane_id >> 3
        row1 = row0 + 4

        G0 = G[batch, row0, col]
        G1 = G[batch, row1, col]
        T0 = T_out[batch, row0, col]
        T1 = T_out[batch, row1, col]

        Gamma0 = G0 * T0
        Gamma1 = G1 * T1
        mur0 = _warp_reduce_sum_row(Gamma0)
        mur1 = _warp_reduce_sum_row(Gamma1)
        muc = _warp_reduce_sum_col(Gamma0, Gamma1)
        Tmur = _warp_reduce_sum_col(T0 * mur0, T1 * mur1)
        w_rhs = T.if_then_else(col < _REDUCED_SIZE, muc - Tmur, 0.0)

        diagonal = T0 * 0.0 + 1.0
        _build_hessian(T0, T1, diagonal, 0.0, lane_id, col, H)
        if lane_id < _REDUCED_SIZE:
            rhs[lane_id] = w_rhs
        T.sync_warp()

        if lane_id == 0:
            _cholesky_solve(H, rhs, x)
        T.sync_warp()

        w = T.if_then_else(col < _REDUCED_SIZE, x[col], 0.0)
        v0 = mur0 - _warp_reduce_sum_row(T0 * w)
        v1 = mur1 - _warp_reduce_sum_row(T1 * w)

        D[batch, row0, col] = (G0 - v0 - w) * T0
        D[batch, row1, col] = (G1 - v1 - w) * T1


def birkhoff_proj_n8_forward(
    R: torch.Tensor, tol: float = 1e-6
) -> dict[str, torch.Tensor]:
    _check_n8_tensor("R", R)
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")
    src_options = {"device": R.device, "dtype": R.dtype}
    R_work = _float32_contiguous(R)
    T_out = torch.empty_like(R_work)
    if R.shape[0] > 0:
        _birkhoff_proj_n8_forward_kernel(R_work, T_out, float(tol))
    return {"T": T_out.to(**src_options)}


def birkhoff_proj_n8_backward(
    G: torch.Tensor, T_proj: torch.Tensor
) -> dict[str, torch.Tensor]:
    _check_n8_tensor("G", G)
    _check_n8_tensor("T_proj", T_proj)
    if G.shape != T_proj.shape:
        raise ValueError("G and T_proj must have the same shape")
    if G.device != T_proj.device:
        raise ValueError("G and T_proj must be on the same device")

    src_options = {"device": G.device, "dtype": G.dtype}
    G_work = _float32_contiguous(G)
    T_work = _float32_contiguous(T_proj)
    D = torch.empty_like(G_work)
    if G.shape[0] > 0:
        _birkhoff_proj_n8_backward_kernel(G_work, T_work, D)
    return {"D": D.to(**src_options)}
