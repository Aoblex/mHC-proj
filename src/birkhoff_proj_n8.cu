// One warp processes one 8x8 matrix.  Lane l owns two entries in one column:
//   (row0, col) = (l / 8, l % 8)
//   (row1, col) = (row0 + 4, col)

#include <cmath>
#include <cuda_runtime.h>

namespace birkhoff_n8 {

constexpr int MATRIX_SIZE = 8;
constexpr int REDUCED_SIZE = 7;
constexpr int BLOCK_DIM = 128;
constexpr int WARPS_PER_BLOCK = BLOCK_DIM / 32;
constexpr int NEWTON_MAX_ITERS = 20;
constexpr int LINE_SEARCH_MAX_ITERS = 5;
// Cholesky minimizes latency; packed LDL^T wins once the batch exposes enough
// independent matrices to hide its serial dependency chain.
constexpr int LDLT_MIN_BATCH_SIZE = 32768;
constexpr float EPSILON = 1e-8f;
constexpr unsigned int FULL_MASK = 0xffffffffu;

__device__ __forceinline__ float warp_reduce_sum_row(float value)
{
    value += __shfl_xor_sync(FULL_MASK, value, 1, 8);
    value += __shfl_xor_sync(FULL_MASK, value, 2, 8);
    value += __shfl_xor_sync(FULL_MASK, value, 4, 8);
    return value;
}

__device__ __forceinline__ float warp_reduce_max_row(float value)
{
    value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 1, 8));
    value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 2, 8));
    value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 4, 8));
    return value;
}

__device__ __forceinline__ float warp_reduce_sum_col(float value0, float value1)
{
    float value = value0 + value1;
    value += __shfl_xor_sync(FULL_MASK, value, 8);
    value += __shfl_xor_sync(FULL_MASK, value, 16);
    return value;
}

__device__ __forceinline__ float warp_reduce_max_col(float value0, float value1)
{
    float value = fmaxf(value0, value1);
    value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 8));
    value = fmaxf(value, __shfl_xor_sync(FULL_MASK, value, 16));
    return value;
}

// Compute the reduced dual objective and row-normalized transport matrix for
// the current beta.  Keeping this separate from the gradient lets line search
// reject candidates whose objective does not improve before doing more
// reductions.  beta[7] is fixed to zero.
__device__ __forceinline__ void compute_f_objective(
    float beta,
    float beta_sum,
    float R0,
    float R1,
    float& out_T0,
    float& out_T1,
    float& out_objective
)
{
    const float u0 = R0 + beta;
    const float u1 = R1 + beta;
    const float row_max0 = warp_reduce_max_row(u0);
    const float row_max1 = warp_reduce_max_row(u1);
    const float exp0 = expf(u0 - row_max0);
    const float exp1 = expf(u1 - row_max1);
    const float row_sum0 = warp_reduce_sum_row(exp0);
    const float row_sum1 = warp_reduce_sum_row(exp1);

    const float T0 = exp0 / row_sum0;
    const float T1 = exp1 / row_sum1;
    const float alpha0 = -row_max0 - logf(row_sum0);
    const float alpha1 = -row_max1 - logf(row_sum1);

    out_T0 = T0;
    out_T1 = T1;
    out_objective =
        -warp_reduce_sum_col(alpha0, alpha1) - beta_sum;
}

// For a small step s, evaluate T' = T * exp(s) / row_sum(T * exp(s))
// and delta_f = sum_i log(sum_j T_ij exp(s_j)) - sum_j s_j.
// expm1/log1p avoid subtracting rounded absolute objectives near convergence.
// Account for the FP32 row mass instead of assuming it is exactly one; the
// column-only exponential is shared by both rows owned by each lane.
__device__ __forceinline__ void compute_relative_candidate(
    float T0,
    float T1,
    float step,
    float step_sum,
    float& out_T0,
    float& out_T1,
    float& out_delta
)
{
    const float change = expm1f(step);
    const float diff0 = warp_reduce_sum_row(T0 * change);
    const float diff1 = warp_reduce_sum_row(T1 * change);
    const float mass0 = warp_reduce_sum_row(T0);
    const float mass1 = warp_reduce_sum_row(T1);
    out_T0 = fmaf(T0, change, T0) / (mass0 + diff0);
    out_T1 = fmaf(T1, change, T1) / (mass1 + diff1);
    out_delta = warp_reduce_sum_col(
        log1pf(diff0 / mass0), log1pf(diff1 / mass1)
    ) - step_sum;
}

__device__ __forceinline__ float compute_gradient(
    float T0,
    float T1,
    int col,
    float& out_c
)
{
    const float c = warp_reduce_sum_col(T0, T1);
    const float g = (col < REDUCED_SIZE) ? (c - 1.0f) : 0.0f;

    out_c = c;
    return warp_reduce_sum_row(fabsf(g));
}

__device__ __forceinline__ float compute_f_gradient(
    float beta,
    float beta_sum,
    float R0,
    float R1,
    int col,
    float& out_c,
    float& out_T0,
    float& out_T1,
    float& out_objective
)
{
    compute_f_objective(
        beta, beta_sum, R0, R1, out_T0, out_T1, out_objective
    );
    return compute_gradient(out_T0, out_T1, col, out_c);
}

// Initialize by setting alpha=0 and normalizing columns, then shift beta so
// beta[7]=0.  The objective is invariant to this gauge shift.
__device__ __forceinline__ float initialize_beta_alpha0(
    float R0,
    float R1,
    float& out_objective,
    float& out_beta_sum
)
{
    const float col_max = warp_reduce_max_col(R0, R1);
    const float col_sum = warp_reduce_sum_col(
        expf(R0 - col_max),
        expf(R1 - col_max)
    );
    float beta = -col_max - logf(col_sum);
    const float beta_sum = warp_reduce_sum_row(beta);
    out_objective = -beta_sum;
    const float beta_last = __shfl_sync(FULL_MASK, beta, 7, 8);
    out_beta_sum = beta_sum - MATRIX_SIZE * beta_last;
    return beta - beta_last;
}

// One complete Sinkhorn cycle used only when all Newton line-search steps fail.
__device__ __forceinline__ float sinkhorn_iteration(
    float beta,
    float R0,
    float R1
)
{
    const float u0 = R0 + beta;
    const float u1 = R1 + beta;
    const float row_max0 = warp_reduce_max_row(u0);
    const float row_max1 = warp_reduce_max_row(u1);
    const float alpha0 =
        -row_max0 - logf(warp_reduce_sum_row(expf(u0 - row_max0)));
    const float alpha1 =
        -row_max1 - logf(warp_reduce_sum_row(expf(u1 - row_max1)));

    const float v0 = R0 + alpha0;
    const float v1 = R1 + alpha1;
    const float col_max = warp_reduce_max_col(v0, v1);
    const float col_sum = warp_reduce_sum_col(
        expf(v0 - col_max),
        expf(v1 - col_max)
    );
    float next_beta = -col_max - logf(col_sum);
    const float beta_last = __shfl_sync(FULL_MASK, next_beta, 7, 8);
    return next_beta - beta_last;
}

// Build the lower triangle of
//   diag(diagonal[:7]) - T[:, :7]^T T[:, :7] + damping * I.
// Lane v owns (v, v + round mod 7), so one endpoint is always local.  The
// cyclic schedule covers all 28 lower-triangle entries exactly once.
__device__ __forceinline__ void build_hessian(
    float T0,
    float T1,
    float diagonal,
    float damping,
    int lane_id,
    int col,
    float* H
)
{
    #pragma unroll
    for (int round = 0; round < 4; ++round)
    {
        const bool valid = col < REDUCED_SIZE;
        const int owner = valid ? col : 0;
        int other = owner + round;
        if (other >= REDUCED_SIZE)
        {
            other -= REDUCED_SIZE;
        }

        float other_T0 = T0;
        float other_T1 = T1;
        if (round != 0)
        {
            other_T0 = __shfl_sync(FULL_MASK, T0, other, 8);
            other_T1 = __shfl_sync(FULL_MASK, T1, other, 8);
        }

        float dot = T0 * other_T0 + T1 * other_T1;
        dot += __shfl_xor_sync(FULL_MASK, dot, 8);
        dot += __shfl_xor_sync(FULL_MASK, dot, 16);

        float h = -dot;
        if (round == 0)
        {
            h += diagonal + damping;
        }

        if (lane_id < REDUCED_SIZE)
        {
            const int j = max(owner, other);
            const int k = min(owner, other);
            H[j * REDUCED_SIZE + k] = h;
        }
    }
}

// Lane 0 factors H in place, then solves Hx=rhs.  The strict lower triangle
// stores the Cholesky factor, while the diagonal stores its reciprocal so all
// subsequent divisions become multiplications.  Returning false lets forward
// fall back to the gradient direction if finite-precision pivots are invalid.
__device__ __forceinline__ bool cholesky_solve(
    float* H,
    const float* rhs,
    float* x
)
{
    bool success = true;

    #pragma unroll
    for (int i = 0; i < REDUCED_SIZE; ++i)
    {
        #pragma unroll
        for (int j = 0; j <= i; ++j)
        {
            float value = H[i * REDUCED_SIZE + j];
            #pragma unroll
            for (int k = 0; k < j; ++k)
            {
                value -=
                    H[i * REDUCED_SIZE + k] *
                    H[j * REDUCED_SIZE + k];
            }

            if (i == j)
            {
                if (!(value > EPSILON) || !isfinite(value))
                {
                    success = false;
                    value = EPSILON;
                }
                H[i * REDUCED_SIZE + i] = rsqrtf(value);
            }
            else
            {
                H[i * REDUCED_SIZE + j] =
                    value * H[j * REDUCED_SIZE + j];
            }
        }
    }

    if (!success)
    {
        #pragma unroll
        for (int i = 0; i < REDUCED_SIZE; ++i)
        {
            x[i] = rhs[i];
        }
        return false;
    }

    // Forward substitution: Ly=rhs.  x temporarily stores y.
    #pragma unroll
    for (int i = 0; i < REDUCED_SIZE; ++i)
    {
        float value = rhs[i];
        #pragma unroll
        for (int k = 0; k < i; ++k)
        {
            value -= H[i * REDUCED_SIZE + k] * x[k];
        }
        x[i] = value * H[i * REDUCED_SIZE + i];
    }

    // Back substitution: L^T x=y, performed in place.
    #pragma unroll
    for (int i = REDUCED_SIZE - 1; i >= 0; --i)
    {
        float value = x[i];
        #pragma unroll
        for (int k = i + 1; k < REDUCED_SIZE; ++k)
        {
            value -= H[k * REDUCED_SIZE + i] * x[k];
        }
        x[i] = value * H[i * REDUCED_SIZE + i];
    }

    return true;
}

// Lane 0 solves Hx=rhs using a packed, register-resident LDL^T factor.
// L has an implicit unit diagonal and stores only its 21 strict-lower entries.
// Returning false lets forward fall back to the gradient direction when a
// finite-precision pivot is invalid.
__device__ __forceinline__ bool ldlt_solve(
    const float* H,
    const float* rhs,
    float* x
)
{
    float L[REDUCED_SIZE * (REDUCED_SIZE - 1) / 2];
    float D[REDUCED_SIZE];
    float work[REDUCED_SIZE];

    #pragma unroll
    for (int k = 0; k < REDUCED_SIZE; ++k)
    {
        const int base_k = k * (k - 1) / 2;
        float diagonal = H[k * REDUCED_SIZE + k];
        #pragma unroll
        for (int r = 0; r < k; ++r)
        {
            const float lkr = L[base_k + r];
            diagonal -= lkr * lkr * D[r];
        }
        if (!(diagonal > EPSILON) || !isfinite(diagonal))
        {
            #pragma unroll
            for (int i = 0; i < REDUCED_SIZE; ++i)
            {
                x[i] = rhs[i];
            }
            return false;
        }
        D[k] = diagonal;

        #pragma unroll
        for (int i = k + 1; i < REDUCED_SIZE; ++i)
        {
            const int base_i = i * (i - 1) / 2;
            float value = H[i * REDUCED_SIZE + k];
            #pragma unroll
            for (int r = 0; r < k; ++r)
            {
                value -= L[base_i + r] * D[r] * L[base_k + r];
            }
            value /= diagonal;
            if (!isfinite(value))
            {
                #pragma unroll
                for (int j = 0; j < REDUCED_SIZE; ++j)
                {
                    x[j] = rhs[j];
                }
                return false;
            }
            L[base_i + k] = value;
        }
    }

    #pragma unroll
    for (int i = 0; i < REDUCED_SIZE; ++i)
    {
        const int base_i = i * (i - 1) / 2;
        float value = rhs[i];
        #pragma unroll
        for (int j = 0; j < i; ++j)
        {
            value -= L[base_i + j] * work[j];
        }
        work[i] = value;
    }

    #pragma unroll
    for (int i = 0; i < REDUCED_SIZE; ++i)
    {
        work[i] /= D[i];
    }

    #pragma unroll
    for (int i = REDUCED_SIZE - 1; i >= 0; --i)
    {
        float value = work[i];
        #pragma unroll
        for (int j = i + 1; j < REDUCED_SIZE; ++j)
        {
            value -= L[j * (j - 1) / 2 + i] * work[j];
        }
        work[i] = value;
    }

    #pragma unroll
    for (int i = 0; i < REDUCED_SIZE; ++i)
    {
        x[i] = work[i];
    }
    return true;
}

template<bool USE_LDLT>
__device__ __forceinline__ void solve_linear_system(
    float* H,
    const float* rhs,
    float* x
)
{
    if constexpr (USE_LDLT)
    {
        ldlt_solve(H, rhs, x);
    }
    else
    {
        cholesky_solve(H, rhs, x);
    }
}

template<bool USE_LDLT>
__global__ void birkhoff_proj_n8_kernel(
    const float* __restrict__ R,
    float* __restrict__ T,
    float tol,
    int batch_size
)
{
    __shared__ float shared_H[WARPS_PER_BLOCK][REDUCED_SIZE * REDUCED_SIZE];
    __shared__ float shared_rhs[WARPS_PER_BLOCK][REDUCED_SIZE];
    __shared__ float shared_x[WARPS_PER_BLOCK][REDUCED_SIZE];

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int instance_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (instance_id >= batch_size)
    {
        return;
    }

    const int col = lane_id & 7;
    const int row0 = lane_id >> 3;
    const int row1 = row0 + 4;
    const int matrix_offset = instance_id * MATRIX_SIZE * MATRIX_SIZE;
    const int index0 = matrix_offset + row0 * MATRIX_SIZE + col;
    const int index1 = matrix_offset + row1 * MATRIX_SIZE + col;
    const float R0 = R[index0];
    const float R1 = R[index1];

    float beta = 0.0f;
    float beta_sum = 0.0f;
    float c, T0, T1, objective;
    float gnorm = compute_f_gradient(
        beta, beta_sum, R0, R1, col, c, T0, T1, objective
    );

    float objective_alpha0;
    float beta_sum_alpha0;
    const float beta_alpha0 =
        initialize_beta_alpha0(
            R0, R1, objective_alpha0, beta_sum_alpha0
        );
    if (objective_alpha0 < objective)
    {
        beta = beta_alpha0;
        beta_sum = beta_sum_alpha0;
        gnorm = compute_f_gradient(
            beta, beta_sum, R0, R1, col, c, T0, T1, objective
        );
    }

    constexpr float gamma_list[LINE_SEARCH_MAX_ITERS] = {
        1.0f, 0.5f, 0.1f, 0.05f, 0.01f
    };

    if (gnorm >= tol)
    {
        #pragma unroll 1
        for (int iter = 0; iter < NEWTON_MAX_ITERS; ++iter)
        {
            const float damping = fminf(gnorm * gnorm, 1e-3f);
            build_hessian(
                T0,
                T1,
                c,
                damping,
                lane_id,
                col,
                shared_H[warp_id]
            );

            const float g =
                (col < REDUCED_SIZE) ? (c - 1.0f) : 0.0f;
            if (lane_id < REDUCED_SIZE)
            {
                shared_rhs[warp_id][lane_id] = -g;
            }
            __syncwarp(FULL_MASK);

            if (lane_id == 0)
            {
                solve_linear_system<USE_LDLT>(
                    shared_H[warp_id],
                    shared_rhs[warp_id],
                    shared_x[warp_id]
                );
            }
            __syncwarp(FULL_MASK);

            const float direction =
                (col < REDUCED_SIZE)
                    ? shared_x[warp_id][col]
                    : 0.0f;
            const float direction_sum = warp_reduce_sum_row(direction);
            const float direction_max = warp_reduce_max_row(fabsf(direction));
            bool accepted = false;

            #pragma unroll 1
            for (int k = 0; k < LINE_SEARCH_MAX_ITERS; ++k)
            {
                const float candidate_beta =
                    beta + gamma_list[k] * direction;
                const float candidate_beta_sum =
                    beta_sum + gamma_list[k] * direction_sum;
                float candidate_c;
                float candidate_T0;
                float candidate_T1;
                float objective_delta;
                // Keep the multiplicative update close to one. Large steps
                // use logits so entries lost to underflow can be recovered.
                if (gamma_list[k] * direction_max < 0.5f)
                {
                    compute_relative_candidate(
                        T0,
                        T1,
                        gamma_list[k] * direction,
                        gamma_list[k] * direction_sum,
                        candidate_T0,
                        candidate_T1,
                        objective_delta
                    );
                }
                else
                {
                    float candidate_objective;
                    compute_f_objective(
                        candidate_beta, candidate_beta_sum, R0, R1,
                        candidate_T0, candidate_T1, candidate_objective
                    );
                    objective_delta = candidate_objective - objective;
                }

                if (!(objective_delta < 0.0f))
                {
                    continue;
                }

                const float candidate_gnorm = compute_gradient(
                    candidate_T0, candidate_T1, col, candidate_c
                );
                if (!(candidate_gnorm < gnorm))
                {
                    continue;
                }

                beta = candidate_beta;
                beta_sum = candidate_beta_sum;
                c = candidate_c;
                T0 = candidate_T0;
                T1 = candidate_T1;
                objective += objective_delta;
                gnorm = candidate_gnorm;
                accepted = true;
                break;
            }

            if (!accepted)
            {
                beta = sinkhorn_iteration(beta, R0, R1);
                beta_sum = warp_reduce_sum_row(beta);
                gnorm = compute_f_gradient(
                    beta,
                    beta_sum,
                    R0,
                    R1,
                    col,
                    c,
                    T0,
                    T1,
                    objective
                );
            }

            if (gnorm < tol)
            {
                break;
            }
        }
    }

    T[index0] = T0;
    T[index1] = T1;
}

__global__ void birkhoff_proj_n8_backward_kernel(
    const float* __restrict__ G,
    const float* __restrict__ T,
    float* __restrict__ D,
    int batch_size
)
{
    __shared__ float shared_H[WARPS_PER_BLOCK][REDUCED_SIZE * REDUCED_SIZE];
    __shared__ float shared_rhs[WARPS_PER_BLOCK][REDUCED_SIZE];
    __shared__ float shared_x[WARPS_PER_BLOCK][REDUCED_SIZE];

    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int instance_id = blockIdx.x * WARPS_PER_BLOCK + warp_id;
    if (instance_id >= batch_size)
    {
        return;
    }

    const int col = lane_id & 7;
    const int row0 = lane_id >> 3;
    const int row1 = row0 + 4;
    const int matrix_offset = instance_id * MATRIX_SIZE * MATRIX_SIZE;
    const int index0 = matrix_offset + row0 * MATRIX_SIZE + col;
    const int index1 = matrix_offset + row1 * MATRIX_SIZE + col;
    const float G0 = G[index0];
    const float G1 = G[index1];
    const float T0 = T[index0];
    const float T1 = T[index1];

    const float Gamma0 = G0 * T0;
    const float Gamma1 = G1 * T1;
    const float mur0 = warp_reduce_sum_row(Gamma0);
    const float mur1 = warp_reduce_sum_row(Gamma1);
    const float muc = warp_reduce_sum_col(Gamma0, Gamma1);
    const float Tmur = warp_reduce_sum_col(T0 * mur0, T1 * mur1);
    const float rhs =
        (col < REDUCED_SIZE) ? (muc - Tmur) : 0.0f;

    build_hessian(
        T0,
        T1,
        1.0f,
        0.0f,
        lane_id,
        col,
        shared_H[warp_id]
    );
    if (lane_id < REDUCED_SIZE)
    {
        shared_rhs[warp_id][lane_id] = rhs;
    }
    __syncwarp(FULL_MASK);

    if (lane_id == 0)
    {
        cholesky_solve(
            shared_H[warp_id],
            shared_rhs[warp_id],
            shared_x[warp_id]
        );
    }
    __syncwarp(FULL_MASK);

    const float w =
        (col < REDUCED_SIZE) ? shared_x[warp_id][col] : 0.0f;
    const float v0 = mur0 - warp_reduce_sum_row(T0 * w);
    const float v1 = mur1 - warp_reduce_sum_row(T1 * w);

    D[index0] = (G0 - v0 - w) * T0;
    D[index1] = (G1 - v1 - w) * T1;
}

}  // namespace birkhoff_n8

void birkhoff_proj_n8(
    const float* R,
    float* T,
    float tol,
    int batch_size,
    cudaStream_t stream
)
{
    const int num_blocks =
        (batch_size + birkhoff_n8::WARPS_PER_BLOCK - 1) /
        birkhoff_n8::WARPS_PER_BLOCK;
    if (batch_size >= birkhoff_n8::LDLT_MIN_BATCH_SIZE)
    {
        birkhoff_n8::birkhoff_proj_n8_kernel<true><<<
            num_blocks,
            birkhoff_n8::BLOCK_DIM,
            0,
            stream
        >>>(R, T, tol, batch_size);
    }
    else
    {
        birkhoff_n8::birkhoff_proj_n8_kernel<false><<<
            num_blocks,
            birkhoff_n8::BLOCK_DIM,
            0,
            stream
        >>>(R, T, tol, batch_size);
    }
}

void birkhoff_proj_n8_backward(
    const float* G,
    const float* T,
    float* D,
    int batch_size,
    cudaStream_t stream
)
{
    const int num_blocks =
        (batch_size + birkhoff_n8::WARPS_PER_BLOCK - 1) /
        birkhoff_n8::WARPS_PER_BLOCK;
    birkhoff_n8::birkhoff_proj_n8_backward_kernel<<<
        num_blocks,
        birkhoff_n8::BLOCK_DIM,
        0,
        stream
    >>>(G, T, D, batch_size);
}
