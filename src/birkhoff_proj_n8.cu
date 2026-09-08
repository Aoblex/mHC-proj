// Small-batch schedule: one warp processes one 8x8 matrix.
// Lane l owns two entries in one column:
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
// All schedules use Cholesky. Saturated batches use row-owned forward
// and half-warp backward; small batches retain one warp per matrix.
constexpr int LARGE_BATCH_MIN_SIZE = 32768;
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
                cholesky_solve(
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

// Large-batch backward: two independent matrices per warp, four entries
// per thread.
namespace birkhoff_n8::halfwarp {
constexpr int THREADS_PER_MATRIX = 16;
constexpr int ROW_GROUPS = THREADS_PER_MATRIX / MATRIX_SIZE;
constexpr int ROWS_PER_THREAD = MATRIX_SIZE / ROW_GROUPS;
constexpr int MATRICES_PER_BLOCK = BLOCK_DIM / THREADS_PER_MATRIX;
using Values = float[ROWS_PER_THREAD];

__device__ __forceinline__ unsigned int matrix_mask()
{
    const int base = (threadIdx.x & 31) & ~(THREADS_PER_MATRIX - 1);
    return (0xffffffffu >> (32 - THREADS_PER_MATRIX)) << base;
}

__device__ __forceinline__ float warp_reduce_sum_row(float value)
{
    value += __shfl_xor_sync(matrix_mask(), value, 1, 8);
    value += __shfl_xor_sync(matrix_mask(), value, 2, 8);
    value += __shfl_xor_sync(matrix_mask(), value, 4, 8);
    return value;
}

__device__ __forceinline__ float warp_reduce_sum_col(const Values& values)
{
    // Match the original 32-thread summation tree, including its grouping.
    float first = values[0] + values[2];
    float second = values[1] + values[3];
    first += __shfl_xor_sync(matrix_mask(), first, 8, THREADS_PER_MATRIX);
    second += __shfl_xor_sync(matrix_mask(), second, 8, THREADS_PER_MATRIX);
    return first + second;
}

// Preserve the cyclic schedule for the 28 lower-triangle entries.
__device__ __forceinline__ void build_hessian(
    const Values& T, float diagonal, float damping, int lane, int col, float* H)
{
    #pragma unroll
    for (int round = 0; round < 4; ++round)
    {
        const int owner = col < REDUCED_SIZE ? col : 0;
        int other = owner + round;
        if (other >= REDUCED_SIZE) other -= REDUCED_SIZE;
        Values products;
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r)
        {
            const float t = round == 0 ? T[r] : __shfl_sync(matrix_mask(), T[r], other, 8);
            products[r] = T[r] * t;
        }
        float h = -warp_reduce_sum_col(products);
        if (round == 0) h += diagonal + damping;
        if (lane < REDUCED_SIZE)
            H[max(owner, other) * REDUCED_SIZE + min(owner, other)] = h;
    }
}

__global__ void birkhoff_proj_n8_backward_kernel(
    const float* __restrict__ G, const float* __restrict__ T,
    float* __restrict__ D, int batch_size)
{
    __shared__ float shared_H[MATRICES_PER_BLOCK][REDUCED_SIZE * REDUCED_SIZE];
    __shared__ float shared_rhs[MATRICES_PER_BLOCK][REDUCED_SIZE];
    __shared__ float shared_x[MATRICES_PER_BLOCK][REDUCED_SIZE];
    const int group = threadIdx.x / THREADS_PER_MATRIX;
    const int lane = threadIdx.x % THREADS_PER_MATRIX;
    const int instance = blockIdx.x * MATRICES_PER_BLOCK + group;
    if (instance >= batch_size) return;
    const int col = lane & 7;
    const int row = lane >> 3;
    const int offset = instance * MATRIX_SIZE * MATRIX_SIZE;
    Values values_G, values_T, gamma, mur, weighted_mur;
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r)
    {
        const int index = offset + (row + r * ROW_GROUPS) * MATRIX_SIZE + col;
        values_G[r] = G[index];
        values_T[r] = T[index];
        gamma[r] = values_G[r] * values_T[r];
        mur[r] = warp_reduce_sum_row(gamma[r]);
        weighted_mur[r] = values_T[r] * mur[r];
    }
    const float muc = warp_reduce_sum_col(gamma);
    const float Tmur = warp_reduce_sum_col(weighted_mur);
    const float rhs = col < REDUCED_SIZE ? muc - Tmur : 0.0f;
    build_hessian(values_T, 1.0f, 0.0f, lane, col, shared_H[group]);
    if (lane < REDUCED_SIZE) shared_rhs[group][lane] = rhs;
    __syncwarp(matrix_mask());
    if (lane == 0) cholesky_solve(shared_H[group], shared_rhs[group], shared_x[group]);
    __syncwarp(matrix_mask());
    const float w = col < REDUCED_SIZE ? shared_x[group][col] : 0.0f;
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r)
    {
        const float v = mur[r] - warp_reduce_sum_row(values_T[r] * w);
        D[offset + (row + r * ROW_GROUPS) * MATRIX_SIZE + col] = (values_G[r] - v - w) * values_T[r];
    }
}
} // namespace birkhoff_n8::halfwarp

// Large-batch forward: four threads own two complete rows each.
// Row reductions stay in registers; a warp advances eight independent matrices.
namespace birkhoff_n8::quarterwarp {
constexpr int THREADS_PER_MATRIX = 4;
constexpr int MATRICES_PER_BLOCK = BLOCK_DIM / THREADS_PER_MATRIX;
using Row = float[8];
using Matrix = float[2][8];

__device__ __forceinline__ unsigned matrix_mask()
{
    return 0xfu << (threadIdx.x & 28);
}

__device__ __forceinline__ float column_sum(float v)
{
    v += __shfl_xor_sync(matrix_mask(), v, 1, 4);
    v += __shfl_xor_sync(matrix_mask(), v, 2, 4);
    return v;
}

__device__ __forceinline__ float column_max(float v)
{
    v = fmaxf(v, __shfl_xor_sync(matrix_mask(), v, 1, 4));
    v = fmaxf(v, __shfl_xor_sync(matrix_mask(), v, 2, 4));
    return v;
}

__device__ __forceinline__ float row_sum(const Row& v)
{
    // Preserve the warp reduction tree without contraction across its stages.
    const float a = __fadd_rn(v[0], v[1]);
    const float b = __fadd_rn(v[2], v[3]);
    const float c = __fadd_rn(v[4], v[5]);
    const float d = __fadd_rn(v[6], v[7]);
    return __fadd_rn(__fadd_rn(a, b), __fadd_rn(c, d));
}

__device__ __forceinline__ float row_max(const Row& v)
{
    return fmaxf(fmaxf(fmaxf(v[0], v[1]), fmaxf(v[2], v[3])),
                 fmaxf(fmaxf(v[4], v[5]), fmaxf(v[6], v[7])));
}

__device__ __forceinline__ float objective(
    const Matrix& R, const Row& beta, float beta_sum, Matrix& T)
{
    float alpha[2];
    #pragma unroll
    for (int r = 0; r < 2; ++r)
    {
        Row u, e;
        #pragma unroll
        for (int j = 0; j < 8; ++j) u[j] = R[r][j] + beta[j];
        const float maximum = row_max(u);
        #pragma unroll
        for (int j = 0; j < 8; ++j) e[j] = expf(u[j] - maximum);
        const float mass = row_sum(e);
        #pragma unroll
        for (int j = 0; j < 8; ++j) T[r][j] = e[j] / mass;
        alpha[r] = -maximum - logf(mass);
    }
    return -column_sum(alpha[0] + alpha[1]) - beta_sum;
}

__device__ __forceinline__ float gradient(const Matrix& T, Row& c)
{
    Row g;
    #pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        c[j] = column_sum(T[0][j] + T[1][j]);
        g[j] = j < 7 ? fabsf(c[j] - 1.0f) : 0.0f;
    }
    return row_sum(g);
}

__device__ __forceinline__ float initialize(
    const Matrix& R, Row& beta, float& beta_sum)
{
    #pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        const float maximum = column_max(fmaxf(R[0][j], R[1][j]));
        const float mass = column_sum(expf(R[0][j] - maximum) + expf(R[1][j] - maximum));
        beta[j] = -maximum - logf(mass);
    }
    const float sum = row_sum(beta);
    const float last = beta[7];
    #pragma unroll
    for (int j = 0; j < 8; ++j) beta[j] -= last;
    beta_sum = sum - 8 * last;
    return -sum;
}

__device__ __forceinline__ float relative_candidate(
    const Matrix& T, const Row& step, float step_sum, Matrix& candidate)
{
    Row change;
    #pragma unroll
    for (int j = 0; j < 8; ++j) change[j] = expm1f(step[j]);
    float logarithm[2];
    #pragma unroll
    for (int r = 0; r < 2; ++r)
    {
        Row products;
        #pragma unroll
        for (int j = 0; j < 8; ++j) products[j] = T[r][j] * change[j];
        const float diff = row_sum(products), mass = row_sum(T[r]);
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            candidate[r][j] = fmaf(T[r][j], change[j], T[r][j]) / (mass + diff);
        logarithm[r] = log1pf(diff / mass);
    }
    return column_sum(logarithm[0] + logarithm[1]) - step_sum;
}

__device__ __forceinline__ void sinkhorn(const Matrix& R, Row& beta)
{
    Matrix v;
    #pragma unroll
    for (int r = 0; r < 2; ++r)
    {
        Row u, e;
        #pragma unroll
        for (int j = 0; j < 8; ++j) u[j] = R[r][j] + beta[j];
        const float maximum = row_max(u);
        #pragma unroll
        for (int j = 0; j < 8; ++j) e[j] = expf(u[j] - maximum);
        const float alpha = -maximum - logf(row_sum(e));
        #pragma unroll
        for (int j = 0; j < 8; ++j) v[r][j] = R[r][j] + alpha;
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        const float maximum = column_max(fmaxf(v[0][j], v[1][j]));
        const float mass = column_sum(expf(v[0][j] - maximum) + expf(v[1][j] - maximum));
        beta[j] = -maximum - logf(mass);
    }
    const float last = beta[7];
    #pragma unroll
    for (int j = 0; j < 8; ++j) beta[j] -= last;
}

__device__ __forceinline__ void hessian(
    const Matrix& T, const Row& c, float damping, int lane, float* H)
{
    #pragma unroll
    for (int round = 0; round < 4; ++round)
    {
        #pragma unroll
        for (int owner = 0; owner < 7; ++owner)
        {
            const int other = (owner + round) % 7;
            float h = -column_sum(T[0][owner] * T[0][other] + T[1][owner] * T[1][other]);
            if (round == 0) h += c[owner] + damping;
            if (lane == owner % 4) H[max(owner, other) * 7 + min(owner, other)] = h;
        }
    }
}

__global__ void birkhoff_proj_n8_kernel(
    const float* __restrict__ R, float* __restrict__ T, float tol, int batch_size)
{
    __shared__ float shared_H[MATRICES_PER_BLOCK][49];
    __shared__ float shared_rhs[MATRICES_PER_BLOCK][7];
    __shared__ float shared_x[MATRICES_PER_BLOCK][7];
    const int group = threadIdx.x / 4, lane = threadIdx.x % 4;
    const int instance = blockIdx.x * MATRICES_PER_BLOCK + group;
    if (instance >= batch_size) return;
    Matrix values_R, values_T;
    #pragma unroll
    for (int r = 0; r < 2; ++r)
    {
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            values_R[r][j] = R[instance * 64 + (lane + 4 * r) * 8 + j];
    }
    Row beta = {}, c;
    float beta_sum = 0.0f;
    float f = objective(values_R, beta, beta_sum, values_T);
    float gnorm = gradient(values_T, c);
    Row beta0;
    float sum0;
    const float f0 = initialize(values_R, beta0, sum0);
    if (f0 < f)
    {
        #pragma unroll
        for (int j = 0; j < 8; ++j) beta[j] = beta0[j];
        beta_sum = sum0;
        f = objective(values_R, beta, beta_sum, values_T);
        gnorm = gradient(values_T, c);
    }
    constexpr float gammas[LINE_SEARCH_MAX_ITERS] = {1.0f, 0.5f, 0.1f, 0.05f, 0.01f};
    if (gnorm >= tol)
    {
        #pragma unroll 1
        for (int iter = 0; iter < NEWTON_MAX_ITERS; ++iter)
        {
            const float damping = fminf(gnorm * gnorm, 1e-3f);
            hessian(values_T, c, damping, lane, shared_H[group]);
            #pragma unroll
            for (int j = 0; j < 7; ++j)
                if (lane == j % 4) shared_rhs[group][j] = 1.0f - c[j];
            __syncwarp(matrix_mask());
            if (lane == 0) cholesky_solve(shared_H[group], shared_rhs[group], shared_x[group]);
            __syncwarp(matrix_mask());
            Row direction, magnitude;
            #pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                direction[j] = j < 7 ? shared_x[group][j] : 0.0f;
                magnitude[j] = fabsf(direction[j]);
            }
            const float direction_sum = row_sum(direction);
            const float direction_max = row_max(magnitude);
            bool accepted = false;
            #pragma unroll 1
            for (int k = 0; k < LINE_SEARCH_MAX_ITERS; ++k)
            {
                Row candidate_beta, step;
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    step[j] = gammas[k] * direction[j];
                    // Match the existing separately rounded step and update.
                    candidate_beta[j] = __fadd_rn(beta[j], step[j]);
                }
                const float candidate_sum = __fadd_rn(beta_sum, gammas[k] * direction_sum);
                Matrix candidate_T;
                float delta;
                if (gammas[k] * direction_max < 0.5f)
                    delta = relative_candidate(values_T, step, gammas[k] * direction_sum, candidate_T);
                else
                    delta = objective(values_R, candidate_beta, candidate_sum, candidate_T) - f;
                if (!(delta < 0.0f)) continue;
                Row candidate_c;
                const float candidate_gnorm = gradient(candidate_T, candidate_c);
                if (!(candidate_gnorm < gnorm)) continue;
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                {
                    beta[j] = candidate_beta[j];
                    c[j] = candidate_c[j];
                    values_T[0][j] = candidate_T[0][j];
                    values_T[1][j] = candidate_T[1][j];
                }
                beta_sum = candidate_sum;
                f += delta;
                gnorm = candidate_gnorm;
                accepted = true;
                break;
            }
            if (!accepted)
            {
                sinkhorn(values_R, beta);
                beta_sum = row_sum(beta);
                f = objective(values_R, beta, beta_sum, values_T);
                gnorm = gradient(values_T, c);
            }
            if (gnorm < tol) break;
        }
    }
    #pragma unroll
    for (int r = 0; r < 2; ++r)
    {
        #pragma unroll
        for (int j = 0; j < 8; ++j)
            T[instance * 64 + (lane + 4 * r) * 8 + j] = values_T[r][j];
    }
}
} // namespace birkhoff_n8::quarterwarp

void birkhoff_proj_n8(
    const float* R,
    float* T,
    float tol,
    int batch_size,
    cudaStream_t stream
)
{
    if (batch_size >= birkhoff_n8::LARGE_BATCH_MIN_SIZE)
    {
        const int num_blocks =
            (batch_size + birkhoff_n8::quarterwarp::MATRICES_PER_BLOCK - 1) /
            birkhoff_n8::quarterwarp::MATRICES_PER_BLOCK;
        birkhoff_n8::quarterwarp::birkhoff_proj_n8_kernel<<<
            num_blocks, birkhoff_n8::BLOCK_DIM, 0, stream
        >>>(R, T, tol, batch_size);
    }
    else
    {
        const int num_blocks =
            (batch_size + birkhoff_n8::WARPS_PER_BLOCK - 1) /
            birkhoff_n8::WARPS_PER_BLOCK;
        birkhoff_n8::birkhoff_proj_n8_kernel<<<
            num_blocks, birkhoff_n8::BLOCK_DIM, 0, stream
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
    if (batch_size >= birkhoff_n8::LARGE_BATCH_MIN_SIZE)
    {
        const int num_blocks =
            (batch_size + birkhoff_n8::halfwarp::MATRICES_PER_BLOCK - 1) /
            birkhoff_n8::halfwarp::MATRICES_PER_BLOCK;
        birkhoff_n8::halfwarp::birkhoff_proj_n8_backward_kernel<<<
            num_blocks, birkhoff_n8::BLOCK_DIM, 0, stream
        >>>(G, T, D, batch_size);
    }
    else
    {
        const int num_blocks =
            (batch_size + birkhoff_n8::WARPS_PER_BLOCK - 1) /
            birkhoff_n8::WARPS_PER_BLOCK;
        birkhoff_n8::birkhoff_proj_n8_backward_kernel<<<
            num_blocks, birkhoff_n8::BLOCK_DIM, 0, stream
        >>>(G, T, D, batch_size);
    }
}
