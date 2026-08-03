#include <cmath>
#include <cuda_runtime.h>
#include "sinkhorn_knopp.cuh"

namespace {

constexpr int N8 = 8;
constexpr int N8_ELEMS = N8 * N8;
constexpr int N8_MAX_ITERS = 20;

template<int MAX_ITERS>
__global__ void sinkhorn_knopp_batched_backward_n8_kernel(
    float* __restrict__ d_inp,
    const float* __restrict__ grad,
    const float* __restrict__ M_inp,
    int batch_size,
    int num_iters,
    float eps
)
{
    const int batch_idx = blockIdx.x;
    if (batch_idx >= batch_size)
    {
        return;
    }

    __shared__ float checkpoints[MAX_ITERS * N8_ELEMS];
    __shared__ float row_inv[MAX_ITERS * N8];
    __shared__ float col_inv[MAX_ITERS * N8];
    __shared__ float work[N8_ELEMS];
    __shared__ float d_work[N8_ELEMS];
    __shared__ float reduction[N8];

    const int offset = batch_idx * N8_ELEMS;
    for (int i = threadIdx.x; i < N8_ELEMS; i += blockDim.x)
    {
        work[i] = M_inp[offset + i];
    }
    __syncthreads();

    for (int iter = 0; iter < num_iters; ++iter)
    {
        for (int i = threadIdx.x; i < N8_ELEMS; i += blockDim.x)
        {
            checkpoints[iter * N8_ELEMS + i] = work[i];
        }
        __syncthreads();

        if (threadIdx.x < N8)
        {
            const int row = threadIdx.x;
            float sum = 0.0f;
            #pragma unroll
            for (int col = 0; col < N8; ++col)
            {
                sum += work[row * N8 + col];
            }
            row_inv[iter * N8 + row] =
                (sum > eps) ? __frcp_rn(sum) : 0.0f;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < N8_ELEMS; i += blockDim.x)
        {
            work[i] *= row_inv[iter * N8 + i / N8];
        }
        __syncthreads();

        if (threadIdx.x < N8)
        {
            const int col = threadIdx.x;
            float sum = 0.0f;
            #pragma unroll
            for (int row = 0; row < N8; ++row)
            {
                sum += work[row * N8 + col];
            }
            col_inv[iter * N8 + col] =
                (sum > eps) ? __frcp_rn(sum) : 0.0f;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < N8_ELEMS; i += blockDim.x)
        {
            work[i] *= col_inv[iter * N8 + i % N8];
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < N8_ELEMS; i += blockDim.x)
    {
        d_work[i] = grad[offset + i];
    }
    __syncthreads();

    for (int iter = num_iters - 1; iter >= 0; --iter)
    {
        for (int i = threadIdx.x; i < N8_ELEMS; i += blockDim.x)
        {
            work[i] =
                checkpoints[iter * N8_ELEMS + i] *
                row_inv[iter * N8 + i / N8];
        }
        __syncthreads();

        if (threadIdx.x < N8)
        {
            const int col = threadIdx.x;
            float dot = 0.0f;
            #pragma unroll
            for (int row = 0; row < N8; ++row)
            {
                dot +=
                    d_work[row * N8 + col] *
                    work[row * N8 + col];
            }
            reduction[col] = dot;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < N8_ELEMS; i += blockDim.x)
        {
            const int col = i % N8;
            const float inv = col_inv[iter * N8 + col];
            d_work[i] = (d_work[i] - reduction[col] * inv) * inv;
        }
        __syncthreads();

        if (threadIdx.x < N8)
        {
            const int row = threadIdx.x;
            float dot = 0.0f;
            #pragma unroll
            for (int col = 0; col < N8; ++col)
            {
                dot +=
                    d_work[row * N8 + col] *
                    checkpoints[
                        iter * N8_ELEMS + row * N8 + col
                    ];
            }
            reduction[row] = dot;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < N8_ELEMS; i += blockDim.x)
        {
            const int row = i / N8;
            const float inv = row_inv[iter * N8 + row];
            d_work[i] = (d_work[i] - reduction[row] * inv) * inv;
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < N8_ELEMS; i += blockDim.x)
    {
        d_inp[offset + i] = d_work[i];
    }
}

}  // namespace

void sinkhorn_knopp_n4(
    const float* d_M, float* d_T, int N, int num_iters,
    cudaStream_t stream = cudaStreamPerThread
)
{
    mhc::sinkhorn_knopp_forward_batched(d_T, d_M, N, 4, num_iters, 1e-8, stream);
}

void sinkhorn_knopp_n4_backward(
    const float* d_G, const float* d_M, float* d_D, int N, int num_iters,
    cudaStream_t stream = cudaStreamPerThread
)
{
    mhc::sinkhorn_knopp_backward_batched(d_D, d_G, d_M, N, 4, num_iters, 1e-8, stream);
}

void sinkhorn_knopp_n8(
    const float* d_M, float* d_T, int N, int num_iters,
    cudaStream_t stream = cudaStreamPerThread
)
{
    mhc::sinkhorn_knopp_forward_batched(
        d_T, d_M, N, 8, num_iters, 1e-8, stream
    );
}

void sinkhorn_knopp_n8_backward(
    const float* d_G, const float* d_M, float* d_D, int N, int num_iters,
    cudaStream_t stream = cudaStreamPerThread
)
{
    sinkhorn_knopp_batched_backward_n8_kernel<N8_MAX_ITERS>
        <<<N, N8_ELEMS, 0, stream>>>(
            d_D, d_G, d_M, N, num_iters, 1e-8
        );
}
