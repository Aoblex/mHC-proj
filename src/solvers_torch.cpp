// Only compile this file when PyTorch is available.
#if defined(TORCH_BUILD)

#include <cmath>
#include <stdexcept>
#include <string>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

void birkhoff_proj_n4(
    const float*, float*, float, int, cudaStream_t = cudaStreamPerThread
);
void birkhoff_proj_n4_backward(
    const float*, const float*, float*, int, cudaStream_t = cudaStreamPerThread
);
void birkhoff_proj_n8(
    const float*, float*, float, int, cudaStream_t = cudaStreamPerThread
);
void birkhoff_proj_n8_backward(
    const float*, const float*, float*, int, cudaStream_t = cudaStreamPerThread
);
void sinkhorn_knopp_n4(
    const float*, float*, int, int, cudaStream_t = cudaStreamPerThread
);
void sinkhorn_knopp_n4_backward(
    const float*, const float*, float*, int, int,
    cudaStream_t = cudaStreamPerThread
);
void sinkhorn_knopp_n8(
    const float*, float*, int, int, cudaStream_t = cudaStreamPerThread
);
void sinkhorn_knopp_n8_backward(
    const float*, const float*, float*, int, int,
    cudaStream_t = cudaStreamPerThread
);

namespace py = pybind11;

namespace {

using ProjectionForward = void (*)(
    const float*, float*, float, int, cudaStream_t
);
using ProjectionBackward = void (*)(
    const float*, const float*, float*, int, cudaStream_t
);
using SinkhornForward = void (*)(
    const float*, float*, int, int, cudaStream_t
);
using SinkhornBackward = void (*)(
    const float*, const float*, float*, int, int, cudaStream_t
);

void check_matrix(const torch::Tensor& tensor, const char* name, int n)
{
    if (tensor.dim() != 3 || tensor.size(1) != n || tensor.size(2) != n)
    {
        const std::string actual = tensor.dim() == 3
            ? std::to_string(tensor.size(0)) + " x " +
                std::to_string(tensor.size(1)) + " x " +
                std::to_string(tensor.size(2))
            : std::to_string(tensor.dim()) + " dimensions";
        throw std::runtime_error(
            std::string(name) + " must be a tensor of size B x " +
            std::to_string(n) + " x " + std::to_string(n) +
            ", got " + actual
        );
    }
    if (!tensor.is_floating_point())
    {
        throw std::invalid_argument(
            std::string(name) + " must be a floating-point tensor"
        );
    }
}

void check_pair(
    const torch::Tensor& lhs,
    const torch::Tensor& rhs,
    const char* lhs_name,
    const char* rhs_name,
    int n
)
{
    check_matrix(lhs, lhs_name, n);
    check_matrix(rhs, rhs_name, n);
    if (lhs.sizes() != rhs.sizes())
    {
        throw std::runtime_error(
            std::string(rhs_name) + " must have the same shape as " + lhs_name
        );
    }
    if (lhs.device() != rhs.device())
    {
        throw std::invalid_argument(
            std::string(lhs_name) + " and " + rhs_name +
            " must be on the same device"
        );
    }
}

torch::TensorOptions cuda_float_options(const torch::Tensor& tensor)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
    if (tensor.is_cuda())
    {
        options = options.device(torch::kCUDA, tensor.device().index());
    }
    return options;
}

cudaStream_t current_cuda_stream(const c10::Device& device)
{
    return c10::cuda::getCurrentCUDAStream(device.index());
}

py::dict projection_forward(
    torch::Tensor input,
    float tolerance,
    int n,
    ProjectionForward kernel
)
{
    if (!(tolerance > 0.0f) || !std::isfinite(tolerance))
    {
        throw std::invalid_argument("tol must be positive and finite");
    }
    check_matrix(input, "R", n);

    const int batch_size = input.size(0);
    const auto source_options = input.options();
    const auto options = cuda_float_options(input);
    const at::cuda::CUDAGuard device_guard(options.device());
    input = input.to(options).contiguous();
    auto output = torch::empty({batch_size, n, n}, options);

    if (batch_size > 0)
    {
        kernel(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            tolerance,
            batch_size,
            current_cuda_stream(options.device())
        );
    }

    py::dict result;
    result["T"] = output.to(source_options);
    return result;
}

py::dict projection_backward(
    torch::Tensor gradient,
    torch::Tensor output,
    int n,
    ProjectionBackward kernel
)
{
    check_pair(gradient, output, "G", "T", n);

    const int batch_size = gradient.size(0);
    const auto source_options = gradient.options();
    const auto options = cuda_float_options(gradient);
    const at::cuda::CUDAGuard device_guard(options.device());
    gradient = gradient.to(options).contiguous();
    output = output.to(options).contiguous();
    auto input_gradient = torch::empty({batch_size, n, n}, options);

    if (batch_size > 0)
    {
        kernel(
            gradient.data_ptr<float>(),
            output.data_ptr<float>(),
            input_gradient.data_ptr<float>(),
            batch_size,
            current_cuda_stream(options.device())
        );
    }

    py::dict result;
    result["D"] = input_gradient.to(source_options);
    return result;
}

void check_iterations(int iterations)
{
    if (iterations < 1 || iterations > 20)
    {
        throw std::invalid_argument("max_iter must be between 1 and 20");
    }
}

py::dict sinkhorn_forward(
    torch::Tensor input,
    int iterations,
    int n,
    SinkhornForward kernel
)
{
    check_iterations(iterations);
    check_matrix(input, "M", n);

    const int batch_size = input.size(0);
    const auto source_options = input.options();
    const auto options = cuda_float_options(input);
    const at::cuda::CUDAGuard device_guard(options.device());
    input = input.to(options).contiguous();
    auto output = torch::empty({batch_size, n, n}, options);

    if (batch_size > 0)
    {
        kernel(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            iterations,
            current_cuda_stream(options.device())
        );
    }

    py::dict result;
    result["T"] = output.to(source_options);
    return result;
}

py::dict sinkhorn_backward(
    torch::Tensor gradient,
    torch::Tensor input,
    int iterations,
    int n,
    SinkhornBackward kernel
)
{
    check_iterations(iterations);
    check_pair(gradient, input, "G", "M", n);

    const int batch_size = gradient.size(0);
    const auto source_options = gradient.options();
    const auto options = cuda_float_options(gradient);
    const at::cuda::CUDAGuard device_guard(options.device());
    gradient = gradient.to(options).contiguous();
    input = input.to(options).contiguous();
    auto input_gradient = torch::empty({batch_size, n, n}, options);

    if (batch_size > 0)
    {
        kernel(
            gradient.data_ptr<float>(),
            input.data_ptr<float>(),
            input_gradient.data_ptr<float>(),
            batch_size,
            iterations,
            current_cuda_stream(options.device())
        );
    }

    py::dict result;
    result["D"] = input_gradient.to(source_options);
    return result;
}

}  // namespace

py::dict birkhoff_proj_n4(torch::Tensor input, float tolerance)
{
    return projection_forward(input, tolerance, 4, ::birkhoff_proj_n4);
}

py::dict birkhoff_proj_n4_backward(
    torch::Tensor gradient, torch::Tensor output
)
{
    return projection_backward(
        gradient, output, 4, ::birkhoff_proj_n4_backward
    );
}

py::dict birkhoff_proj_n8(torch::Tensor input, float tolerance)
{
    return projection_forward(input, tolerance, 8, ::birkhoff_proj_n8);
}

py::dict birkhoff_proj_n8_backward(
    torch::Tensor gradient, torch::Tensor output
)
{
    return projection_backward(
        gradient, output, 8, ::birkhoff_proj_n8_backward
    );
}

py::dict sinkhorn_knopp_n4(torch::Tensor input, int iterations)
{
    return sinkhorn_forward(input, iterations, 4, ::sinkhorn_knopp_n4);
}

py::dict sinkhorn_knopp_n4_backward(
    torch::Tensor gradient, torch::Tensor input, int iterations
)
{
    return sinkhorn_backward(
        gradient, input, iterations, 4, ::sinkhorn_knopp_n4_backward
    );
}

py::dict sinkhorn_knopp_n8(torch::Tensor input, int iterations)
{
    return sinkhorn_forward(input, iterations, 8, ::sinkhorn_knopp_n8);
}

py::dict sinkhorn_knopp_n8_backward(
    torch::Tensor gradient, torch::Tensor input, int iterations
)
{
    return sinkhorn_backward(
        gradient, input, iterations, 8, ::sinkhorn_knopp_n8_backward
    );
}

#endif  // defined(TORCH_BUILD)
