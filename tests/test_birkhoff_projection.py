"""Focused correctness tests for the public n=4 and n=8 CUDA solvers."""

from collections.abc import Callable

import pytest
import torch

import mhc_proj

BATCH_SIZE = 8
FORWARD_RTOL = 2e-3
FORWARD_ATOL = 2e-4
BACKWARD_RTOL = 2e-2
BACKWARD_ATOL = 2e-4

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the projection implementations require CUDA",
)


PROJECTION_MODULES = [
    pytest.param(4, mhc_proj.MHCProjectionN4, id="n4"),
    pytest.param(8, mhc_proj.MHCProjectionN8, id="n8"),
]
SINKHORN_MODULES = [
    pytest.param(4, mhc_proj.MHCSinkhornN4, id="n4"),
    pytest.param(8, mhc_proj.MHCSinkhornN8, id="n8"),
]


def _known_solution(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    eye = torch.eye(n, dtype=torch.float64)
    uniform = torch.full((n, n), 1.0 / n, dtype=torch.float64)
    permutation = torch.arange(n).roll(2)
    target = (
        0.55 * eye + 0.25 * eye.roll(1, 1) + 0.15 * eye[permutation] + 0.05 * uniform
    )
    row_shift = torch.linspace(-0.7, 0.7, n, dtype=torch.float64)[:, None]
    col_shift = torch.linspace(0.6, -0.6, n, dtype=torch.float64)[None, :]
    logits = target.log() + row_shift + col_shift
    return (
        target.float().cuda().expand(BATCH_SIZE, -1, -1).contiguous(),
        logits.float().cuda().expand(BATCH_SIZE, -1, -1).contiguous(),
    )


def _check_forward_backward(module: torch.nn.Module, n: int) -> None:
    target, logits = _known_solution(n)
    logits.requires_grad_(True)
    torch.manual_seed(123)
    upstream = torch.randn_like(target)
    output = module(logits)
    gradient = torch.autograd.grad(output, logits, upstream)[0]

    assert torch.isfinite(output).all()
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(output, target, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)

    delta = torch.randn_like(target)
    delta = (
        delta
        - delta.mean(dim=2, keepdim=True)
        - delta.mean(dim=1, keepdim=True)
        + delta.mean(dim=(1, 2), keepdim=True)
    )
    delta = 1e-3 * delta / delta.abs().flatten(1).max(dim=1).values[:, None, None]
    actual = (gradient.double() * (delta / target).double()).sum(dim=(1, 2))
    expected = (upstream.double() * delta.double()).sum(dim=(1, 2))
    torch.testing.assert_close(actual, expected, rtol=BACKWARD_RTOL, atol=BACKWARD_ATOL)


@pytest.mark.parametrize(("n", "module_type"), PROJECTION_MODULES)
def test_projection_forward_backward(
    n: int, module_type: type[torch.nn.Module]
) -> None:
    _check_forward_backward(module_type(tol=1e-6), n)


@pytest.mark.parametrize(("n", "module_type"), PROJECTION_MODULES)
@pytest.mark.parametrize("batch_size", [128, 129, 32768, 32769])
def test_projection_matches_float64_reference(
    n: int, module_type: type[torch.nn.Module], batch_size: int
) -> None:
    """Exercise small-step convergence and partial groups in both n8 schedules."""
    torch.manual_seed(2026)
    logits = torch.randn(batch_size, n, n, device="cuda", requires_grad=True)
    upstream = torch.randn_like(logits)

    # Independent FP64 Sinkhorn reference on well-conditioned random inputs.
    target = logits.detach().double().exp()
    for _ in range(200):
        target = target / target.sum(dim=-1, keepdim=True)
        target = target / target.sum(dim=-2, keepdim=True)
    torch.testing.assert_close(
        target.sum(dim=-1), torch.ones_like(target[:, :, 0]), rtol=0, atol=1e-12
    )
    weighted = upstream.double() * target
    row_rhs = weighted.sum(dim=-1)
    rhs = weighted.sum(dim=-2) - (
        target.transpose(-1, -2) @ row_rhs.unsqueeze(-1)
    ).squeeze(-1)
    reduced = target[:, :, : n - 1]
    hessian = torch.eye(n - 1, dtype=torch.float64, device="cuda") - (
        reduced.transpose(-1, -2) @ reduced
    )
    column_dual = torch.linalg.solve(hessian, rhs[:, : n - 1])
    column_dual = torch.cat((column_dual, torch.zeros_like(rhs[:, :1])), dim=-1)
    row_dual = row_rhs - (target @ column_dual.unsqueeze(-1)).squeeze(-1)
    expected_gradient = (
        upstream.double() - row_dual.unsqueeze(-1) - column_dual.unsqueeze(-2)
    ) * target

    output = module_type(tol=1e-6)(logits)
    gradient = torch.autograd.grad(output, logits, upstream)[0]
    torch.testing.assert_close(output.double(), target, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        gradient.double(), expected_gradient, rtol=1e-5, atol=2e-6
    )
    error = (output.double().sum(dim=-1) - 1).abs().sum(dim=-1) + (
        output.double().sum(dim=-2) - 1
    ).abs().sum(dim=-1)
    assert error.max() < 3e-6


@pytest.mark.parametrize("batch_size", [2048, 32768])
def test_n8_small_step_convergence_on_difficult_inputs(batch_size: int) -> None:
    """Small valid steps must not be lost to absolute-objective rounding."""
    torch.manual_seed(321 + batch_size)
    logits = 10 * torch.randn(batch_size, 8, 8, device="cuda")
    output = mhc_proj.MHCProjectionN8(tol=1e-6)(logits)
    assert torch.isfinite(output).all()
    error = (output.double().sum(dim=-1) - 1).abs().sum(dim=-1) + (
        output.double().sum(dim=-2) - 1
    ).abs().sum(dim=-1)
    assert error.mean() < 2e-6
    assert error.quantile(0.99) < 5e-6
    assert error.max() < 5e-4


@pytest.mark.parametrize(("n", "module_type"), SINKHORN_MODULES)
def test_sinkhorn_forward_backward(n: int, module_type: type[torch.nn.Module]) -> None:
    _check_forward_backward(module_type(max_iter=20), n)


@pytest.mark.parametrize(
    ("n", "module_type", "invalid_keyword"),
    [
        pytest.param(4, mhc_proj.MHCProjectionN4, {"tol": 0.0}, id="projection-n4"),
        pytest.param(8, mhc_proj.MHCProjectionN8, {"tol": 0.0}, id="projection-n8"),
        pytest.param(4, mhc_proj.MHCSinkhornN4, {"max_iter": 0}, id="sinkhorn-n4"),
        pytest.param(8, mhc_proj.MHCSinkhornN8, {"max_iter": 21}, id="sinkhorn-n8"),
    ],
)
def test_module_contract(
    n: int,
    module_type: type[torch.nn.Module],
    invalid_keyword: dict,
) -> None:
    with pytest.raises(ValueError):
        module_type(**invalid_keyword)

    module = module_type()
    with pytest.raises(ValueError, match=f"{n} x {n}"):
        module(torch.empty(2, n, n + 1, device="cuda"))
    with pytest.raises(TypeError, match="floating-point"):
        module(torch.ones(2, n, n, device="cuda", dtype=torch.int32))

    value = torch.randn(2, n, n, device="cuda").transpose(1, 2)
    assert not value.is_contiguous()
    value.requires_grad_(True)
    gradient = torch.randn_like(value)
    output = module(value)
    actual = torch.autograd.grad(output, value, gradient)[0]

    reference = value.detach().contiguous().requires_grad_(True)
    reference_output = module(reference)
    expected = torch.autograd.grad(reference_output, reference, gradient.contiguous())[
        0
    ]
    torch.testing.assert_close(output, reference_output)
    torch.testing.assert_close(actual, expected)

    empty = torch.empty(0, n, n, device="cuda", requires_grad=True)
    empty_output = module(empty)
    empty_output.sum().backward()
    assert empty_output.shape == empty.shape
    assert empty.grad is not None and empty.grad.shape == empty.shape


@pytest.mark.parametrize(
    ("n", "forward", "prepare", "argument"),
    [
        pytest.param(
            4, mhc_proj.torch.birkhoff_proj_n4, lambda x: x, 1e-6, id="projection-n4"
        ),
        pytest.param(
            8, mhc_proj.torch.birkhoff_proj_n8, lambda x: x, 1e-6, id="projection-n8"
        ),
        pytest.param(
            4, mhc_proj.torch.sinkhorn_knopp_n4, torch.exp, 20, id="sinkhorn-n4"
        ),
        pytest.param(
            8, mhc_proj.torch.sinkhorn_knopp_n8, torch.exp, 20, id="sinkhorn-n8"
        ),
    ],
)
def test_solver_uses_current_stream(
    n: int,
    forward: Callable,
    prepare: Callable[[torch.Tensor], torch.Tensor],
    argument: float,
) -> None:
    desired = 4.0 * torch.eye(n, device="cuda").unsqueeze(0)
    delayed = torch.zeros_like(desired)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(20_000_000)
        delayed.copy_(desired)
        actual = forward(prepare(delayed), argument)["T"].clone()
    stream.synchronize()
    expected = forward(prepare(desired), argument)["T"]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("n", "batch_size", "forward"),
    [
        pytest.param(4, 4096, mhc_proj.torch.birkhoff_proj_n4, id="n4"),
        pytest.param(8, 32768, mhc_proj.torch.birkhoff_proj_n8, id="n8"),
    ],
)
def test_projection_converges_on_difficult_inputs(
    n: int,
    batch_size: int,
    forward: Callable,
) -> None:
    torch.manual_seed(123)
    logits = 10.0 * torch.randn(batch_size, n, n, device="cuda")
    output = forward(logits, 1e-6)["T"]
    error = (output.sum(dim=-1) - 1.0).abs().sum(dim=-1) + (
        output.sum(dim=-2) - 1.0
    ).abs().sum(dim=-1)
    assert error.mean() < 2e-3
    assert error.max() < 5e-2
