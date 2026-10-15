from typing import Union

import torch
import triton
import triton.language as tl
from flag_blas.ops.level2.hpr import _check_hpr_args
from flag_blas.utils import libentry

from flag_blas.runtime import torch_device_fn

from ..utils import CORE_NUM

try:
    from torch_npu._C import _npu_getCurrentRawStreamNoWait as _current_raw_stream
except ImportError:

    def _current_raw_stream(device):
        return triton.runtime.driver.active.get_current_stream(device)


try:
    from torch_npu._C import _npu_getDevice as _current_device
except ImportError:
    _current_device = torch_device_fn.current_device

try:
    from triton.backends.ascend.driver import NPULauncher as _NPU_LAUNCHER
except ImportError:
    _NPU_LAUNCHER = None

ScalarType = Union[float, int, torch.Tensor]

# Only compiled launch metadata is retained, never tensors, pointers or outputs.
# The direct runner ABI is verified on Triton-Ascend 3.5. Other versions use
# CompiledKernel's standard entry, which still avoids LibEntry argument packing.
_LAUNCH_CACHE = {}
_FAST_LAUNCH_SUPPORTED = triton.__version__.split(".")[:2] == ["3", "5"]
_HOOK_CHAIN_TYPE = getattr(getattr(triton, "knobs", None), "HookChain", None)


class _DevicePointer(int):
    """Fresh pointer retaining its tensor and complex shape for msprof."""

    def size(self):
        return (*self.tensor.shape, 2)


@libentry()
@triton.jit
def chpr_scalar_kernel(ap_ptr, x_ptr, alpha: tl.float32):
    ar = tl.load(ap_ptr)
    xr = tl.load(x_ptr)
    xi = tl.load(x_ptr + 1)
    tl.store(ap_ptr, ar + alpha * (xr * xr + xi * xi))
    tl.store(ap_ptr + 1, 0.0)


@triton.jit
def _chpr_packed_block(
    ap_ptr,
    xv,
    alpha,
    block,
    N: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Invert packed offsets on-core; AP itself is read/written contiguously,
    # including both components of each complex number.
    count: tl.constexpr = N * (N + 1) // 2
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    safe_offsets = tl.minimum(offsets, count - 1)
    if UPLO == 0:
        triangle = safe_offsets
    else:
        triangle = count - 1 - safe_offsets
    major = ((tl.sqrt(8.0 * triangle.to(tl.float32) + 1.0) - 1.0) * 0.5).to(tl.int32)
    # Correct FP32 sqrt rounding at triangular boundaries. Integer shifts
    # avoid the Ascend vector division lowering's floating-point rounding.
    major = tl.where(((major * (major + 1)) >> 1) > triangle, major - 1, major)
    major = tl.where((((major + 1) * (major + 2)) >> 1) <= triangle, major + 1, major)
    minor = triangle - ((major * (major + 1)) >> 1)
    if UPLO == 0:
        rows, cols = major, minor
    else:
        rows, cols = N - 1 - major, N - 1 - minor
    xr = tl.gather(xv, rows * 2, axis=0)
    xi = tl.gather(xv, rows * 2 + 1, axis=0)
    cr = tl.gather(xv, cols * 2, axis=0)
    ci = tl.gather(xv, cols * 2 + 1, axis=0)
    update_r = alpha * (xr * cr + xi * ci)
    update_i = alpha * (xi * cr - xr * ci)
    float_offsets = block * (2 * BLOCK_SIZE) + tl.arange(0, 2 * BLOCK_SIZE)
    mask = float_offsets < count * 2
    av = tl.load(ap_ptr + float_offsets, mask=mask, other=0.0)
    ar, ai = tl.split(tl.reshape(av, (BLOCK_SIZE, 2)))
    out_i = tl.where(rows == cols, 0.0, ai + update_i)
    out = tl.reshape(tl.join(ar + update_r, out_i), (2 * BLOCK_SIZE,))
    tl.store(ap_ptr + float_offsets, out, mask=mask)


@libentry()
@triton.jit
def chpr_packed_kernel(
    ap_ptr,
    x_ptr,
    alpha: tl.float32,
    N: tl.constexpr,
    INCX: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    VECTOR_SIZE: tl.constexpr,
    SINGLE_TILE: tl.constexpr,
):
    lanes = tl.arange(0, 2 * VECTOR_SIZE)
    if INCX == 1:
        xoff = lanes
    else:
        xoff = (lanes.to(tl.int64) >> 1) * (2 * INCX) + (lanes & 1)
    xv = tl.load(x_ptr + xoff, mask=lanes < 2 * N, other=0.0)
    block = tl.program_id(0)
    if SINGLE_TILE:
        _chpr_packed_block(ap_ptr, xv, alpha, block, N, UPLO, BLOCK_SIZE)
    else:
        while block * BLOCK_SIZE < N * (N + 1) // 2:
            _chpr_packed_block(ap_ptr, xv, alpha, block, N, UPLO, BLOCK_SIZE)
            block += tl.num_programs(0)


@libentry()
@triton.jit
def chpr_row_kernel(
    ap_ptr,
    x_ptr,
    alpha: tl.float32,
    N: tl.constexpr,
    INCX: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Cyclic rows balance the triangular workload across vector cores. Only
    # scalar bases use int64; each row strip has consecutive packed AP lanes.
    row = tl.program_id(0)
    lanes = tl.arange(0, 2 * BLOCK_SIZE)
    parity = lanes & 1
    exchange = lanes ^ 1
    while row < N:
        row64 = row.to(tl.int64)
        if UPLO == 0:
            base = (row64 * (row64 + 1)) >> 1
            begin, end = 0, row + 1
        else:
            base = row64 * N - ((row64 * (row64 + 1)) >> 1)
            begin, end = row, N
        row_xoff = row64 * (2 * INCX)
        xr = tl.load(x_ptr + row_xoff)
        xi = tl.load(x_ptr + row_xoff + 1)
        for start in range(begin, end, BLOCK_SIZE):
            float_positions = start * 2 + lanes
            mask = float_positions < end * 2
            if INCX == 1:
                xoff = float_positions
            else:
                xoff = (
                    start.to(tl.int64) * (2 * INCX)
                    + (lanes.to(tl.int64) >> 1) * (2 * INCX)
                    + parity
                )
            xv = tl.load(x_ptr + xoff, mask=mask, other=0.0)
            swapped = tl.gather(xv, exchange, axis=0)
            # Compute real/imaginary updates in interleaved lanes, avoiding
            # split/join buffers. Even: xr*cr+xi*ci; odd: -xr*ci+xi*cr.
            term1 = xv * xr
            term1 = tl.where(parity == 0, term1, -term1)
            update = alpha * (term1 + swapped * xi)
            apoff = (base + start) * 2 + lanes
            av = tl.load(ap_ptr + apoff, mask=mask, other=0.0)
            out = tl.where(float_positions == row * 2 + 1, 0.0, av + update)
            tl.store(ap_ptr + apoff, out, mask=mask)
        row += tl.num_programs(0)


def chpr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    AP: torch.Tensor,
):
    _check_hpr_args(torch.complex64, uplo, n, x, incx, AP)
    if n == 0:
        return AP
    alpha_value = (
        alpha
        if type(alpha) is float
        else float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    )
    if alpha_value == 0.0:
        return AP
    ap_device = AP.device
    assert ap_device.type == "npu"
    device = ap_device.index
    if device != _current_device():
        with torch_device_fn.device(ap_device):
            # Re-enter only once and do not repeat Tensor.item()/__float__.
            return chpr(uplo, n, alpha_value, x, incx, AP)

    # Raw addresses must not bypass view_as_real's conjugate checks or Tensor
    # subclass dispatch. Unusual views keep the original Tensor launch path.
    if (
        type(AP) is not torch.Tensor
        or type(x) is not torch.Tensor
        or AP.is_conj()
        or x.is_conj()
        or AP.is_neg()
        or x.is_neg()
    ):
        _compile_and_launch_chpr(None, uplo, n, alpha_value, x, incx, AP)
        return AP

    pointers = (AP.data_ptr(), x.data_ptr())
    # alpha is an FP32 runtime argument, not a constexpr. Neither its value
    # nor addresses/streams are captured by the compiled program.
    key = (device, n, uplo, incx, pointers[0] % 16, pointers[1] % 16)
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        _compile_and_launch_chpr(key, uplo, n, alpha_value, x, incx, AP)
        return AP
    compiled, grid, constants = entry
    run = compiled.run
    stream = _current_raw_stream(device)
    knobs = getattr(triton, "knobs", None)
    runtime_knobs = getattr(knobs, "runtime", None)
    enter_hook = getattr(runtime_knobs, "launch_enter_hook", None)
    exit_hook = getattr(runtime_knobs, "launch_exit_hook", None)
    has_enter = enter_hook is not None and (
        type(enter_hook) is not _HOOK_CHAIN_TYPE or bool(enter_hook.calls)
    )
    has_exit = exit_hook is not None and (
        type(exit_hook) is not _HOOK_CHAIN_TYPE or bool(exit_hook.calls)
    )
    # Inspect special modes on every hit; hooks/profiling may be enabled after
    # warm-up. Keep the backend runner, including its profiler bookkeeping.
    if (
        not _FAST_LAUNCH_SUPPORTED
        or type(run) is not _NPU_LAUNCHER
        or runtime_knobs is None
        or getattr(run, "compile_only", False)
        or getattr(run, "enable_msprof_register_tensor", False)
        or getattr(compiled.metadata, "debug_enabled", False)
        or getattr(getattr(run, "metadata", None), "debug_enabled", False)
        or has_enter
        or has_exit
    ):
        compiled[(grid, 1, 1)](
            torch.view_as_real(AP),
            torch.view_as_real(x),
            alpha_value,
            *constants,
            stream=stream,
        )
        return AP
    ap_arg = _DevicePointer(pointers[0])
    ap_arg.tensor = AP
    x_arg = _DevicePointer(pointers[1])
    x_arg.tensor = x
    run(
        grid,
        1,
        1,
        stream,
        compiled.function,
        compiled.packed_metadata,
        None,
        None,
        None,
        ap_arg,
        x_arg,
        alpha_value,
        *constants,
    )
    return AP


def _compile_and_launch_chpr(key, uplo, n, alpha, x, incx, AP):
    """Preserve kernel selection; a cache miss executes exactly one update."""
    if n == 1:
        kernel, grid, constants = chpr_scalar_kernel, 1, ()
    elif n <= 1024:
        count = n * (n + 1) // 2
        core_budget = min(CORE_NUM, 20) if n <= 128 else CORE_NUM
        block = min(
            1024, max(64, triton.next_power_of_2(triton.cdiv(count, core_budget)))
        )
        blocks = triton.cdiv(count, block)
        grid = min(blocks, CORE_NUM)
        kernel = chpr_packed_kernel
        constants = (n, incx, uplo, block, triton.next_power_of_2(n), blocks == grid)
    else:
        kernel, grid = chpr_row_kernel, min(n, CORE_NUM)
        constants = (n, incx, uplo, min(triton.next_power_of_2(n), 2048))
    # Scalar's original default compiler options are intentionally unchanged.
    options = {} if n == 1 else {"num_warps": 1}
    compiled, _ = kernel[(grid,)](
        torch.view_as_real(AP),
        torch.view_as_real(x),
        alpha,
        *constants,
        **options,
    )
    if key is not None:
        if len(_LAUNCH_CACHE) >= 512:
            _LAUNCH_CACHE.clear()
        _LAUNCH_CACHE[key] = (compiled, grid, constants)
