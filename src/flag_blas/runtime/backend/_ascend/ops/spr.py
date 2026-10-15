# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union

import torch
import triton
import triton.language as tl
from flag_blas.utils import libentry

from flag_blas.runtime import torch_device_fn

from ..utils import CORE_NUM

try:
    from triton.backends.ascend import utils as _ascend_driver_utils
    from triton.backends.ascend.driver import NPULauncher as _NPU_LAUNCHER
except ImportError:
    _NPU_LAUNCHER = None

try:
    from torch_npu._C import _npu_getCurrentRawStreamNoWait as _current_raw_stream
except ImportError:

    def _current_raw_stream(device):
        return triton.runtime.driver.active.get_current_stream(device)


try:
    from torch_npu._C import _npu_getDevice as _current_device
except ImportError:
    _current_device = torch_device_fn.current_device

ScalarType = Union[float, int, complex, torch.Tensor]

# Only compiled launch metadata is cached. Every call reads the current inputs
# and launches a new update on the current stream; no tensor or result is cached.
_LAUNCH_CACHE = {}
_HOOK_CHAIN_TYPE = getattr(triton.knobs, "HookChain", None)


class _DevicePointer(int):
    """Validated NPU address with fresh tensor metadata for the profiler."""

    def size(self):
        return self.tensor.size()


@libentry()
@triton.jit
def sspr_scalar_kernel(ap_ptr, x_ptr, alpha: tl.float32):
    ap = tl.load(ap_ptr)
    x = tl.load(x_ptr)
    tl.store(ap_ptr, ap + (alpha * x) * x)


@triton.jit
def _sspr_packed_block(
    ap_ptr,
    xv,
    alpha,
    block,
    N: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    count: tl.constexpr = N * (N + 1) // 2
    off = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Clamp padding before gathering: tl.gather has no mask argument.
    safe_off = tl.minimum(off, count - 1)
    if UPLO == 0:
        triangle = safe_off
    else:
        triangle = count - 1 - safe_off
    major = ((tl.sqrt(8.0 * triangle.to(tl.float32) + 1.0) - 1.0) * 0.5).to(tl.int32)
    # Exact integer corrections protect packed row boundaries from FP32 sqrt
    # rounding. Shifts avoid float-based lowering of vector integer division.
    major = tl.where(((major * (major + 1)) >> 1) > triangle, major - 1, major)
    major = tl.where((((major + 1) * (major + 2)) >> 1) <= triangle, major + 1, major)
    minor = triangle - ((major * (major + 1)) >> 1)
    if UPLO == 0:
        rows, cols = major, minor
    else:
        rows, cols = N - 1 - major, N - 1 - minor
    xr = tl.gather(xv, rows, axis=0)
    xc = tl.gather(xv, cols, axis=0)
    ap = tl.load(ap_ptr + off, off < count, 0.0)
    tl.store(ap_ptr + off, ap + (alpha * xr) * xc, off < count)


@libentry()
@triton.jit
def sspr_packed_kernel(
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
    """Stream consecutive packed entries and gather x from an on-core vector."""
    lanes = tl.arange(0, VECTOR_SIZE)
    xv = tl.load(x_ptr + lanes.to(tl.int64) * INCX, lanes < N, 0.0)
    block = tl.program_id(0)
    if SINGLE_TILE:
        _sspr_packed_block(ap_ptr, xv, alpha, block, N, UPLO, BLOCK_SIZE)
    else:
        while block * BLOCK_SIZE < N * (N + 1) // 2:
            _sspr_packed_block(ap_ptr, xv, alpha, block, N, UPLO, BLOCK_SIZE)
            block += tl.num_programs(0)


@triton.jit
def _sspr_small_row(triangle):
    # The caller restricts N to 513. Test every offset in this domain on NPU.
    inverse = (tl.sqrt(8.0 * triangle.to(tl.float32) + 1.0) - 1.0) * 0.5
    return (inverse + 0.000244140625).to(tl.int32)


@libentry()
@triton.jit
def sspr_small_kernel(
    ap_ptr,
    x_ptr,
    alpha: tl.float32,
    N: tl.constexpr,
    INCX: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    VECTOR_SIZE: tl.constexpr,
):
    """One contiguous tile per core, with a bounded small-N row inverse."""
    tl.static_assert(N <= 513)
    lanes = tl.arange(0, VECTOR_SIZE)
    xv = tl.load(x_ptr + lanes.to(tl.int64) * INCX, lanes < N, 0.0)
    count: tl.constexpr = N * (N + 1) // 2
    off = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    safe_off = tl.minimum(off, count - 1)
    if UPLO == 0:
        triangle = safe_off
    else:
        triangle = count - 1 - safe_off
    # For N <= 513 the inverse at a non-boundary is at least 1/514
    # below the next integer. 2**-12 covers FP32 sqrt rounding at a
    # boundary without crossing a non-boundary. Larger N uses the
    # general kernel's exact integer corrections instead.
    major = _sspr_small_row(triangle)
    minor = triangle - ((major * (major + 1)) >> 1)
    if UPLO == 0:
        rows, cols = major, minor
    else:
        rows, cols = N - 1 - major, N - 1 - minor
    xr = tl.gather(xv, rows, axis=0)
    xc = tl.gather(xv, cols, axis=0)
    ap = tl.load(ap_ptr + off, off < count, 0.0)
    tl.store(ap_ptr + off, ap + (alpha * xr) * xc, off < count)


@libentry()
@triton.jit
def sspr_row_kernel(
    ap_ptr,
    x_ptr,
    alpha: tl.float32,
    N: tl.constexpr,
    INCX: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Balance complete rows cyclically across cores; every AP lane is contiguous."""
    row = tl.program_id(0)
    lanes = tl.arange(0, BLOCK_SIZE)
    while row < N:
        row64 = row.to(tl.int64)
        if UPLO == 0:
            base = (row64 * (row64 + 1)) >> 1
            begin, end = 0, row + 1
        else:
            base = row64 * N - ((row64 * (row64 + 1)) >> 1)
            begin, end = row, N
        scaled_xr = alpha * tl.load(x_ptr + row64 * INCX)
        for start in range(begin, end, BLOCK_SIZE):
            col = start + lanes
            mask = col < end
            xc = tl.load(x_ptr + col.to(tl.int64) * INCX, mask, 0.0)
            ap = tl.load(ap_ptr + base + col, mask, 0.0)
            tl.store(ap_ptr + base + col, ap + scaled_xr * xc, mask)
        row += tl.num_programs(0)


def sspr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    AP: torch.Tensor,
) -> None:
    # Preserve the generic SPR argument checks, including on launch-cache hits.
    assert AP.dtype == torch.float32 == x.dtype
    assert AP.is_contiguous() and x.is_contiguous()
    ap_device = AP.device
    assert ap_device == x.device
    assert uplo in (0, 1)
    assert n >= 0
    assert incx > 0
    if n > 0:
        assert x.numel() >= 1 + (n - 1) * incx
        assert AP.numel() >= n * (n + 1) // 2
    if n == 0:
        return
    if type(alpha) is not float:
        alpha = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha == 0.0:
        return
    assert ap_device.type == "npu"
    device = ap_device.index
    if device != _current_device():
        with torch_device_fn.device(ap_device):
            return sspr(uplo, n, alpha, x, incx, AP)

    # Both kernels address row-packed AP directly; no column-major uplo flip.
    pointers = (AP.data_ptr(), x.data_ptr())
    key = (device, n, uplo, incx, alpha, pointers[0] % 16, pointers[1] % 16)
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        _compile_and_launch_sspr(key, uplo, n, alpha, x, incx, AP)
        return
    compiled, run, grid, constants, special_launch = entry
    stream = _current_raw_stream(device)
    runtime_knobs = triton.knobs.runtime
    enter_hook = runtime_knobs.launch_enter_hook
    exit_hook = runtime_knobs.launch_exit_hook
    has_enter = enter_hook is not None and (
        type(enter_hook) is not _HOOK_CHAIN_TYPE or bool(enter_hook.calls)
    )
    has_exit = exit_hook is not None and (
        type(exit_hook) is not _HOOK_CHAIN_TYPE or bool(exit_hook.calls)
    )
    if special_launch or has_enter or has_exit:
        compiled[(grid, 1, 1)](AP, x, alpha, *constants, stream=stream)
        return
    # Already validated same-device addresses avoid redundant pointer queries.
    # Wrappers are created afresh, retaining shape information for profiling.
    ap_arg = _DevicePointer(pointers[0])
    ap_arg.tensor = AP
    x_arg = _DevicePointer(pointers[1])
    x_arg.tensor = x
    direct_launch = (
        type(run) is _NPU_LAUNCHER
        and not run.compile_only
        and not run.enable_msprof_register_tensor
        and not getattr(run.metadata, "debug_enabled", False)
    )
    launch = run.launch if direct_launch else run
    profiler_registered = launch(
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
        alpha,
        *constants,
    )
    if direct_launch:
        _ascend_driver_utils.TRITON_PROFILER_REGISTERED = profiler_registered == 1


def _compile_and_launch_sspr(key, uplo, n, alpha, x, incx, AP):
    """Launch once on a miss and cache only the compiled launch metadata."""
    if n == 1:
        kernel, grid, constants = sspr_scalar_kernel, 1, ()
    elif n <= 513:
        # Small triangles need fewer, wider tiles, not one tiny tile on
        # every core. The size bands keep the grid bounded to 37 programs
        # and were swept independently of the equivalent-performance score.
        block = 512 if n <= 96 else 1024 if n <= 192 else 2048 if n <= 384 else 4096
        grid = triton.cdiv(n * (n + 1) // 2, block)
        kernel = sspr_small_kernel
        constants = (n, incx, uplo, block, triton.next_power_of_2(n))
    elif n <= 3072:
        count = n * (n + 1) // 2
        block = 1024
        blocks = triton.cdiv(count, block)
        grid = min(blocks, CORE_NUM)
        kernel = sspr_packed_kernel
        constants = (n, incx, uplo, block, triton.next_power_of_2(n), blocks == grid)
    else:
        kernel, grid = sspr_row_kernel, min(n, CORE_NUM)
        constants = (n, incx, uplo, min(triton.next_power_of_2(n), 4096))
    compiled, _ = kernel[(grid,)](
        AP,
        x,
        alpha,
        *constants,
        num_warps=1,
        num_stages=1 if n <= 513 else 2,
    )
    if len(_LAUNCH_CACHE) >= 512:
        _LAUNCH_CACHE.clear()
    run = compiled.run
    special_launch = (
        getattr(run, "compile_only", False)
        or getattr(run, "enable_msprof_register_tensor", False)
        or getattr(compiled.metadata, "debug_enabled", False)
    )
    _LAUNCH_CACHE[key] = (compiled, run, grid, constants, special_launch)
