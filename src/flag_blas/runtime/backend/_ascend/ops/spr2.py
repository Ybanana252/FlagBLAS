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
    # A validated NPU tensor necessarily has an initialized runtime.
    from torch_npu._C import _npu_getDevice as _current_device
except ImportError:
    _current_device = torch_device_fn.current_device

ScalarType = Union[float, int, torch.Tensor]

# Cache only compiled launch metadata, never tensor data or completed results.
_LAUNCH_CACHE = {}
_HOOK_CHAIN_TYPE = getattr(triton.knobs, "HookChain", None)


class _DevicePointer(int):
    """Validated NPU address with shape metadata for the profiler."""

    def size(self):
        return self.tensor.size()


@libentry()
@triton.jit
def sspr2_scalar_kernel(ap_ptr, x_ptr, y_ptr, alpha: tl.float32):
    ap = tl.load(ap_ptr)
    x = tl.load(x_ptr)
    y = tl.load(y_ptr)
    tl.store(ap_ptr, ap + alpha * (x * y + y * x))


@libentry()
@triton.jit
def sspr2_small_kernel(
    ap_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    N: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    VECTOR_SIZE: tl.constexpr,
):
    """One packed tile per program: no persistent loop for small triangles."""
    vector_lanes = tl.arange(0, VECTOR_SIZE)
    xv = tl.load(x_ptr + vector_lanes.to(tl.int64) * INCX, vector_lanes < N, 0.0)
    yv = tl.load(y_ptr + vector_lanes.to(tl.int64) * INCY, vector_lanes < N, 0.0)
    packed_size: tl.constexpr = N * (N + 1) // 2
    off = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    safe_off = tl.minimum(off, packed_size - 1)
    if UPLO == 0:
        triangular_off = safe_off
    else:
        triangular_off = packed_size - 1 - safe_off
    major = ((tl.sqrt(8.0 * triangular_off.to(tl.float32) + 1.0) - 1.0) * 0.5).to(
        tl.int32
    )
    # Preserve the generic packed kernel's exact row-boundary corrections.
    major = tl.where((major * (major + 1) >> 1) > triangular_off, major - 1, major)
    major = tl.where(
        ((major + 1) * (major + 2) >> 1) <= triangular_off, major + 1, major
    )
    minor = triangular_off - (major * (major + 1) >> 1)
    if UPLO == 0:
        rows, cols = major, minor
    else:
        rows, cols = N - 1 - major, N - 1 - minor
    xr = tl.gather(xv, rows, axis=0)
    yr = tl.gather(yv, rows, axis=0)
    xc = tl.gather(xv, cols, axis=0)
    yc = tl.gather(yv, cols, axis=0)
    ap = tl.load(ap_ptr + off, off < packed_size, 0.0)
    tl.store(ap_ptr + off, ap + alpha * (xr * yc + yr * xc), off < packed_size)


@libentry()
@triton.jit
def sspr2_packed_kernel(
    ap_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    N: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    VECTOR_SIZE: tl.constexpr,
):
    """Stream packed AP, gathering x/y from on-core vectors, not from GM."""
    vector_lanes = tl.arange(0, VECTOR_SIZE)
    xv = tl.load(x_ptr + vector_lanes.to(tl.int64) * INCX, vector_lanes < N, 0.0)
    yv = tl.load(y_ptr + vector_lanes.to(tl.int64) * INCY, vector_lanes < N, 0.0)
    packed_size: tl.constexpr = N * (N + 1) // 2
    block = tl.program_id(0)
    lanes = tl.arange(0, BLOCK_SIZE)
    while block * BLOCK_SIZE < packed_size:
        off = block * BLOCK_SIZE + lanes
        safe_off = tl.minimum(off, packed_size - 1)
        if UPLO == 0:
            triangular_off = safe_off
        else:
            triangular_off = packed_size - 1 - safe_off
        major = ((tl.sqrt(8.0 * triangular_off.to(tl.float32) + 1.0) - 1.0) * 0.5).to(
            tl.int32
        )
        # Ascend vector integer division can round through float32. Shifts
        # preserve exact triangular numbers; correct sqrt at row boundaries.
        major = tl.where((major * (major + 1) >> 1) > triangular_off, major - 1, major)
        major = tl.where(
            ((major + 1) * (major + 2) >> 1) <= triangular_off, major + 1, major
        )
        minor = triangular_off - (major * (major + 1) >> 1)
        if UPLO == 0:
            rows, cols = major, minor
        else:
            rows, cols = N - 1 - major, N - 1 - minor
        xr = tl.gather(xv, rows, axis=0)
        yr = tl.gather(yv, rows, axis=0)
        xc = tl.gather(xv, cols, axis=0)
        yc = tl.gather(yv, cols, axis=0)
        ap = tl.load(ap_ptr + off, off < packed_size, 0.0)
        tl.store(ap_ptr + off, ap + alpha * (xr * yc + yr * xc), off < packed_size)
        block += tl.num_programs(0)


@libentry()
@triton.jit
def sspr2_row_kernel(
    ap_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    N: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Cyclic row ownership balances work and makes AP accesses contiguous."""
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
        xr = tl.load(x_ptr + row64 * INCX)
        yr = tl.load(y_ptr + row64 * INCY)
        for start in range(begin, end, BLOCK_SIZE):
            col = start + lanes
            mask = col < end
            xc = tl.load(x_ptr + col.to(tl.int64) * INCX, mask, 0.0)
            yc = tl.load(y_ptr + col.to(tl.int64) * INCY, mask, 0.0)
            ap = tl.load(ap_ptr + base + col, mask, 0.0)
            tl.store(ap_ptr + base + col, ap + alpha * (xr * yc + yr * xc), mask)
        row += tl.num_programs(0)


def sspr2(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    AP: torch.Tensor,
) -> None:
    # Preserve the shared frontend's checks on every call, including cache
    # hits. Reuse only local metadata; no tensor/address/result is cached.
    assert AP.dtype == torch.float32 == x.dtype == y.dtype
    assert AP.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    ap_device = AP.device
    assert ap_device == x.device == y.device
    assert uplo in (0, 1)
    assert n >= 0
    assert incx > 0 and incy > 0
    if n > 0:
        assert x.numel() >= 1 + (n - 1) * incx
        assert y.numel() >= 1 + (n - 1) * incy
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
            # Re-enter once on the target device; never repeat scalar item().
            return sspr2(uplo, n, alpha, x, incx, y, incy, AP)

    pointers = (AP.data_ptr(), x.data_ptr(), y.data_ptr())
    key = (
        device,
        n,
        uplo,
        incx,
        incy,
        alpha,
        pointers[0] % 16,
        pointers[1] % 16,
        pointers[2] % 16,
    )
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        _compile_and_launch_sspr2(key, uplo, n, alpha, x, incx, y, incy, AP)
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
        compiled[(grid, 1, 1)](AP, x, y, alpha, *constants, stream=stream)
        return
    # int's native constructor avoids three Python __new__ calls. Attach
    # fresh shape metadata for profiling without retaining the tensors.
    ap_arg = _DevicePointer(pointers[0])
    ap_arg.tensor = AP
    x_arg = _DevicePointer(pointers[1])
    x_arg.tensor = x
    y_arg = _DevicePointer(pointers[2])
    y_arg.tensor = y
    # The ordinary Ascend launcher forwards to its generated launch entry.
    # Avoid the extra Python *args forwarding only for the known launcher,
    # with its special modes checked afresh. Hooks above still use Triton.
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
        y_arg,
        alpha,
        *constants,
    )
    if direct_launch:
        # Match NPULauncher.__call__'s post-launch profiler bookkeeping.
        _ascend_driver_utils.TRITON_PROFILER_REGISTERED = profiler_registered == 1


def _compile_and_launch_sspr2(key, uplo, n, alpha, x, incx, y, incy, AP):
    """Launch once on a miss and cache only the compiled launch metadata."""
    if n == 1:
        kernel, grid, constants = sspr2_scalar_kernel, 1, ()
    elif n <= 513:
        # Small triangles fit in a single tile per program. Eliminate the
        # persistent loop and amortize index setup with wider tiles, while
        # bounding UB storage (8192-wide tiles exceed the 910B4 budget).
        packed_size = n * (n + 1) // 2
        core_budget = min(CORE_NUM, 20) if n <= 128 else CORE_NUM
        block_size = min(
            4096,
            max(256, triton.next_power_of_2(triton.cdiv(packed_size, core_budget))),
        )
        kernel = sspr2_small_kernel
        grid = triton.cdiv(packed_size, block_size)
        constants = (n, incx, incy, uplo, block_size, triton.next_power_of_2(n))
    elif n <= 3072:
        block_size = 1024
        kernel = sspr2_packed_kernel
        grid = min(triton.cdiv(n * (n + 1) // 2, block_size), CORE_NUM)
        constants = (n, incx, incy, uplo, block_size, triton.next_power_of_2(n))
    else:
        kernel, grid = sspr2_row_kernel, min(n, CORE_NUM)
        constants = (n, incx, incy, uplo, min(triton.next_power_of_2(n), 4096))
    compiled, _ = kernel[(grid,)](
        AP,
        x,
        y,
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
