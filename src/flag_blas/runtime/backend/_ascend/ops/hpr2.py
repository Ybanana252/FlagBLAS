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
from flag_blas.ops.level2.hpr2 import _complex_scalar
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

ScalarType = Union[float, int, complex, torch.Tensor]

# Cache compiled launchers only, never tensor values, pointers, or results.
# The backend's CompiledKernel runner preserves launch hooks and stream ordering.
_LAUNCH_CACHE = {}
_HOOK_CHAIN_TYPE = getattr(triton.knobs, "HookChain", None)


class _DevicePointer(int):
    """Validated pointer with shape metadata for the Ascend msprof launcher."""

    def size(self):
        return (*self.tensor.shape, 2)


@libentry()
@triton.jit
def chpr2_scalar_kernel(
    ap_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
):
    ar = tl.load(ap_ptr)
    xr = tl.load(x_ptr)
    xi = tl.load(x_ptr + 1)
    yr = tl.load(y_ptr)
    yi = tl.load(y_ptr + 1)
    prod_r = xr * yr + xi * yi
    prod_i = xi * yr - xr * yi
    update_r = 2.0 * (alpha_r * prod_r - alpha_i * prod_i)
    tl.store(ap_ptr, ar + update_r)
    tl.store(ap_ptr + 1, 0.0)


@libentry()
@triton.jit
def chpr2_row_kernel(
    ap_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    N: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Own complete packed rows; move interleaved complex values contiguously.

    A bounded, cyclic row assignment balances the increasing/decreasing row
    lengths without a triangular inverse or a large grid of tiny programs.
    Only the scalar row base needs 64-bit arithmetic. No AP atomics or scratch
    allocations are needed: each packed element has exactly one writer.
    """
    row = tl.program_id(0)
    lanes = tl.arange(0, 2 * BLOCK_SIZE)
    columns = tl.arange(0, BLOCK_SIZE)
    while row < N:
        row64 = row.to(tl.int64)
        if UPLO == 0:
            base = row64 * (row64 + 1) // 2
            begin = 0
            end = row + 1
        else:
            base = row64 * N - row64 * (row64 + 1) // 2
            begin = row
            end = N

        pair = tl.arange(0, 2)
        xr, xi = tl.split(
            tl.reshape(tl.load(x_ptr + row64 * (2 * INCX) + pair), (1, 2))
        )
        yr, yi = tl.split(
            tl.reshape(tl.load(y_ptr + row64 * (2 * INCY) + pair), (1, 2))
        )
        # alpha*x[row] and conj(alpha)*y[row], reused across the whole row.
        ur = alpha_r * xr - alpha_i * xi
        ui = alpha_r * xi + alpha_i * xr
        vr = alpha_r * yr + alpha_i * yi
        vi = alpha_r * yi - alpha_i * yr

        for start in range(begin, end, BLOCK_SIZE):
            col = start + columns
            float_mask = start * 2 + lanes < end * 2
            # INC{X,Y}=1 yields contiguous loads of both real/imag components.
            if INCX == 1:
                xoff = start * 2 + lanes
            else:
                xoff = (
                    start.to(tl.int64) * (2 * INCX)
                    + (lanes.to(tl.int64) >> 1) * (2 * INCX)
                    + (lanes & 1)
                )
            if INCY == 1:
                yoff = start * 2 + lanes
            else:
                yoff = (
                    start.to(tl.int64) * (2 * INCY)
                    + (lanes.to(tl.int64) >> 1) * (2 * INCY)
                    + (lanes & 1)
                )
            xv = tl.load(x_ptr + xoff, mask=float_mask, other=0.0)
            yv = tl.load(y_ptr + yoff, mask=float_mask, other=0.0)
            xcr, xci = tl.split(tl.reshape(xv, (BLOCK_SIZE, 2)))
            ycr, yci = tl.split(tl.reshape(yv, (BLOCK_SIZE, 2)))
            update_r = ur * ycr + ui * yci + vr * xcr + vi * xci
            update_i = ui * ycr - ur * yci + vi * xcr - vr * xci

            ap_off = (base + start) * 2 + lanes
            av = tl.load(ap_ptr + ap_off, mask=float_mask, other=0.0)
            ar, ai = tl.split(tl.reshape(av, (BLOCK_SIZE, 2)))
            out_i = tl.where(col == row, 0.0, ai + update_i)
            out = tl.reshape(tl.join(ar + update_r, out_i), (2 * BLOCK_SIZE,))
            tl.store(ap_ptr + ap_off, out, mask=float_mask)
        row += tl.num_programs(0)


@triton.jit
def _chpr2_packed_row(triangular_off, N: tl.constexpr):
    """Invert a triangular offset, with a bounded small-matrix specialization."""
    inverse = (tl.sqrt(8.0 * triangular_off.to(tl.float32) + 1.0) - 1.0) * 0.5
    if N <= 257:
        # For these sizes the exact inverse below the next integer is at
        # least 1/258 away from it. A 2**-12 bias absorbs FP32 sqrt roundoff
        # at triangular boundaries without crossing a non-boundary index.
        # This removes both vector correction passes, not any HPR2 arithmetic.
        # The device regression checks every valid offset and padded lane.
        return (inverse + 0.000244140625).to(tl.int32)
    else:
        major = inverse.to(tl.int32)
        # Keep exact integer corrections outside the verified tiny domain.
        # Shifts avoid the backend's float-based vector integer division.
        major = tl.where((major * (major + 1) >> 1) > triangular_off, major - 1, major)
        return tl.where(
            ((major + 1) * (major + 2) >> 1) <= triangular_off, major + 1, major
        )


@triton.jit
def _chpr2_packed_block(
    ap_ptr,
    xv,
    yv,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    block,
    N: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Update one contiguous AP tile using already-loaded on-core vectors."""
    packed_size: tl.constexpr = N * (N + 1) // 2
    lanes = tl.arange(0, BLOCK_SIZE)
    float_lanes = tl.arange(0, 2 * BLOCK_SIZE)
    off = block * BLOCK_SIZE + lanes
    safe_off = tl.minimum(off, packed_size - 1)
    if UPLO == 0:
        triangular_off = safe_off
    else:
        triangular_off = packed_size - 1 - safe_off
    major = _chpr2_packed_row(triangular_off, N)
    minor = triangular_off - (major * (major + 1) >> 1)
    if UPLO == 0:
        rows, cols = major, minor
    else:
        rows, cols = N - 1 - major, N - 1 - minor

    xr = tl.gather(xv, 2 * rows, axis=0)
    xi = tl.gather(xv, 2 * rows + 1, axis=0)
    yr = tl.gather(yv, 2 * rows, axis=0)
    yi = tl.gather(yv, 2 * rows + 1, axis=0)
    xcr = tl.gather(xv, 2 * cols, axis=0)
    xci = tl.gather(xv, 2 * cols + 1, axis=0)
    ycr = tl.gather(yv, 2 * cols, axis=0)
    yci = tl.gather(yv, 2 * cols + 1, axis=0)
    p1r = xr * ycr + xi * yci
    p1i = xi * ycr - xr * yci
    p2r = yr * xcr + yi * xci
    p2i = yi * xcr - yr * xci
    update_r = alpha_r * p1r - alpha_i * p1i + alpha_r * p2r + alpha_i * p2i
    update_i = alpha_r * p1i + alpha_i * p1r + alpha_r * p2i - alpha_i * p2r
    ap_off = block * (2 * BLOCK_SIZE) + float_lanes
    ap_mask = ap_off < 2 * packed_size
    av = tl.load(ap_ptr + ap_off, mask=ap_mask, other=0.0)
    ar, ai = tl.split(tl.reshape(av, (BLOCK_SIZE, 2)))
    out_i = tl.where(rows == cols, 0.0, ai + update_i)
    out = tl.reshape(tl.join(ar + update_r, out_i), (2 * BLOCK_SIZE,))
    tl.store(ap_ptr + ap_off, out, mask=ap_mask)


@libentry()
@triton.jit
def chpr2_packed_kernel(
    ap_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    N: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    VECTOR_SIZE: tl.constexpr,
    SINGLE_TILE: tl.constexpr,
):
    """Stream AP and gather x/y from on-core buffers, not scattered GM loads."""
    vector_lanes = tl.arange(0, 2 * VECTOR_SIZE)
    vector_mask = vector_lanes < 2 * N
    if INCX == 1:
        xoff = vector_lanes
    else:
        xoff = (vector_lanes.to(tl.int64) >> 1) * (2 * INCX) + (vector_lanes & 1)
    if INCY == 1:
        yoff = vector_lanes
    else:
        yoff = (vector_lanes.to(tl.int64) >> 1) * (2 * INCY) + (vector_lanes & 1)
    xv = tl.load(x_ptr + xoff, mask=vector_mask, other=0.0)
    yv = tl.load(y_ptr + yoff, mask=vector_mask, other=0.0)
    block = tl.program_id(0)
    if SINGLE_TILE:
        # The launch grid covers all tiles. Avoid scalar loop bookkeeping on
        # tiny triangles where each program has exactly one tile to update.
        _chpr2_packed_block(
            ap_ptr, xv, yv, alpha_r, alpha_i, block, N, UPLO, BLOCK_SIZE
        )
    else:
        packed_size: tl.constexpr = N * (N + 1) // 2
        while block * BLOCK_SIZE < packed_size:
            _chpr2_packed_block(
                ap_ptr, xv, yv, alpha_r, alpha_i, block, N, UPLO, BLOCK_SIZE
            )
            block += tl.num_programs(0)


def chpr2(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    AP: torch.Tensor,
) -> None:
    # Keep the same checks as _check_hpr2_args on every call, including cache
    # hits. Reuse the device object locally instead of fetching AP.device four
    # times through Python/Tensor bindings; never cache input tensor metadata.
    assert AP.dtype == torch.complex64 == x.dtype == y.dtype
    assert AP.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    ap_device = AP.device
    assert ap_device == x.device == y.device
    assert uplo in (0, 1)
    assert n >= 0
    assert incx > 0 and incy > 0
    if n == 0:
        return
    assert x.numel() >= 1 + (n - 1) * incx
    assert y.numel() >= 1 + (n - 1) * incy
    assert AP.numel() >= n * (n + 1) // 2

    # Native Python scalars need no Tensor isinstance/item dispatch. Tensor
    # scalars and subclasses keep the shared frontend's conversion behavior.
    scalar_type = type(alpha)
    if scalar_type is complex:
        ar, ai = alpha.real, alpha.imag
    elif scalar_type is float:
        ar, ai = alpha, 0.0
    elif scalar_type is int:
        ar, ai = float(alpha), 0.0
    else:
        ar, ai = _complex_scalar(alpha)
    if ar == 0.0 and ai == 0.0:
        return AP
    assert ap_device.type == "npu"
    device = ap_device.index
    # A validated NPU tensor already has an initialized runtime. Query the
    # current device afresh without repeating torch_npu's Python lazy-init.
    if device != _current_device():
        with torch_device_fn.device(ap_device):
            # The recursive call now sees the target device, so it re-enters
            # at most once. Reuse the parsed scalar instead of repeating a
            # Tensor.item() or a scalar subclass's conversion side effects.
            return chpr2(uplo, n, complex(ar, ai), x, incx, y, incy, AP)

    # Keep a cache hit in this frame: small HPR2 calls are sensitive to the
    # Python frontend-to-launcher boundary as well as device kernel work.
    pointers = (AP.data_ptr(), x.data_ptr(), y.data_ptr())
    # Include all specialization inputs, device, and pointer-alignment hints.
    # alpha's float32 components are runtime arguments, not specialization
    # constants: a new scalar reuses the compiled code, not a previous result.
    # Addresses themselves are supplied afresh on every invocation, including
    # fresh correctness inputs, different storage offsets, and other streams.
    key = (
        device,
        n,
        uplo,
        incx,
        incy,
        pointers[0] % 16,
        pointers[1] % 16,
        pointers[2] % 16,
    )
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        _compile_and_launch_chpr2(key, uplo, n, ar, ai, x, incx, y, incy, AP)
        return AP
    compiled, run, grid, constants, special_launch = entry
    stream = _current_raw_stream(device)
    runtime_knobs = triton.knobs.runtime
    enter_hook = runtime_knobs.launch_enter_hook
    exit_hook = runtime_knobs.launch_exit_hook
    # Triton 3.5 uses empty HookChain objects instead of None by default.
    # Check their current contents so dynamically registered hooks still run.
    has_enter = enter_hook is not None and (
        type(enter_hook) is not _HOOK_CHAIN_TYPE or bool(enter_hook.calls)
    )
    has_exit = exit_hook is not None and (
        type(exit_hook) is not _HOOK_CHAIN_TYPE or bool(exit_hook.calls)
    )
    if special_launch or has_enter or has_exit:
        compiled[(grid, 1, 1)](
            torch.view_as_real(AP),
            torch.view_as_real(x),
            torch.view_as_real(y),
            ar,
            ai,
            *constants,
            stream=stream,
        )
        return AP
    # Validated same-device pointers avoid redundant aclrtPointerGetAttributes
    # in the Ascend launcher. Shape-carrying ints support dynamic msprof L1.
    # No graph/replay/output cache: this is one ordinary asynchronous launch.
    # Use int's native constructor, attaching fresh shape metadata immediately
    # instead of calling a Python __new__ once per input. Never reuse wrappers.
    ap_arg = _DevicePointer(pointers[0])
    ap_arg.tensor = AP
    x_arg = _DevicePointer(pointers[1])
    x_arg.tensor = x
    y_arg = _DevicePointer(pointers[2])
    y_arg.tensor = y
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
        y_arg,
        ar,
        ai,
        *constants,
    )
    return AP


def _compile_and_launch_chpr2(key, uplo, n, ar, ai, x, incx, y, incy, AP):
    """Select/compile the kernel, launch once, and cache only launch metadata."""
    if n == 1:
        kernel, grid, constants = chpr2_scalar_kernel, 1, ()
    elif n <= 1024:
        # Tiny triangles are launch/setup dominated: fewer, slightly wider
        # programs avoid replicating vector loads and packed-index setup on
        # every core. Larger triangles still use the full core budget.
        core_budget = min(CORE_NUM, 20) if n <= 128 else CORE_NUM
        # Cover small triangles in one tile per core where possible.
        # Cap tile storage for larger triangles while amortizing the
        # packed-index arithmetic and scalar loop control across lanes.
        packed_size = n * (n + 1) // 2
        block_size = min(
            1024, max(64, triton.next_power_of_2(triton.cdiv(packed_size, core_budget)))
        )
        kernel = chpr2_packed_kernel
        blocks = triton.cdiv(packed_size, block_size)
        grid = min(blocks, CORE_NUM)
        constants = (
            n,
            incx,
            incy,
            uplo,
            block_size,
            triton.next_power_of_2(n),
            blocks == grid,
        )
    else:
        kernel, grid = chpr2_row_kernel, min(n, CORE_NUM)
        constants = (n, incx, incy, uplo, min(triton.next_power_of_2(n), 1024))
    compiled, _ = kernel[(grid,)](
        triton.reinterpret(AP, tl.float32),
        triton.reinterpret(x, tl.float32),
        triton.reinterpret(y, tl.float32),
        ar,
        ai,
        *constants,
        num_warps=1,
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
