# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Ascend row-major SYR: ordinary transpose, never conjugate x."""

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

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

_LAUNCH_CACHE = {}
_HOOK_CHAIN_TYPE = getattr(triton.knobs, "HookChain", None)


class _DevicePointer(int):
    """Fresh validated pointer with metadata for the profiling launcher."""

    def size(self):
        shape = self.tensor.shape
        return (*shape, 2) if self.tensor.is_complex() else shape


@libentry()
@triton.jit
def syr_block_kernel(
    a_ptr,
    x_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    N: tl.constexpr,
    LDA: tl.constexpr,
    INCX: tl.constexpr,
    UPLO: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    """Own row groups; use rectangular DMA and preserve unselected elements.

    Masked 2-D triangular stores are particularly expensive on this backend.
    Every full rectangle is in bounds and belongs to one program. A shifted
    final rectangle preserves its overlap, which was updated by this program.
    """
    tl.static_assert(N >= COLS)
    groups: tl.constexpr = N // ROWS
    full_cols: tl.constexpr = N // COLS * COLS
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    # Contiguous partitions approximately balance the quadratic triangle.
    linear: tl.constexpr = 2 * COLS // ROWS
    total: tl.constexpr = groups * (groups + linear)
    if N >= 2048:
        group = (
            (tl.sqrt(linear * linear + 4.0 * total * pid / programs) - linear) * 0.5
        ).to(tl.int32)
        last = (
            (tl.sqrt(linear * linear + 4.0 * total * (pid + 1) / programs) - linear)
            * 0.5
        ).to(tl.int32)
        group_end = tl.where(pid + 1 == programs, groups, last)
        group_step = 1
    else:
        group, group_end, group_step = pid, groups, programs
    while group < group_end:
        if N >= 2048 and UPLO == 1:
            rb = (groups - 1 - group) * ROWS
        else:
            rb = group * ROWS
        rows = rb + tl.arange(0, ROWS)
        if IS_COMPLEX:
            rr = tl.arange(0, 2 * ROWS)
            ri = tl.arange(0, ROWS) * 2
            cc = tl.arange(0, 2 * COLS)
            if INCX == 1:
                rxo = rb * 2 + rr
            else:
                rxo = (rb.to(tl.int64) + (rr >> 1)) * (2 * INCX) + (rr & 1)
            xv = tl.load(x_ptr + rxo)
            xr, xi = tl.gather(xv, ri, 0), tl.gather(xv, ri + 1, 0)
            ur, ui = alpha_r * xr - alpha_i * xi, alpha_r * xi + alpha_i * xr
            sign = 2.0 * (cc & 1) - 1.0
        else:
            cc = tl.arange(0, COLS)
            xr = tl.load(x_ptr + rows.to(tl.int64) * INCX)
        begin = 0 if UPLO == 0 else rb // COLS * COLS
        end = rb + ROWS if UPLO == 0 else N
        for logical_cb in range(begin, end, COLS):
            cb = tl.minimum(logical_cb, N - COLS) if N % COLS else logical_cb
            if IS_COMPLEX:
                cols = cb + (cc >> 1)
                if INCX == 1:
                    xo = cb * 2 + cc
                else:
                    xo = cols.to(tl.int64) * (2 * INCX) + (cc & 1)
                xc = tl.load(x_ptr + xo)
                xs = tl.gather(xc, cc ^ 1, 0) * sign
                update = ur[:, None] * xc[None, :] + ui[:, None] * xs[None, :]
                if 2 * N * LDA < 2147483648:
                    off = rows[:, None] * (2 * LDA) + cb * 2 + cc[None, :]
                else:
                    off = rows.to(tl.int64)[:, None] * (2 * LDA) + cb * 2 + cc[None, :]
            else:
                cols = cb + cc
                xc = tl.load(x_ptr + cols.to(tl.int64) * INCX)
                update = (alpha_r * xr[:, None]) * xc[None, :]
                if N * LDA < 2147483648:
                    off = rows[:, None] * LDA + cols[None, :]
                else:
                    off = rows.to(tl.int64)[:, None] * LDA + cols[None, :]
            old = tl.load(a_ptr + off)
            result = old + update
            if (UPLO == 0 and cb + COLS > rb) or (UPLO == 1 and cb < rb + ROWS):
                if UPLO == 0:
                    selected = rows[:, None] >= cols[None, :]
                else:
                    selected = rows[:, None] <= cols[None, :]
                result = tl.where(selected, result, old)
            if N % COLS:
                if logical_cb == full_cols:
                    result = tl.where(cols[None, :] >= full_cols, result, old)
            tl.store(a_ptr + off, result)
        group += group_step
    # Fewer than ROWS leftover rows. A disjoint 1-D prefix mask handles them
    # without overlapping row-group stores or writing lda padding.
    row = groups * ROWS + pid
    while row < N:
        _syr_tail_row(
            a_ptr,
            x_ptr,
            alpha_r,
            alpha_i,
            row,
            N,
            LDA,
            INCX,
            UPLO,
            IS_COMPLEX,
            1024,
        )
        row += programs


@triton.jit
def _syr_tail_row(
    a_ptr,
    x_ptr,
    alpha_r,
    alpha_i,
    row,
    N: tl.constexpr,
    LDA: tl.constexpr,
    INCX: tl.constexpr,
    UPLO: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row64 = row.to(tl.int64)
    begin = 0 if UPLO == 0 else row
    end = row + 1 if UPLO == 0 else N
    if IS_COMPLEX:
        xr = tl.load(x_ptr + row64 * (2 * INCX))
        xi = tl.load(x_ptr + row64 * (2 * INCX) + 1)
        ur, ui = alpha_r * xr - alpha_i * xi, alpha_r * xi + alpha_i * xr
        lanes = tl.arange(0, 2 * BLOCK)
        sign = 2.0 * (lanes & 1) - 1.0
    else:
        xr = tl.load(x_ptr + row64 * INCX)
        lanes = tl.arange(0, BLOCK)
    for cb in range(begin, end, BLOCK):
        if IS_COMPLEX:
            cols = cb + (lanes >> 1)
            mask = cb * 2 + lanes < end * 2
            if INCX == 1:
                xo = cb * 2 + lanes
            else:
                xo = cols.to(tl.int64) * (2 * INCX) + (lanes & 1)
            xc = tl.load(x_ptr + xo, mask, other=0.0)
            xs = tl.gather(xc, lanes ^ 1, 0) * sign
            update = ur * xc + ui * xs
            off = (row64 * LDA + cb) * 2 + lanes
        else:
            cols = cb + lanes
            mask = cols < end
            xc = tl.load(x_ptr + cols.to(tl.int64) * INCX, mask, other=0.0)
            update = (alpha_r * xr) * xc
            off = row64 * LDA + cols
        old = tl.load(a_ptr + off, mask, other=0.0)
        tl.store(a_ptr + off, old + update, mask)


@libentry()
@triton.jit
def syr_row_kernel(
    a_ptr,
    x_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    N: tl.constexpr,
    LDA: tl.constexpr,
    INCX: tl.constexpr,
    UPLO: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Tiny matrices: one cyclic row owner, bounded by the physical core count."""
    row = tl.program_id(0)
    while row < N:
        _syr_tail_row(
            a_ptr,
            x_ptr,
            alpha_r,
            alpha_i,
            row,
            N,
            LDA,
            INCX,
            UPLO,
            IS_COMPLEX,
            BLOCK,
        )
        row += tl.num_programs(0)


def _syr(uplo, n, alpha, x, incx, A, lda, is_complex):
    dtype = torch.complex64 if is_complex else torch.float32
    assert A.dtype == dtype == x.dtype
    assert A.is_contiguous() and x.is_contiguous()
    device_obj = A.device
    assert device_obj == x.device
    assert device_obj.type == "npu"
    assert uplo in (0, 1) and n >= 0
    assert incx > 0 and lda >= max(1, n)
    if n == 0:
        return A
    assert x.numel() >= 1 + (n - 1) * incx
    assert A.numel() >= n * lda
    scalar_type = type(alpha)
    if scalar_type is float:
        ar, ai = alpha, 0.0
    elif scalar_type is int:
        ar, ai = float(alpha), 0.0
    elif is_complex and scalar_type is complex:
        ar, ai = alpha.real, alpha.imag
    else:
        value = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
        value = complex(value) if is_complex else float(value)
        ar, ai = (value.real, value.imag) if is_complex else (value, 0.0)
    if ar == 0.0 and ai == 0.0:
        return A
    if A.is_conj() or x.is_conj() or A.is_neg() or x.is_neg():
        raise RuntimeError(
            "Ascend SYR requires resolved conjugate/negative tensor views"
        )
    device = device_obj.index
    if device != _current_device():
        with torch_device_fn.device(device_obj):
            return _syr(uplo, n, alpha, x, incx, A, lda, is_complex)
    pointers = A.data_ptr(), x.data_ptr()
    key = (device, is_complex, n, lda, incx, uplo, pointers[0] % 16, pointers[1] % 16)
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        _compile_and_launch(key, uplo, n, ar, ai, x, incx, A, lda, is_complex)
        return A
    compiled, run, grid, constants, special = entry
    stream = _current_raw_stream(device)
    knobs = triton.knobs.runtime
    enter, leave = knobs.launch_enter_hook, knobs.launch_exit_hook
    has_enter = enter is not None and (
        type(enter) is not _HOOK_CHAIN_TYPE or bool(enter.calls)
    )
    has_leave = leave is not None and (
        type(leave) is not _HOOK_CHAIN_TYPE or bool(leave.calls)
    )
    if special or has_enter or has_leave:
        args = (
            (torch.view_as_real(A), torch.view_as_real(x)) if is_complex else (A, x)
        )
        compiled[(grid, 1, 1)](*args, ar, ai, *constants, stream=stream)
    else:
        # Cache compiled metadata only: validate every call and pass fresh
        # addresses, scalars and current stream. No output cache or replay.
        a_arg = _DevicePointer(pointers[0])
        a_arg.tensor = A
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
            a_arg,
            x_arg,
            ar,
            ai,
            *constants,
        )
        if direct_launch:
            _ascend_driver_utils.TRITON_PROFILER_REGISTERED = profiler_registered == 1
    return A


def _compile_and_launch(key, uplo, n, ar, ai, x, incx, A, lda, is_complex):
    constants = (n, lda, incx, uplo, is_complex)
    if n >= 64:
        rows, cols = (8, 128) if n < 512 else (16, 256)
        if n < 128:
            cols = 64
            if not is_complex:
                rows = 16
        if not is_complex and n == 192:
            # Avoid the shifted/overlapping 128-column tail and use more cores.
            rows, cols = 4, 64
        if is_complex and n >= 4096:
            rows, cols = 8, 512
        kernel = syr_block_kernel
        grid = min(CORE_NUM, max(n // rows, n % rows, 1))
        constants += (rows, cols)
    else:
        kernel, grid = syr_row_kernel, min(n, CORE_NUM)
        constants += (min(triton.next_power_of_2(n), 1024),)
    args = (
        tuple(triton.reinterpret(t, tl.float32) for t in (A, x))
        if is_complex
        else (A, x)
    )
    compiled, _ = kernel[(grid,)](*args, ar, ai, *constants, num_warps=1)
    run = compiled.run
    special = (
        getattr(run, "compile_only", False)
        or getattr(run, "enable_msprof_register_tensor", False)
        or getattr(compiled.metadata, "debug_enabled", False)
    )
    if len(_LAUNCH_CACHE) >= 512:
        _LAUNCH_CACHE.clear()
    _LAUNCH_CACHE[key] = compiled, run, grid, constants, special


def ssyr(uplo, n, alpha, x, incx, A, lda):
    """In-place float32 symmetric rank-1 update of the selected triangle."""
    return _syr(uplo, n, alpha, x, incx, A, lda, False)


def csyr(uplo, n, alpha, x, incx, A, lda):
    """In-place complex64 symmetric rank-1 update, without conjugation."""
    return _syr(uplo, n, alpha, x, incx, A, lda, True)
