# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Row-major GER with exclusive row ownership and contiguous complex transfers."""

import torch
import triton
import triton.language as tl

from flag_blas.ops.level2.ger import _check_ger_common, _scalar_to_complex_parts
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
    """Fresh tensor metadata for the Ascend profiling launcher."""

    def size(self):
        shape = self.tensor.shape
        return (*shape, 2) if self.tensor.is_complex() else shape


@triton.jit
def _ger_row(
    A, X, Y, ar, ai, row,
    N: tl.constexpr, LDA: tl.constexpr, INCX: tl.constexpr, INCY: tl.constexpr,
    COMPLEX: tl.constexpr, CONJ: tl.constexpr, BLOCK: tl.constexpr,
):
    if COMPLEX:
        xr = tl.load(X + row.to(tl.int64) * (2 * INCX))
        xi = tl.load(X + row.to(tl.int64) * (2 * INCX) + 1)
        ur, ui = ar * xr - ai * xi, ar * xi + ai * xr
        lane = tl.arange(0, 2 * BLOCK)
        sign = 2.0 * (lane & 1) - 1.0
    else:
        ur = ar * tl.load(X + row.to(tl.int64) * INCX)
        lane = tl.arange(0, BLOCK)
    for cb in range(0, N, BLOCK):
        if COMPLEX:
            cols = cb + (lane >> 1)
            if INCY == 1:
                yo = cb * 2 + lane
            else:
                yo = cols.to(tl.int64) * (2 * INCY) + (lane & 1)
            yy = tl.load(Y + yo, cols < N, other=0)
            if CONJ:
                yy = yy * -sign
            swapped = tl.gather(yy, lane ^ 1, 0) * sign
            off = row.to(tl.int64) * (2 * LDA) + 2 * cb + lane
            update = ur * yy + ui * swapped
        else:
            cols = cb + lane
            yy = tl.load(Y + cols.to(tl.int64) * INCY, cols < N, other=0)
            off = row.to(tl.int64) * LDA + cols
            update = ur * yy
        old = tl.load(A + off, cols < N, other=0)
        tl.store(A + off, old + update, cols < N)


@libentry()
@triton.jit
def ger_row_kernel(
    A, X, Y, ar: tl.float32, ai: tl.float32,
    M: tl.constexpr, N: tl.constexpr, LDA: tl.constexpr,
    INCX: tl.constexpr, INCY: tl.constexpr,
    COMPLEX: tl.constexpr, CONJ: tl.constexpr, BLOCK: tl.constexpr,
):
    for row in range(tl.program_id(0), M, tl.num_programs(0)):
        _ger_row(A, X, Y, ar, ai, row, N, LDA, INCX, INCY, COMPLEX, CONJ, BLOCK)


@libentry()
@triton.jit
def ger_block_kernel(
    A, X, Y, ar: tl.float32, ai: tl.float32,
    M: tl.constexpr, N: tl.constexpr, LDA: tl.constexpr,
    INCX: tl.constexpr, INCY: tl.constexpr,
    COMPLEX: tl.constexpr, CONJ: tl.constexpr,
    ROWS: tl.constexpr, COLS: tl.constexpr,
):
    tl.static_assert(N >= COLS)
    tl.static_assert(ROWS % 8 == 0)
    groups: tl.constexpr = triton.cdiv(M, ROWS)
    full_cols: tl.constexpr = N // COLS * COLS
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    # Row groups start at DMA-aligned boundaries (ROWS is a multiple of eight).
    # One program owns the whole group, including the shifted column tail, so
    # unaligned stores in adjacent rows cannot race with another program.
    for group in range(pid, groups, programs):
        rb = group * ROWS
        rows = rb + tl.arange(0, ROWS)
        if COMPLEX:
            rr = tl.arange(0, 2 * ROWS)
            ri = tl.arange(0, ROWS) * 2
            if INCX == 1:
                xo = rb * 2 + rr
            else:
                xo = (rb.to(tl.int64) + (rr >> 1)) * (2 * INCX) + (rr & 1)
            xx = tl.load(X + xo, rb + (rr >> 1) < M, other=0)
            xr, xi = tl.gather(xx, ri, 0), tl.gather(xx, ri + 1, 0)
            ur, ui = ar * xr - ai * xi, ar * xi + ai * xr
            cc = tl.arange(0, 2 * COLS)
            sign = 2.0 * (cc & 1) - 1.0
        else:
            ur = ar * tl.load(X + rows.to(tl.int64) * INCX, rows < M, other=0)
            cc = tl.arange(0, COLS)
        for logical_cb in range(0, N, COLS):
            # Keep column transfers contiguous even for odd N. The final tile
            # overlaps the preceding tile: retain values already updated there.
            cb = tl.minimum(logical_cb, N - COLS) if N % COLS else logical_cb
            if COMPLEX:
                cols = cb + (cc >> 1)
                if INCY == 1:
                    yo = cb * 2 + cc
                else:
                    yo = cols.to(tl.int64) * (2 * INCY) + (cc & 1)
                yy = tl.load(Y + yo)
                if CONJ:
                    yy = yy * -sign
                swapped = tl.gather(yy, cc ^ 1, 0) * sign
                update = ur[:, None] * yy[None, :] + ui[:, None] * swapped[None, :]
                if 2 * M * LDA < 2147483648:
                    off = rows[:, None] * (2 * LDA) + cb * 2 + cc[None, :]
                else:
                    off = rows.to(tl.int64)[:, None] * (2 * LDA) + cb * 2 + cc[None, :]
            else:
                cols = cb + cc
                yy = tl.load(Y + cols.to(tl.int64) * INCY)
                update = ur[:, None] * yy[None, :]
                if M * LDA < 2147483648:
                    off = rows[:, None] * LDA + cols[None, :]
                else:
                    off = rows.to(tl.int64)[:, None] * LDA + cols[None, :]
            old = tl.load(A + off, rows[:, None] < M, other=0)
            result = old + update
            if N % COLS:
                if logical_cb == full_cols:
                    result = tl.where(cols[None, :] >= full_cols, result, old)
            tl.store(A + off, result, rows[:, None] < M)


def _compile_and_launch(key, m, n, ar, ai, x, incx, y, incy, A, lda, complex_data, conj):
    constants = (m, n, lda, incx, incy, complex_data, conj)
    if m >= 8 and n >= 64:
        if n < 256:
            rows, cols = 8, 64
        elif complex_data:
            # Tail masks need additional UB space; keep those tiles smaller.
            rows, cols = (8, 512) if n % 512 == 0 and m % 8 == 0 else (8, 256)
        else:
            rows, cols = (8, 1024) if n >= 1024 else (32, 256)
        kernel = ger_block_kernel
        grid = min(CORE_NUM, triton.cdiv(m, rows))
        constants += (rows, cols)
    else:
        kernel, grid = ger_row_kernel, min(m, CORE_NUM)
        constants += (min(triton.next_power_of_2(n), 1024),)
        if lda * (8 if complex_data else 4) % 32:
            grid = 1
    if A.data_ptr() % 32:
        grid = 1
    args = (
        tuple(triton.reinterpret(t, tl.float32) for t in (A, x, y))
        if complex_data else (A, x, y)
    )
    compiled, _ = kernel[(grid,)](*args, ar, ai, *constants, num_warps=1)
    if len(_LAUNCH_CACHE) >= 512:
        _LAUNCH_CACHE.clear()
    _LAUNCH_CACHE[key] = compiled, grid, constants


def _ger(m, n, alpha, x, incx, y, incy, A, lda, complex_data, conj):
    dtype = torch.complex64 if complex_data else torch.float32
    if not _check_ger_common(m, n, x, incx, y, incy, A, lda, dtype):
        return
    if complex_data:
        ar, ai = _scalar_to_complex_parts(alpha)
    else:
        ar = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
        ai = 0.0
    if ar == 0.0 and ai == 0.0:
        return
    if A.is_conj() or x.is_conj() or y.is_conj() or A.is_neg() or x.is_neg() or y.is_neg():
        raise RuntimeError("Ascend GER requires resolved conjugate/negative tensor views")
    device = A.device.index
    if device != _current_device():
        with torch_device_fn.device(A.device):
            return _ger(m, n, alpha, x, incx, y, incy, A, lda, complex_data, conj)
    key = (
        device, m, n, lda, incx, incy, complex_data, conj,
        A.data_ptr() % 32, x.data_ptr() % 16, y.data_ptr() % 16,
    )
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        _compile_and_launch(key, m, n, ar, ai, x, incx, y, incy, A, lda, complex_data, conj)
        return
    compiled, grid, constants = entry
    stream = _current_raw_stream(device)
    run = compiled.run
    knobs = triton.knobs.runtime
    enter, leave = knobs.launch_enter_hook, knobs.launch_exit_hook
    has_enter = enter is not None and (type(enter) is not _HOOK_CHAIN_TYPE or bool(enter.calls))
    has_leave = leave is not None and (type(leave) is not _HOOK_CHAIN_TYPE or bool(leave.calls))
    special = (
        getattr(run, "compile_only", False)
        or getattr(run, "enable_msprof_register_tensor", False)
        or getattr(compiled.metadata, "debug_enabled", False)
    )
    if special or has_enter or has_leave:
        args = (
            tuple(triton.reinterpret(t, tl.float32) for t in (A, x, y))
            if complex_data else (A, x, y)
        )
        compiled[(grid, 1, 1)](*args, ar, ai, *constants, stream=stream)
    else:
        # Cache code/metadata only. Validate inputs and use current pointers,
        # scalar values and stream on every invocation.
        pointers = []
        for tensor in (A, x, y):
            pointer = _DevicePointer(tensor.data_ptr())
            pointer.tensor = tensor
            pointers.append(pointer)
        direct = type(run) is _NPU_LAUNCHER
        launch = run.launch if direct else run
        registered = launch(
            grid, 1, 1, stream, compiled.function,
            compiled.packed_metadata, None, None, None,
            *pointers, ar, ai, *constants,
        )
        if direct:
            _ascend_driver_utils.TRITON_PROFILER_REGISTERED = registered == 1


def sger(m, n, alpha, x, incx, y, incy, A, lda):
    _ger(m, n, alpha, x, incx, y, incy, A, lda, False, False)


def cgeru(m, n, alpha, x, incx, y, incy, A, lda):
    _ger(m, n, alpha, x, incx, y, incy, A, lda, True, False)


def cgerc(m, n, alpha, x, incx, y, incy, A, lda):
    _ger(m, n, alpha, x, incx, y, incy, A, lda, True, True)
