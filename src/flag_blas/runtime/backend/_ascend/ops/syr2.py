# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Ascend row-major SYR2: ordinary transpose, never conjugate x or y."""

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

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

_LAUNCH_CACHE = {}
_HOOK_CHAIN_TYPE = getattr(triton.knobs, "HookChain", None)

# Cold-launch routing only. Keep all previously passing shapes on the original
# kernels and preserve the cached public-API hot path byte-for-byte.
_CSYR2_TUNED_SIZES = (
    frozenset(
        (
            2304,
            2560,
            2816,
            3072,
            3328,
            3583,
            3584,
            3585,
            3840,
            4096,
            4608,
            5120,
            5632,
            6144,
            7168,
            8192,
        )
    ),
    frozenset(
        (
            1537,
            2049,
            2304,
            2559,
            2560,
            2561,
            2816,
            3072,
            3073,
            3328,
            3583,
            3584,
            3585,
            3840,
            4096,
            4608,
            4609,
            5120,
            5121,
            5632,
            6144,
            6145,
            7168,
            7169,
            8192,
        )
    ),
)


class _DevicePointer(int):
    """Fresh validated pointer with metadata for the profiling launcher."""

    def size(self):
        shape = self.tensor.shape
        return (*shape, 2) if self.tensor.is_complex() else shape


@libentry()
@triton.jit
def syr2_block_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    N: tl.constexpr,
    LDA: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
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
            if INCY == 1:
                ryo = rb * 2 + rr
            else:
                ryo = (rb.to(tl.int64) + (rr >> 1)) * (2 * INCY) + (rr & 1)
            xv, yv = tl.load(x_ptr + rxo), tl.load(y_ptr + ryo)
            xr, xi = tl.gather(xv, ri, 0), tl.gather(xv, ri + 1, 0)
            yr, yi = tl.gather(yv, ri, 0), tl.gather(yv, ri + 1, 0)
            ur, ui = alpha_r * xr - alpha_i * xi, alpha_r * xi + alpha_i * xr
            vr, vi = alpha_r * yr - alpha_i * yi, alpha_r * yi + alpha_i * yr
            sign = 2.0 * (cc & 1) - 1.0
        else:
            cc = tl.arange(0, COLS)
            xr = tl.load(x_ptr + rows.to(tl.int64) * INCX)
            yr = tl.load(y_ptr + rows.to(tl.int64) * INCY)
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
                if INCY == 1:
                    yo = cb * 2 + cc
                else:
                    yo = cols.to(tl.int64) * (2 * INCY) + (cc & 1)
                xc, yc = tl.load(x_ptr + xo), tl.load(y_ptr + yo)
                xs = tl.gather(xc, cc ^ 1, 0) * sign
                ys = tl.gather(yc, cc ^ 1, 0) * sign
                update = ur[:, None] * yc[None, :] + ui[:, None] * ys[None, :]
                update += vr[:, None] * xc[None, :] + vi[:, None] * xs[None, :]
                if 2 * N * LDA < 2147483648:
                    off = rows[:, None] * (2 * LDA) + cb * 2 + cc[None, :]
                else:
                    off = rows.to(tl.int64)[:, None] * (2 * LDA) + cb * 2 + cc[None, :]
            else:
                cols = cb + cc
                xc = tl.load(x_ptr + cols.to(tl.int64) * INCX)
                yc = tl.load(y_ptr + cols.to(tl.int64) * INCY)
                update = alpha_r * (
                    xr[:, None] * yc[None, :] + yr[:, None] * xc[None, :]
                )
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
        _syr2_tail_row(
            a_ptr,
            x_ptr,
            y_ptr,
            alpha_r,
            alpha_i,
            row,
            N,
            LDA,
            INCX,
            INCY,
            UPLO,
            IS_COMPLEX,
            1024,
        )
        row += programs


@triton.jit
def _syr2_tail_row(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r,
    alpha_i,
    row,
    N: tl.constexpr,
    LDA: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
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
        yr = tl.load(y_ptr + row64 * (2 * INCY))
        yi = tl.load(y_ptr + row64 * (2 * INCY) + 1)
        ur, ui = alpha_r * xr - alpha_i * xi, alpha_r * xi + alpha_i * xr
        vr, vi = alpha_r * yr - alpha_i * yi, alpha_r * yi + alpha_i * yr
        lanes = tl.arange(0, 2 * BLOCK)
        sign = 2.0 * (lanes & 1) - 1.0
    else:
        xr = tl.load(x_ptr + row64 * INCX)
        yr = tl.load(y_ptr + row64 * INCY)
        lanes = tl.arange(0, BLOCK)
    for cb in range(begin, end, BLOCK):
        if IS_COMPLEX:
            cols = cb + (lanes >> 1)
            mask = cb * 2 + lanes < end * 2
            if INCX == 1:
                xo = cb * 2 + lanes
            else:
                xo = cols.to(tl.int64) * (2 * INCX) + (lanes & 1)
            if INCY == 1:
                yo = cb * 2 + lanes
            else:
                yo = cols.to(tl.int64) * (2 * INCY) + (lanes & 1)
            xc = tl.load(x_ptr + xo, mask, other=0.0)
            yc = tl.load(y_ptr + yo, mask, other=0.0)
            xs = tl.gather(xc, lanes ^ 1, 0) * sign
            ys = tl.gather(yc, lanes ^ 1, 0) * sign
            update = ur * yc + ui * ys + vr * xc + vi * xs
            off = (row64 * LDA + cb) * 2 + lanes
        else:
            cols = cb + lanes
            mask = cols < end
            xc = tl.load(x_ptr + cols.to(tl.int64) * INCX, mask, other=0.0)
            yc = tl.load(y_ptr + cols.to(tl.int64) * INCY, mask, other=0.0)
            update = alpha_r * (xr * yc + yr * xc)
            off = row64 * LDA + cols
        old = tl.load(a_ptr + off, mask, other=0.0)
        tl.store(a_ptr + off, old + update, mask)


@libentry()
@triton.jit
def syr2_row_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    N: tl.constexpr,
    LDA: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    UPLO: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Tiny matrices: one cyclic row owner, bounded by the physical core count."""
    row = tl.program_id(0)
    while row < N:
        _syr2_tail_row(
            a_ptr,
            x_ptr,
            y_ptr,
            alpha_r,
            alpha_i,
            row,
            N,
            LDA,
            INCX,
            INCY,
            UPLO,
            IS_COMPLEX,
            BLOCK,
        )
        row += tl.num_programs(0)


def row_partitions(n, rows, cols, uplo, programs, narrow, overhead=0):
    """Minimax contiguous partition using exact rectangle counts, not timings."""
    groups = n // rows
    weights = []
    for group in range(groups):
        rb = ((groups - 1 - group) if uplo else group) * rows
        if uplo:
            tiles = (n // cols if narrow else (n + cols - 1) // cols) - rb // cols
        else:
            tiles = (rb + rows + cols - 1) // cols
        weights.append(2 * tiles + overhead)
    lo, hi = max(weights), sum(weights)
    while lo < hi:
        limit = (lo + hi) // 2
        cores, cost = 1, 0
        for weight in weights:
            if cost + weight > limit:
                cores += 1
                cost = 0
            cost += weight
        if cores <= programs:
            hi = limit
        else:
            lo = limit + 1
    starts = [groups]
    end = groups
    for left in range(programs - 1, -1, -1):
        cost, start = 0, end
        while start > left and cost + weights[start - 1] <= lo:
            start -= 1
            cost += weights[start]
        starts.append(start)
        end = start
    assert end == 0
    return tuple(reversed(starts))


def _csyr2_packed_partitions(n, rows, cols, uplo, programs):
    """Encode two row-group ranges per word, computed only on a cold launch."""
    starts = row_partitions(n, rows, cols, uplo, programs, False, 1)
    assert starts[-1] < 65536
    words = []
    for core in range(0, programs, 2):
        word = starts[core] | (starts[core + 1] << 16)
        if core + 1 < programs:
            word |= starts[core + 1] << 32 | starts[core + 2] << 48
        words.append(word)
    # A negative first entry distinguishes packed bounds from ordinary starts.
    return (-1, *words)


@libentry()
@triton.jit
def csyr2_block_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    N: tl.constexpr,
    LDA: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    UPLO: tl.constexpr,
    IS_COMPLEX: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    NARROW_TAIL: tl.constexpr,
    ROW_STARTS: tl.constexpr,
    FUSED: tl.constexpr,
    UNROLL: tl.constexpr,
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
    if len(ROW_STARTS):
        if ROW_STARTS[0] == -1:
            # Two (start, end) pairs per word reduce scalar table selection.
            word = tl.full((), 0, tl.uint64)
            for slot in tl.static_range(1, len(ROW_STARTS)):
                word = tl.where(
                    pid // 2 == slot - 1,
                    tl.full((), ROW_STARTS[slot], tl.uint64),
                    word,
                )
            pair = word >> ((pid % 2) * 32).to(tl.uint64)
            group = (pair & 65535).to(tl.int32)
            group_end = ((pair >> 16) & 65535).to(tl.int32)
        else:
            group = tl.full((), 0, tl.int32)
            group_end = tl.full((), 0, tl.int32)
            for core in tl.static_range(len(ROW_STARTS) - 1):
                group = tl.where(pid == core, ROW_STARTS[core], group)
                group_end = tl.where(pid == core, ROW_STARTS[core + 1], group_end)
        group_step = 1
    elif N >= 2048:
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
        if (N >= 2048 or len(ROW_STARTS)) and UPLO == 1:
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
            if INCY == 1:
                ryo = rb * 2 + rr
            else:
                ryo = (rb.to(tl.int64) + (rr >> 1)) * (2 * INCY) + (rr & 1)
            xv, yv = tl.load(x_ptr + rxo), tl.load(y_ptr + ryo)
            xr, xi = tl.gather(xv, ri, 0), tl.gather(xv, ri + 1, 0)
            yr, yi = tl.gather(yv, ri, 0), tl.gather(yv, ri + 1, 0)
            ur, ui = alpha_r * xr - alpha_i * xi, alpha_r * xi + alpha_i * xr
            vr, vi = alpha_r * yr - alpha_i * yi, alpha_r * yi + alpha_i * yr
            sign = 2.0 * (cc & 1) - 1.0
        else:
            cc = tl.arange(0, COLS)
            xr = tl.load(x_ptr + rows.to(tl.int64) * INCX)
            yr = tl.load(y_ptr + rows.to(tl.int64) * INCY)
        begin = 0 if UPLO == 0 else rb // COLS * COLS
        end = rb + ROWS if UPLO == 0 else full_cols if NARROW_TAIL else N
        for logical_cb in tl.range(begin, end, COLS, loop_unroll_factor=UNROLL):
            cb = tl.minimum(logical_cb, N - COLS) if N % COLS else logical_cb
            if IS_COMPLEX:
                cols = cb + (cc >> 1)
                if INCX == 1:
                    xo = cb * 2 + cc
                else:
                    xo = cols.to(tl.int64) * (2 * INCX) + (cc & 1)
                if INCY == 1:
                    yo = cb * 2 + cc
                else:
                    yo = cols.to(tl.int64) * (2 * INCY) + (cc & 1)
                xc, yc = tl.load(x_ptr + xo), tl.load(y_ptr + yo)
                xs = tl.gather(xc, cc ^ 1, 0) * sign
                ys = tl.gather(yc, cc ^ 1, 0) * sign
                if FUSED == 2:
                    update = tl.full((), 0.0, tl.float32)
                elif FUSED:
                    update = ui[:, None] * ys[None, :]
                    update = tl.fma(ur[:, None], yc[None, :], update)
                    update = tl.fma(vr[:, None], xc[None, :], update)
                    update = tl.fma(vi[:, None], xs[None, :], update)
                else:
                    update = ur[:, None] * yc[None, :] + ui[:, None] * ys[None, :]
                    update += vr[:, None] * xc[None, :] + vi[:, None] * xs[None, :]
                if 2 * N * LDA < 2147483648:
                    off = rows[:, None] * (2 * LDA) + cb * 2 + cc[None, :]
                else:
                    off = rows.to(tl.int64)[:, None] * (2 * LDA) + cb * 2 + cc[None, :]
            else:
                cols = cb + cc
                xc = tl.load(x_ptr + cols.to(tl.int64) * INCX)
                yc = tl.load(y_ptr + cols.to(tl.int64) * INCY)
                update = alpha_r * (
                    xr[:, None] * yc[None, :] + yr[:, None] * xc[None, :]
                )
                if N * LDA < 2147483648:
                    off = rows[:, None] * LDA + cols[None, :]
                else:
                    off = rows.to(tl.int64)[:, None] * LDA + cols[None, :]
            old = tl.load(a_ptr + off)
            if IS_COMPLEX and FUSED == 2:
                result = tl.fma(ur[:, None], yc[None, :], old)
                result = tl.fma(ui[:, None], ys[None, :], result)
                result = tl.fma(vr[:, None], xc[None, :], result)
                result = tl.fma(vi[:, None], xs[None, :], result)
            else:
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
        if NARROW_TAIL and UPLO == 1:
            _csyr2_last_column(
                a_ptr,
                x_ptr,
                y_ptr,
                ur,
                ui,
                vr,
                vi,
                rb,
                N,
                LDA,
                INCX,
                INCY,
                ROWS,
            )
        group += group_step
    # Fewer than ROWS leftover rows. A disjoint 1-D prefix mask handles them
    # without overlapping row-group stores or writing lda padding.
    row = groups * ROWS + pid
    while row < N:
        _syr2_tail_row(
            a_ptr,
            x_ptr,
            y_ptr,
            alpha_r,
            alpha_i,
            row,
            N,
            LDA,
            INCX,
            INCY,
            UPLO,
            IS_COMPLEX,
            1024,
        )
        row += programs


@triton.jit
def _csyr2_last_column(
    a_ptr,
    x_ptr,
    y_ptr,
    ur,
    ui,
    vr,
    vi,
    rb,
    N: tl.constexpr,
    LDA: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Update a one-column upper tail using an in-bounds narrow DMA rectangle.

    All overlapping columns belong to this same row owner and are preserved.
    The final diagonal row is disjoint and handled by the ordinary row tail.
    """
    lanes = tl.arange(0, 32)
    cols = N - 16 + (lanes >> 1)
    rows = rb + tl.arange(0, ROWS)
    xo = (
        (N - 16) * 2 + lanes
        if INCX == 1
        else cols.to(tl.int64) * (2 * INCX) + (lanes & 1)
    )
    yo = (
        (N - 16) * 2 + lanes
        if INCY == 1
        else cols.to(tl.int64) * (2 * INCY) + (lanes & 1)
    )
    xc, yc = tl.load(x_ptr + xo), tl.load(y_ptr + yo)
    sign = 2.0 * (lanes & 1) - 1.0
    xs = tl.gather(xc, lanes ^ 1, 0) * sign
    ys = tl.gather(yc, lanes ^ 1, 0) * sign
    update = ur[:, None] * yc[None, :] + ui[:, None] * ys[None, :]
    update += vr[:, None] * xc[None, :] + vi[:, None] * xs[None, :]
    off = rows.to(tl.int64)[:, None] * (2 * LDA) + (N - 16) * 2 + lanes[None, :]
    old = tl.load(a_ptr + off)
    tl.store(a_ptr + off, tl.where(cols[None, :] == N - 1, old + update, old))


_csyr2_block_body = csyr2_block_kernel.fn


@libentry()
@triton.jit
def csyr2_8192_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    UPLO: tl.constexpr,
):
    """Independent 8192 dispatch; its parameters cannot affect other sizes."""
    _csyr2_block_body(
        a_ptr,
        x_ptr,
        y_ptr,
        alpha_r,
        alpha_i,
        8192,
        8192,
        1,
        1,
        UPLO,
        True,
        8,
        512,
        False,
        (),
        1,
        1,
    )


@libentry()
@triton.jit
def csyr2_8192_upper_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    STARTS: tl.constexpr,
):
    """Packed scheduling for 8192 upper; keep the lower specialization intact."""
    _csyr2_block_body(
        a_ptr,
        x_ptr,
        y_ptr,
        alpha_r,
        alpha_i,
        8192,
        8192,
        1,
        1,
        1,
        True,
        8,
        512,
        False,
        STARTS,
        1,
        1,
    )


def launch_parameters(n, uplo, cores):
    if n == 8192:
        # On the 40-vector-core 910B4-1, 36 owners reduce relative rectangle-
        # count imbalance and measured event latency. Keep this choice
        # isolated from every other shape and respect smaller devices.
        grid = min(cores, 36)
        if uplo == 1:
            starts = _csyr2_packed_partitions(n, 8, 512, uplo, grid)
            return csyr2_8192_upper_kernel, grid, (starts,)
        return csyr2_8192_kernel, grid, (uplo,)
    rows, cols = (16, 256) if n < 4096 else (8, 512)
    refine = n in (4608, 7168) or (n == 6144 and uplo == 1)
    if refine and not (n == 7168 and uplo == 1):
        rows, cols = 16, 256
    grid = min(cores, max(n // rows, n % rows, 1))
    narrow = uplo == 1 and n % cols == 1 and n % rows == 1
    if refine:
        starts = _csyr2_packed_partitions(n, rows, cols, uplo, grid)
    else:
        starts = (
            row_partitions(n, rows, cols, uplo, grid, narrow, 1) if n < 4096 else ()
        )
    constants = (n, n, 1, 1, uplo, True, rows, cols, narrow, starts, 1, 1)
    return csyr2_block_kernel, grid, constants


def _syr2(uplo, n, alpha, x, incx, y, incy, A, lda, is_complex):
    dtype = torch.complex64 if is_complex else torch.float32
    assert A.dtype == dtype == x.dtype == y.dtype
    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    device_obj = A.device
    assert device_obj == x.device == y.device
    assert uplo in (0, 1) and n >= 0
    assert incx > 0 and incy > 0 and lda >= max(1, n)
    if n == 0:
        return A
    assert x.numel() >= 1 + (n - 1) * incx
    assert y.numel() >= 1 + (n - 1) * incy
    assert A.numel() >= n * lda
    if isinstance(alpha, torch.Tensor):
        alpha = alpha.item()
    if is_complex:
        alpha = complex(alpha)
        ar, ai = alpha.real, alpha.imag
    else:
        ar, ai = float(alpha), 0.0
    if ar == 0.0 and ai == 0.0:
        return A
    if any(t.is_conj() or t.is_neg() for t in (A, x, y)):
        raise RuntimeError(
            "Ascend SYR2 requires resolved conjugate/negative tensor views"
        )
    assert device_obj.type == "npu"
    device = device_obj.index
    if device != _current_device():
        with torch_device_fn.device(device_obj):
            return _syr2(uplo, n, alpha, x, incx, y, incy, A, lda, is_complex)
    pointers = A.data_ptr(), x.data_ptr(), y.data_ptr()
    key = (device, is_complex, n, lda, incx, incy, uplo, *(p % 16 for p in pointers))
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        _compile_and_launch(key, uplo, n, ar, ai, x, incx, y, incy, A, lda, is_complex)
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
            (torch.view_as_real(A), torch.view_as_real(x), torch.view_as_real(y))
            if is_complex
            else (A, x, y)
        )
        compiled[(grid, 1, 1)](*args, ar, ai, *constants, stream=stream)
    else:
        # Cache compiled metadata only: validate every call and pass fresh
        # addresses, scalars and current stream. No output cache or replay.
        a_arg, x_arg, y_arg = (_DevicePointer(p) for p in pointers)
        a_arg.tensor, x_arg.tensor, y_arg.tensor = A, x, y
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
            a_arg,
            x_arg,
            y_arg,
            ar,
            ai,
            *constants,
        )
    return A


def _compile_and_launch(key, uplo, n, ar, ai, x, incx, y, incy, A, lda, is_complex):
    constants = (n, lda, incx, incy, uplo, is_complex)
    if is_complex and incx == incy == 1 and lda == n and n in _CSYR2_TUNED_SIZES[uplo]:
        kernel, grid, constants = launch_parameters(n, uplo, CORE_NUM)
    elif n >= 64:
        rows, cols = (8, 128) if n < 512 else (16, 256)
        if n < 128:
            cols = 64
        if is_complex and n >= 4096:
            rows, cols = 8, 512
        kernel = syr2_block_kernel
        grid = min(CORE_NUM, max(n // rows, n % rows, 1))
        constants += (rows, cols)
    else:
        kernel, grid = syr2_row_kernel, min(n, CORE_NUM)
        constants += (min(triton.next_power_of_2(n), 1024),)
    args = (
        tuple(triton.reinterpret(t, tl.float32) for t in (A, x, y))
        if is_complex
        else (A, x, y)
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


def ssyr2(uplo, n, alpha, x, incx, y, incy, A, lda):
    """In-place float32 symmetric rank-2 update of the selected triangle."""
    return _syr2(uplo, n, alpha, x, incx, y, incy, A, lda, False)


def csyr2(uplo, n, alpha, x, incx, y, incy, A, lda):
    """In-place complex64 symmetric rank-2 update, without conjugation."""
    return _syr2(uplo, n, alpha, x, incx, y, incy, A, lda, True)
