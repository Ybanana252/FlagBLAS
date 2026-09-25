from typing import Union

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
def cher2_scalar_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
):
    ar = tl.load(a_ptr)
    xr = tl.load(x_ptr)
    xi = tl.load(x_ptr + 1)
    yr = tl.load(y_ptr)
    yi = tl.load(y_ptr + 1)
    prod_r = xr * yr + xi * yi
    prod_i = xi * yr - xr * yi
    update_r = 2.0 * (alpha_r * prod_r - alpha_i * prod_i)
    tl.store(a_ptr, ar + update_r)
    tl.store(a_ptr + 1, 0.0)


@libentry()
@triton.jit
def cher2_kernel(
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
    BLOCK_SIZE: tl.constexpr,
):
    """Own disjoint rows and stream interleaved complex values in place.

    Cyclic rows balance triangular lengths across physical cores. Each element
    has one writer; neither a matrix snapshot nor atomics are needed.
    """
    row = tl.program_id(0)
    lanes = tl.arange(0, 2 * BLOCK_SIZE)
    columns = tl.arange(0, BLOCK_SIZE)
    pair = tl.arange(0, 2)
    while row < N:
        row64 = row.to(tl.int64)
        if UPLO == 0:
            begin, end = 0, row + 1
        else:
            begin, end = row, N
        xr, xi = tl.split(
            tl.reshape(tl.load(x_ptr + row64 * (2 * INCX) + pair), (1, 2))
        )
        yr, yi = tl.split(
            tl.reshape(tl.load(y_ptr + row64 * (2 * INCY) + pair), (1, 2))
        )
        ur = alpha_r * xr - alpha_i * xi
        ui = alpha_r * xi + alpha_i * xr
        vr = alpha_r * yr + alpha_i * yi
        vi = alpha_r * yi - alpha_i * yr
        for start in range(begin, end, BLOCK_SIZE):
            col = start + columns
            mask = start * 2 + lanes < end * 2
            if INCX == 1:
                xoff = start * 2 + lanes
            else:
                xoff = (start.to(tl.int64) + (lanes >> 1)) * (2 * INCX) + (lanes & 1)
            if INCY == 1:
                yoff = start * 2 + lanes
            else:
                yoff = (start.to(tl.int64) + (lanes >> 1)) * (2 * INCY) + (lanes & 1)
            xc = tl.load(x_ptr + xoff, mask, other=0.0)
            yc = tl.load(y_ptr + yoff, mask, other=0.0)
            xcr, xci = tl.split(tl.reshape(xc, (BLOCK_SIZE, 2)))
            ycr, yci = tl.split(tl.reshape(yc, (BLOCK_SIZE, 2)))
            update_r = ur * ycr + ui * yci + vr * xcr + vi * xci
            update_i = ui * ycr - ur * yci + vi * xcr - vr * xci
            offset = (row64 * LDA + start) * 2 + lanes
            ar, ai = tl.split(
                tl.reshape(tl.load(a_ptr + offset, mask, other=0.0), (BLOCK_SIZE, 2))
            )
            out_i = tl.where(col == row, 0.0, ai + update_i)
            out = tl.reshape(tl.join(ar + update_r, out_i), (2 * BLOCK_SIZE,))
            tl.store(a_ptr + offset, out, mask)
        row += tl.num_programs(0)


@triton.jit
def _cher2_last_column(
    a_ptr, x_ptr, y_ptr, ur, ui, vr, vi, rb,
    N: tl.constexpr, LDA: tl.constexpr,
    INCX: tl.constexpr, INCY: tl.constexpr, ROWS: tl.constexpr,
):
    # A narrow, fully valid rectangle avoids scalar/scatter GM stores.
    # This program already owns and updated the overlapping columns.
    # Its full row group precedes the diagonal's last row.
    lanes = tl.arange(0, 32)
    cols = N - 16 + (lanes >> 1)
    rows = rb + tl.arange(0, ROWS)
    if INCX == 1:
        xo = (N - 16) * 2 + lanes
    else:
        xo = cols.to(tl.int64) * (2 * INCX) + (lanes & 1)
    if INCY == 1:
        yo = (N - 16) * 2 + lanes
    else:
        yo = cols.to(tl.int64) * (2 * INCY) + (lanes & 1)
    xc, yc = tl.load(x_ptr + xo), tl.load(y_ptr + yo)
    sign = 1.0 - 2.0 * (lanes & 1)
    x0, x1 = xc * sign, tl.gather(xc, lanes ^ 1, 0)
    y0, y1 = yc * sign, tl.gather(yc, lanes ^ 1, 0)
    update = ur[:, None] * y0[None, :] + ui[:, None] * y1[None, :]
    update += vr[:, None] * x0[None, :] + vi[:, None] * x1[None, :]
    offsets = rows.to(tl.int64)[:, None] * (2 * LDA) + (N - 16) * 2 + lanes[None, :]
    old = tl.load(a_ptr + offsets)
    tl.store(a_ptr + offsets, tl.where(cols[None, :] == N - 1, old + update, old))


@triton.jit
def _cher2_tail_row(
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
    BLOCK: tl.constexpr,
):
    row64 = row.to(tl.int64)
    xr = tl.load(x_ptr + row64 * (2 * INCX))
    xi = tl.load(x_ptr + row64 * (2 * INCX) + 1)
    yr = tl.load(y_ptr + row64 * (2 * INCY))
    yi = tl.load(y_ptr + row64 * (2 * INCY) + 1)
    ur, ui = alpha_r * xr - alpha_i * xi, alpha_r * xi + alpha_i * xr
    vr, vi = alpha_r * yr + alpha_i * yi, alpha_r * yi - alpha_i * yr
    lanes = tl.arange(0, 2 * BLOCK)
    sign = 1.0 - 2.0 * (lanes & 1)
    begin = 0 if UPLO == 0 else row
    end = row + 1 if UPLO == 0 else N
    for cb in range(begin, end, BLOCK):
        cols = cb + (lanes >> 1)
        # A linear float-lane prefix mask is important here: a mask derived
        # from complex-column division can lower to overlapping writeback.
        mask = cb * 2 + lanes < end * 2
        if INCX == 1:
            xo = cb * 2 + lanes
        else:
            xo = cols.to(tl.int64) * (2 * INCX) + (lanes & 1)
        if INCY == 1:
            yo = cb * 2 + lanes
        else:
            yo = cols.to(tl.int64) * (2 * INCY) + (lanes & 1)
        xv = tl.load(x_ptr + xo, mask, other=0.0)
        yv = tl.load(y_ptr + yo, mask, other=0.0)
        x0, x1 = xv * sign, tl.gather(xv, lanes ^ 1, 0)
        y0, y1 = yv * sign, tl.gather(yv, lanes ^ 1, 0)
        offsets = (row64 * LDA + cb) * 2 + lanes
        old = tl.load(a_ptr + offsets, mask, other=0.0)
        result = tl.fma(ur, y0, old)
        result = tl.fma(ui, y1, result)
        result = tl.fma(vr, x0, result)
        result = tl.fma(vi, x1, result)
        result = tl.where((cols == row) & ((lanes & 1) != 0), 0.0, result)
        tl.store(a_ptr + offsets, result, mask)


@libentry()
@triton.jit
def cher2_block_kernel(
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
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BALANCED: tl.constexpr,
    ROW_STARTS: tl.constexpr,
    PAIR_ROWS: tl.constexpr,
):
    # Shift the last column tile left until the entire rectangle is valid.
    # Preserve its overlap with the preceding tile (owned by this program).
    # This avoids both masked 2-D stores and serial per-row column tails.
    tl.static_assert(N >= COLS)
    full_rows: tl.constexpr = N // ROWS * ROWS
    full_cols: tl.constexpr = N // COLS * COLS
    groups: tl.constexpr = N // ROWS
    single_column_tail: tl.constexpr = N % COLS == 1 and N % ROWS == 1
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    if BALANCED:
        # Triangular work grows quadratically with the row-group index;
        # the linear term accounts for per-group vector/diagonal overhead.
        linear: tl.constexpr = 2 * COLS // ROWS
        total: tl.constexpr = groups * (groups + linear)
        if len(ROW_STARTS):
            # Compile the partition into scalar selects: no GM lookup table,
            # workspace tensor, host copy, or per-launch scheduling work.
            group = tl.full((), 0, tl.int32)
            group_end = tl.full((), 0, tl.int32)
            for core in tl.static_range(len(ROW_STARTS) - 1):
                group = tl.where(pid == core, ROW_STARTS[core], group)
                group_end = tl.where(pid == core, ROW_STARTS[core + 1], group_end)
        else:
            group = (
                (tl.sqrt(linear * linear + 4.0 * total * pid / programs) - linear)
                * 0.5
            ).to(tl.int32)
            next_group = (
                (tl.sqrt(linear * linear + 4.0 * total * (pid + 1) / programs) - linear)
                * 0.5
            ).to(tl.int32)
            group_end = tl.where(pid + 1 == programs, groups, next_group)
        group_step = 1
    else:
        group, group_end, group_step = pid, groups, programs
    rr = tl.arange(0, 2 * ROWS)
    ri = tl.arange(0, ROWS) * 2
    cc = tl.arange(0, 2 * COLS)
    sign = 1.0 - 2.0 * (cc & 1)
    while group < group_end:
        if BALANCED and UPLO == 1:
            rb = (groups - 1 - group) * ROWS
        elif PAIR_ROWS:
            # With P < groups <= 2P, give the extra cores two short
            # groups each and every other core one long group. The first
            # pass owns [0, extra) and [2*extra, groups); the second owns
            # [extra, 2*extra) in reverse order. No row is shared or skipped.
            extra = groups - programs
            paired = tl.where(
                group < programs,
                tl.where(pid < extra, pid, pid + extra),
                2 * extra - 1 - pid,
            )
            if UPLO == 1:
                paired = groups - 1 - paired
            rb = paired * ROWS
        else:
            rb = group * ROWS
        rows = rb + tl.arange(0, ROWS)
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
        vr, vi = alpha_r * yr + alpha_i * yi, alpha_r * yi - alpha_i * yr
        begin = 0 if UPLO == 0 else (rb // COLS) * COLS
        end = rb + ROWS if UPLO == 0 else full_cols if single_column_tail else N
        for logical_cb in range(begin, end, COLS):
            if N % COLS and not single_column_tail:
                cb = tl.minimum(logical_cb, N - COLS)
            else:
                cb = logical_cb
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
            x0, x1 = xc * sign, tl.gather(xc, cc ^ 1, 0)
            y0, y1 = yc * sign, tl.gather(yc, cc ^ 1, 0)
            update = ur[:, None] * y0[None, :] + ui[:, None] * y1[None, :]
            update += vr[:, None] * x0[None, :] + vi[:, None] * x1[None, :]
            if 2 * N * LDA < 2147483648:
                offsets = rows[:, None] * (2 * LDA) + cb * 2 + cc[None, :]
            else:
                offsets = (
                    rows.to(tl.int64)[:, None] * (2 * LDA) + cb * 2 + cc[None, :]
                )
            old = tl.load(a_ptr + offsets)
            result = old + update
            if (UPLO == 0 and cb + COLS > rb) or (UPLO == 1 and cb < rb + ROWS):
                if UPLO == 0:
                    selected = rows[:, None] >= cols[None, :]
                else:
                    selected = rows[:, None] <= cols[None, :]
                diag_imag = (rows[:, None] == cols[None, :]) & ((cc[None, :] & 1) != 0)
                result = tl.where(selected, tl.where(diag_imag, 0.0, result), old)
            if N % COLS and not single_column_tail:
                if logical_cb == full_cols:
                    result = tl.where(cols[None, :] >= full_cols, result, old)
            tl.store(a_ptr + offsets, result)
        if single_column_tail and UPLO == 1:
            _cher2_last_column(
                a_ptr, x_ptr, y_ptr, ur, ui, vr, vi, rb,
                N, LDA, INCX, INCY, ROWS,
            )
        group += group_step
    if N % ROWS:
        tail_width: tl.constexpr = 1024 if N >= 1024 else 512 if N >= 512 else COLS
        row = full_rows + pid
        while row < N:
            _cher2_tail_row(
                a_ptr, x_ptr, y_ptr, alpha_r, alpha_i, row,
                N, LDA, INCX, INCY, UPLO, tail_width,
            )
            row += programs


# Inline the unchanged main-row implementation in the odd-size specialization.
# The original entry point and all of its existing callers remain untouched.
_cher2_main_rows = cher2_block_kernel.fn


@libentry()
@triton.jit
def cher2_lower_tail_kernel(
    a_ptr, x_ptr, y_ptr,
    alpha_r: tl.float32, alpha_i: tl.float32,
    N: tl.constexpr, LDA: tl.constexpr,
    ROWS: tl.constexpr, COLS: tl.constexpr,
    TAIL_START: tl.constexpr, TAIL_WIDTH: tl.constexpr,
):
    full_rows: tl.constexpr = N // ROWS * ROWS
    # Lower main rows do not need any column at or beyond full_rows.
    # Passing this exact logical extent suppresses the old tail path while
    # retaining the real matrix leading dimension and disjoint row ownership.
    _cher2_main_rows(
        a_ptr, x_ptr, y_ptr, alpha_r, alpha_i,
        full_rows, LDA, 1, 1, 0, ROWS, COLS, False, (), False,
    )
    tail = tl.program_id(0) - TAIL_START
    if tail >= 0 and tail < N % ROWS:
        _cher2_tail_row(
            a_ptr, x_ptr, y_ptr, alpha_r, alpha_i, full_rows + tail,
            N, LDA, 1, 1, 0, TAIL_WIDTH,
        )


_cher2_lower_tail_body = cher2_lower_tail_kernel.fn


@libentry()
@triton.jit
def cher2_lower_tail_compact_kernel(
    a_ptr, x_ptr, y_ptr,
    alpha_r: tl.float32, alpha_i: tl.float32,
    N: tl.constexpr,
):
    # Only the logical size crosses the launch boundary. Tile dimensions and
    # tail ownership are immutable properties of these contiguous cases.
    tl.static_assert(N == 129 or N == 513 or N == 1023)
    rows: tl.constexpr = 8 if N == 129 else 16
    cols: tl.constexpr = 128 if N == 129 else 256
    start: tl.constexpr = N // rows if N // rows < 40 else (N // rows) % 40
    width: tl.constexpr = 256 if N == 129 else 1024
    _cher2_lower_tail_body(
        a_ptr, x_ptr, y_ptr, alpha_r, alpha_i, N, N, rows, cols, start, width,
    )


@libentry()
@triton.jit
def cher2_lower_448_kernel(
    a_ptr, x_ptr, y_ptr,
    alpha_r: tl.float32, alpha_i: tl.float32,
):
    # Forty programs each own exactly two 8-by-256 rectangles. The first
    # sixteen own two short row groups; the others own one long row group.
    # Overlapping column rectangles always belong to the same program.
    pid = tl.program_id(0)
    rr = tl.arange(0, 16)
    ri = tl.arange(0, 8) * 2
    cc = tl.arange(0, 512)
    sign = 1.0 - 2.0 * (cc & 1)
    rb = tl.where(pid < 16, pid, pid + 16) * 8
    xv, yv = tl.load(x_ptr + rb * 2 + rr), tl.load(y_ptr + rb * 2 + rr)
    xr, xi = tl.gather(xv, ri, 0), tl.gather(xv, ri + 1, 0)
    yr, yi = tl.gather(yv, ri, 0), tl.gather(yv, ri + 1, 0)
    ur, ui = alpha_r * xr - alpha_i * xi, alpha_r * xi + alpha_i * xr
    vr, vi = alpha_r * yr + alpha_i * yi, alpha_r * yi - alpha_i * yr
    for step in tl.static_range(2):
        if step == 1:
            if pid < 16:
                rb = (31 - pid) * 8
                xv, yv = tl.load(x_ptr + rb * 2 + rr), tl.load(y_ptr + rb * 2 + rr)
                xr, xi = tl.gather(xv, ri, 0), tl.gather(xv, ri + 1, 0)
                yr, yi = tl.gather(yv, ri, 0), tl.gather(yv, ri + 1, 0)
                ur, ui = alpha_r * xr - alpha_i * xi, alpha_r * xi + alpha_i * xr
                vr, vi = alpha_r * yr + alpha_i * yi, alpha_r * yi - alpha_i * yr
        cb = 0 if step == 0 else tl.where(pid < 16, 0, 192)
        rows = rb + tl.arange(0, 8)
        cols = cb + (cc >> 1)
        xc, yc = tl.load(x_ptr + cb * 2 + cc), tl.load(y_ptr + cb * 2 + cc)
        x0, x1 = xc * sign, tl.gather(xc, cc ^ 1, 0)
        y0, y1 = yc * sign, tl.gather(yc, cc ^ 1, 0)
        update = ur[:, None] * y0[None, :] + ui[:, None] * y1[None, :]
        update += vr[:, None] * x0[None, :] + vi[:, None] * x1[None, :]
        offsets = rows[:, None] * 896 + cb * 2 + cc[None, :]
        old = tl.load(a_ptr + offsets)
        selected = rows[:, None] >= cols[None, :]
        if step == 1:
            selected = selected & ((pid < 16) | (cols[None, :] >= 256))
        diagonal_imag = (rows[:, None] == cols[None, :]) & ((cc[None, :] & 1) != 0)
        result = tl.where(selected, tl.where(diagonal_imag, 0.0, old + update), old)
        tl.store(a_ptr + offsets, result)


def cher2(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    A: torch.Tensor,
    lda: int,
):
    # Keep the same checks as _check_her2_args on every call, including cache
    # hits. Reuse the device object locally instead of fetching A.device four
    # times through Python/Tensor bindings; never cache input tensor metadata.
    assert A.dtype == torch.complex64 == x.dtype == y.dtype
    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    a_device = A.device
    assert a_device == x.device == y.device
    assert uplo in (0, 1)
    assert n >= 0
    assert incx > 0 and incy > 0
    assert lda >= max(1, n)
    if n == 0:
        return A
    assert x.numel() >= 1 + (n - 1) * incx
    assert y.numel() >= 1 + (n - 1) * incy
    assert A.numel() >= n * lda

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
        value = complex(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
        ar, ai = value.real, value.imag
    if ar == 0.0 and ai == 0.0:
        return A
    assert a_device.type == "npu"
    device = a_device.index
    # A validated NPU tensor already has an initialized runtime. Query the
    # current device afresh without repeating torch_npu's Python lazy-init.
    if device != _current_device():
        with torch_device_fn.device(a_device):
            # The recursive call now sees the target device, so it re-enters
            # at most once. Reuse the parsed scalar instead of repeating a
            # Tensor.item() or a scalar subclass's conversion side effects.
            return cher2(uplo, n, complex(ar, ai), x, incx, y, incy, A, lda)

    # Preserve view_as_real's validation and Tensor subclass dispatch. Raw
    # pointers must never silently reinterpret a lazy conjugate/negative view.
    if (
        type(A) is not torch.Tensor
        or type(x) is not torch.Tensor
        or type(y) is not torch.Tensor
        or A.is_conj()
        or x.is_conj()
        or y.is_conj()
        or A.is_neg()
        or x.is_neg()
        or y.is_neg()
    ):
        _compile_and_launch_cher2(None, uplo, n, ar, ai, x, incx, y, incy, A, lda)
        return A

    # Keep a cache hit in this frame: small HER2 calls are sensitive to the
    # Python frontend-to-launcher boundary as well as device kernel work.
    pointers = (A.data_ptr(), x.data_ptr(), y.data_ptr())
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
        lda,
        pointers[0] % 16,
        pointers[1] % 16,
        pointers[2] % 16,
    )
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        _compile_and_launch_cher2(key, uplo, n, ar, ai, x, incx, y, incy, A, lda)
        return A
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
            torch.view_as_real(A),
            torch.view_as_real(x),
            torch.view_as_real(y),
            ar,
            ai,
            *constants,
            stream=stream,
        )
        return A
    # Validated same-device pointers avoid redundant aclrtPointerGetAttributes
    # in the Ascend launcher. Shape-carrying ints support dynamic msprof L1.
    # No graph/replay/output cache: this is one ordinary asynchronous launch.
    # Use int's native constructor, attaching fresh shape metadata immediately
    # instead of calling a Python __new__ once per input. Never reuse wrappers.
    a_arg = _DevicePointer(pointers[0])
    a_arg.tensor = A
    x_arg = _DevicePointer(pointers[1])
    x_arg.tensor = x
    y_arg = _DevicePointer(pointers[2])
    y_arg.tensor = y
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
        y_arg,
        ar,
        ai,
        *constants,
    )
    if direct_launch:
        _ascend_driver_utils.TRITON_PROFILER_REGISTERED = profiler_registered == 1
    return A


def _cher2_row_partition(groups, groups_per_column, programs):
    """Minimize the heaviest core's integer row-group cost at compile time.

    A row group costs its number of column rectangles plus one vector/diagonal
    setup unit. Unlike a rounded continuous square-root partition, this accounts
    for the staircase of actual loop trip counts. Upper groups are mirrored by
    the kernel, so aligned upper and lower matrices share the same partition.
    Only immutable shape-derived integers enter the compiled-launcher cache.
    """
    weights = [
        (group + groups_per_column) // groups_per_column + 1
        for group in range(groups)
    ]
    low, high = max(weights), sum(weights)
    while low < high:
        capacity = (low + high) // 2
        used, parts = 0, 1
        for weight in weights:
            if used + weight > capacity:
                parts += 1
                used = 0
            used += weight
        if parts <= programs:
            high = capacity
        else:
            low = capacity + 1

    # Build exactly programs nonempty, contiguous partitions. Reserve one row
    # group for each remaining core even when the greedy capacity needs fewer.
    boundaries, used = [groups], 0
    for group in range(groups - 1, -1, -1):
        if used + weights[group] > low or group + 1 < programs - len(boundaries) + 1:
            boundaries.append(group + 1)
            used = 0
        used += weights[group]
    boundaries.append(0)
    return tuple(reversed(boundaries))


def _compile_and_launch_cher2(key, uplo, n, ar, ai, x, incx, y, incy, A, lda):
    """Cache compiled code only; pointers, scalar values and streams stay live."""
    if n == 1:
        kernel, grid, constants = cher2_scalar_kernel, 1, ()
    elif (
        CORE_NUM == 40 and uplo == 0 and n in (129, 448, 513, 1023)
        and lda == n and incx == incy == 1
    ):
        # Narrow, independently checked specializations. Every other shape,
        # triangle, leading dimension and vector stride keeps the old route.
        if n == 448:
            kernel, grid, constants = cher2_lower_448_kernel, 40, ()
        else:
            rows = 8 if n == 129 else 16
            kernel = cher2_lower_tail_compact_kernel
            grid = min(n // rows + 1, CORE_NUM)
            constants = (n,)
    elif n >= 64:
        rows, cols = (8, 128) if n < 512 else (16, 256)
        if n < 128:
            cols = 64
        # Bound the rectangle by the vector-core UB and keep every load
        # inside the matrix. Partial row groups use the 1-D row helper.
        kernel = cher2_block_kernel
        grid = min(CORE_NUM, max(n // rows, n % rows, 1))
        balanced = n >= 2048
        row_starts = ()
        if balanced and n % cols == 0:
            row_starts = _cher2_row_partition(n // rows, cols // rows, grid)
        # At most two row groups per core: pair light triangular work instead
        # of leaving a core with two heavy groups in a cyclic assignment.
        # Larger matrices retain their existing minimax/continuous partition.
        pair_rows = n >= 512 and not balanced and grid < n // rows <= 2 * grid
        # These lower cases showed no full-call gain (and occasional cold
        # regressions) in paired A/B runs; retain their cyclic assignment.
        if uplo == 0 and n in (1023, 1024):
            pair_rows = False
        constants = (
            n, lda, incx, incy, uplo, rows, cols, balanced, row_starts, pair_rows,
        )
    else:
        kernel, grid = cher2_kernel, min(n, CORE_NUM)
        # A 1024-complex block balances contiguous transfers and UB pressure.
        # Larger blocks lose occupancy or exceed the 910B4 vector-core UB.
        constants = (n, lda, incx, incy, uplo, min(triton.next_power_of_2(n), 1024))
    compiled, _ = kernel[(grid,)](
        torch.view_as_real(A),
        torch.view_as_real(x),
        torch.view_as_real(y),
        ar,
        ai,
        *constants,
        num_warps=1,
    )
    if key is None:
        return
    if len(_LAUNCH_CACHE) >= 512:
        _LAUNCH_CACHE.clear()
    run = compiled.run
    special_launch = (
        getattr(run, "compile_only", False)
        or getattr(run, "enable_msprof_register_tensor", False)
        or getattr(compiled.metadata, "debug_enabled", False)
    )
    _LAUNCH_CACHE[key] = (compiled, run, grid, constants, special_launch)
