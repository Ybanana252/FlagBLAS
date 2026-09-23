from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas.ops.level2.her import _check_her_args
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

from ..utils import CORE_NUM
from .her2 import (
    _LAUNCH_CACHE as _HER2_LAUNCH_CACHE,
    _cher2_row_partition,
    _compile_and_launch_cher2,
)

ScalarType = Union[float, int, torch.Tensor]

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
    """Validated address carrying complex-tensor shape for msprof."""

    def size(self):
        return (*self.tensor.shape, 2)


@triton.jit
def _cher_last_column(
    a_ptr,
    x_ptr,
    ur,
    ui,
    rb,
    N: tl.constexpr,
    LDA: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Update the final upper-triangle column without a masked 2-D store."""
    lanes = tl.arange(0, 32)
    cols = N - 16 + (lanes >> 1)
    rows = rb + tl.arange(0, ROWS)
    xc = tl.load(x_ptr + (N - 16) * 2 + lanes)
    sign = 1.0 - 2.0 * (lanes & 1)
    x0, x1 = xc * sign, tl.gather(xc, lanes ^ 1, 0)
    update = ur[:, None] * x0[None, :] + ui[:, None] * x1[None, :]
    offsets = rows.to(tl.int64)[:, None] * (2 * LDA) + (N - 16) * 2 + lanes[None, :]
    old = tl.load(a_ptr + offsets)
    tl.store(a_ptr + offsets, tl.where(cols[None, :] == N - 1, old + update, old))


@triton.jit
def _cher_tail_row(
    a_ptr,
    x_ptr,
    alpha,
    row,
    N: tl.constexpr,
    LDA: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Handle rows outside the final complete ROWS-wide row group."""
    row64 = row.to(tl.int64)
    xr = tl.load(x_ptr + row64 * 2)
    xi = tl.load(x_ptr + row64 * 2 + 1)
    ur, ui = alpha * xr, alpha * xi
    lanes = tl.arange(0, 2 * BLOCK)
    sign = 1.0 - 2.0 * (lanes & 1)
    begin = 0 if UPLO == 0 else row
    end = row + 1 if UPLO == 0 else N
    for cb in range(begin, end, BLOCK):
        cols = cb + (lanes >> 1)
        mask = cb * 2 + lanes < end * 2
        xv = tl.load(x_ptr + cb * 2 + lanes, mask, other=0.0)
        x0, x1 = xv * sign, tl.gather(xv, lanes ^ 1, 0)
        offsets = (row64 * LDA + cb) * 2 + lanes
        old = tl.load(a_ptr + offsets, mask, other=0.0)
        result = tl.fma(ur, x0, old)
        result = tl.fma(ui, x1, result)
        result = tl.where((cols == row) & ((lanes & 1) != 0), 0.0, result)
        tl.store(a_ptr + offsets, result, mask)


@libentry()
@triton.jit
def cher_block_kernel(
    a_ptr,
    x_ptr,
    alpha: tl.float32,
    N: tl.constexpr,
    LDA: tl.constexpr,
    UPLO: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    BALANCED: tl.constexpr,
    ROW_STARTS: tl.constexpr,
    PAIR_ROWS: tl.constexpr,
):
    """Stream disjoint row blocks and update the selected triangle in place."""
    tl.static_assert(N >= COLS)
    full_rows: tl.constexpr = N // ROWS * ROWS
    full_cols: tl.constexpr = N // COLS * COLS
    groups: tl.constexpr = N // ROWS
    single_column_tail: tl.constexpr = N % COLS == 1 and N % ROWS == 1
    pid = tl.program_id(0)
    programs = tl.num_programs(0)
    if BALANCED:
        linear: tl.constexpr = 2 * COLS // ROWS
        total: tl.constexpr = groups * (groups + linear)
        if len(ROW_STARTS):
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
        xv = tl.load(x_ptr + rb * 2 + rr)
        xr, xi = tl.gather(xv, ri, 0), tl.gather(xv, ri + 1, 0)
        ur, ui = alpha * xr, alpha * xi
        begin = 0 if UPLO == 0 else (rb // COLS) * COLS
        end = rb + ROWS if UPLO == 0 else full_cols if single_column_tail else N
        for logical_cb in range(begin, end, COLS):
            if N % COLS and not single_column_tail:
                cb = tl.minimum(logical_cb, N - COLS)
            else:
                cb = logical_cb
            cols = cb + (cc >> 1)
            xc = tl.load(x_ptr + cb * 2 + cc)
            x0, x1 = xc * sign, tl.gather(xc, cc ^ 1, 0)
            update = ur[:, None] * x0[None, :] + ui[:, None] * x1[None, :]
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
            _cher_last_column(a_ptr, x_ptr, ur, ui, rb, N, LDA, ROWS)
        group += group_step

    if N % ROWS:
        tail_width: tl.constexpr = 1024 if N >= 1024 else 512 if N >= 512 else COLS
        row = full_rows + pid
        while row < N:
            _cher_tail_row(a_ptr, x_ptr, alpha, row, N, LDA, UPLO, tail_width)
            row += programs


def _compile_and_launch_rank1(key, uplo, n, alpha, x, A, lda):
    rows, cols = (8, 128) if n < 512 else (16, 256)
    if n < 128:
        cols = 64
    grid = min(CORE_NUM, max(n // rows, n % rows, 1))
    balanced = n >= 2048
    row_starts = ()
    if balanced and n % cols == 0:
        row_starts = _cher2_row_partition(n // rows, cols // rows, grid)
    pair_rows = n >= 512 and not balanced and grid < n // rows <= 2 * grid
    if uplo == 0 and n in (1023, 1024):
        pair_rows = False
    constants = (
        n,
        lda,
        uplo,
        rows,
        cols,
        balanced,
        row_starts,
        pair_rows,
    )
    compiled, _ = cher_block_kernel[(grid,)](
        torch.view_as_real(A),
        torch.view_as_real(x),
        alpha,
        *constants,
        num_warps=1,
    )
    if key is not None:
        if len(_LAUNCH_CACHE) >= 512:
            _LAUNCH_CACHE.clear()
        run = compiled.run
        special_launch = (
            getattr(run, "compile_only", False)
            or getattr(run, "enable_msprof_register_tensor", False)
            or getattr(compiled.metadata, "debug_enabled", False)
        )
        _LAUNCH_CACHE[key] = (compiled, run, grid, constants, special_launch)


def _launch_cached_rank1(uplo, n, alpha, x, A, lda):
    """Launch with fresh pointers/stream while caching only compiled code."""
    a_device = A.device
    assert a_device.type == "npu"
    device = a_device.index
    if device != _current_device():
        with torch_device_fn.device(a_device):
            return _launch_cached_rank1(uplo, n, alpha, x, A, lda)

    if (
        type(A) is not torch.Tensor
        or type(x) is not torch.Tensor
        or A.is_conj()
        or x.is_conj()
        or A.is_neg()
        or x.is_neg()
    ):
        _compile_and_launch_rank1(None, uplo, n, alpha, x, A, lda)
        return

    pointers = (A.data_ptr(), x.data_ptr())
    key = (device, n, uplo, lda, pointers[0] % 16, pointers[1] % 16)
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        _compile_and_launch_rank1(key, uplo, n, alpha, x, A, lda)
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
        compiled[(grid, 1, 1)](
            torch.view_as_real(A),
            torch.view_as_real(x),
            alpha,
            *constants,
            stream=stream,
        )
        return

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
        alpha,
        *constants,
    )
    if direct_launch:
        _ascend_driver_utils.TRITON_PROFILER_REGISTERED = profiler_registered == 1


def _launch_her2_specialization(uplo, n, alpha, x, incx, A, lda):
    """Launch HER2(alpha/2, x, x) after HER has already validated inputs."""
    a_device = A.device
    assert a_device.type == "npu"
    device = a_device.index
    if device != _current_device():
        with torch_device_fn.device(a_device):
            return _launch_her2_specialization(uplo, n, alpha, x, incx, A, lda)

    ar, ai = alpha * 0.5, 0.0
    if (
        type(A) is not torch.Tensor
        or type(x) is not torch.Tensor
        or A.is_conj()
        or x.is_conj()
        or A.is_neg()
        or x.is_neg()
    ):
        _compile_and_launch_cher2(
            None, uplo, n, ar, ai, x, incx, x, incx, A, lda
        )
        return

    pointers = (A.data_ptr(), x.data_ptr(), x.data_ptr())
    key = (
        device,
        n,
        uplo,
        incx,
        incx,
        lda,
        pointers[0] % 16,
        pointers[1] % 16,
        pointers[2] % 16,
    )
    entry = _HER2_LAUNCH_CACHE.get(key)
    if entry is None:
        _compile_and_launch_cher2(
            key, uplo, n, ar, ai, x, incx, x, incx, A, lda
        )
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
        compiled[(grid, 1, 1)](
            torch.view_as_real(A),
            torch.view_as_real(x),
            torch.view_as_real(x),
            ar,
            ai,
            *constants,
            stream=stream,
        )
        return

    a_arg = _DevicePointer(pointers[0])
    a_arg.tensor = A
    x_arg = _DevicePointer(pointers[1])
    x_arg.tensor = x
    y_arg = _DevicePointer(pointers[2])
    y_arg.tensor = x
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


def cher(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    A: torch.Tensor,
    lda: int,
):
    # Validate on every call, including cache hits. Reuse metadata locally:
    # the shared multi-backend checker repeatedly crosses the Tensor/Python
    # boundary to fetch devices and shapes, which dominates small HER calls.
    # Subclasses retain the shared checker's dispatch behaviour.
    if type(A) is not torch.Tensor or type(x) is not torch.Tensor:
        _check_her_args(
            "cher", uplo, n, alpha, x, incx, A, lda, torch.complex64, torch.float32
        )
    else:
        assert uplo in (
            0, 1
        ), "uplo must be CUBLAS_FILL_MODE_LOWER or CUBLAS_FILL_MODE_UPPER"
        assert isinstance(n, int) and n >= 0, "n must be a non-negative integer"
        assert A.dtype == torch.complex64, "A must be torch.complex64 for cher"
        assert x.dtype == torch.complex64, "x must be torch.complex64 for cher"
        a_device = A.device
        assert a_device.type == "npu", "A and x must be NPU tensors"
        assert a_device == x.device, "A and x must be on the same device"
        assert A.is_contiguous() and x.is_contiguous(), "A and x must be contiguous"
        shape = A.shape
        assert len(shape) == 2, "A must be a 2-D tensor"
        assert x.ndim == 1, "x must be a 1-D tensor"
        assert isinstance(incx, int) and incx > 0, "incx must be a positive integer"
        assert isinstance(lda, int) and lda >= max(
            1, n
        ), "lda must be at least max(1, n)"
        assert shape[0] >= n, "A has too few rows"
        assert shape[1] >= lda, "A leading dimension is too small"
        if n > 0:
            assert x.numel() >= 1 + (n - 1) * incx, "x is too small for n and incx"
            # shape[0] >= n and shape[1] >= lda already imply A.numel() >= n*lda.
        scalar_type = type(alpha)
        if scalar_type is not float and scalar_type is not int:
            if isinstance(alpha, torch.Tensor):
                assert alpha.numel() == 1, "alpha tensor must contain one value"
                assert (
                    alpha.dtype == torch.float32
                ), "alpha tensor must be torch.float32"
            else:
                assert isinstance(alpha, (float, int)), "alpha must be real"
    if n == 0:
        return A
    alpha_value = (
        alpha
        if type(alpha) is float
        else float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    )
    if alpha_value == 0.0:
        return A

    # Dense vectors/matrices use rank-1 blocks at small sizes too. Paired
    # full-call measurements show that avoiding HER2's duplicate vector loads,
    # rank-2 arithmetic and third pointer argument benefits these shapes.
    # The existing large-matrix tile sizes and partitions stay the same.
    if n >= 64 and incx == 1 and lda == n:
        _launch_cached_rank1(uplo, n, alpha_value, x, A, lda)
        return A

    # HER2's kernels cover small, strided and padded cases. HER already did
    # the public validation, so enter its compiled-launch cache directly and
    # avoid repeating the entire Python frontend on every timed call.
    _launch_her2_specialization(uplo, n, alpha_value, x, incx, A, lda)
    return A
