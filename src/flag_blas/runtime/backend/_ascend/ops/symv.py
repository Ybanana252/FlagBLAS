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

"""Ascend-specific SYMV kernels.

SSYMV uses fused row reductions for small matrices and read-once triangular
panels for large matrices. Neither path allocates a workspace.
CSYMV uses packed complex reads in a row-owned small-matrix kernel and a
triangular panel for larger matrices. Neither path allocates a workspace.
"""

from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level2.symv import (
    _check_common,
    _complex_scalars,
    _row_major_uplo,
    _strided_y,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

ScalarType = Union[float, int, complex, torch.Tensor]

_MAX_CORE_DIM = 65535
_SYMV_KEY = ["n"]
_RESTORE = ["y_ptr"]


@triton.jit
def _triangular_tile_ids(tile_id, UPLO: tl.constexpr):
    high = ((tl.sqrt(8.0 * tile_id + 1.0) - 1.0) * 0.5).to(tl.int32)
    low = tile_id - high * (high + 1) // 2
    if UPLO == 0:
        return high, low
    return low, high


def _triangular_grid(n, max_programs=_MAX_CORE_DIM):
    def grid(meta):
        tiles = triton.cdiv(n, meta["BLOCK_SIZE"])
        return (min(tiles * (tiles + 1) // 2, max_programs),)

    return grid


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ssymv"),
    key=_SYMV_KEY,
    restore_value=_RESTORE,
)
@triton.jit
def ssymv_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    n,
    LDA,
    INCX,
    INCY,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tile_id = tl.program_id(0)
    tiles = tl.cdiv(n, BLOCK_SIZE)
    tile_count = tiles * (tiles + 1) // 2
    program_count = tl.num_programs(0)

    while tile_id < tile_count:
        pid_m, pid_n = _triangular_tile_ids(tile_id, UPLO)
        rows = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        cols = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        row_mask = rows < n
        col_mask = cols < n
        mask2d = row_mask[:, None] & col_mask[None, :]

        x_rows = tl.load(x_ptr + rows * INCX, mask=row_mask, other=0.0)
        x_cols = tl.load(x_ptr + cols * INCX, mask=col_mask, other=0.0)

        if pid_m == pid_n:
            i = rows[:, None]
            j = cols[None, :]
            if UPLO == 0:
                use_direct = j <= i
            else:
                use_direct = j >= i
            off = tl.where(use_direct, i + j * LDA, j + i * LDA)
            a_vals = tl.load(a_ptr + off, mask=mask2d, other=0.0)
            acc = tl.sum(a_vals * x_cols[None, :], axis=1)
            tl.atomic_add(
                y_ptr + rows * INCY, alpha * acc, mask=row_mask, sem="relaxed"
            )
        else:
            off = rows[:, None] + cols[None, :] * LDA
            a_vals = tl.load(a_ptr + off, mask=mask2d, other=0.0)
            acc_rows = tl.sum(a_vals * x_cols[None, :], axis=1)
            acc_cols = tl.sum(a_vals * x_rows[:, None], axis=0)
            tl.atomic_add(
                y_ptr + rows * INCY,
                alpha * acc_rows,
                mask=row_mask,
                sem="relaxed",
            )
            tl.atomic_add(
                y_ptr + cols * INCY,
                alpha * acc_cols,
                mask=col_mask,
                sem="relaxed",
            )
        tile_id += program_count


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("csymv"),
    key=_SYMV_KEY,
    restore_value=_RESTORE,
)
@triton.jit
def csymv_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    n,
    LDA,
    INCX,
    INCY,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tile_id = tl.program_id(0)
    tiles = tl.cdiv(n, BLOCK_SIZE)
    tile_count = tiles * (tiles + 1) // 2
    program_count = tl.num_programs(0)

    while tile_id < tile_count:
        pid_m, pid_n = _triangular_tile_ids(tile_id, UPLO)
        rows = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        cols = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        row_mask = rows < n
        col_mask = cols < n
        mask2d = row_mask[:, None] & col_mask[None, :]
        y_rows_off = rows * INCY * 2
        y_cols_off = cols * INCY * 2

        x_rows_off = rows * INCX * 2
        x_cols_off = cols * INCX * 2
        xrr = tl.load(x_ptr + x_rows_off, mask=row_mask, other=0.0)
        xri = tl.load(x_ptr + x_rows_off + 1, mask=row_mask, other=0.0)
        xcr = tl.load(x_ptr + x_cols_off, mask=col_mask, other=0.0)
        xci = tl.load(x_ptr + x_cols_off + 1, mask=col_mask, other=0.0)

        if pid_m == pid_n:
            i = rows[:, None]
            j = cols[None, :]
            if UPLO == 0:
                use_direct = j <= i
            else:
                use_direct = j >= i
            elem_off = tl.where(use_direct, i + j * LDA, j + i * LDA)
            a_off = elem_off * 2
            ar = tl.load(a_ptr + a_off, mask=mask2d, other=0.0)
            ai = tl.load(a_ptr + a_off + 1, mask=mask2d, other=0.0)
            acc_r = tl.sum(ar * xcr[None, :] - ai * xci[None, :], axis=1)
            acc_i = tl.sum(ar * xci[None, :] + ai * xcr[None, :], axis=1)
            res_r = alpha_r * acc_r - alpha_i * acc_i
            res_i = alpha_r * acc_i + alpha_i * acc_r
            tl.atomic_add(y_ptr + y_rows_off, res_r, mask=row_mask, sem="relaxed")
            tl.atomic_add(y_ptr + y_rows_off + 1, res_i, mask=row_mask, sem="relaxed")
        else:
            elem_off = rows[:, None] + cols[None, :] * LDA
            a_off = elem_off * 2
            ar = tl.load(a_ptr + a_off, mask=mask2d, other=0.0)
            ai = tl.load(a_ptr + a_off + 1, mask=mask2d, other=0.0)

            acc_rows_r = tl.sum(ar * xcr[None, :] - ai * xci[None, :], axis=1)
            acc_rows_i = tl.sum(ar * xci[None, :] + ai * xcr[None, :], axis=1)
            acc_cols_r = tl.sum(ar * xrr[:, None] - ai * xri[:, None], axis=0)
            acc_cols_i = tl.sum(ar * xri[:, None] + ai * xrr[:, None], axis=0)

            row_res_r = alpha_r * acc_rows_r - alpha_i * acc_rows_i
            row_res_i = alpha_r * acc_rows_i + alpha_i * acc_rows_r
            col_res_r = alpha_r * acc_cols_r - alpha_i * acc_cols_i
            col_res_i = alpha_r * acc_cols_i + alpha_i * acc_cols_r

            tl.atomic_add(y_ptr + y_rows_off, row_res_r, mask=row_mask, sem="relaxed")
            tl.atomic_add(
                y_ptr + y_rows_off + 1, row_res_i, mask=row_mask, sem="relaxed"
            )
            tl.atomic_add(y_ptr + y_cols_off, col_res_r, mask=col_mask, sem="relaxed")
            tl.atomic_add(
                y_ptr + y_cols_off + 1, col_res_i, mask=col_mask, sem="relaxed"
            )
        tile_id += program_count


@libentry()
@triton.jit
def _csymv_packed_rows_kernel(
    A, X, Y,
    AR: tl.float32, AI: tl.float32,
    BR: tl.float32, BI: tl.float32,
    N: tl.constexpr, LDA: tl.constexpr, IX: tl.constexpr,
    IY: tl.constexpr, UPLO: tl.constexpr, ZERO: tl.constexpr,
    BM: tl.constexpr, BK: tl.constexpr,
):
    first = tl.program_id(0) * BM
    rows = first + tl.arange(0, BM)
    ks = tl.arange(0, BK)
    lane = tl.arange(0, 2)
    rr = tl.zeros((BM, BK), tl.float32)
    ri = tl.zeros((BM, BK), tl.float32)

    if UPLO == 0:
        direct_begin, direct_end = 0, tl.minimum(first + BM, N)
        mirror_begin, mirror_end = first, N
    else:
        direct_begin, direct_end = first, N
        mirror_begin, mirror_end = 0, tl.minimum(first + BM, N)

    for start in range(direct_begin, direct_end, BK):
        cols = start + ks
        col_mask = cols < N
        xp = tl.load(X + cols[:, None] * IX * 2 + lane[None, :],
                     col_mask[:, None], 0)
        xr, xi = tl.split(xp)
        if UPLO == 0:
            mask = ((rows[:, None] < N) & col_mask[None, :]
                    & (rows[:, None] >= cols[None, :]))
        else:
            mask = ((rows[:, None] < N) & col_mask[None, :]
                    & (rows[:, None] <= cols[None, :]))
        off = (rows[:, None] * LDA + cols[None, :]) * 2
        ap = tl.load(A + off[:, :, None] + lane[None, None, :],
                     mask[:, :, None], 0)
        ar, ai = tl.split(ap)
        rr += ar * xr[None, :] - ai * xi[None, :]
        ri += ar * xi[None, :] + ai * xr[None, :]

    for start in range(mirror_begin, mirror_end, BK):
        cols = start + ks
        col_mask = cols < N
        xp = tl.load(X + cols[:, None] * IX * 2 + lane[None, :],
                     col_mask[:, None], 0)
        xr, xi = tl.split(xp)
        if UPLO == 0:
            mask = ((rows[None, :] < N) & col_mask[:, None]
                    & (cols[:, None] > rows[None, :]))
        else:
            mask = ((rows[None, :] < N) & col_mask[:, None]
                    & (cols[:, None] < rows[None, :]))
        off = (cols[:, None] * LDA + rows[None, :]) * 2
        ap = tl.load(A + off[:, :, None] + lane[None, None, :],
                     mask[:, :, None], 0)
        avr, avi = tl.split(ap)
        ar, ai = tl.trans(avr), tl.trans(avi)
        rr += ar * xr[None, :] - ai * xi[None, :]
        ri += ar * xi[None, :] + ai * xr[None, :]

    sr = tl.sum(rr, 1)
    si = tl.sum(ri, 1)
    yr = AR * sr - AI * si
    yi = AR * si + AI * sr
    if not ZERO:
        oldr = tl.load(Y + rows * IY * 2, rows < N, 0)
        oldi = tl.load(Y + rows * IY * 2 + 1, rows < N, 0)
        yr += BR * oldr - BI * oldi
        yi += BR * oldi + BI * oldr
    tl.store(Y + rows * IY * 2, yr, rows < N)
    tl.store(Y + rows * IY * 2 + 1, yi, rows < N)


@libentry()
@triton.jit
def _csymv_scale_kernel(
    Y, BR: tl.float32, BI: tl.float32,
    N: tl.constexpr, IY: tl.constexpr, ZERO: tl.constexpr,
    BM: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    yr = tl.full((BM,), 0, tl.float32)
    yi = tl.full((BM,), 0, tl.float32)
    if not ZERO:
        oldr = tl.load(Y + rows * IY * 2, rows < N, 0)
        oldi = tl.load(Y + rows * IY * 2 + 1, rows < N, 0)
        yr = BR * oldr - BI * oldi
        yi = BR * oldi + BI * oldr
    tl.store(Y + rows * IY * 2, yr, rows < N)
    tl.store(Y + rows * IY * 2 + 1, yi, rows < N)


@libentry()
@triton.jit
def _csymv_packed_panel_kernel(
    A, X, Y, AR: tl.float32, AI: tl.float32,
    N: tl.constexpr, LDA: tl.constexpr, IX: tl.constexpr,
    IY: tl.constexpr, UPLO: tl.constexpr,
    BM: tl.constexpr, BK: tl.constexpr, REVERSE: tl.constexpr,
    COL_FIRST: tl.constexpr,
):
    if REVERSE:
        first = (tl.cdiv(N, BM) - 1 - tl.program_id(0)) * BM
    else:
        first = tl.program_id(0) * BM
    rows = first + tl.arange(0, BM)
    k = tl.arange(0, BK)
    lane = tl.arange(0, 2)
    xp = tl.load(X + rows[:, None] * IX * 2 + lane[None, :],
                 rows[:, None] < N, 0)
    xrr, xri = tl.split(xp)
    rr = tl.full((BM,), 0, tl.float32)
    ri = tl.full((BM,), 0, tl.float32)
    diag_begin = first // BK * BK
    diag_end = tl.cdiv(first + BM, BK) * BK
    if UPLO == 0:
        begin, end = 0, diag_begin
    else:
        begin, end = diag_end, N
    for start in range(begin, end, BK):
        cols = start + k
        xc = tl.load(X + cols[:, None] * IX * 2 + lane[None, :],
                     cols[:, None] < N, 0)
        xcr, xci = tl.split(xc)
        off = (rows[:, None] * LDA + cols[None, :]) * 2
        mask = (rows[:, None] < N) & (cols[None, :] < N)
        ap = tl.load(A + off[:, :, None] + lane[None, None, :],
                     mask[:, :, None], 0)
        ar, ai = tl.split(ap)
        if COL_FIRST:
            cr = tl.sum(ar * xrr[:, None] - ai * xri[:, None], 0)
            ci = tl.sum(ar * xri[:, None] + ai * xrr[:, None], 0)
            yr = AR * cr - AI * ci
            yi = AR * ci + AI * cr
            tl.atomic_add(Y + cols[:, None] * IY * 2 + lane[None, :],
                          tl.join(yr, yi), cols[:, None] < N, sem="relaxed")
        rr += tl.sum(ar * xcr[None, :] - ai * xci[None, :], 1)
        ri += tl.sum(ar * xci[None, :] + ai * xcr[None, :], 1)
        if not COL_FIRST:
            cr = tl.sum(ar * xrr[:, None] - ai * xri[:, None], 0)
            ci = tl.sum(ar * xri[:, None] + ai * xrr[:, None], 0)
            yr = AR * cr - AI * ci
            yi = AR * ci + AI * cr
            tl.atomic_add(Y + cols[:, None] * IY * 2 + lane[None, :],
                          tl.join(yr, yi), cols[:, None] < N, sem="relaxed")
    for start in range(diag_begin, diag_end, BK):
        cols = start + k
        xc = tl.load(X + cols[:, None] * IX * 2 + lane[None, :],
                     cols[:, None] < N, 0)
        xcr, xci = tl.split(xc)
        off = (rows[:, None] * LDA + cols[None, :]) * 2
        mask = (rows[:, None] < N) & (cols[None, :] < N)
        ap = tl.load(A + off[:, :, None] + lane[None, None, :],
                     mask[:, :, None], 0)
        ar, ai = tl.split(ap)
        if UPLO == 0:
            valid = rows[:, None] >= cols[None, :]
        else:
            valid = rows[:, None] <= cols[None, :]
        ar, ai = tl.where(valid, ar, 0), tl.where(valid, ai, 0)
        rr += tl.sum(ar * xcr[None, :] - ai * xci[None, :], 1)
        ri += tl.sum(ar * xci[None, :] + ai * xcr[None, :], 1)
        ar = tl.where(rows[:, None] != cols[None, :], ar, 0)
        ai = tl.where(rows[:, None] != cols[None, :], ai, 0)
        cr = tl.sum(ar * xrr[:, None] - ai * xri[:, None], 0)
        ci = tl.sum(ar * xri[:, None] + ai * xrr[:, None], 0)
        yr = AR * cr - AI * ci
        yi = AR * ci + AI * cr
        tl.atomic_add(Y + cols[:, None] * IY * 2 + lane[None, :],
                      tl.join(yr, yi), cols[:, None] < N, sem="relaxed")
    yr = AR * rr - AI * ri
    yi = AR * ri + AI * rr
    tl.atomic_add(Y + rows[:, None] * IY * 2 + lane[None, :],
                  tl.join(yr, yi), rows[:, None] < N, sem="relaxed")


def ssymv(
    uplo: int,
    n: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    if n >= 4095 or incy != 1:
        # Non-unit output strides require bounded panel tiles: the row-owned
        # autotune pool can exceed Ascend's local-memory budget here.
        assert A.dtype == torch.float32 == x.dtype == y.dtype
        _check_common(A, x, y, uplo, n, lda, incx, incy)
        alpha = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
        beta = float(beta.item() if isinstance(beta, torch.Tensor) else beta)
        if alpha == 0 and beta == 1:
            return
        if beta != 1:
            _ssymv_rows_launch(
                uplo, n, 1.0, A, lda, x, incx, beta, y, incy, kind="scale"
            )
        if alpha != 0:
            _ssymv_rows_launch(
                uplo,
                n,
                alpha,
                A,
                lda,
                x,
                incx,
                beta,
                y,
                incy,
                kind="panel_strided" if incy != 1 else "panel",
            )
        return
    _ssymv_rows_launch(uplo, n, alpha, A, lda, x, incx, beta, y, incy)


def csymv(
    uplo: int,
    n: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    assert A.dtype == torch.complex64 == x.dtype == y.dtype
    _check_common(A, x, y, uplo, n, lda, incx, incy)
    if n == 0:
        return

    ar, ai, br, bi = _complex_scalars(alpha, beta)
    if lda == n and incx == incy == 1 and (ar != 0 or ai != 0):
        if 128 <= n <= 1024 or n in (1536, 2048):
            _csymv_packed_rows_launch(
                A, x, y, ar, ai, br, bi, n, lda, incx, incy, uplo
            )
            return
        if 1536 <= n <= 16384:
            A_real = torch.view_as_real(A)
            x_real = torch.view_as_real(x)
            y_real = torch.view_as_real(y)
            col_first = uplo == 1 and n >= 6144
            bm, bk = ((32, 128) if col_first else
                      (16, 128 if n < 4095 else 256))
            with torch_device_fn.device(A.device):
                if br != 1.0 or bi != 0.0:
                    _csymv_scale_kernel[(triton.cdiv(n, 256),)](
                        y_real, br, bi, n, incy, br == 0 and bi == 0, 256,
                        num_warps=1, num_stages=1,
                    )
                _csymv_packed_panel_kernel[(triton.cdiv(n, bm),)](
                    A_real, x_real, y_real, ar, ai, n, lda, incx, incy,
                    uplo, bm, bk, uplo == 0, col_first,
                    num_warps=1, num_stages=1,
                )
            return
    y_view = _strided_y(y, n, incy)
    if ar == 0.0 and ai == 0.0:
        if br == 0.0 and bi == 0.0:
            y_view.zero_()
        elif br != 1.0 or bi != 0.0:
            y_view.mul_(complex(br, bi))
        return

    A_real = torch.view_as_real(A)
    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)
    with torch_device_fn.device(A.device):
        if br == 0.0 and bi == 0.0:
            y_view.zero_()
        elif br != 1.0 or bi != 0.0:
            y_view.mul_(complex(br, bi))
        # Bound active programs to reduce launch and atomic contention.
        core_limit = (
            (4096 if uplo == 0 else 512)
            if n >= 16384
            else max(256, min(2048, triton.next_power_of_2(n) // 4))
        )
        _csymv_direct_launch(
            A_real,
            x_real,
            y_real,
            ar,
            ai,
            n,
            lda,
            incx,
            incy,
            _row_major_uplo(uplo),
            core_limit,
        )


_CSYMV_PACKED_ROWS_CACHE = {}


def _csymv_packed_rows_launch(A, X, Y, ar, ai, br, bi, n, lda, incx, incy, uplo):
    from . import spr

    device = A.device.index
    if device != spr._current_device():
        with torch_device_fn.device(A.device):
            return _csymv_packed_rows_launch(
                A, X, Y, ar, ai, br, bi, n, lda, incx, incy, uplo
            )
    tensors = (A, X, Y)
    pointers = tuple(t.data_ptr() for t in tensors)
    if uplo == 1 and n in (128, 192, 255):
        bm, bk, warps = 16, 64, 1
    elif 512 <= n <= 1024:
        bm, bk, warps = 32, 64, 2
    elif n == 2048:
        bm, bk, warps = 32, 64, 8 if uplo == 0 else 4
    else:
        bm, bk, warps = 16, 128, 4
    values = (ar, ai, br, bi, n, lda, incx, incy, uplo,
              br == 0 and bi == 0, bm, bk)
    key = (device, *values, warps, *(p % 16 for p in pointers))
    cached = _CSYMV_PACKED_ROWS_CACHE.get(key)
    grid = triton.cdiv(n, bm)
    if cached is None:
        real_tensors = tuple(torch.view_as_real(t) for t in tensors)
        compiled, _ = _csymv_packed_rows_kernel[(grid,)](
            *real_tensors, *values, num_warps=warps, num_stages=1
        )
        run = compiled.run
        special = (
            getattr(run, "compile_only", False)
            or getattr(run, "enable_msprof_register_tensor", False)
            or getattr(compiled.metadata, "debug_enabled", False)
        )
        if len(_CSYMV_PACKED_ROWS_CACHE) >= 512:
            _CSYMV_PACKED_ROWS_CACHE.clear()
        _CSYMV_PACKED_ROWS_CACHE[key] = (compiled, run, special)
        return

    compiled, run, special = cached
    stream = spr._current_raw_stream(device)
    knobs = triton.knobs.runtime
    enter, exit = knobs.launch_enter_hook, knobs.launch_exit_hook
    has_enter = enter is not None and (
        type(enter) is not spr._HOOK_CHAIN_TYPE or bool(enter.calls)
    )
    has_exit = exit is not None and (
        type(exit) is not spr._HOOK_CHAIN_TYPE or bool(exit.calls)
    )
    if special or has_enter or has_exit:
        real_tensors = tuple(torch.view_as_real(t) for t in tensors)
        compiled[(grid, 1, 1)](*real_tensors, *values, stream=stream)
        return
    args = []
    for tensor, pointer in zip(tensors, pointers):
        arg = spr._DevicePointer(pointer)
        arg.tensor = tensor
        args.append(arg)
    direct = (
        type(run) is spr._NPU_LAUNCHER
        and not run.compile_only
        and not run.enable_msprof_register_tensor
        and not getattr(run.metadata, "debug_enabled", False)
    )
    launch = run.launch if direct else run
    registered = launch(
        grid, 1, 1, stream, compiled.function, compiled.packed_metadata,
        None, None, None, *args, *values,
    )
    if direct:
        spr._ascend_driver_utils.TRITON_PROFILER_REGISTERED = registered == 1


_CSYMV_DIRECT_ENTRIES = {}
_CSYMV_DIRECT_CACHE = {}
_CSYMV_SPR = None


def _csymv_direct_launch(A, X, Y, ar, ai, n, lda, incx, incy, uplo, core_limit):
    global _CSYMV_SPR
    if _CSYMV_SPR is None:
        from . import spr

        _CSYMV_SPR = spr
    spr = _CSYMV_SPR
    device = A.device.index
    if device != spr._current_device():
        with torch_device_fn.device(A.device):
            return _csymv_direct_launch(
                A, X, Y, ar, ai, n, lda, incx, incy, uplo, core_limit
            )

    tensors = (A, X, Y)
    pointers = tuple(t.data_ptr() for t in tensors)
    key = (
        device,
        ar,
        ai,
        n,
        lda,
        incx,
        incy,
        uplo,
        core_limit,
        *(p % 16 for p in pointers),
    )
    cached = _CSYMV_DIRECT_CACHE.get(key)
    if cached is None:
        entry = _CSYMV_DIRECT_ENTRIES.get(device)
        if entry is None:
            entry = libentry()(
                libtuner(
                    configs=runtime.get_tuned_config("csymv"),
                    key=_SYMV_KEY,
                    restore_value=_RESTORE,
                )(csymv_kernel.jit_function)
            )
            _CSYMV_DIRECT_ENTRIES[device] = entry
        compiled, meta = entry[_triangular_grid(n, core_limit)](
            A, X, Y, ar, ai, n, lda, incx, incy, UPLO=uplo
        )
        block_size = meta["BLOCK_SIZE"]
        run = compiled.run
        special = (
            getattr(run, "compile_only", False)
            or getattr(run, "enable_msprof_register_tensor", False)
            or getattr(compiled.metadata, "debug_enabled", False)
        )
        if len(_CSYMV_DIRECT_CACHE) >= 512:
            _CSYMV_DIRECT_CACHE.clear()
        _CSYMV_DIRECT_CACHE[key] = (compiled, run, special, block_size)
        return

    compiled, run, special, block_size = cached
    tiles = triton.cdiv(n, block_size)
    grid = min(tiles * (tiles + 1) // 2, core_limit)
    values = (ar, ai, n, lda, incx, incy, uplo, block_size)
    stream = spr._current_raw_stream(device)
    knobs = triton.knobs.runtime
    enter, exit = knobs.launch_enter_hook, knobs.launch_exit_hook
    has_enter = enter is not None and (
        type(enter) is not spr._HOOK_CHAIN_TYPE or bool(enter.calls)
    )
    has_exit = exit is not None and (
        type(exit) is not spr._HOOK_CHAIN_TYPE or bool(exit.calls)
    )
    if special or has_enter or has_exit:
        compiled[(grid, 1, 1)](*tensors, *values, stream=stream)
        return
    args = []
    for tensor, pointer in zip(tensors, pointers):
        arg = spr._DevicePointer(pointer)
        arg.tensor = tensor
        args.append(arg)
    direct = (
        type(run) is spr._NPU_LAUNCHER
        and not run.compile_only
        and not run.enable_msprof_register_tensor
        and not getattr(run.metadata, "debug_enabled", False)
    )
    launch = run.launch if direct else run
    registered = launch(
        grid,
        1,
        1,
        stream,
        compiled.function,
        compiled.packed_metadata,
        None,
        None,
        None,
        *args,
        *values,
    )
    if direct:
        spr._ascend_driver_utils.TRITON_PROFILER_REGISTERED = registered == 1


@libentry()
@triton.jit
def _ssymv_rows_kernel(
    A,
    X,
    Y,
    alpha: tl.float32,
    beta: tl.float32,
    N: tl.constexpr,
    LDA: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    UPLO: tl.constexpr,
    ZERO: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    WIDE: tl.constexpr = (
        N * LDA >= 1073741824 or N * IX >= 1073741824 or N * IY >= 1073741824
    )
    if WIDE:
        pid = pid.to(tl.int64)
    first = pid * BM
    rows = first + tl.arange(0, BM)
    acc = tl.full((BM,), 0, tl.float32)
    if alpha != 0:
        if UPLO == 0:
            direct_begin, direct_end = 0, tl.minimum(first + BM, N)
            mirror_begin, mirror_end = first, N
        else:
            direct_begin, direct_end = first, N
            mirror_begin, mirror_end = 0, tl.minimum(first + BM, N)
        for start in range(direct_begin, direct_end, BK):
            cols = start + tl.arange(0, BK)
            if WIDE:
                cols = cols.to(tl.int64)
            xv = tl.load(X + cols * IX, cols < N, 0)
            if UPLO == 0:
                triangle = rows[:, None] >= cols[None, :]
            else:
                triangle = rows[:, None] <= cols[None, :]
            a = tl.load(
                A + rows[:, None] * LDA + cols[None, :],
                (rows[:, None] < N) & (cols[None, :] < N),
                0,
            )
            # Keep rectangular memory reads; discard the unused triangle
            # before multiplying so its NaNs cannot enter the reduction.
            a = tl.where(triangle, a, 0)
            acc += tl.sum(a * xv[None, :], 1)
        for start in range(mirror_begin, mirror_end, BK):
            cols = start + tl.arange(0, BK)
            if WIDE:
                cols = cols.to(tl.int64)
            xv = tl.load(X + cols * IX, cols < N, 0)
            if UPLO == 0:
                triangle = cols[:, None] > rows[None, :]
            else:
                triangle = cols[:, None] < rows[None, :]
            a = tl.load(
                A + cols[:, None] * LDA + rows[None, :],
                (cols[:, None] < N) & (rows[None, :] < N),
                0,
            )
            a = tl.where(triangle, a, 0)
            acc += tl.sum(a * xv[:, None], 0)
    result = alpha * acc
    if not ZERO:
        result += beta * tl.load(Y + rows * IY, rows < N, 0)
    tl.store(Y + rows * IY, result, rows < N)


@libentry()
@triton.jit
def _ssymv_panel_kernel(
    A,
    X,
    Y,
    alpha: tl.float32,
    beta: tl.float32,
    N: tl.constexpr,
    LDA: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    UPLO: tl.constexpr,
    ZERO: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    WIDE: tl.constexpr = (
        N * LDA >= 1073741824 or N * IX >= 1073741824 or N * IY >= 1073741824
    )
    if WIDE:
        pid = pid.to(tl.int64)
    first = pid * BM
    rows = first + tl.arange(0, BM)
    k = tl.arange(0, BK)
    if WIDE:
        k = k.to(tl.int64)
    xr = tl.load(X + rows * IX, rows < N, 0)
    acc = tl.full((BM,), 0, tl.float32)
    diagonal_begin = first // BK * BK
    diagonal_end = tl.cdiv(first + BM, BK) * BK
    if UPLO == 0:
        begin, end = 0, diagonal_begin
    else:
        begin, end = diagonal_end, N
    # Each stored off-diagonal tile contributes to both symmetric halves.
    for start in range(begin, end, BK):
        cols = start + k
        xc = tl.load(X + cols * IX, cols < N, 0)
        a = tl.load(
            A + rows[:, None] * LDA + cols[None, :],
            (rows[:, None] < N) & (cols[None, :] < N),
            0,
        )
        acc += tl.sum(a * xc[None, :], 1)
        column_acc = tl.sum(a * xr[:, None], 0)
        tl.atomic_add(Y + cols * IY, alpha * column_acc, cols < N, sem="relaxed")
    # A small diagonal tile bounds local memory and masks the unused triangle.
    kd = tl.arange(0, 32)
    if WIDE:
        kd = kd.to(tl.int64)
    for start in range(diagonal_begin, diagonal_end, 32):
        cols = start + kd
        xc = tl.load(X + cols * IX, cols < N, 0)
        a = tl.load(
            A + rows[:, None] * LDA + cols[None, :],
            (rows[:, None] < N) & (cols[None, :] < N),
            0,
        )
        if UPLO == 0:
            valid = rows[:, None] >= cols[None, :]
        else:
            valid = rows[:, None] <= cols[None, :]
        a = tl.where(valid, a, 0)
        acc += tl.sum(a * xc[None, :], 1)
        mirrored = tl.where(rows[:, None] != cols[None, :], a, 0)
        column_acc = tl.sum(mirrored * xr[:, None], 0)
        tl.atomic_add(Y + cols * IY, alpha * column_acc, cols < N, sem="relaxed")
    tl.atomic_add(Y + rows * IY, alpha * acc, rows < N, sem="relaxed")


@libentry()
@triton.jit
def _ssymv_scale_kernel(
    A,
    X,
    Y,
    alpha: tl.float32,
    beta: tl.float32,
    N: tl.constexpr,
    LDA: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    UPLO: tl.constexpr,
    ZERO: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    value = tl.full((BM,), 0, tl.float32)
    if not ZERO:
        value = beta * tl.load(Y + offsets * IY, offsets < N, 0)
    tl.store(Y + offsets * IY, value, offsets < N)


_SSYMV_ROWS_ENTRIES = {}
_SSYMV_ROWS_CACHE = {}
_SSYMV_SPR = None


def _ssymv_rows_entry(device, kind="rows"):
    entry = _SSYMV_ROWS_ENTRIES.get((device, kind))
    if entry is None:
        kernel = _ssymv_panel_kernel if kind == "panel" else _ssymv_rows_kernel
        entry = libentry()(
            libtuner(
                configs=runtime.get_tuned_config(f"ssymv_{kind}_ascend"),
                key=["N", "LDA", "IX", "IY", "UPLO", "ZERO"],
                restore_value=["Y"],
            )(kernel.jit_function)
        )
        _SSYMV_ROWS_ENTRIES[device, kind] = entry
    return entry


def _ssymv_rows_launch(uplo, n, alpha, A, lda, x, incx, beta, y, incy, kind="rows"):
    global _SSYMV_SPR
    assert A.dtype == torch.float32 == x.dtype == y.dtype
    _check_common(A, x, y, uplo, n, lda, incx, incy)
    if n == 0:
        return
    alpha = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta = float(beta.item() if isinstance(beta, torch.Tensor) else beta)
    if alpha == 0 and beta == 1:
        return
    assert A.device.type == "npu"
    if _SSYMV_SPR is None:
        from . import spr

        _SSYMV_SPR = spr
    spr = _SSYMV_SPR
    device = A.device.index
    if device != spr._current_device():
        with torch_device_fn.device(A.device):
            return _ssymv_rows_launch(
                uplo, n, alpha, A, lda, x, incx, beta, y, incy, kind=kind
            )
    pointers = (A.data_ptr(), x.data_ptr(), y.data_ptr())
    values = (alpha, beta, n, lda, incx, incy, uplo, beta == 0)
    key = (device, kind, *values, *(p % 16 for p in pointers))
    cached = _SSYMV_ROWS_CACHE.get(key)
    if cached is None:
        if kind == "scale":
            bm, bk = 256, 1
            compiled, _ = _ssymv_scale_kernel[(triton.cdiv(n, bm),)](
                A, x, y, *values, bm, bk, num_warps=1, num_stages=1
            )
        elif kind == "panel_strided":
            bm, bk = 8, 32
            compiled, _ = _ssymv_panel_kernel[(triton.cdiv(n, bm),)](
                A, x, y, *values, bm, bk, num_warps=1, num_stages=1
            )
        elif alpha == 0:
            # Scaling needs no configuration search and must not tune the
            # shared matrix-vector pool using an empty reduction.
            bm, bk = 32, 128
            compiled, _ = _ssymv_rows_kernel[(triton.cdiv(n, bm),)](
                A, x, y, *values, bm, bk, num_warps=1, num_stages=1
            )
        elif kind == "rows" and uplo == 1 and n in (192, 255):
            bm, bk = 32, 128
            compiled, _ = _ssymv_rows_kernel[(triton.cdiv(n, bm),)](
                A, x, y, *values, bm, bk, num_warps=1, num_stages=1
            )
        elif (
            kind == "panel"
            and uplo == 1
            and n == 4096
            and lda == n
            and incx == incy == 1
        ):
            bm, bk = 32, 256
            compiled, _ = _ssymv_panel_kernel[(triton.cdiv(n, bm),)](
                A, x, y, *values, bm, bk, num_warps=1, num_stages=1
            )
        else:
            compiled, meta = _ssymv_rows_entry(device, kind)[
                lambda config: (triton.cdiv(n, config["BM"]),)
            ](A, x, y, *values)
            bm, bk = meta["BM"], meta["BK"]
        run = compiled.run
        special = (
            getattr(run, "compile_only", False)
            or getattr(run, "enable_msprof_register_tensor", False)
            or getattr(compiled.metadata, "debug_enabled", False)
        )
        if len(_SSYMV_ROWS_CACHE) >= 512:
            _SSYMV_ROWS_CACHE.clear()
        _SSYMV_ROWS_CACHE[key] = (compiled, run, special, bm, bk)
        return
    compiled, run, special, bm, bk = cached
    values = (*values, bm, bk)
    grid = triton.cdiv(n, bm)
    stream = spr._current_raw_stream(device)
    knobs = triton.knobs.runtime
    enter, exit = knobs.launch_enter_hook, knobs.launch_exit_hook
    has_enter = enter is not None and (
        type(enter) is not spr._HOOK_CHAIN_TYPE or bool(enter.calls)
    )
    has_exit = exit is not None and (
        type(exit) is not spr._HOOK_CHAIN_TYPE or bool(exit.calls)
    )
    if special or has_enter or has_exit:
        compiled[(grid, 1, 1)](A, x, y, *values, stream=stream)
        return
    args = []
    for tensor, pointer in zip((A, x, y), pointers):
        arg = spr._DevicePointer(pointer)
        arg.tensor = tensor
        args.append(arg)
    direct = (
        type(run) is spr._NPU_LAUNCHER
        and not run.compile_only
        and not run.enable_msprof_register_tensor
        and not getattr(run.metadata, "debug_enabled", False)
    )
    launch = run.launch if direct else run
    registered = launch(
        grid,
        1,
        1,
        stream,
        compiled.function,
        compiled.packed_metadata,
        None,
        None,
        None,
        *args,
        *values,
    )
    if direct:
        spr._ascend_driver_utils.TRITON_PROFILER_REGISTERED = registered == 1
