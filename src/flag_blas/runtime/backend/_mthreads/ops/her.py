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

import torch
import triton
import triton.language as tl

from flag_blas.ops.level2.her import ScalarType, _check_her_args, _f64_to_i64
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry


@libentry()
@triton.jit
def _cher_kernel(
    a_ptr,
    x_ptr,
    alpha,
    N: tl.constexpr,
    LDA: tl.constexpr,
    INCX: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TRIANGULAR: tl.constexpr,
    ROW_SLICES_FIRST: tl.constexpr,
):
    if TRIANGULAR:
        if ROW_SLICES_FIRST:
            tile = tl.program_id(0) // (BLOCK_N // BLOCK_M)
            row_slice = tl.program_id(0) % (BLOCK_N // BLOCK_M)
        else:
            tile = tl.program_id(0)
            row_slice = tl.program_id(1)
        major = ((tl.sqrt(8.0 * tile + 1.0) - 1.0) * 0.5).to(tl.int32)
        base = major * (major + 1) // 2
        major = tl.where(base > tile, major - 1, major)
        next_base = (major + 1) * (major + 2) // 2
        major = tl.where(next_base <= tile, major + 1, major)
        minor = tile - major * (major + 1) // 2
        if UPLO == 0:
            row_tile, col_tile = major, minor
        else:
            row_tile, col_tile = minor, major
        row_start = row_tile * BLOCK_N + row_slice * BLOCK_M
        col_start = col_tile * BLOCK_N
    else:
        row_start = tl.program_id(0) * BLOCK_M
        col_start = tl.program_id(1) * BLOCK_N
        if UPLO == 0:
            if row_start + BLOCK_M <= col_start:
                return
        else:
            if col_start + BLOCK_N <= row_start:
                return

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)
    r64, c64 = rows.to(tl.int64), cols.to(tl.int64)
    lane = tl.arange(0, 2)
    xr, xi = tl.split(
        tl.load(
            x_ptr + r64[:, None] * INCX * 2 + lane[None, :],
            (rows < N)[:, None],
            other=0.0,
        )
    )
    yr, yi = tl.split(
        tl.load(
            x_ptr + c64[:, None] * INCX * 2 + lane[None, :],
            (cols < N)[:, None],
            other=0.0,
        )
    )
    update_r = alpha * (xr[:, None] * yr[None, :] + xi[:, None] * yi[None, :])
    update_i = alpha * (xi[:, None] * yr[None, :] - xr[:, None] * yi[None, :])
    mask = (rows[:, None] < N) & (cols[None, :] < N)
    if UPLO == 0:
        mask = mask & (rows[:, None] >= cols[None, :])
    else:
        mask = mask & (rows[:, None] <= cols[None, :])
    ptr = (
        a_ptr
        + (r64[:, None] * LDA + c64[None, :])[:, :, None] * 2
        + lane[None, None, :]
    )
    ar, ai = tl.split(tl.load(ptr, mask[:, :, None], other=0.0))
    out_i = tl.where(rows[:, None] == cols[None, :], 0.0, ai + update_i)
    tl.store(ptr, tl.join(ar + update_r, out_i), mask[:, :, None])


@triton.jit
def _zher_kernel(
    a_ptr,
    x_ptr,
    alpha_bits,
    N: tl.constexpr,
    LDA: tl.constexpr,
    INCX: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TRIANGULAR: tl.constexpr,
):
    if TRIANGULAR:
        tile = tl.program_id(0)
        row_slice = tl.program_id(1)
        major = ((tl.sqrt(8.0 * tile + 1.0) - 1.0) * 0.5).to(tl.int32)
        base = major * (major + 1) // 2
        major = tl.where(base > tile, major - 1, major)
        next_base = (major + 1) * (major + 2) // 2
        major = tl.where(next_base <= tile, major + 1, major)
        minor = tile - major * (major + 1) // 2
        if UPLO == 0:
            row_tile, col_tile = major, minor
        else:
            row_tile, col_tile = minor, major
        row_start = row_tile * BLOCK_N + row_slice * BLOCK_M
        col_start = col_tile * BLOCK_N
    else:
        row_start = tl.program_id(0) * BLOCK_M
        col_start = tl.program_id(1) * BLOCK_N
        if UPLO == 0:
            if row_start + BLOCK_M <= col_start:
                return
        else:
            if col_start + BLOCK_N <= row_start:
                return
    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)
    r64, c64 = rows.to(tl.int64), cols.to(tl.int64)
    xr = tl.load(x_ptr + r64 * INCX * 2, rows < N, 0)
    xi = tl.load(x_ptr + r64 * INCX * 2 + 1, rows < N, 0)
    yr = tl.load(x_ptr + c64 * INCX * 2, cols < N, 0)
    yi = tl.load(x_ptr + c64 * INCX * 2 + 1, cols < N, 0)
    alpha = alpha_bits.to(tl.float64, bitcast=True)
    update_r = alpha * (xr[:, None] * yr[None, :] + xi[:, None] * yi[None, :])
    update_i = alpha * (xi[:, None] * yr[None, :] - xr[:, None] * yi[None, :])
    mask = (rows[:, None] < N) & (cols[None, :] < N)
    if UPLO == 0:
        mask &= rows[:, None] >= cols[None, :]
    else:
        mask &= rows[:, None] <= cols[None, :]
    off = (r64[:, None] * LDA + c64[None, :]) * 2
    ar = tl.load(a_ptr + off, mask, 0)
    ai = tl.load(a_ptr + off + 1, mask, 0)
    tl.store(a_ptr + off, ar + update_r, mask)
    tl.store(
        a_ptr + off + 1,
        tl.where(rows[:, None] == cols[None, :], 0.0, ai + update_i),
        mask,
    )


def _cher_config(n, uplo):
    # S5000 measurements: (rows, columns, triangular grid, adjacent slices, warps).
    # Small matrices benefit from avoiding triangular-index inversion.
    if n <= 512:
        return 1, 128, False, False, 4
    if n <= 1536:
        return 4, 32, True, False, 4
    if n <= 2048:
        return 1, 128, True, True, 4
    if uplo == 0:
        return 4, 64, True, False, 4
    # Upper triangles around 3072 prefer adjacent row slices within a tile.
    return 4, 128, True, n <= 3072, 8


def cher(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    A: torch.Tensor,
    lda: int,
):
    _check_her_args(
        "cher", uplo, n, alpha, x, incx, A, lda, torch.complex64, torch.float32
    )
    if n == 0:
        return A
    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    bm, bn, triangular, row_slices_first, warps = _cher_config(n, uplo)
    if triangular:
        tiles = triton.cdiv(n, bn)
        slices = triton.cdiv(min(n, bn), bm)
        if row_slices_first:
            grid = (tiles * (tiles + 1) // 2 * (bn // bm),)
        else:
            grid = (tiles * (tiles + 1) // 2, slices)
    else:
        grid = (triton.cdiv(n, bm), triton.cdiv(n, bn))
    # Coordinates are logical row-major indices; UPLO is not flipped.
    # Preserve the common implementation's diagonal handling even for alpha=0.
    with torch_device_fn.device(A.device):
        _cher_kernel[grid](
            torch.view_as_real(A),
            torch.view_as_real(x),
            alpha_value,
            N=n,
            LDA=lda,
            INCX=incx,
            UPLO=uplo,
            BLOCK_M=bm,
            BLOCK_N=bn,
            TRIANGULAR=triangular,
            ROW_SLICES_FIRST=row_slices_first,
            num_warps=warps,
            num_stages=1,
        )
    return A


def zher(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    A: torch.Tensor,
    lda: int,
):
    _check_her_args(
        "zher", uplo, n, alpha, x, incx, A, lda, torch.complex128, torch.float64
    )
    if n == 0:
        return A
    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if n <= 512:
        bm, bn, triangular, warps = 1, 128, False, 4
    elif n <= 1024:
        bm, bn, triangular, warps = 2, 64, True, 8
    elif n <= 1536:
        bm, bn, triangular, warps = 1, 128, False, 8
    elif n <= 2048:
        bm, bn, triangular, warps = 1, 128, True, 8
    elif uplo == 0:
        bm, bn, triangular, warps = 2, 64, True, 8
    else:
        bm, bn, triangular, warps = 1, 128, False, 4
    tiles = triton.cdiv(n, bn)
    grid = (
        (tiles * (tiles + 1) // 2, triton.cdiv(min(n, bn), bm))
        if triangular
        else (triton.cdiv(n, bm), tiles)
    )
    with torch_device_fn.device(A.device):
        _zher_kernel[grid](
            torch.view_as_real(A),
            torch.view_as_real(x),
            _f64_to_i64(alpha_value),
            N=n,
            LDA=lda,
            INCX=incx,
            UPLO=uplo,
            BLOCK_M=bm,
            BLOCK_N=bn,
            TRIANGULAR=triangular,
            num_warps=warps,
            num_stages=1,
        )
    return A


__all__ = ["cher", "zher"]
