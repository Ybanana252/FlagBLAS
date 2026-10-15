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

from flag_blas.ops.level2.syr2 import ScalarType, _check_ssyr2_args
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry


@triton.jit
def _ssyr2_rows_jit(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    N: tl.constexpr,
    LDA: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
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
    xr = tl.load(x_ptr + r64 * INCX, rows < N, other=0.0)
    yr = tl.load(y_ptr + r64 * INCY, rows < N, other=0.0)
    xc = tl.load(x_ptr + c64 * INCX, cols < N, other=0.0)
    yc = tl.load(y_ptr + c64 * INCY, cols < N, other=0.0)
    mask = (rows[:, None] < N) & (cols[None, :] < N)
    if UPLO == 0:
        mask = mask & (rows[:, None] >= cols[None, :])
    else:
        mask = mask & (rows[:, None] <= cols[None, :])
    offsets = r64[:, None] * LDA + c64[None, :]
    a = tl.load(a_ptr + offsets, mask, other=0.0)
    update = alpha * (xr[:, None] * yc[None, :] + yr[:, None] * xc[None, :])
    tl.store(a_ptr + offsets, a + update, mask)


@triton.jit
def _ssyr2_tiles_jit(
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
    TRIANGULAR: tl.constexpr,
):
    if TRIANGULAR:
        tile = tl.program_id(0)
        major = ((tl.sqrt(8.0 * tile + 1.0) - 1.0) * 0.5).to(tl.int32)
        # Correct rounding in the inverse triangular index at tile boundaries.
        base = major * (major + 1) // 2
        major = tl.where(base > tile, major - 1, major)
        next_base = (major + 1) * (major + 2) // 2
        major = tl.where(next_base <= tile, major + 1, major)
        minor = tile - major * (major + 1) // 2
        if UPLO == 0:
            pid_m, pid_n = major, minor
        else:
            pid_m, pid_n = minor, major
    else:
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        if UPLO == 0:
            if pid_m < pid_n:
                return
        else:
            if pid_m > pid_n:
                return
    rows = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    cols = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_mask = rows < n
    col_mask = cols < n
    mask = row_mask[:, None] & col_mask[None, :]
    if UPLO == 0:
        mask = mask & (rows[:, None] >= cols[None, :])
    else:
        mask = mask & (rows[:, None] <= cols[None, :])
    xr = tl.load(x_ptr + rows * INCX, row_mask, other=0.0)
    yr = tl.load(y_ptr + rows * INCY, row_mask, other=0.0)
    xc = tl.load(x_ptr + cols * INCX, col_mask, other=0.0)
    yc = tl.load(y_ptr + cols * INCY, col_mask, other=0.0)
    offsets = rows[:, None].to(tl.int64) + cols[None, :].to(tl.int64) * LDA
    a = tl.load(a_ptr + offsets, mask, other=0.0)
    update = alpha * (xr[:, None] * yc[None, :] + yr[:, None] * xc[None, :])
    tl.store(a_ptr + offsets, a + update, mask)


# Retain the original square candidates for shapes whose row pitch makes
# triangular traversal slower. Only the measured complementary candidate is added.
_SSYR2_CONFIGS = [
    triton.Config({"BLOCK_SIZE": 8, "TRIANGULAR": False}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_SIZE": 16, "TRIANGULAR": False}, num_warps=1, num_stages=2),
    triton.Config({"BLOCK_SIZE": 16, "TRIANGULAR": False}, num_warps=2, num_stages=2),
    triton.Config({"BLOCK_SIZE": 32, "TRIANGULAR": False}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_SIZE": 16, "TRIANGULAR": True}, num_warps=2, num_stages=1),
]

ssyr2_rows_kernel = libentry()(_ssyr2_rows_jit)
ssyr2_tiles_kernel = libentry()(
    triton.autotune(
        configs=_SSYR2_CONFIGS,
        key=["n", "LDA", "INCX", "INCY", "UPLO"],
        restore_value=["a_ptr"],
    )(_ssyr2_tiles_jit)
)


def _ssyr2_grid(n):
    def grid(meta):
        tiles = triton.cdiv(n, meta["BLOCK_SIZE"])
        if meta["TRIANGULAR"]:
            return (tiles * (tiles + 1) // 2,)
        return (tiles, tiles)

    return grid


def ssyr2(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    A: torch.Tensor,
    lda: int,
) -> None:
    _check_ssyr2_args(uplo, n, x, incx, y, incy, A, lda)
    if n == 0:
        return
    alpha = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha == 0.0:
        return

    with torch_device_fn.device(A.device):
        # Row slices remove the small square tile overhead below the measured
        # crossover. They address logical row-major A, without flipping UPLO.
        if n < 512:
            ssyr2_rows_kernel[(n, triton.cdiv(n, 128))](
                A,
                x,
                y,
                alpha,
                N=n,
                LDA=lda,
                INCX=incx,
                INCY=incy,
                UPLO=uplo,
                BLOCK_M=1,
                BLOCK_N=128,
                num_warps=4,
                num_stages=1,
            )
        else:
            # Square tiles use column-major coordinates over the same buffer.
            # SYR2 is symmetric, so transposing coordinates flips the triangle.
            ssyr2_tiles_kernel[_ssyr2_grid(n)](
                A, x, y, alpha, n, lda, incx, incy, UPLO=1 - uplo
            )
