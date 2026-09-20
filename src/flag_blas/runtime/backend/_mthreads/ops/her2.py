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

from flag_blas.ops.level2.her2 import ScalarType, _check_her2_args, _complex_scalar
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

# Small row slices keep register pressure low. Triangular scheduling benefits
# larger matrices; rectangular scheduling avoids its indexing cost on small ones.
_CHER2_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 1, "BLOCK_N": 128, "TRIANGULAR": False},
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 4, "BLOCK_N": 32, "TRIANGULAR": True},
        num_warps=4,
        num_stages=1,
    ),
    triton.Config(
        {"BLOCK_M": 1, "BLOCK_N": 256, "TRIANGULAR": False},
        num_warps=4,
        num_stages=1,
    ),
]


@triton.jit
def _cher2_tiled_jit(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_r,
    alpha_i,
    N: tl.constexpr,
    LDA: tl.constexpr,
    TRIANGULAR: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Map only valid square macro tiles; split each into short row slices.
    if TRIANGULAR:
        tile = tl.program_id(0)
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
        row_start = row_tile * BLOCK_N + tl.program_id(1) * BLOCK_M
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
    # Keep each complex pair together in memory, then split for arithmetic.
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
            y_ptr + r64[:, None] * INCY * 2 + lane[None, :],
            (rows < N)[:, None],
            other=0.0,
        )
    )
    xc_r, xc_i = tl.split(
        tl.load(
            x_ptr + c64[:, None] * INCX * 2 + lane[None, :],
            (cols < N)[:, None],
            other=0.0,
        )
    )
    yc_r, yc_i = tl.split(
        tl.load(
            y_ptr + c64[:, None] * INCY * 2 + lane[None, :],
            (cols < N)[:, None],
            other=0.0,
        )
    )

    # Scale alpha*x and conj(alpha)*y once per row, then broadcast.
    # A[i,j] += alpha*x[i]*conj(y[j]) + conj(alpha)*y[i]*conj(x[j]).
    ax_r = alpha_r * xr - alpha_i * xi
    ax_i = alpha_r * xi + alpha_i * xr
    ay_r = alpha_r * yr + alpha_i * yi
    ay_i = alpha_r * yi - alpha_i * yr
    update_r = (
        ax_r[:, None] * yc_r[None, :]
        + ax_i[:, None] * yc_i[None, :]
        + ay_r[:, None] * xc_r[None, :]
        + ay_i[:, None] * xc_i[None, :]
    )
    update_i = (
        ax_i[:, None] * yc_r[None, :]
        - ax_r[:, None] * yc_i[None, :]
        + ay_i[:, None] * xc_r[None, :]
        - ay_r[:, None] * xc_i[None, :]
    )

    mask = (rows[:, None] < N) & (cols[None, :] < N)
    offsets = r64[:, None] * LDA + c64[None, :]
    if UPLO == 0:
        mask = mask & (rows[:, None] >= cols[None, :])
    else:
        mask = mask & (rows[:, None] <= cols[None, :])
    ptr = a_ptr + offsets[:, :, None] * 2 + lane[None, None, :]
    ar, ai = tl.split(tl.load(ptr, mask[:, :, None], other=0.0))
    out_i = tl.where(rows[:, None] == cols[None, :], 0.0, ai + update_i)
    tl.store(ptr, tl.join(ar + update_r, out_i), mask[:, :, None])


cher2_mthreads_kernel = libentry()(
    triton.autotune(
        configs=_CHER2_CONFIGS,
        key=["N", "LDA", "INCX", "INCY", "UPLO"],
        restore_value=["a_ptr"],
    )(_cher2_tiled_jit)
)


def _cher2_grid(n):
    def grid(meta):
        columns = triton.cdiv(n, meta["BLOCK_N"])
        if meta["TRIANGULAR"]:
            row_slices = triton.cdiv(min(n, meta["BLOCK_N"]), meta["BLOCK_M"])
            return (columns * (columns + 1) // 2, row_slices)
        return (triton.cdiv(n, meta["BLOCK_M"]), columns)

    return grid


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
) -> None:
    _check_her2_args(torch.complex64, uplo, n, x, incx, y, incy, A, lda)
    if n == 0:
        return
    ar, ai = _complex_scalar(alpha)
    if ar == 0.0 and ai == 0.0:
        return

    # The kernel uses logical row-major coordinates, so UPLO is not flipped.
    with torch_device_fn.device(A.device):
        cher2_mthreads_kernel[_cher2_grid(n)](
            torch.view_as_real(A),
            torch.view_as_real(x),
            torch.view_as_real(y),
            ar,
            ai,
            N=n,
            LDA=lda,
            INCX=incx,
            INCY=incy,
            UPLO=uplo,
        )
