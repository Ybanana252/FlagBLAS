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

from flag_blas.ops.level2.hpr2 import ScalarType, _check_hpr2_args, _complex_scalar
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

_CHPR2_CONFIGS = [
    triton.Config({"BLOCK_M": 1, "BLOCK_N": 128}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 4, "BLOCK_N": 64}, num_warps=4, num_stages=1),
]


@triton.jit
def _chpr2_tiled_jit(
    ap_ptr,
    x_ptr,
    y_ptr,
    alpha_r,
    alpha_i,
    N: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # One scalar inverse-triangle calculation per tile, not per matrix entry.
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

    rows = row_tile * BLOCK_N + tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = col_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    r64, c64 = rows.to(tl.int64), cols.to(tl.int64)
    xr = tl.load(x_ptr + r64 * INCX * 2, rows < N, other=0.0)
    xi = tl.load(x_ptr + r64 * INCX * 2 + 1, rows < N, other=0.0)
    yr = tl.load(y_ptr + r64 * INCY * 2, rows < N, other=0.0)
    yi = tl.load(y_ptr + r64 * INCY * 2 + 1, rows < N, other=0.0)
    xc_r = tl.load(x_ptr + c64 * INCX * 2, cols < N, other=0.0)
    xc_i = tl.load(x_ptr + c64 * INCX * 2 + 1, cols < N, other=0.0)
    yc_r = tl.load(y_ptr + c64 * INCY * 2, cols < N, other=0.0)
    yc_i = tl.load(y_ptr + c64 * INCY * 2 + 1, cols < N, other=0.0)

    # Scale each row value once and broadcast it across the tile.
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
    if UPLO == 0:
        row_base = r64 * (r64 + 1) // 2
        offsets = row_base[:, None] + c64[None, :]
        mask = mask & (rows[:, None] >= cols[None, :])
    else:
        row_base = r64 * (2 * N - r64 + 1) // 2
        offsets = row_base[:, None] + c64[None, :] - r64[:, None]
        mask = mask & (rows[:, None] <= cols[None, :])
    ar = tl.load(ap_ptr + offsets * 2, mask, other=0.0)
    ai = tl.load(ap_ptr + offsets * 2 + 1, mask, other=0.0)
    out_i = tl.where(rows[:, None] == cols[None, :], 0.0, ai + update_i)
    tl.store(ap_ptr + offsets * 2, ar + update_r, mask)
    tl.store(ap_ptr + offsets * 2 + 1, out_i, mask)


chpr2_mthreads_kernel = libentry()(
    triton.autotune(
        configs=_CHPR2_CONFIGS,
        key=["N", "INCX", "INCY", "UPLO"],
        restore_value=["ap_ptr"],
    )(_chpr2_tiled_jit)
)


def _chpr2_grid(n):
    def grid(meta):
        tiles = triton.cdiv(n, meta["BLOCK_N"])
        row_slices = triton.cdiv(min(n, meta["BLOCK_N"]), meta["BLOCK_M"])
        return (tiles * (tiles + 1) // 2, row_slices)

    return grid


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
    _check_hpr2_args(torch.complex64, uplo, n, x, incx, y, incy, AP)
    if n == 0:
        return
    ar, ai = _complex_scalar(alpha)
    if ar == 0.0 and ai == 0.0:
        return

    with torch_device_fn.device(AP.device):
        chpr2_mthreads_kernel[_chpr2_grid(n)](
            torch.view_as_real(AP),
            torch.view_as_real(x),
            torch.view_as_real(y),
            ar,
            ai,
            N=n,
            INCX=incx,
            INCY=incy,
            UPLO=uplo,
        )
