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

from flag_blas.ops.level2.spr2 import ScalarType, _check_spr2_args
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

_SSPR2_CONFIGS = [
    triton.Config({"BLOCK_M": 1, "BLOCK_N": 128}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 4, "BLOCK_N": 64}, num_warps=4, num_stages=1),
    triton.Config({"BLOCK_M": 16, "BLOCK_N": 64}, num_warps=4, num_stages=1),
]


@libentry()
@triton.autotune(
    configs=_SSPR2_CONFIGS,
    key=["N", "INCX", "INCY", "UPLO"],
    restore_value=["ap_ptr"],
)
@triton.jit
def sspr2_mthreads_kernel(
    ap_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    N: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile = tl.program_id(0)
    major = ((tl.sqrt(8.0 * tile + 1.0) - 1.0) * 0.5).to(tl.int32)
    # Correct rounding at triangular boundaries.
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
    xr = tl.load(x_ptr + r64 * INCX, rows < N, other=0.0)
    yr = tl.load(y_ptr + r64 * INCY, rows < N, other=0.0)
    xc = tl.load(x_ptr + c64 * INCX, cols < N, other=0.0)
    yc = tl.load(y_ptr + c64 * INCY, cols < N, other=0.0)

    mask = (rows[:, None] < N) & (cols[None, :] < N)
    if UPLO == 0:
        row_base = r64 * (r64 + 1) // 2
        offsets = row_base[:, None] + c64[None, :]
        mask = mask & (rows[:, None] >= cols[None, :])
    else:
        # The public interface uses row-packed upper storage.
        row_base = r64 * (2 * N - r64 + 1) // 2
        offsets = row_base[:, None] + c64[None, :] - r64[:, None]
        mask = mask & (rows[:, None] <= cols[None, :])

    ap = tl.load(ap_ptr + offsets, mask, other=0.0)
    update = alpha * (xr[:, None] * yc[None, :] + yr[:, None] * xc[None, :])
    tl.store(ap_ptr + offsets, ap + update, mask)


def _sspr2_grid(n):
    def grid(meta):
        tiles = triton.cdiv(n, meta["BLOCK_N"])
        row_slices = triton.cdiv(min(n, meta["BLOCK_N"]), meta["BLOCK_M"])
        return (tiles * (tiles + 1) // 2, row_slices)

    return grid


def sspr2(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    AP: torch.Tensor,
) -> None:
    _check_spr2_args(torch.float32, uplo, n, x, incx, y, incy, AP)
    if n == 0:
        return
    alpha = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha == 0.0:
        return
    with torch_device_fn.device(AP.device):
        sspr2_mthreads_kernel[_sspr2_grid(n)](
            AP, x, y, alpha, N=n, INCX=incx, INCY=incy, UPLO=uplo
        )
