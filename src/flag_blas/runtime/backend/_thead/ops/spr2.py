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

from flag_blas import runtime
from flag_blas.ops.level2._constants import CUBLAS_FILL_MODE_UPPER
from flag_blas.ops.level2.spr2 import (
    ScalarType,
    _check_spr2_args,
    _f64_to_i64,
    dspr2 as _generic_dspr2,
    dspr2_kernel as _generic_dspr2_kernel,
    sspr2 as _generic_sspr2,
    sspr2_kernel as _generic_sspr2_kernel,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

from ._triangular import triangular_grid, triangular_tile_ids


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_spr2_row"),
    key=["n", "INCX", "INCY"],
    restore_value=["ap_ptr"],
)
@triton.jit
def sspr2_thead_kernel(
    ap_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    n,
    INCX,
    INCY,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    program_count = tl.num_programs(0)
    lanes = tl.arange(0, BLOCK_SIZE)

    while row < n:
        row64 = row.to(tl.int64)
        packed_start = row64 * (row64 + 1) // 2
        x_row = tl.load(x_ptr + row * INCX)
        y_row = tl.load(y_ptr + row * INCY)
        chunk = 0
        while chunk <= row:
            cols = chunk + lanes
            mask = cols <= row
            x_cols = tl.load(x_ptr + cols * INCX, mask=mask, other=0.0)
            y_cols = tl.load(y_ptr + cols * INCY, mask=mask, other=0.0)
            offsets = packed_start + chunk + lanes
            packed = tl.load(ap_ptr + offsets, mask=mask, other=0.0)
            update = alpha * (x_row * y_cols + y_row * x_cols)
            tl.store(ap_ptr + offsets, packed + update, mask=mask)
            chunk += BLOCK_SIZE
        row += program_count


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_spr2_row"),
    key=["n", "INCX", "INCY"],
    restore_value=["ap_ptr"],
)
@triton.jit
def dspr2_thead_kernel(
    ap_ptr,
    x_ptr,
    y_ptr,
    alpha_int: tl.int64,
    n,
    INCX,
    INCY,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    program_count = tl.num_programs(0)
    lanes = tl.arange(0, BLOCK_SIZE)
    alpha = alpha_int.to(tl.float64, bitcast=True)

    while row < n:
        row64 = row.to(tl.int64)
        packed_start = row64 * (row64 + 1) // 2
        x_row = tl.load(x_ptr + row * INCX)
        y_row = tl.load(y_ptr + row * INCY)
        chunk = 0
        while chunk <= row:
            cols = chunk + lanes
            mask = cols <= row
            x_cols = tl.load(x_ptr + cols * INCX, mask=mask, other=0.0)
            y_cols = tl.load(y_ptr + cols * INCY, mask=mask, other=0.0)
            offsets = packed_start + chunk + lanes
            packed = tl.load(ap_ptr + offsets, mask=mask, other=0.0)
            update = alpha * (x_row * y_cols + y_row * x_cols)
            tl.store(ap_ptr + offsets, packed + update, mask=mask)
            chunk += BLOCK_SIZE
        row += program_count


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_dspr2_tri"),
    key=["n", "INCX", "INCY"],
    restore_value=["ap_ptr"],
)
@triton.jit
def dspr2_tri_thead_kernel(
    ap_ptr,
    x_ptr,
    y_ptr,
    alpha_int: tl.int64,
    n,
    INCX,
    INCY,
    BLOCK_SIZE: tl.constexpr,
):
    tile_id = tl.program_id(0)
    tiles = tl.cdiv(n, BLOCK_SIZE)
    tile_count = tiles * (tiles + 1) // 2
    program_count = tl.num_programs(0)
    alpha = alpha_int.to(tl.float64, bitcast=True)

    while tile_id < tile_count:
        pid_m, pid_n = triangular_tile_ids(tile_id, 0)
        rows = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        cols = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        row_mask = rows < n
        col_mask = cols < n
        tri_mask = rows[:, None] >= cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :] & tri_mask
        rows64 = rows.to(tl.int64)
        cols64 = cols.to(tl.int64)
        offsets = rows64[:, None] * (rows64[:, None] + 1) // 2 + cols64[None, :]
        x_rows = tl.load(x_ptr + rows * INCX, mask=row_mask, other=0.0)
        y_rows = tl.load(y_ptr + rows * INCY, mask=row_mask, other=0.0)
        x_cols = tl.load(x_ptr + cols * INCX, mask=col_mask, other=0.0)
        y_cols = tl.load(y_ptr + cols * INCY, mask=col_mask, other=0.0)
        packed = tl.load(ap_ptr + offsets, mask=mask, other=0.0)
        update = alpha * (
            x_rows[:, None] * y_cols[None, :]
            + y_rows[:, None] * x_cols[None, :]
        )
        tl.store(ap_ptr + offsets, packed + update, mask=mask)
        tile_id += program_count


# The shared SPR2 tile kernel already uses row-major packed indexing.
dspr2_thead_small_kernel = libentry()(_generic_dspr2_kernel.jit_function)
sspr2_thead_small_kernel = libentry()(_generic_sspr2_kernel.jit_function)
_SPR2_TILE_SMALL_CONFIG = runtime.get_tuned_config("thead_spr2_tile_small")[0]
_SPR2_TILE_MEDIUM_CONFIG = runtime.get_tuned_config("thead_spr2_tile_medium")[0]


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
    if uplo == CUBLAS_FILL_MODE_UPPER:
        _generic_sspr2(uplo, n, alpha, x, incx, y, incy, AP)
        return
    _check_spr2_args(torch.float32, uplo, n, x, incx, y, incy, AP)
    if n == 0:
        return
    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha_value == 0.0:
        return
    with torch_device_fn.device(AP.device):
        if (256 <= n <= 320 or 512 < n <= 640) and incx == 1 and incy == 1:
            config = _SPR2_TILE_SMALL_CONFIG if n <= 320 else _SPR2_TILE_MEDIUM_CONFIG
            grid = (triton.cdiv(n, config.kwargs["BLOCK_SIZE"]),) * 2
            sspr2_thead_small_kernel[grid](
                AP, x, y, alpha_value, n, incx, incy,
                UPLO=uplo, **config.all_kwargs(),
            )
            return
        sspr2_thead_kernel[(min(n, 65535),)](
            AP, x, y, alpha_value, n, incx, incy
        )


def dspr2(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    AP: torch.Tensor,
) -> None:
    if uplo == CUBLAS_FILL_MODE_UPPER:
        _generic_dspr2(uplo, n, alpha, x, incx, y, incy, AP)
        return
    _check_spr2_args(torch.float64, uplo, n, x, incx, y, incy, AP)
    if n == 0:
        return
    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha_value == 0.0:
        return
    with torch_device_fn.device(AP.device):
        if (n == 193 or 512 < n <= 768) and incx == 1 and incy == 1:
            config = (
                _SPR2_TILE_SMALL_CONFIG if n == 193 else _SPR2_TILE_MEDIUM_CONFIG
            )
            grid = (triton.cdiv(n, config.kwargs["BLOCK_SIZE"]),) * 2
            dspr2_thead_small_kernel[grid](
                AP,
                x,
                y,
                _f64_to_i64(alpha_value),
                n,
                incx,
                incy,
                UPLO=uplo,
                **config.all_kwargs(),
            )
        elif n >= 1024:
            dspr2_tri_thead_kernel[triangular_grid(n)](
                AP, x, y, _f64_to_i64(alpha_value), n, incx, incy
            )
        else:
            dspr2_thead_kernel[(min(n, 65535),)](
                AP, x, y, _f64_to_i64(alpha_value), n, incx, incy
            )
