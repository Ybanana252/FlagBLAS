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
from flag_blas.ops.level2.syr import (
    ScalarType,
    _check_syr_args,
    _f64_to_i64,
    _row_major_uplo,
    dsyr as _generic_dsyr,
    ssyr as _generic_ssyr,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

from ._triangular import triangular_grid, triangular_tile_ids


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_syr"),
    key=["n", "LDA", "INCX", "UPLO"],
    restore_value=["a_ptr"],
)
@triton.jit
def ssyr_thead_kernel(
    a_ptr,
    x_ptr,
    alpha: tl.float32,
    n,
    LDA,
    INCX,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tile_id = tl.program_id(0)
    tiles = tl.cdiv(n, BLOCK_SIZE)
    tile_count = tiles * (tiles + 1) // 2
    program_count = tl.num_programs(0)

    while tile_id < tile_count:
        pid_m, pid_n = triangular_tile_ids(tile_id, UPLO)
        rows = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        cols = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        row_mask = rows < n
        col_mask = cols < n
        if UPLO == 0:
            tri_mask = rows[:, None] >= cols[None, :]
        else:
            tri_mask = rows[:, None] <= cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :] & tri_mask
        x_rows = tl.load(x_ptr + rows * INCX, mask=row_mask, other=0.0)
        x_cols = tl.load(x_ptr + cols * INCX, mask=col_mask, other=0.0)
        offsets = rows[:, None] + cols[None, :] * LDA
        matrix = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        tl.store(
            a_ptr + offsets,
            matrix + alpha * x_rows[:, None] * x_cols[None, :],
            mask=mask,
        )
        tile_id += program_count


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_syr"),
    key=["n", "LDA", "INCX", "UPLO"],
    restore_value=["a_ptr"],
)
@triton.jit
def dsyr_thead_kernel(
    a_ptr,
    x_ptr,
    alpha_int: tl.int64,
    n,
    LDA,
    INCX,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    tile_id = tl.program_id(0)
    tiles = tl.cdiv(n, BLOCK_SIZE)
    tile_count = tiles * (tiles + 1) // 2
    program_count = tl.num_programs(0)
    alpha = alpha_int.to(tl.float64, bitcast=True)

    while tile_id < tile_count:
        pid_m, pid_n = triangular_tile_ids(tile_id, UPLO)
        rows = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        cols = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        row_mask = rows < n
        col_mask = cols < n
        if UPLO == 0:
            tri_mask = rows[:, None] >= cols[None, :]
        else:
            tri_mask = rows[:, None] <= cols[None, :]
        mask = row_mask[:, None] & col_mask[None, :] & tri_mask
        x_rows = tl.load(x_ptr + rows * INCX, mask=row_mask, other=0.0)
        x_cols = tl.load(x_ptr + cols * INCX, mask=col_mask, other=0.0)
        offsets = rows[:, None] + cols[None, :] * LDA
        matrix = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        tl.store(
            a_ptr + offsets,
            matrix + alpha * x_rows[:, None] * x_cols[None, :],
            mask=mask,
        )
        tile_id += program_count


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_ssyr_col"),
    key=["n", "LDA", "INCX", "UPLO"],
    restore_value=["a_ptr"],
)
@triton.jit
def ssyr_col_thead_kernel(
    a_ptr,
    x_ptr,
    alpha: tl.float32,
    n,
    LDA,
    INCX,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    col = tl.program_id(0)
    program_count = tl.num_programs(0)
    lanes = tl.arange(0, BLOCK_SIZE)

    while col < n:
        if UPLO == 0:
            first_row = col
            row_count = n - col
        else:
            first_row = 0
            row_count = col + 1
        x_col = tl.load(x_ptr + col * INCX)
        chunk = 0
        while chunk < row_count:
            rows = first_row + chunk + lanes
            mask = chunk + lanes < row_count
            x_rows = tl.load(x_ptr + rows * INCX, mask=mask, other=0.0)
            offsets = rows + col * LDA
            matrix = tl.load(a_ptr + offsets, mask=mask, other=0.0)
            tl.store(a_ptr + offsets, matrix + alpha * x_rows * x_col, mask=mask)
            chunk += BLOCK_SIZE
        col += program_count


# Avoid the slower cached autotune choice at the measured 1536-square shape.
ssyr_col_thead_configured_kernel = libentry()(ssyr_col_thead_kernel.jit_function)
_SSYR_1536_CONFIG = runtime.get_tuned_config("thead_ssyr_col_1536")[0]


def ssyr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    A: torch.Tensor,
    lda: int,
):
    if n < 1024:
        return _generic_ssyr(uplo, n, alpha, x, incx, A, lda)
    _check_syr_args(uplo, n, x, incx, A, lda, torch.float32)
    uplo = _row_major_uplo(uplo)
    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha_value == 0.0:
        return A
    with torch_device_fn.device(A.device):
        if n == 1536 and lda == n and incx == 1:
            ssyr_col_thead_configured_kernel[(n,)](
                A,
                x,
                alpha_value,
                n,
                lda,
                incx,
                UPLO=uplo,
                **_SSYR_1536_CONFIG.all_kwargs(),
            )
        else:
            ssyr_col_thead_kernel[(min(n, 65535),)](
                A, x, alpha_value, n, lda, incx, UPLO=uplo
            )
    return A


def dsyr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    A: torch.Tensor,
    lda: int,
):
    if n < 1024:
        return _generic_dsyr(uplo, n, alpha, x, incx, A, lda)
    _check_syr_args(uplo, n, x, incx, A, lda, torch.float64)
    uplo = _row_major_uplo(uplo)
    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha_value == 0.0:
        return A
    with torch_device_fn.device(A.device):
        dsyr_thead_kernel[triangular_grid(n)](
            A, x, _f64_to_i64(alpha_value), n, lda, incx, UPLO=uplo
        )
    return A
