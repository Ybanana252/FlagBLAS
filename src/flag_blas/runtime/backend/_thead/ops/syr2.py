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
from flag_blas.ops.level2._constants import CUBLAS_FILL_MODE_LOWER
from flag_blas.ops.level2.syr2 import (
    ScalarType,
    _check_ssyr2_args,
    _check_syr2_args,
    _f64_to_i64,
    _row_major_uplo,
    dsyr2 as _generic_dsyr2,
    dsyr2_kernel as _generic_dsyr2_kernel,
    ssyr2 as _generic_ssyr2,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

from ._triangular import triangular_grid, triangular_tile_ids


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_ssyr2_tri"),
    key=["n", "LDA", "INCX", "INCY", "UPLO"],
    restore_value=["a_ptr"],
)
@triton.jit
def ssyr2_thead_kernel(
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
        y_rows = tl.load(y_ptr + rows * INCY, mask=row_mask, other=0.0)
        x_cols = tl.load(x_ptr + cols * INCX, mask=col_mask, other=0.0)
        y_cols = tl.load(y_ptr + cols * INCY, mask=col_mask, other=0.0)
        offsets = rows[:, None] + cols[None, :] * LDA
        matrix = tl.load(a_ptr + offsets, mask=mask, other=0.0)
        update = alpha * (
            x_rows[:, None] * y_cols[None, :]
            + y_rows[:, None] * x_cols[None, :]
        )
        tl.store(a_ptr + offsets, matrix + update, mask=mask)
        tile_id += program_count


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_ssyr2_aligned"),
    key=["n", "LDA", "UPLO"],
    restore_value=["a_ptr"],
)
@triton.jit
def ssyr2_aligned_thead_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    n,
    LDA,
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
        x_rows = tl.load(x_ptr + rows)
        y_rows = tl.load(y_ptr + rows)
        x_cols = tl.load(x_ptr + cols)
        y_cols = tl.load(y_ptr + cols)
        offsets = cols[:, None] * LDA + rows[None, :]
        update = alpha * (
            x_cols[:, None] * y_rows[None, :]
            + y_cols[:, None] * x_rows[None, :]
        )
        if pid_m != pid_n:
            matrix = tl.load(a_ptr + offsets)
            tl.store(a_ptr + offsets, matrix + update)
        else:
            if UPLO == 0:
                mask = rows[None, :] >= cols[:, None]
            else:
                mask = rows[None, :] <= cols[:, None]
            matrix = tl.load(a_ptr + offsets, mask=mask, other=0.0)
            tl.store(a_ptr + offsets, matrix + update, mask=mask)
        tile_id += program_count


# Reuse the existing JIT bodies; only the measured shapes bypass autotuning.
ssyr2_aligned_configured_kernel = libentry()(ssyr2_aligned_thead_kernel.jit_function)
dsyr2_configured_kernel = libentry()(_generic_dsyr2_kernel.jit_function)
_SSYR2_FOCUSED_CONFIG = runtime.get_tuned_config("thead_ssyr2_aligned_focused")[0]
_DSYR2_1024_CONFIG = runtime.get_tuned_config("thead_dsyr2_1024")[0]


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
    if n < 1024:
        _generic_ssyr2(uplo, n, alpha, x, incx, y, incy, A, lda)
        return
    _check_ssyr2_args(uplo, n, x, incx, y, incy, A, lda)
    use_focused_config = n in (1024, 1280) or (
        n == 2560 and uplo == CUBLAS_FILL_MODE_LOWER
    )
    uplo = _row_major_uplo(uplo)
    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha_value == 0.0:
        return
    with torch_device_fn.device(A.device):
        if n % 256 == 0 and lda % 4 == 0 and incx == 1 and incy == 1:
            if use_focused_config:
                ssyr2_aligned_configured_kernel[triangular_grid(n)](
                    A,
                    x,
                    y,
                    alpha_value,
                    n,
                    lda,
                    UPLO=uplo,
                    **_SSYR2_FOCUSED_CONFIG.all_kwargs(),
                )
            else:
                ssyr2_aligned_thead_kernel[triangular_grid(n)](
                    A, x, y, alpha_value, n, lda, UPLO=uplo
                )
        else:
            ssyr2_thead_kernel[triangular_grid(n)](
                A, x, y, alpha_value, n, lda, incx, incy, UPLO=uplo
            )


def dsyr2(
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
    if not (
        n == 1024
        and uplo == CUBLAS_FILL_MODE_LOWER
        and lda == 1024
        and incx == 1
        and incy == 1
    ):
        _generic_dsyr2(uplo, n, alpha, x, incx, y, incy, A, lda)
        return
    _check_syr2_args(torch.float64, n, x, incx, y, incy, A, lda, uplo)
    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha_value == 0.0:
        return
    config = _DSYR2_1024_CONFIG
    blocks = triton.cdiv(n, config.kwargs["BLOCK_SIZE"])
    with torch_device_fn.device(A.device):
        dsyr2_configured_kernel[(blocks, blocks)](
            A,
            x,
            y,
            _f64_to_i64(alpha_value),
            n,
            lda,
            incx,
            incy,
            UPLO=_row_major_uplo(uplo),
            **config.all_kwargs(),
        )
