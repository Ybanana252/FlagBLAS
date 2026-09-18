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
from flag_blas.ops.level2.ger import (
    ScalarType,
    _check_ger_common,
    _f64_to_i64,
    _scalar_to_float,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner
from flag_blas.utils import triton_lang_extension as tle


def _grid(m: int, n: int):
    def grid(meta):
        return (
            triton.cdiv(m, meta["BLOCK_SIZE_M"]),
            triton.cdiv(n, meta["BLOCK_SIZE_N"]),
        )

    return grid


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_sger"),
    key=["m", "n", "LDA", "INCX", "INCY"],
    restore_value=["A_ptr"],
)
@triton.jit
def sger_thead_kernel(
    x_ptr,
    y_ptr,
    A_ptr,
    alpha: tl.float32,
    m,
    n,
    INCX,
    INCY,
    LDA,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    cols = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_mask = rows < m
    col_mask = cols < n
    x_vals = tl.load(x_ptr + rows * INCX, mask=row_mask, other=0.0)
    y_vals = tl.load(y_ptr + cols * INCY, mask=col_mask, other=0.0)
    offsets = rows[:, None] * LDA + cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :]
    matrix = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    tl.store(
        A_ptr + offsets,
        matrix + alpha * x_vals[:, None] * y_vals[None, :],
        mask=mask,
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_dger"),
    key=["m", "n", "LDA", "INCX", "INCY"],
    restore_value=["A_ptr"],
)
@triton.jit
def dger_thead_kernel(
    x_ptr,
    y_ptr,
    A_ptr,
    alpha_int: tl.int64,
    m,
    n,
    INCX,
    INCY,
    LDA,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)
    alpha = alpha_int.to(tl.float64, bitcast=True)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    cols = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_mask = rows < m
    col_mask = cols < n
    x_vals = tl.load(x_ptr + rows * INCX, mask=row_mask, other=0.0)
    y_vals = tl.load(y_ptr + cols * INCY, mask=col_mask, other=0.0)
    offsets = rows[:, None] * LDA + cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :]
    matrix = tl.load(A_ptr + offsets, mask=mask, other=0.0)
    tl.store(
        A_ptr + offsets,
        matrix + alpha * x_vals[:, None] * y_vals[None, :],
        mask=mask,
    )


def sger(
    m: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    A: torch.Tensor,
    lda: int,
) -> None:
    if not _check_ger_common(m, n, x, incx, y, incy, A, lda, torch.float32):
        return
    alpha_value = _scalar_to_float(alpha)
    if alpha_value == 0.0:
        return
    with torch_device_fn.device(A.device):
        sger_thead_kernel[_grid(m, n)](
            x, y, A, alpha_value, m, n, incx, incy, lda
        )


def dger(
    m: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    A: torch.Tensor,
    lda: int,
) -> None:
    if not _check_ger_common(m, n, x, incx, y, incy, A, lda, torch.float64):
        return
    alpha_value = _scalar_to_float(alpha)
    if alpha_value == 0.0:
        return
    with torch_device_fn.device(A.device):
        dger_thead_kernel[_grid(m, n)](
            x, y, A, _f64_to_i64(alpha_value), m, n, incx, incy, lda
        )
