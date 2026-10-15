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

from flag_blas.ops.level2.ger import (
    _CGER_CONFIGS,
    ScalarType,
    _check_ger_common,
    _scalar_to_complex_parts,
    _scalar_to_float,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry
from flag_blas.utils import triton_lang_extension as tle


@libentry()
@triton.jit
def _sger_contiguous_kernel(
    X,
    Y,
    A,
    alpha: tl.float32,
    M: tl.constexpr,
    N: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    if max(M * N, M * INCX, N * INCY) > 2147483647:
        pid = pid.to(tl.int64)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    rows = offsets // N
    cols = offsets % N
    mask = offsets < M * N
    x = tl.load(X + rows * INCX, mask, other=0.0)
    y = tl.load(Y + cols * INCY, mask, other=0.0)
    a = tl.load(A + offsets, mask, other=0.0)
    tl.store(A + offsets, a + (alpha * x) * y, mask)


@libentry()
@triton.jit
def _sger_tiled_kernel(
    X,
    Y,
    A,
    alpha: tl.float32,
    M: tl.constexpr,
    N: tl.constexpr,
    LDA: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Consecutive programs update row tiles within the same column tile.
    pid = tl.program_id(0)
    tile_rows = tl.cdiv(M, BLOCK_M)
    pid_m = pid % tile_rows
    pid_n = pid // tile_rows
    if max(M * LDA, M * INCX, N * INCY) > 2147483647:
        pid_m = pid_m.to(tl.int64)
        pid_n = pid_n.to(tl.int64)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x = tl.load(X + rows * INCX, rows < M, other=0.0)
    y = tl.load(Y + cols * INCY, cols < N, other=0.0)
    offsets = rows[:, None] * LDA + cols[None, :]
    mask = (rows[:, None] < M) & (cols[None, :] < N)
    a = tl.load(A + offsets, mask, other=0.0)
    tl.store(A + offsets, a + (alpha * x[:, None]) * y[None, :], mask)


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
    alpha = _scalar_to_float(alpha)
    if alpha == 0.0:
        return

    with torch_device_fn.device(A.device):
        # Aligned dense rows favor linear A accesses. Odd row widths favor
        # the tiled kernel's reuse of x/y and avoid per-element division.
        if lda == n and n % 64 == 0:
            block = 512
            _sger_contiguous_kernel[(triton.cdiv(m * n, block),)](
                x,
                y,
                A,
                alpha,
                m,
                n,
                incx,
                incy,
                block,
                num_warps=4,
                num_stages=1,
            )
        else:
            block_m = 8 if m * n <= 65536 else 4
            block_n = 64
            grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
            _sger_tiled_kernel[grid](
                x,
                y,
                A,
                alpha,
                m,
                n,
                lda,
                incx,
                incy,
                block_m,
                block_n,
                num_warps=2,
                num_stages=1,
            )


@libentry()
@triton.autotune(
    configs=_CGER_CONFIGS,
    key=["m", "n", "LDA", "INCX", "INCY", "CONJ_Y"],
    restore_value=["A_ptr"],
)
@triton.jit
def cger_kernel(
    x_ptr,
    y_ptr,
    A_ptr,
    alpha_real: tl.float32,
    alpha_imag: tl.float32,
    m,
    n,
    INCX,
    INCY,
    LDA,
    CONJ_Y: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_n = tle.program_id(1)
    x_ptr_i64 = x_ptr.to(tl.pointer_type(tl.int64))
    y_ptr_i64 = y_ptr.to(tl.pointer_type(tl.int64))
    A_ptr_i64 = A_ptr.to(tl.pointer_type(tl.int64))

    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    cols = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    row_mask = rows < m
    col_mask = cols < n
    safe_rows = tl.where(row_mask, rows, 0)
    safe_cols = tl.where(col_mask, cols, 0)

    x_val = tl.load(x_ptr_i64 + safe_rows * INCX, mask=row_mask, other=0)
    y_val = tl.load(y_ptr_i64 + safe_cols * INCY, mask=col_mask, other=0)
    x_real = x_val.to(tl.int32).to(tl.float32, bitcast=True)
    x_imag = (x_val >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    y_real = y_val.to(tl.int32).to(tl.float32, bitcast=True)
    y_imag = (y_val >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    if CONJ_Y:
        y_imag = -y_imag

    ax_real = alpha_real * x_real - alpha_imag * x_imag
    ax_imag = alpha_real * x_imag + alpha_imag * x_real
    update_real = (
        ax_real[:, None] * y_real[None, :] - ax_imag[:, None] * y_imag[None, :]
    )
    update_imag = (
        ax_real[:, None] * y_imag[None, :] + ax_imag[:, None] * y_real[None, :]
    )

    a_offsets = safe_rows[:, None] * LDA + safe_cols[None, :]
    mask = row_mask[:, None] & col_mask[None, :]
    a_val = tl.load(A_ptr_i64 + a_offsets, mask=mask, other=0)
    a_real = a_val.to(tl.int32).to(tl.float32, bitcast=True)
    a_imag = (a_val >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    out_real = update_real + a_real
    out_imag = update_imag + a_imag
    out_real_i64 = out_real.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    out_imag_i64 = out_imag.to(tl.int32, bitcast=True).to(tl.int64) << 32
    out_val = out_real_i64 | out_imag_i64

    tl.store(A_ptr_i64 + a_offsets, out_val, mask=mask)


def _grid(m: int, n: int):
    def grid(meta):
        return (
            triton.cdiv(m, meta["BLOCK_SIZE_M"]),
            triton.cdiv(n, meta["BLOCK_SIZE_N"]),
        )

    return grid


def _cger(
    m: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    A: torch.Tensor,
    lda: int,
    conj_y: bool,
) -> None:
    if not _check_ger_common(m, n, x, incx, y, incy, A, lda, torch.complex64):
        return

    alpha_real, alpha_imag = _scalar_to_complex_parts(alpha)
    if alpha_real == 0.0 and alpha_imag == 0.0:
        return

    x_real = torch.view_as_real(x)
    y_real = torch.view_as_real(y)
    A_real = torch.view_as_real(A)

    with torch_device_fn.device(A.device):
        cger_kernel[_grid(m, n)](
            x_real,
            y_real,
            A_real,
            alpha_real,
            alpha_imag,
            m,
            n,
            incx,
            incy,
            lda,
            conj_y,
        )


def cgeru(
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
    _cger(m, n, alpha, x, incx, y, incy, A, lda, False)


def cgerc(
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
    _cger(m, n, alpha, x, incx, y, incy, A, lda, True)
