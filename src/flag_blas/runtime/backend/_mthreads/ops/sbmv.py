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

from flag_blas.ops.level2.sbmv import ScalarType, _check_common, _strided_y
from flag_blas.runtime import torch_device_fn


@triton.jit
def _ssbmv_band(
    A,
    X,
    Y,
    ALPHA: tl.float32,
    BETA: tl.float32,
    N,
    K: tl.constexpr,
    STORED_K: tl.constexpr,
    LDA: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    UPLO: tl.constexpr,
    BETA_ZERO: tl.constexpr,
    M: tl.constexpr,
    B: tl.constexpr,
):
    rows = tl.program_id(0) * M + tl.arange(0, M)
    rm = rows < N
    row_base = rows.to(tl.int64) * LDA
    ds = tl.arange(0, B) + 1
    acc = tl.full((M, B), 0.0, tl.float32)
    for block in range(triton.cdiv(K, B)):
        d = block * B + ds
        if UPLO == 1:
            direct_col = rows[:, None] + d[None, :]
            reflected_col = rows[:, None] - d[None, :]
            direct_off = row_base[:, None] + d[None, :]
            reflected_off = reflected_col.to(tl.int64) * LDA + d[None, :]
        else:
            direct_col = rows[:, None] - d[None, :]
            reflected_col = rows[:, None] + d[None, :]
            direct_off = row_base[:, None] + STORED_K - d[None, :]
            reflected_off = reflected_col.to(tl.int64) * LDA + STORED_K - d[None, :]
        dm = rm[:, None] & (d[None, :] <= K)
        direct_mask = dm & (direct_col >= 0) & (direct_col < N)
        reflected_mask = dm & (reflected_col >= 0) & (reflected_col < N)
        av = tl.load(A + direct_off, direct_mask, 0)
        xv = tl.load(X + direct_col.to(tl.int64) * INCX, direct_mask, 0)
        acc += av * xv
        av = tl.load(A + reflected_off, reflected_mask, 0)
        xv = tl.load(X + reflected_col.to(tl.int64) * INCX, reflected_mask, 0)
        acc += av * xv

    diag_off = row_base
    if UPLO == 0:
        diag_off += STORED_K
    diag = tl.load(A + diag_off, rm, 0)
    xv = tl.load(X + rows.to(tl.int64) * INCX, rm, 0)
    total = tl.sum(acc, axis=1) + diag * xv
    result = ALPHA * total
    yo = rows.to(tl.int64) * INCY
    if not BETA_ZERO:
        result += BETA * tl.load(Y + yo, rm, 0)
    tl.store(Y + yo, result, rm)


def ssbmv(
    uplo: int,
    n: int,
    k: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    assert A.dtype == torch.float32 == x.dtype == y.dtype
    _check_common(A, x, y, uplo, n, k, lda, incx, incy)
    if n == 0:
        return
    alpha = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta = float(beta.item() if isinstance(beta, torch.Tensor) else beta)
    if alpha == 0.0:
        y_view = _strided_y(y, n, incy)
        if beta == 0.0:
            y_view.zero_()
        elif beta != 1.0:
            y_view.mul_(beta)
        return

    # Clamp the work, but retain the original k for lower-band addressing.
    effective_k = min(k, n - 1)
    band = min(256, triton.next_power_of_2(max(1, effective_k)))
    # S5000 measurements favor small row groups for short vectors and
    # narrower band tiles for large matrices. No output-restoring autotune
    # or temporary buffer is needed on this path.
    if effective_k == 0:
        rows_per_program = 128
    elif band == 1:
        rows_per_program = 32 if n <= 2048 else 128
    elif band <= 4:
        rows_per_program = 16 if n <= 2048 else 32
    elif band <= 16:
        rows_per_program = min(32, max(4, triton.next_power_of_2(triton.cdiv(n, 512))))
    elif band <= 64:
        rows_per_program = 1 if n <= 512 else (2 if n <= 1024 else 4)
    elif n <= 512:
        rows_per_program = 1
    elif n <= 2048:
        rows_per_program, band = 1, 128
    else:
        rows_per_program, band = 4, 64
    with torch_device_fn.device(A.device):
        _ssbmv_band[(triton.cdiv(n, rows_per_program),)](
            A,
            x,
            y,
            alpha,
            beta,
            n,
            K=effective_k,
            STORED_K=k,
            LDA=lda,
            INCX=incx,
            INCY=incy,
            UPLO=uplo,
            BETA_ZERO=beta == 0.0,
            M=rows_per_program,
            B=band,
            num_warps=4,
            num_stages=1,
        )
