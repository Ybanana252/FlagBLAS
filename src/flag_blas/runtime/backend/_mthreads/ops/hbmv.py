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

from flag_blas.ops.level2.hbmv import (
    ScalarType,
    _check_common,
    _complex_scalars,
    _strided_y,
)
from flag_blas.ops.level2.hbmv import chbmv as _common_chbmv
from flag_blas.runtime import torch_device_fn


@triton.jit
def _chbmv_band(
    A,
    X,
    Y,
    AR,
    AI,
    BR,
    BI,
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
    acc_r = tl.full((M, B), 0.0, tl.float32)
    acc_i = tl.full((M, B), 0.0, tl.float32)
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
        ar = tl.load(A + 2 * direct_off, direct_mask, 0)
        ai = tl.load(A + 2 * direct_off + 1, direct_mask, 0)
        xo = direct_col.to(tl.int64) * (2 * INCX)
        xr = tl.load(X + xo, direct_mask, 0)
        xi = tl.load(X + xo + 1, direct_mask, 0)
        acc_r += ar * xr - ai * xi
        acc_i += ar * xi + ai * xr

        ar = tl.load(A + 2 * reflected_off, reflected_mask, 0)
        ai = tl.load(A + 2 * reflected_off + 1, reflected_mask, 0)
        xo = reflected_col.to(tl.int64) * (2 * INCX)
        xr = tl.load(X + xo, reflected_mask, 0)
        xi = tl.load(X + xo + 1, reflected_mask, 0)
        acc_r += ar * xr + ai * xi
        acc_i += ar * xi - ai * xr

    diag_off = row_base
    if UPLO == 0:
        diag_off += STORED_K
    diag = tl.load(A + 2 * diag_off, rm, 0)
    xo = rows.to(tl.int64) * (2 * INCX)
    xr = tl.load(X + xo, rm, 0)
    xi = tl.load(X + xo + 1, rm, 0)
    sr = tl.sum(acc_r, axis=1) + diag * xr
    si = tl.sum(acc_i, axis=1) + diag * xi
    rr = AR * sr - AI * si
    ri = AR * si + AI * sr
    yo = rows.to(tl.int64) * (2 * INCY)
    if not BETA_ZERO:
        yr = tl.load(Y + yo, rm, 0)
        yi = tl.load(Y + yo + 1, rm, 0)
        rr += BR * yr - BI * yi
        ri += BR * yi + BI * yr
    tl.store(Y + yo, rr, rm)
    tl.store(Y + yo + 1, ri, rm)


def chbmv(
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
    if k <= 4:
        _common_chbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
        return
    assert A.dtype == torch.complex64 == x.dtype == y.dtype
    _check_common(A, x, y, uplo, n, k, lda, incx, incy)
    if n == 0:
        return
    ar, ai, br, bi = _complex_scalars(alpha, beta)
    if ar == 0.0 and ai == 0.0:
        y_view = _strided_y(y, n, incy)
        if br == 0.0 and bi == 0.0:
            y_view.zero_()
        elif br != 1.0 or bi != 0.0:
            y_view.mul_(complex(br, bi))
        return

    effective_k = min(k, n - 1)
    band = min(256, triton.next_power_of_2(max(1, effective_k)))
    # MTT S5000 measurements: narrow bands benefit from grouping rows;
    # wide bands need more independent programs and fewer live values.
    if band <= 16:
        rows_per_program = 8
    elif band <= 64 and n > 512:
        rows_per_program = 4
    else:
        rows_per_program = 1
    with torch_device_fn.device(A.device):
        _chbmv_band[(triton.cdiv(n, rows_per_program),)](
            torch.view_as_real(A),
            torch.view_as_real(x),
            torch.view_as_real(y),
            ar,
            ai,
            br,
            bi,
            n,
            K=effective_k,
            STORED_K=k,
            LDA=lda,
            INCX=incx,
            INCY=incy,
            UPLO=uplo,
            BETA_ZERO=br == 0.0 and bi == 0.0,
            M=rows_per_program,
            B=band,
            num_warps=4,
            num_stages=1,
        )
