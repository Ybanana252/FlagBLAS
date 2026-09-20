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

from flag_blas.ops.level2.hpmv import (
    ScalarType,
    _check_common,
    _complex_scalars,
    _strided_y,
)
from flag_blas.runtime import torch_device_fn


@triton.jit
def _chpmv_partial(
    AP,
    X,
    Y,
    P,
    AR,
    AI,
    BR,
    BI,
    N: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    UPLO: tl.constexpr,
    BETA_ZERO: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    SPLITS: tl.constexpr,
):
    rows = tl.program_id(0) * M + tl.arange(0, M)
    split = tl.program_id(1)
    chunk: tl.constexpr = triton.cdiv(N, SPLITS * K) * K
    ks = tl.arange(0, K)
    # Bound the intermediate row-base products, not just the final offsets.
    if N <= 32767:
        r = rows
    else:
        r = rows.to(tl.int64)
    if UPLO == 0:
        row_base = r * (r + 1) // 2
    else:
        row_base = r * (2 * N - r - 1) // 2
    acc_r = tl.full((M, K), 0, tl.float32)
    acc_i = tl.full((M, K), 0, tl.float32)
    for kb in range(chunk // K):
        cols = split * chunk + kb * K + ks
        if N <= 32767:
            c = cols
        else:
            c = cols.to(tl.int64)
        lower = rows[:, None] >= cols[None, :]
        if UPLO == 0:
            col_base = c * (c + 1) // 2
            off = tl.where(
                lower,
                row_base[:, None] + c[None, :],
                col_base[None, :] + r[:, None],
            )
            conj = ~lower
        else:
            col_base = c * (2 * N - c - 1) // 2
            off = tl.where(
                lower,
                col_base[None, :] + r[:, None],
                row_base[:, None] + c[None, :],
            )
            conj = lower
        mask = (rows[:, None] < N) & (cols[None, :] < N)
        # Convert before doubling so the 64-bit path also handles large AP.
        off = off.to(tl.int64) * 2
        ar = tl.load(AP + off, mask, 0)
        ai = tl.load(AP + off + 1, mask, 0)
        ai = tl.where(conj, -ai, ai)
        ai = tl.where(rows[:, None] == cols[None, :], 0.0, ai)
        xo = cols.to(tl.int64) * INCX * 2
        xr = tl.load(X + xo, cols < N, 0)
        xi = tl.load(X + xo + 1, cols < N, 0)
        acc_r += ar * xr[None, :] - ai * xi[None, :]
        acc_i += ar * xi[None, :] + ai * xr[None, :]
    sr = tl.sum(acc_r, 1)
    si = tl.sum(acc_i, 1)
    if SPLITS == 1:
        rr = AR * sr - AI * si
        ri = AR * si + AI * sr
        yo = rows.to(tl.int64) * INCY * 2
        if not BETA_ZERO:
            yr = tl.load(Y + yo, rows < N, 0)
            yi = tl.load(Y + yo + 1, rows < N, 0)
            rr += BR * yr - BI * yi
            ri += BR * yi + BI * yr
        tl.store(Y + yo, rr, rows < N)
        tl.store(Y + yo + 1, ri, rows < N)
    else:
        po = (split * N + rows) * 2
        tl.store(P + po, sr, rows < N)
        tl.store(P + po + 1, si, rows < N)


@triton.jit
def _chpmv_finish(
    P,
    Y,
    AR,
    AI,
    BR,
    BI,
    N: tl.constexpr,
    INCY: tl.constexpr,
    BETA_ZERO: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, triton.next_power_of_2(SPLITS))
    off = (splits[:, None] * N + rows[None, :]) * 2
    mask = (splits[:, None] < SPLITS) & (rows[None, :] < N)
    sr = tl.sum(tl.load(P + off, mask, 0), 0)
    si = tl.sum(tl.load(P + off + 1, mask, 0), 0)
    rr = AR * sr - AI * si
    ri = AR * si + AI * sr
    yo = rows.to(tl.int64) * INCY * 2
    if not BETA_ZERO:
        yr = tl.load(Y + yo, rows < N, 0)
        yi = tl.load(Y + yo + 1, rows < N, 0)
        rr += BR * yr - BI * yi
        ri += BR * yi + BI * yr
    tl.store(Y + yo, rr, rows < N)
    tl.store(Y + yo + 1, ri, rows < N)


def chpmv(
    uplo: int,
    n: int,
    alpha: ScalarType,
    AP: torch.Tensor,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    assert AP.dtype == torch.complex64 == x.dtype == y.dtype
    _check_common(AP, x, y, uplo, n, incx, incy)
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
    # Measured on MTT S5000: small row groups expose enough parallelism;
    # large matrices benefit from shorter independent K reductions.
    if n <= 512:
        m, k, splits, warps = 1, 256, 1, 4
    elif n <= 2048:
        m, k, splits, warps = 2, 256, 1, 4
    elif n <= 4096:
        m, k, splits, warps = 8, 128, 1, 4
    else:
        m, k, splits, warps = 16, 128, 8, 4
    with torch_device_fn.device(AP.device):
        partial = (
            torch.empty((splits, n, 2), device=AP.device, dtype=torch.float32)
            if splits > 1
            else torch.view_as_real(y)
        )
        _chpmv_partial[(triton.cdiv(n, m), splits)](
            torch.view_as_real(AP),
            torch.view_as_real(x),
            torch.view_as_real(y),
            partial,
            ar,
            ai,
            br,
            bi,
            n,
            incx,
            incy,
            uplo,
            br == 0.0 and bi == 0.0,
            m,
            k,
            splits,
            num_warps=warps,
            num_stages=1,
        )
        if splits > 1:
            _chpmv_finish[(triton.cdiv(n, 128),)](
                partial,
                torch.view_as_real(y),
                ar,
                ai,
                br,
                bi,
                n,
                incy,
                br == 0.0 and bi == 0.0,
                splits,
                128,
                num_warps=4,
            )
