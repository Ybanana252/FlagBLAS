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

from flag_blas.ops.level2.spmv import ScalarType, _check_common, _strided_y
from flag_blas.runtime import torch_device_fn


@triton.jit
def _sspmv_partial(
    AP,
    X,
    Y,
    P,
    ALPHA,
    BETA,
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
    # Include padded rows/columns when bounding intermediate index products.
    INDEX32: tl.constexpr = triton.cdiv(N, M) * M <= 32768 and chunk * SPLITS <= 32768
    if INDEX32:
        r = rows
    else:
        r = rows.to(tl.int64)
    if UPLO == 0:
        row_base = r * (r + 1) // 2
    else:
        row_base = r * (2 * N - r - 1) // 2
    acc = tl.full((M, K), 0, tl.float32)
    for kb in range(chunk // K):
        cols = split * chunk + kb * K + ks
        if INDEX32:
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
        else:
            col_base = c * (2 * N - c - 1) // 2
            off = tl.where(
                lower,
                col_base[None, :] + r[:, None],
                row_base[:, None] + c[None, :],
            )
        mask = (rows[:, None] < N) & (cols[None, :] < N)
        av = tl.load(AP + off.to(tl.int64), mask, 0)
        xv = tl.load(X + cols.to(tl.int64) * INCX, cols < N, 0)
        acc += av * xv[None, :]
    result = tl.sum(acc, 1)
    if SPLITS == 1:
        out = ALPHA * result
        yp = Y + rows.to(tl.int64) * INCY
        if not BETA_ZERO:
            out += BETA * tl.load(yp, rows < N, 0)
        tl.store(yp, out, rows < N)
    else:
        tl.store(P + split.to(tl.int64) * N + rows, result, rows < N)


@triton.jit
def _sspmv_finish(
    P,
    Y,
    ALPHA,
    BETA,
    N: tl.constexpr,
    INCY: tl.constexpr,
    BETA_ZERO: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    splits = tl.arange(0, triton.next_power_of_2(SPLITS))
    off = splits[:, None].to(tl.int64) * N + rows[None, :]
    mask = (splits[:, None] < SPLITS) & (rows[None, :] < N)
    result = tl.sum(tl.load(P + off, mask, 0), 0)
    out = ALPHA * result
    yp = Y + rows.to(tl.int64) * INCY
    if not BETA_ZERO:
        out += BETA * tl.load(yp, rows < N, 0)
    tl.store(yp, out, rows < N)


def _select_config(n, uplo):
    # MTT S5000 measurements: grow row groups as the matrix grows, then
    # split long reductions. One configuration serves both packed triangles.
    if n <= 512:
        return 1, 128, 1, 4
    if n <= 1024:
        return 2, 128, 1, 4
    if n <= 2048:
        return 4, 128, 1, 4
    if n <= 4096:
        return 8, 128, 1, 4
    if n <= 6144:
        return 16, 128, 1, 4
    if n <= 8192:
        return 32, 128, 1, 4
    return 32, 128, 4, 4


def sspmv(
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
    assert AP.dtype == torch.float32 == x.dtype == y.dtype
    _check_common(AP, x, y, uplo, n, incx, incy)
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

    # UPLO follows the public row-packed interface; no column-major flip.
    m, k, splits, warps = _select_config(n, uplo)
    with torch_device_fn.device(AP.device):
        partial = (
            torch.empty((splits, n), device=AP.device, dtype=torch.float32)
            if splits > 1
            else y
        )
        _sspmv_partial[(triton.cdiv(n, m), splits)](
            AP,
            x,
            y,
            partial,
            alpha,
            beta,
            n,
            incx,
            incy,
            uplo,
            beta == 0.0,
            m,
            k,
            splits,
            num_warps=warps,
            num_stages=1,
        )
        if splits > 1:
            _sspmv_finish[(triton.cdiv(n, 128),)](
                partial,
                y,
                alpha,
                beta,
                n,
                incy,
                beta == 0.0,
                splits,
                128,
                num_warps=4,
                num_stages=1,
            )
