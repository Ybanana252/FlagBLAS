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

"""Single-launch symmetric band matrix-vector product for Ascend."""

import torch
import triton
import triton.language as tl

from flag_blas.ops.level2.sbmv import _check_common, _strided_y
from flag_blas.ops.level2.sbmv import ssbmv as _common_ssbmv
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

from .hpmv import _current_device, _launch


@libentry()
@triton.jit(do_not_specialize=["N"])
def _ssbmv_narrow(
    A, X, Y,
    ALPHA: tl.float32, BETA: tl.float32, N,
    K: tl.constexpr, LDA: tl.constexpr,
    INCX: tl.constexpr, INCY: tl.constexpr,
    UPLO: tl.constexpr, BLOCK: tl.constexpr, BETA_ZERO: tl.constexpr,
):
    # The diagonal-only case is a contiguous scale-and-accumulate.
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid_row = rows < N
    acc = tl.full((BLOCK,), 0.0, tl.float32)
    for d in tl.static_range(-K, K + 1):
        cols = rows + d
        valid = valid_row & (cols >= 0) & (cols < N)
        if UPLO == 0:
            stored_row = tl.maximum(rows, cols)
            band = K - (d if d >= 0 else -d)
        else:
            stored_row = tl.minimum(rows, cols)
            band = d if d >= 0 else -d
        av = tl.load(A + stored_row * LDA + band, valid, 0.0)
        xv = tl.load(X + cols * INCX, valid, 0.0)
        acc += av * xv
    out = ALPHA * acc
    if not BETA_ZERO:
        out += BETA * tl.load(Y + rows * INCY, valid_row, 0.0)
    tl.store(Y + rows * INCY, out, valid_row)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _ssbmv_slab(
    A, X, Y,
    ALPHA: tl.float32, BETA: tl.float32, N,
    K: tl.constexpr, LDA: tl.constexpr,
    INCX: tl.constexpr, INCY: tl.constexpr,
    UPLO: tl.constexpr, BLOCK: tl.constexpr,
    BUFFER: tl.constexpr, BETA_ZERO: tl.constexpr,
):
    # Ascend scalarizes irregular global-memory gathers. Read a contiguous
    # slab and gather the neighbouring band elements from UB instead.
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # Align the contiguous DMA start to 32 bytes. For k=4, starting at the
    # first halo row would otherwise put a 5-float LDA at a 16-byte offset.
    base = tl.program_id(0) * BLOCK * LDA - triton.cdiv(K * LDA, 8) * 8
    offsets = base + tl.arange(0, BUFFER)
    slab = tl.load(A + offsets, (offsets >= 0) & (offsets < N * LDA), 0.0)
    acc = tl.full((BLOCK,), 0.0, tl.float32)
    for d in tl.static_range(-K, K + 1):
        cols = rows + d
        valid = (rows < N) & (cols >= 0) & (cols < N)
        if UPLO == 0:
            stored = rows + (d if d > 0 else 0)
            band = K - (d if d >= 0 else -d)
        else:
            stored = rows + (d if d < 0 else 0)
            band = d if d >= 0 else -d
        index = stored * LDA + band - base
        av = tl.gather(slab, index, 0)
        av = tl.where(valid, av, 0.0)
        xv = tl.load(X + cols * INCX, valid, 0.0)
        acc += av * xv
    out = ALPHA * acc
    if not BETA_ZERO:
        out += BETA * tl.load(Y + rows * INCY, rows < N, 0.0)
    tl.store(Y + rows * INCY, out, rows < N)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _ssbmv_band(
    A, X, Y,
    ALPHA: tl.float32, BETA: tl.float32, N,
    K: tl.constexpr, LDA: tl.constexpr,
    INCX: tl.constexpr, INCY: tl.constexpr,
    UPLO: tl.constexpr, BLOCK: tl.constexpr, BETA_ZERO: tl.constexpr,
):
    rb = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0.0, tl.float32)
    reach: tl.constexpr = triton.cdiv(K, BLOCK)
    for cb in range(
        tl.maximum(0, rb - reach), tl.minimum(tl.cdiv(N, BLOCK), rb + reach + 1)
    ):
        direct = rb >= cb if UPLO == 0 else rb <= cb
        stored_r = tl.where(direct, rb, cb)
        stored_c = tl.where(direct, cb, rb)
        rows = stored_r * BLOCK + lanes
        cols = stored_c * BLOCK + lanes
        if UPLO == 0:
            base = rows * (LDA - 1) + K
            valid = (cols[None, :] >= rows[:, None] - K) & (
                cols[None, :] <= rows[:, None]
            )
        else:
            base = rows * (LDA - 1)
            valid = (cols[None, :] >= rows[:, None]) & (
                cols[None, :] <= rows[:, None] + K
            )
        av = tl.load(
            A + base[:, None] + cols[None, :],
            valid & (rows[:, None] < N) & (cols[None, :] < N), 0.0,
        )
        xv = tl.load(
            X + (cb * BLOCK + lanes) * INCX, cb * BLOCK + lanes < N, 0.0
        )
        if direct:
            acc += tl.sum(av * xv[None, :], 1)
        if (not direct) or rb == cb:
            if rb == cb:
                av = tl.where(rows[:, None] == cols[None, :], 0.0, av)
            acc += tl.sum(av * xv[:, None], 0)
    out_rows = rb * BLOCK + lanes
    out = ALPHA * acc
    if not BETA_ZERO:
        out += BETA * tl.load(Y + out_rows * INCY, out_rows < N, 0.0)
    tl.store(Y + out_rows * INCY, out, out_rows < N)


def ssbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
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
    if incx != 1 or incy != 1 or k > 256:
        return _common_ssbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    if A.device.index != _current_device():
        with torch_device_fn.device(A.device):
            return ssbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)

    beta_zero = beta == 0.0
    if k == 0:
        block = 128
        _launch(
            _ssbmv_narrow, ((n + block - 1) // block,),
            (A, x, y), (alpha, beta, n),
            (k, lda, incx, incy, uplo, block, beta_zero),
        )
    elif k in (1, 4) and (128 + 2 * k) * lda <= 4096:
        block = 128
        elements = (block + 2 * k) * lda
        buffer = 1 << (elements - 1).bit_length()
        _launch(
            _ssbmv_slab, ((n + block - 1) // block,),
            (A, x, y), (alpha, beta, n),
            (k, lda, incx, incy, uplo, block, buffer, beta_zero),
        )
    else:
        block = 32
        _launch(
            _ssbmv_band, ((n + block - 1) // block,),
            (A, x, y), (alpha, beta, n),
            (k, lda, incx, incy, uplo, block, beta_zero),
        )
