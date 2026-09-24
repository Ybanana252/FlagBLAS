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

"""Ascend triangular band matrix-vector products in row-major storage.

Band storage A[r, b] is addressed in logical matrix coordinates as
``r * (lda - 1) + c + (k if lower else 0)``. Both transpose modes therefore
load consecutive columns of stored rows; only tile ownership and the reduction
axis change. Complex values stay interleaved in global-memory transfers.

Small kernels retain the input vector in UB or visit strips in dependency-safe
order. Parallel matrix tiles snapshot the vector to preserve in-place semantics.
"""

import importlib

import torch
import triton
import triton.language as tl

from flag_blas.ops.level2._constants import CUBLAS_DIAG_UNIT
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

from .hpmv import _current_device, _launch

_common = importlib.import_module("flag_blas.ops.level2.tbmv")


@libentry()
@triton.jit(do_not_specialize=["N"])
def _tbmv_narrow(
    A,
    X,
    Y,
    N,
    K: tl.constexpr,
    LDA: tl.constexpr,
    INCX: tl.constexpr,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    COMPLEX: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK: tl.constexpr,
    BUFFER: tl.constexpr,
):
    channels: tl.constexpr = 2 if COMPLEX else 1
    lanes = tl.arange(0, BLOCK * channels)
    rb = tl.program_id(0) * BLOCK
    rows = rb + lanes // channels
    yf = rb * channels + lanes
    # Align the DMA origin, including the halo, to a 32-byte UB block.
    start = rb * LDA * channels - triton.cdiv(K * LDA * channels, 8) * 8
    offsets = start + tl.arange(0, BUFFER)
    slab = tl.load(A + offsets, (offsets >= 0) & (offsets < N * LDA * channels), 0.0)
    acc = tl.full((BLOCK * channels,), 0.0, tl.float32)
    for distance in tl.static_range(K + 1):
        d = -distance if (UPLO == 0) != TRANS else distance
        cols = rows + d
        xf = yf + d * channels
        stored_row = cols if TRANS else rows
        band = (-d if TRANS else d) + (K if UPLO == 0 else 0)
        valid = (yf < N * channels) & (xf >= 0) & (xf < N * channels)
        if UNIT and distance == 0:
            av = tl.full((BLOCK * channels,), 1.0, tl.float32)
            if COMPLEX:
                av = tl.where((lanes & 1) == 0, 1.0, 0.0)
        else:
            av = tl.gather(
                slab, (stored_row * LDA + band) * channels + lanes % channels - start, 0
            )
        av = tl.where(valid, av, 0.0)
        xv = tl.load(X + xf, valid, 0.0)
        if COMPLEX:
            odd = (lanes & 1) != 0
            if CONJ:
                av = tl.where(odd, -av, av)
            xr = tl.gather(xv, lanes & ~1, 0)
            xi = tl.gather(xv, lanes | 1, 0)
            acc += av * xr + tl.where(odd, 1.0, -1.0) * tl.gather(av, lanes ^ 1, 0) * xi
        else:
            acc += av * xv
    out = yf if INCX == 1 else rows * INCX * channels + lanes % channels
    tl.store(Y + out, acc, yf < N * channels)


def narrow(uplo, trans, diag, n, k, A, lda, x, incx):
    channels = 2 if x.is_complex() else 1
    block = 1 << (n - 1).bit_length() if n <= 256 else 128
    # One program reads the entire vector before writing; otherwise snapshot X.
    xin = (
        x
        if (n <= block or k == 0) and incx == 1
        else x.as_strided((n,), (incx,)).clone()
    )
    _launch(
        _tbmv_narrow,
        ((n + block - 1) // block,),
        (A, xin, x),
        (n,),
        (
            k,
            lda,
            incx,
            uplo,
            trans != 0,
            diag == 1,
            channels == 2,
            trans == 2,
            block,
            1 << ((block + 2 * k) * lda * channels + 7).bit_length(),
        ),
    )
    return x



@libentry()
@triton.jit(do_not_specialize=["N"])
def _tbmv_ordered(
    A,
    X,
    N,
    K: tl.constexpr,
    LDA: tl.constexpr,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    COMPLEX: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK: tl.constexpr,
    BUFFER: tl.constexpr,
):
    """Process output strips in a dependency-safe order without snapshotting X."""
    channels: tl.constexpr = 2 if COMPLEX else 1
    lanes = tl.arange(0, BLOCK * channels)
    blocks = tl.cdiv(N, BLOCK)
    lower: tl.constexpr = (UPLO == 0) != TRANS
    for step in tl.range(0, blocks):
        bid = blocks - 1 - step if lower else step
        rb = bid * BLOCK
        rows = rb + lanes // channels
        yf = rb * channels + lanes
        start = rb * LDA * channels - triton.cdiv(K * LDA * channels, 8) * 8
        offsets = start + tl.arange(0, BUFFER)
        slab = tl.load(
            A + offsets,
            (offsets >= 0) & (offsets < N * LDA * channels),
            0.0,
        )
        acc = tl.full((BLOCK * channels,), 0.0, tl.float32)
        for distance in tl.static_range(K + 1):
            d = -distance if (UPLO == 0) != TRANS else distance
            cols = rows + d
            xf = yf + d * channels
            stored_row = cols if TRANS else rows
            band = (-d if TRANS else d) + (K if UPLO == 0 else 0)
            valid = (yf < N * channels) & (xf >= 0) & (xf < N * channels)
            if UNIT and distance == 0:
                av = tl.full((BLOCK * channels,), 1.0, tl.float32)
                if COMPLEX:
                    av = tl.where((lanes & 1) == 0, 1.0, 0.0)
            else:
                av = tl.gather(
                    slab,
                    (stored_row * LDA + band) * channels
                    + lanes % channels
                    - start,
                    0,
                )
            av = tl.where(valid, av, 0.0)
            xv = tl.load(X + xf, valid, 0.0)
            if COMPLEX:
                odd = (lanes & 1) != 0
                if CONJ:
                    av = tl.where(odd, -av, av)
                xr = tl.gather(xv, lanes & ~1, 0)
                xi = tl.gather(xv, lanes | 1, 0)
                acc += (
                    av * xr
                    + tl.where(odd, 1.0, -1.0)
                    * tl.gather(av, lanes ^ 1, 0)
                    * xi
                )
            else:
                acc += av * xv
        tl.store(X + yf, acc, yf < N * channels)


def ordered(uplo, trans, diag, n, k, A, lda, x, incx):
    assert incx == 1
    channels = 2 if x.is_complex() else 1
    if channels == 2:
        block = 512 if n == 1024 or k == 4 or n >= 8192 else 256
    elif n == 1024 or n > 4096:
        block = 1024
    elif k == 4:
        block = 512
    else:
        block = 256
    _launch(
        _tbmv_ordered,
        (1,),
        (A, x),
        (n,),
        (
            k,
            lda,
            uplo,
            trans != 0,
            diag == 1,
            channels == 2,
            trans == 2,
            block,
            1 << ((block + 2 * k) * lda * channels + 7).bit_length(),
        ),
    )
    return x

@libentry()
@triton.jit(do_not_specialize=["N"])
def _tbmv_matrix_tiles(
    A,
    X,
    Y,
    N,
    K: tl.constexpr,
    LDA: tl.constexpr,
    INCX: tl.constexpr,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    COMPLEX: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK: tl.constexpr,
):
    rb = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    reach: tl.constexpr = triton.cdiv(K, BLOCK)
    lower: tl.constexpr = (UPLO == 0) != TRANS
    begin = tl.maximum(0, rb - reach) if lower else rb
    end = rb + 1 if lower else tl.minimum(tl.cdiv(N, BLOCK), rb + reach + 1)
    if COMPLEX:
        floats = tl.arange(0, 2 * BLOCK)
        parity = floats & 1
        sign = tl.where(parity == 0, 1.0, -1.0)
        acc = tl.full((2 * BLOCK,), 0.0, tl.float32)
    else:
        acc = tl.full((BLOCK,), 0.0, tl.float32)
    for cb in range(begin, end):
        sr = cb if TRANS else rb
        sc = rb if TRANS else cb
        rows = sr * BLOCK + lanes
        base = rows * (LDA - 1) + (K if UPLO == 0 else 0)
        if COMPLEX:
            cf = sc * BLOCK * 2 + floats
            cols = sc * BLOCK + (floats >> 1)
            # Express load masks in consecutive float coordinates too: using
            # complex-element coordinates here scalarizes the Ascend loads.
            if UPLO == 0:
                valid = (cf[None, :] >= (rows[:, None] - K) * 2) & (
                    cf[None, :] < (rows[:, None] + 1) * 2
                )
            else:
                valid = (cf[None, :] >= rows[:, None] * 2) & (
                    cf[None, :] < (rows[:, None] + K + 1) * 2
                )
            av = tl.load(
                A + base[:, None] * 2 + cf[None, :],
                valid & (rows[:, None] < N) & (cf[None, :] < N * 2),
                0.0,
            )
            if UNIT:
                av = tl.where(
                    rows[:, None] == cols[None, :],
                    tl.where(parity[None, :] == 0, 1.0, 0.0),
                    av,
                )
            if CONJ:
                av = av * sign[None, :]
            xv = tl.load(
                X + cb * BLOCK * 2 + floats, cb * BLOCK * 2 + floats < N * 2, 0.0
            )
            if TRANS:
                xr = tl.gather(xv, lanes * 2, 0)
                xi = tl.gather(xv, lanes * 2 + 1, 0)
                mr = tl.sum(av * xr[:, None], 0)
                mi = tl.sum(av * xi[:, None], 0)
                acc += mr - sign * tl.gather(mi, floats ^ 1, 0)
            else:
                real = tl.sum(av * (xv * sign)[None, :], 1)
                imag = tl.sum(av * tl.gather(xv, floats ^ 1, 0)[None, :], 1)
                acc += tl.where(
                    parity == 0,
                    tl.gather(real, floats >> 1, 0),
                    tl.gather(imag, floats >> 1, 0),
                )
        else:
            cols = sc * BLOCK + lanes
            if UPLO == 0:
                valid = (cols[None, :] >= rows[:, None] - K) & (
                    cols[None, :] <= rows[:, None]
                )
            else:
                valid = (cols[None, :] >= rows[:, None]) & (
                    cols[None, :] <= rows[:, None] + K
                )
            av = tl.load(
                A + base[:, None] + cols[None, :],
                valid & (rows[:, None] < N) & (cols[None, :] < N),
                0.0,
            )
            if UNIT:
                av = tl.where(rows[:, None] == cols[None, :], 1.0, av)
            xv = tl.load(X + cb * BLOCK + lanes, cb * BLOCK + lanes < N, 0.0)
            if TRANS:
                acc += tl.sum(av * xv[:, None], 0)
            else:
                acc += tl.sum(av * xv[None, :], 1)
    if COMPLEX:
        offsets = (
            rb * BLOCK * 2 + floats
            if INCX == 1
            else (rb * BLOCK + (floats >> 1)) * INCX * 2 + parity
        )
        tl.store(Y + offsets, acc, rb * BLOCK * 2 + floats < N * 2)
    else:
        rows = rb * BLOCK + lanes
        tl.store(Y + rows * INCX, acc, rows < N)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _tbmv_small_slab(
    A,
    X,
    N,
    K: tl.constexpr,
    LDA: tl.constexpr,
    INCX: tl.constexpr,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    COMPLEX: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK: tl.constexpr,
    BUFFER: tl.constexpr,
):
    # A single program owns all of X. Load it before any output store so that
    # the BLAS in-place update needs neither a temporary vector nor a barrier
    # between programs. All irregular A/X accesses below are local gathers.
    channels: tl.constexpr = 2 if COMPLEX else 1
    lanes = tl.arange(0, BLOCK * channels)
    rows = lanes // channels
    offsets = tl.arange(0, BUFFER)
    slab = tl.load(A + offsets, offsets < N * LDA * channels, 0.0)
    xoff = lanes if INCX == 1 else rows * INCX * channels + lanes % channels
    xin = tl.load(X + xoff, lanes < N * channels, 0.0)
    acc = tl.full((BLOCK * channels,), 0.0, tl.float32)
    for distance in range(K + 1):
        d = -distance if (UPLO == 0) != TRANS else distance
        cols = rows + d
        sr = cols if TRANS else rows
        band = (-d if TRANS else d) + (K if UPLO == 0 else 0)
        index = (
            tl.minimum(tl.maximum(sr, 0), N - 1) * LDA + band
        ) * channels + lanes % channels
        av = tl.gather(slab, index, 0)
        if UNIT:
            av = tl.where(distance == 0, tl.where(lanes % channels == 0, 1.0, 0.0), av)
        valid = (rows < N) & (cols >= 0) & (cols < N)
        av = tl.where(valid, av, 0.0)
        xi = tl.minimum(tl.maximum(cols, 0), N - 1) * channels + lanes % channels
        xv = tl.gather(xin, xi, 0)
        if COMPLEX:
            odd = (lanes & 1) != 0
            if CONJ:
                av = tl.where(odd, -av, av)
            xr = tl.gather(xv, lanes & ~1, 0)
            xim = tl.gather(xv, lanes | 1, 0)
            acc += (
                av * xr + tl.where(odd, 1.0, -1.0) * tl.gather(av, lanes ^ 1, 0) * xim
            )
        else:
            acc += av * xv
    tl.store(X + xoff, acc, lanes < N * channels)


def small_slab(uplo, trans, diag, n, k, A, lda, x, incx):
    channels = 2 if x.is_complex() else 1
    _launch(
        _tbmv_small_slab,
        (1,),
        (A, x),
        (n,),
        (
            k,
            lda,
            incx,
            uplo,
            trans != 0,
            diag == 1,
            channels == 2,
            trans == 2,
            1 << (n - 1).bit_length(),
            1 << (n * lda * channels - 1).bit_length(),
        ),
    )
    return x


def matrix_tiles(uplo, trans, diag, n, k, A, lda, x, incx, block=None):
    if block is None:
        # Transposed complex reductions have two full-tile intermediates; a
        # 64x128-float tile exceeds the 910B vector core's local memory.
        block = 64 if k >= 128 and (not x.is_complex() or trans == 0) else 32
    # A snapshot is necessary because the public BLAS operation updates X in place.
    xin = x.as_strided((n,), (incx,)).clone()
    _launch(
        _tbmv_matrix_tiles,
        ((n + block - 1) // block,),
        (A, xin, x),
        (n,),
        (k, lda, incx, uplo, trans != 0, diag == 1, x.is_complex(), trans == 2, block),
    )
    return x


@libentry()
@triton.jit(do_not_specialize=["N"])
def _small_unrolled(
    A,
    X,
    N,
    K: tl.constexpr,
    LDA: tl.constexpr,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    COMPLEX: tl.constexpr,
    CONJ: tl.constexpr,
    B: tl.constexpr,
    BUFFER: tl.constexpr,
):
    channels: tl.constexpr = 2 if COMPLEX else 1
    rows = tl.arange(0, B)
    floats = tl.arange(0, B * channels)
    offsets = tl.arange(0, BUFFER)
    slab = tl.load(A + offsets, offsets < N * LDA * channels, 0.0)
    # Keep the complete original vector local until the final store.
    xin = tl.load(X + floats, floats < N * channels, 0.0)
    real = tl.full((B,), 0.0, tl.float32)
    if COMPLEX:
        imag = tl.full((B,), 0.0, tl.float32)
    for distance in tl.static_range(K + 1):
        d = -distance if (UPLO == 0) != TRANS else distance
        cols = rows + d
        stored_row = cols if TRANS else rows
        band = (-d if TRANS else d) + (K if UPLO == 0 else 0)
        index = (
            tl.minimum(tl.maximum(stored_row, 0), N - 1) * LDA + band
        ) * channels
        valid = (rows < N) & (cols >= 0) & (cols < N)
        xi = tl.minimum(tl.maximum(cols, 0), N - 1) * channels
        xr = tl.gather(xin, xi, 0)
        if UNIT and distance == 0:
            ar = tl.full((B,), 1.0, tl.float32)
            if COMPLEX:
                ai = tl.full((B,), 0.0, tl.float32)
        else:
            ar = tl.gather(slab, index, 0)
            if COMPLEX:
                ai = tl.gather(slab, index + 1, 0)
        ar = tl.where(valid, ar, 0.0)
        if COMPLEX:
            ai = tl.where(valid, ai, 0.0)
            if CONJ:
                ai = -ai
            xim = tl.gather(xin, xi + 1, 0)
            # Separate real/imaginary accumulators avoid pair shuffles at
            # every band distance.
            real += ar * xr - ai * xim
            imag += ar * xim + ai * xr
        else:
            real += ar * xr
    if COMPLEX:
        out = tl.where(
            floats % 2 == 0,
            tl.gather(real, floats // 2, 0),
            tl.gather(imag, floats // 2, 0),
        )
    else:
        out = real
    tl.store(X + floats, out, floats < N * channels)


def small_unrolled(uplo, trans, diag, n, k, A, lda, x, incx):
    assert incx == 1
    channels = 2 if x.is_complex() else 1
    _launch(
        _small_unrolled,
        (1,),
        (A, x),
        (n,),
        (
            k,
            lda,
            uplo,
            trans != 0,
            diag == 1,
            channels == 2,
            trans == 2,
            1 << (n - 1).bit_length(),
            1 << (n * lda * channels - 1).bit_length(),
        ),
    )
    return x


@libentry()
@triton.jit(do_not_specialize=["N"])
def _ordered_tiles(
    A,
    X,
    Y,
    N,
    K: tl.constexpr,
    LDA: tl.constexpr,
    INCX: tl.constexpr,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    COMPLEX: tl.constexpr,
    CONJ: tl.constexpr,
    BLOCK: tl.constexpr,
):
    blocks = tl.cdiv(N, BLOCK)
    for step in range(blocks):
        rb = blocks - 1 - step if ((UPLO == 0) != TRANS) else step
        lanes = tl.arange(0, BLOCK)
        reach: tl.constexpr = triton.cdiv(K, BLOCK)
        lower: tl.constexpr = (UPLO == 0) != TRANS
        begin = tl.maximum(0, rb - reach) if lower else rb
        end = rb + 1 if lower else tl.minimum(tl.cdiv(N, BLOCK), rb + reach + 1)
        if COMPLEX:
            floats = tl.arange(0, 2 * BLOCK)
            parity = floats & 1
            sign = tl.where(parity == 0, 1.0, -1.0)
            acc = tl.full((2 * BLOCK,), 0.0, tl.float32)
        else:
            acc = tl.full((BLOCK,), 0.0, tl.float32)
        for cb in range(begin, end):
            sr = cb if TRANS else rb
            sc = rb if TRANS else cb
            rows = sr * BLOCK + lanes
            base = rows * (LDA - 1) + (K if UPLO == 0 else 0)
            if COMPLEX:
                cf = sc * BLOCK * 2 + floats
                cols = sc * BLOCK + (floats >> 1)
                # Express load masks in consecutive float coordinates too: using
                # complex-element coordinates here scalarizes the Ascend loads.
                if UPLO == 0:
                    valid = (cf[None, :] >= (rows[:, None] - K) * 2) & (
                        cf[None, :] < (rows[:, None] + 1) * 2
                    )
                else:
                    valid = (cf[None, :] >= rows[:, None] * 2) & (
                        cf[None, :] < (rows[:, None] + K + 1) * 2
                    )
                av = tl.load(
                    A + base[:, None] * 2 + cf[None, :],
                    valid & (rows[:, None] < N) & (cf[None, :] < N * 2),
                    0.0,
                )
                if UNIT:
                    av = tl.where(
                        rows[:, None] == cols[None, :],
                        tl.where(parity[None, :] == 0, 1.0, 0.0),
                        av,
                    )
                if CONJ:
                    av = av * sign[None, :]
                xv = tl.load(
                    X + cb * BLOCK * 2 + floats, cb * BLOCK * 2 + floats < N * 2, 0.0
                )
                if TRANS:
                    xr = tl.gather(xv, lanes * 2, 0)
                    xi = tl.gather(xv, lanes * 2 + 1, 0)
                    mr = tl.sum(av * xr[:, None], 0)
                    mi = tl.sum(av * xi[:, None], 0)
                    acc += mr - sign * tl.gather(mi, floats ^ 1, 0)
                else:
                    real = tl.sum(av * (xv * sign)[None, :], 1)
                    imag = tl.sum(av * tl.gather(xv, floats ^ 1, 0)[None, :], 1)
                    acc += tl.where(
                        parity == 0,
                        tl.gather(real, floats >> 1, 0),
                        tl.gather(imag, floats >> 1, 0),
                    )
            else:
                cols = sc * BLOCK + lanes
                if UPLO == 0:
                    valid = (cols[None, :] >= rows[:, None] - K) & (
                        cols[None, :] <= rows[:, None]
                    )
                else:
                    valid = (cols[None, :] >= rows[:, None]) & (
                        cols[None, :] <= rows[:, None] + K
                    )
                av = tl.load(
                    A + base[:, None] + cols[None, :],
                    valid & (rows[:, None] < N) & (cols[None, :] < N),
                    0.0,
                )
                if UNIT:
                    av = tl.where(rows[:, None] == cols[None, :], 1.0, av)
                xv = tl.load(X + cb * BLOCK + lanes, cb * BLOCK + lanes < N, 0.0)
                if TRANS:
                    acc += tl.sum(av * xv[:, None], 0)
                else:
                    acc += tl.sum(av * xv[None, :], 1)
        if COMPLEX:
            offsets = (
                rb * BLOCK * 2 + floats
                if INCX == 1
                else (rb * BLOCK + (floats >> 1)) * INCX * 2 + parity
            )
            tl.store(Y + offsets, acc, rb * BLOCK * 2 + floats < N * 2)
        else:
            rows = rb * BLOCK + lanes
            tl.store(Y + rows * INCX, acc, rows < N)


def ordered_tiles(uplo, trans, diag, n, k, A, lda, x, incx, block=None):
    assert incx == 1
    if block is None:
        # Transposed complex tiles need two full-tile intermediates in UB.
        block = 64 if trans == 0 else 32
    _launch(
        _ordered_tiles,
        (1,),
        (A, x, x),
        (n,),
        (k, lda, incx, uplo, trans != 0, diag == 1, x.is_complex(), trans == 2, block),
    )
    return x


def _compute(uplo, trans, diag, n, k, A, lda, x, incx):
    if n == 0 or (diag == CUBLAS_DIAG_UNIT and k == 0):
        return x
    if incx != 1:
        # Normalize the vector layout too. In particular, masked strided
        # stores at an odd tail are not reliable with the Ascend compiler.
        view = x.as_strided((n,), (incx,))
        logical_x = view.clone()
        _compute(uplo, trans, diag, n, k, A, lda, logical_x, 1)
        view.copy_(logical_x)
        return x
    # These kernels use the public row-major orientation directly. Do not apply
    # the column-major argument conversion used by the generic implementation.
    channels = 2 if x.is_complex() else 1
    # The complete-vector slab kernel is still best for small problems.  For
    # larger narrow bands, use ordered strips so the matrix does not consume
    # the whole UB and in-place dependencies remain valid without a snapshot.
    slab_floats = n * lda * channels
    vector_floats = n * channels
    if n <= 256 and k <= 4 and slab_floats + 2 * vector_floats <= 32768:
        implementation = small_unrolled
    elif n <= 256 and channels == 2 and k in (32, 48):
        implementation = ordered_tiles
    elif n <= 256 and slab_floats + 2 * vector_floats <= 32768:
        implementation = small_slab
    elif (
        k <= 4
        and n <= (16384 if channels == 1 else (8192 if k == 1 else 4096))
        and (256 + 2 * k) * lda * channels + 8 <= 8192
    ):
        implementation = ordered
    else:
        # Bound local slab storage even when the caller supplies a large LDA.
        implementation = matrix_tiles
    return implementation(uplo, trans, diag, n, k, A, lda, x, incx)


def stbmv(uplo, trans, diag, n, k, A, lda, x, incx):
    assert A.dtype == torch.float32 == x.dtype
    _common._check_tbmv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=False)
    if A.device.index != _current_device():
        with torch_device_fn.device(A.device):
            return _compute(uplo, trans, diag, n, k, A, lda, x, incx)
    return _compute(uplo, trans, diag, n, k, A, lda, x, incx)


def ctbmv(uplo, trans, diag, n, k, A, lda, x, incx):
    assert A.dtype == torch.complex64 == x.dtype
    _common._check_tbmv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=True)
    if A.device.index != _current_device():
        with torch_device_fn.device(A.device):
            return _compute(uplo, trans, diag, n, k, A, lda, x, incx)
    return _compute(uplo, trans, diag, n, k, A, lda, x, incx)
