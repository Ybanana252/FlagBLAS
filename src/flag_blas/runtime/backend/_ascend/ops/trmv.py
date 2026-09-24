# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Ascend TRMV kernels for row-major full triangular matrices."""

import importlib

import torch
import triton
import triton.language as tl

from flag_blas.ops.level2._constants import CUBLAS_DIAG_UNIT, CUBLAS_OP_C, CUBLAS_OP_N
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

from .trsv import _launch

_common = importlib.import_module("flag_blas.ops.level2.trmv")


@libentry()
@triton.jit
def _small(A, X, N: tl.constexpr, LDA: tl.constexpr, INCX: tl.constexpr,
           UPLO: tl.constexpr, TRANS: tl.constexpr, UNIT: tl.constexpr,
           CONJ: tl.constexpr, COMPLEX: tl.constexpr, B: tl.constexpr):
    r = tl.arange(0, B)
    c = tl.arange(0, B)
    valid = (r[:, None] < N) & (c[None, :] < N)
    if UPLO == TRANS:
        valid &= r[:, None] >= c[None, :]
    else:
        valid &= r[:, None] <= c[None, :]
    if UNIT:
        valid &= r[:, None] != c[None, :]
    if TRANS:
        aoff = c[None, :] * LDA + r[:, None]
    else:
        aoff = r[:, None] * LDA + c[None, :]
    aoff = tl.where(valid, aoff, 0)
    if COMPLEX:
        xr = tl.load(X + c * INCX * 2, c < N, 0.0)
        xi = tl.load(X + c * INCX * 2 + 1, c < N, 0.0)
        ar = tl.load(A + aoff * 2, valid, 0.0)
        ai = tl.load(A + aoff * 2 + 1, valid, 0.0)
        if CONJ:
            ai = -ai
        yr = tl.sum(ar * xr[None, :] - ai * xi[None, :], 1)
        yi = tl.sum(ar * xi[None, :] + ai * xr[None, :], 1)
        if UNIT:
            yr += tl.load(X + r * INCX * 2, r < N, 0.0)
            yi += tl.load(X + r * INCX * 2 + 1, r < N, 0.0)
        tl.store(X + r * INCX * 2, yr, r < N)
        tl.store(X + r * INCX * 2 + 1, yi, r < N)
    else:
        xv = tl.load(X + c * INCX, c < N, 0.0)
        av = tl.load(A + aoff, valid, 0.0)
        y = tl.sum(av * xv[None, :], 1)
        if UNIT:
            y += tl.load(X + r * INCX, r < N, 0.0)
        tl.store(X + r * INCX, y, r < N)


@libentry()
@triton.jit
def _copy(X, TMP, N: tl.constexpr, INCX: tl.constexpr,
          COMPLEX: tl.constexpr, B: tl.constexpr):
    j = tl.program_id(0) * B + tl.arange(0, B)
    if COMPLEX:
        v = tl.load(X + j * INCX * 2, j < N, 0.0)
        w = tl.load(X + j * INCX * 2 + 1, j < N, 0.0)
        tl.store(TMP + j * 2, v, j < N)
        tl.store(TMP + j * 2 + 1, w, j < N)
    else:
        v = tl.load(X + j * INCX, j < N, 0.0)
        tl.store(TMP + j, v, j < N)


@libentry()
@triton.jit
def _row(A, X, Y, N: tl.constexpr, LDA: tl.constexpr, INCX: tl.constexpr,
         UPLO: tl.constexpr, UNIT: tl.constexpr, CONJ: tl.constexpr,
         COMPLEX: tl.constexpr, K: tl.constexpr):
    r = tl.program_id(0)
    k = tl.arange(0, K)
    lo = 0 if UPLO == 0 else r
    hi = r + 1 if UPLO == 0 else N
    ar_acc = tl.full((K,), 0.0, tl.float32)
    if COMPLEX:
        ai_acc = tl.full((K,), 0.0, tl.float32)
    for kb in range(lo, hi, K):
        c = kb + k
        valid = (c < hi) & (c < N)
        if UNIT:
            valid &= c != r
        off = r * LDA + c
        if COMPLEX:
            ar = tl.load(A + off * 2, valid, 0.0)
            ai = tl.load(A + off * 2 + 1, valid, 0.0)
            if CONJ:
                ai = -ai
            xr = tl.load(X + c * 2, valid, 0.0)
            xi = tl.load(X + c * 2 + 1, valid, 0.0)
            ar_acc += ar * xr - ai * xi
            ai_acc += ar * xi + ai * xr
        else:
            av = tl.load(A + off, valid, 0.0)
            xv = tl.load(X + c, valid, 0.0)
            ar_acc += av * xv
    yr = tl.sum(ar_acc, 0)
    if UNIT:
        yr += tl.load(X + r * (2 if COMPLEX else 1))
    if COMPLEX:
        yi = tl.sum(ai_acc, 0)
        if UNIT:
            yi += tl.load(X + r * 2 + 1)
        tl.store(Y + r * INCX * 2, yr)
        tl.store(Y + r * INCX * 2 + 1, yi)
    else:
        tl.store(Y + r * INCX, yr)


@libentry()
@triton.jit
def _rows_tiled(A, X, Y, N: tl.constexpr, LDA: tl.constexpr,
                INCX: tl.constexpr, UPLO: tl.constexpr, UNIT: tl.constexpr,
                CONJ: tl.constexpr, TRANS: tl.constexpr,
                BR: tl.constexpr, BK: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.arange(0, BK)
    valid = (r[:, None] < N) & (c[None, :] < N)
    if UPLO == TRANS:
        valid &= r[:, None] >= c[None, :]
    else:
        valid &= r[:, None] <= c[None, :]
    if UNIT:
        valid &= r[:, None] != c[None, :]
    if TRANS:
        off = c[None, :] * LDA + r[:, None]
    else:
        off = r[:, None] * LDA + c[None, :]
    ar = tl.load(A + 2 * off, valid, 0.0)
    ai = tl.load(A + 2 * off + 1, valid, 0.0)
    xr = tl.load(X + 2 * c, c < N, 0.0)
    xi = tl.load(X + 2 * c + 1, c < N, 0.0)
    if CONJ:
        ai = -ai
    yr = tl.sum(ar * xr[None, :] - ai * xi[None, :], 1)
    yi = tl.sum(ar * xi[None, :] + ai * xr[None, :], 1)
    if UNIT:
        yr += tl.load(X + 2 * r, r < N, 0.0)
        yi += tl.load(X + 2 * r + 1, r < N, 0.0)
    tl.store(Y + r * INCX * 2, yr, r < N)
    tl.store(Y + r * INCX * 2 + 1, yi, r < N)


@libentry()
@triton.jit
def _complex_make_rhs(X, RHS, N: tl.constexpr, INCX: tl.constexpr,
                      CONJ: tl.constexpr, B: tl.constexpr):
    j = tl.program_id(0) * B + tl.arange(0, B)
    xr = tl.load(X + j * INCX * 2, j < N, 0.0)
    xi = tl.load(X + j * INCX * 2 + 1, j < N, 0.0)
    base = j * 4
    tl.store(RHS + base, xr, j < N)
    tl.store(RHS + base + 1, xi, j < N)
    if CONJ:
        tl.store(RHS + base + 2, xi, j < N)
        tl.store(RHS + base + 3, -xr, j < N)
    else:
        tl.store(RHS + base + 2, -xi, j < N)
        tl.store(RHS + base + 3, xr, j < N)


@libentry()
@triton.jit
def _complex_store(X, Y, N: tl.constexpr, INCX: tl.constexpr,
                   UNIT: tl.constexpr, B: tl.constexpr):
    j = tl.program_id(0) * B + tl.arange(0, B)
    yr = tl.load(Y + j * 2, j < N, 0.0)
    yi = tl.load(Y + j * 2 + 1, j < N, 0.0)
    if UNIT:
        yr += tl.load(X + j * INCX * 2, j < N, 0.0)
        yi += tl.load(X + j * INCX * 2 + 1, j < N, 0.0)
    tl.store(X + j * INCX * 2, yr, j < N)
    tl.store(X + j * INCX * 2 + 1, yi, j < N)


@libentry()
@triton.jit
def _complex_make_source8(X, RHS, N: tl.constexpr, INCX: tl.constexpr,
                          B: tl.constexpr):
    j = tl.program_id(0) * B + tl.arange(0, B)
    lane = tl.arange(0, 8)
    xr = tl.load(X + j * INCX * 2, j < N, 0.0)
    xi = tl.load(X + j * INCX * 2 + 1, j < N, 0.0)
    value = tl.where(lane[None, :] == 0, xr[:, None],
                     tl.where(lane[None, :] == 1, xi[:, None], 0.0))
    tl.store(RHS + j[:, None] * 8 + lane[None, :],
             value, j[:, None] < N)


@libentry()
@triton.jit
def _complex_transpose_store(X, Y, N: tl.constexpr, INCX: tl.constexpr,
                             UNIT: tl.constexpr, CONJ: tl.constexpr,
                             LDY: tl.constexpr, B: tl.constexpr):
    j = tl.program_id(0) * B + tl.arange(0, B)
    p = tl.load(Y + j * 2 * LDY, j < N, 0.0)
    q = tl.load(Y + j * 2 * LDY + 1, j < N, 0.0)
    r = tl.load(Y + (j * 2 + 1) * LDY, j < N, 0.0)
    s = tl.load(Y + (j * 2 + 1) * LDY + 1, j < N, 0.0)
    if CONJ:
        yr = p + s
        yi = q - r
    else:
        yr = p - s
        yi = q + r
    if UNIT:
        yr += tl.load(X + j * INCX * 2, j < N, 0.0)
        yi += tl.load(X + j * INCX * 2 + 1, j < N, 0.0)
    tl.store(X + j * INCX * 2, yr, j < N)
    tl.store(X + j * INCX * 2 + 1, yi, j < N)


def _trmv(uplo, trans, diag, n, A, lda, x, incx, complex_data):
    _common._check_trmv(A, x, uplo, trans, diag, n, lda, incx, complex_data)
    if n == 0:
        return
    unit = int(diag == CUBLAS_DIAG_UNIT)
    transpose = int(trans != CUBLAS_OP_N)
    conjugate = int(trans == CUBLAS_OP_C)
    if not complex_data and n >= 31 and (transpose or n <= 512):
        matrix = A if lda == n else A[:, :n]
        if uplo == 0:
            matrix = torch.tril(matrix, diagonal=-1 if unit else 0)
        else:
            matrix = torch.triu(matrix, diagonal=1 if unit else 0)
        source = x if incx == 1 and x.numel() == n else x[: n * incx : incx]
        result = torch.mv(matrix.T if transpose else matrix, source)
        if unit:
            source.add_(result)
        else:
            source.copy_(result)
        return
    with torch_device_fn.device(A.device):
        if n <= (31 if complex_data else 64):
            block = 32 if n in (1, 8) else triton.next_power_of_2(n)
            _launch(_small, (1,), (A, x),
                    (n, lda, incx, uplo, transpose, unit, conjugate,
                     complex_data, block))
        elif complex_data and (n <= 256 or n == 512):
            tmp = torch.empty((n,), dtype=x.dtype, device=x.device)
            _launch(_copy, (triton.cdiv(n, 1024),), (x, tmp),
                    (n, incx, True, 1024))
            rows = 16 if n == 256 and not transpose else 8
            _launch(_rows_tiled, (triton.cdiv(n, rows),), (A, tmp, x),
                    (n, lda, incx, uplo, unit, conjugate, transpose,
                     rows, triton.next_power_of_2(n)))
        elif complex_data and n >= 127:
            packed = torch.view_as_real(A)[:, :n].view(torch.int64)
            packed = packed.squeeze(-1)
            if uplo == 0:
                packed = torch.tril(packed, diagonal=-1 if unit else 0)
            else:
                packed = torch.triu(packed, diagonal=1 if unit else 0)
            matrix = packed.unsqueeze(-1).view(torch.float32).view(n, 2 * n)
            if transpose:
                if 8192 <= n < 10000:
                    source = torch.empty((n, 8), dtype=torch.float32,
                                         device=x.device)
                    _launch(_complex_make_source8, (triton.cdiv(n, 128),),
                            (x, source), (n, incx, 128))
                    result_stride = 8
                else:
                    source_x = (
                        x if incx == 1 and x.numel() == n
                        else x[: n * incx : incx]
                    )
                    source = torch.view_as_real(source_x).contiguous()
                    result_stride = 2
                result = torch.matmul(matrix.T, source)
                _launch(_complex_transpose_store, (triton.cdiv(n, 128),),
                        (x, result),
                        (n, incx, unit, conjugate, result_stride, 128))
            else:
                rhs = torch.empty((2 * n, 2), dtype=torch.float32,
                                  device=x.device)
                _launch(_complex_make_rhs, (triton.cdiv(n, 128),), (x, rhs),
                        (n, incx, 0, 128))
                result = torch.matmul(matrix, rhs)
                _launch(_complex_store, (triton.cdiv(n, 128),), (x, result),
                        (n, incx, unit, 128))
        else:
            tmp = torch.empty((n,), dtype=x.dtype, device=x.device)
            _launch(_copy, (triton.cdiv(n, 1024),), (x, tmp),
                    (n, incx, complex_data, 1024))
            _launch(_row, (n,), (A, tmp, x),
                    (n, lda, incx, uplo, unit, 0, complex_data,
                     128 if complex_data else 512))


def strmv(uplo, trans, diag, n, A, lda, x, incx):
    assert A.dtype == x.dtype == torch.float32
    return _trmv(uplo, trans, diag, n, A, lda, x, incx, False)


def ctrmv(uplo, trans, diag, n, A, lda, x, incx):
    assert A.dtype == x.dtype == torch.complex64
    return _trmv(uplo, trans, diag, n, A, lda, x, incx, True)
