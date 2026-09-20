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

import importlib

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

_common = importlib.import_module("flag_blas.ops.level2.tbsv")


@libentry()
@triton.jit
def _prepare_inverse(
    A,
    inverse,
    n,
    k,
    lda,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    FORWARD: tl.constexpr,
    COMPLEX: tl.constexpr,
    B: tl.constexpr,
):
    block = tl.program_id(0)
    offs = tl.arange(0, B)
    valid = block * B + offs < n
    out = block * B * B * (2 if COMPLEX else 1) + offs[:, None] * B + offs[None, :]
    if COMPLEX:
        real, imag = _common._complex_tbsv_diag_block_inv(
            A,
            block * B,
            n,
            k,
            lda,
            offs,
            valid,
            UPLO,
            TRANS,
            UNIT,
            CONJ,
            FORWARD,
            False,
            B,
        )
        tl.store(inverse + out, real)
        tl.store(inverse + out + B * B, imag)
    else:
        inv = _common._real_tbsv_diag_block_inv(
            A,
            block * B,
            n,
            k,
            lda,
            offs,
            valid,
            UPLO,
            TRANS,
            UNIT,
            FORWARD,
            False,
            B,
        )
        tl.store(inverse + out, inv)


@libentry()
@triton.jit
def _prepare_transfer(
    A,
    x,
    inverse,
    coeff,
    rhs,
    n,
    k,
    lda,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    CONJ: tl.constexpr,
    FORWARD: tl.constexpr,
    COMPLEX: tl.constexpr,
    B: tl.constexpr,
    BAND: tl.constexpr,
    C: tl.constexpr,
):
    block = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tl.arange(0, B)
    band = tile * C + tl.arange(0, C)
    start = block * B
    end = tl.minimum(start + B, n)
    rows = start + offs
    valid = rows < end
    if FORWARD:
        first = tl.maximum(0, start - k)
        last = start
    else:
        first = end
        last = tl.minimum(n, end + k)
    cols = first + band
    cv = (cols < last) & (band < BAND)
    if TRANS == 0:
        aoff = _common._tbsv_band_offset(rows[:, None], cols[None, :], k, lda, UPLO)
    else:
        aoff = _common._tbsv_band_offset(cols[None, :], rows[:, None], k, lda, UPLO)
    mask = valid[:, None] & cv[None, :] & (tl.abs(rows[:, None] - cols[None, :]) <= k)
    ar = tl.load(A + aoff * (2 if COMPLEX else 1), mask, 0)
    inv_off = block * B * B * (2 if COMPLEX else 1) + offs[:, None] * B + offs[None, :]
    ir = tl.load(inverse + inv_off)
    if COMPLEX:
        ai = tl.load(A + aoff * 2 + 1, mask, 0)
        if CONJ:
            ai = -ai
        ii = tl.load(inverse + inv_off + B * B)
        cr = tl.dot(ir, ar, input_precision="ieee") - tl.dot(
            ii, ai, input_precision="ieee"
        )
        ci = tl.dot(ir, ai, input_precision="ieee") + tl.dot(
            ii, ar, input_precision="ieee"
        )
    else:
        cr = tl.dot(ir, ar, input_precision="ieee")
    out = block * B * BAND * (2 if COMPLEX else 1) + band[None, :] * B + offs[:, None]
    tl.store(coeff + out, cr, band[None, :] < BAND)
    if COMPLEX:
        tl.store(coeff + out + B * BAND, ci, band[None, :] < BAND)
    if tile == 0:
        xr = tl.load(x + rows * (2 if COMPLEX else 1), valid, 0)
        if COMPLEX:
            xi = tl.load(x + rows * 2 + 1, valid, 0)
            rr = tl.sum(ir * xr[None, :] - ii * xi[None, :], 1)
            ri = tl.sum(ir * xi[None, :] + ii * xr[None, :], 1)
            tl.store(rhs + rows * 2, rr, valid)
            tl.store(rhs + rows * 2 + 1, ri, valid)
        else:
            rr = tl.sum(ir * xr[None, :], 1)
            tl.store(rhs + rows, rr, valid)


@libentry()
@triton.jit
def _solve_transfers(
    coeff,
    rhs,
    x,
    n,
    k,
    FORWARD: tl.constexpr,
    COMPLEX: tl.constexpr,
    B: tl.constexpr,
    BAND: tl.constexpr,
):
    count = tl.cdiv(n, B)
    for step in range(count):
        block = step if FORWARD else count - 1 - step
        offs = tl.arange(0, B)
        band = tl.arange(0, BAND)
        start = block * B
        end = tl.minimum(start + B, n)
        rows = start + offs
        valid = rows < end
        if FORWARD:
            first = tl.maximum(0, start - k)
            last = start
        else:
            first = end
            last = tl.minimum(n, end + k)
        cols = first + band
        cv = cols < last
        pos = (
            block * B * BAND * (2 if COMPLEX else 1) + band[:, None] * B + offs[None, :]
        )
        ar = tl.load(coeff + pos)
        rr = tl.load(rhs + rows * (2 if COMPLEX else 1), valid, 0)
        if COMPLEX:
            ai = tl.load(coeff + pos + B * BAND)
            ri = tl.load(rhs + rows * 2 + 1, valid, 0)
        xr = tl.load(x + cols * (2 if COMPLEX else 1), cv, 0)
        if COMPLEX:
            xi = tl.load(x + cols * 2 + 1, cv, 0)
            rr -= tl.sum(ar * xr[:, None] - ai * xi[:, None], 0)
            ri -= tl.sum(ar * xi[:, None] + ai * xr[:, None], 0)
            tl.store(x + rows * 2, rr, valid)
            tl.store(x + rows * 2 + 1, ri, valid)
        else:
            rr -= tl.sum(ar * xr[:, None], 0)
            tl.store(x + rows, rr, valid)


def _tbsv(uplo, trans, diag, n, k, A, lda, x, incx, complex_):
    dtype = torch.complex64 if complex_ else torch.float32
    assert A.dtype == dtype == x.dtype
    _common._check_tbsv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=complex_)
    if n == 0:
        return

    # Keep the common paths for tiny systems, scalar recurrences, strides and
    # bands outside the range measured on MUSA. FP64 entry points are unchanged.
    if n < 32 or incx != 1 or not 2 <= k <= 256:
        op = _common.ctbsv if complex_ else _common.stbsv
        return op(uplo, trans, diag, n, k, A, lda, x, incx)

    physical_uplo, physical_trans, conj = _common._row_major_tbsv_args(uplo, trans)
    forward = bool((physical_uplo == 0) ^ (physical_trans == 1))
    block = 16
    band = max(32, triton.next_power_of_2(k))
    count = triton.cdiv(n, block)
    factor = 2 if complex_ else 1

    with torch_device_fn.device(A.device):
        ar = torch.view_as_real(A) if complex_ else A
        xr = torch.view_as_real(x) if complex_ else x
        inverse = torch.empty(
            (count * block * block * factor,), device=A.device, dtype=torch.float32
        )
        coeff = torch.empty(
            (count * block * band * factor,), device=A.device, dtype=torch.float32
        )
        rhs = torch.empty((n * factor,), device=A.device, dtype=torch.float32)
        _prepare_inverse[(count,)](
            ar,
            inverse,
            n,
            k,
            lda,
            UPLO=physical_uplo,
            TRANS=physical_trans,
            UNIT=bool(diag),
            CONJ=conj,
            FORWARD=forward,
            COMPLEX=complex_,
            B=block,
            num_warps=4,
            num_stages=1,
        )
        # IEEE float32 products preserve the public precision requirements;
        # the preprocessing must not use TF32 approximations.
        _prepare_transfer[(count, triton.cdiv(band, 32))](
            ar,
            xr,
            inverse,
            coeff,
            rhs,
            n,
            k,
            lda,
            UPLO=physical_uplo,
            TRANS=physical_trans,
            CONJ=conj,
            FORWARD=forward,
            COMPLEX=complex_,
            B=block,
            BAND=band,
            C=32,
            num_warps=4,
            num_stages=1,
        )
        _solve_transfers[(1,)](
            coeff,
            rhs,
            xr,
            n,
            k,
            FORWARD=forward,
            COMPLEX=complex_,
            B=block,
            BAND=band,
            num_warps=4,
            num_stages=1,
        )


def stbsv(uplo, trans, diag, n, k, A, lda, x, incx):
    """Solve a real single-precision triangular banded system in-place."""
    return _tbsv(uplo, trans, diag, n, k, A, lda, x, incx, False)


def ctbsv(uplo, trans, diag, n, k, A, lda, x, incx):
    """Solve a complex single-precision triangular banded system in-place."""
    return _tbsv(uplo, trans, diag, n, k, A, lda, x, incx, True)
