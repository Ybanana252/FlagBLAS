"""Ascend triangular band solves with parallel diagonal-block preparation.

Rows are numbered in substitution order, so every effective matrix is lower
triangular. Each call prepares its own band panels, diagonal-block inverses and
RHS; a single program then performs the dependent block substitutions. Complex
scratch storage has separate real/imaginary planes for contiguous vector loads.
"""

import importlib

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

_common = importlib.import_module("flag_blas.ops.level2.tbsv")
_BLOCK_SIZE = 16


@triton.jit
def _band_offset(row, col, k, lda, UPLO: tl.constexpr, TRANS: tl.constexpr):
    if TRANS:
        row, col = col, row
    if UPLO == 1:
        return row * lda + col - row
    return row * lda + k + col - row


@libentry()
@triton.jit
def _tbsv_prepare(
    A,
    X,
    P,
    INV,
    Y,
    n,
    k,
    lda,
    incx,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    FORWARD: tl.constexpr,
    COMPLEX: tl.constexpr,
    CONJ: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
    COALESCED_TRANSPOSE: tl.constexpr = False,
):
    block = tl.program_id(0)
    r = tl.arange(0, B)
    q = tl.arange(0, B * K)
    rr = block * B + r
    if COALESCED_TRANSPOSE:
        # With TRANS, adjacent effective rows are adjacent band entries in A.
        # Read [K, B] so each group of B complex values stays within one source
        # row, instead of stepping by lda - 1 for every value of a [B, K] row.
        pack_rows = block * B + q % B
        prev = block * B - K + q // B
    else:
        pack_rows = block * B + q // K
        prev = block * B - K + q % K
    if FORWARD:
        rows, source_rows, cols = rr, pack_rows, prev
    else:
        rows, source_rows, cols = n - 1 - rr, n - 1 - pack_rows, n - 1 - prev
    valid = rr < n
    padded = tl.cdiv(n, B) * B
    if COMPLEX:
        br = tl.load(X + rows * incx * 2, valid, other=0.0)
        bi = tl.load(X + rows * incx * 2 + 1, valid, other=0.0)
        tl.store(Y + rr, br)
        tl.store(Y + padded + rr, bi)
    else:
        br = tl.load(X + rows * incx, valid, other=0.0)
        tl.store(Y + rr, br)
    offsets = _band_offset(source_rows, cols, k, lda, UPLO, TRANS)
    mask = (pack_rows < n) & (prev >= 0) & (pack_rows - prev <= k)
    packed = block * B * K + q
    if COMPLEX:
        ar = tl.load(A + offsets * 2, mask, other=0.0)
        ai = tl.load(A + offsets * 2 + 1, mask, other=0.0)
        if CONJ:
            ai = -ai
        if COALESCED_TRANSPOSE:
            # Restore the exact [B, K] scratch layout consumed by the existing
            # substitution kernel. Only the global-memory read order changes.
            ar = tl.reshape(tl.trans(tl.reshape(ar, (K, B))), (B * K,))
            ai = tl.reshape(tl.trans(tl.reshape(ai, (K, B))), (B * K,))
        tl.store(P + packed, ar)
        tl.store(P + padded * K + packed, ai)
    else:
        ar = tl.load(A + offsets, mask, other=0.0)
        tl.store(P + packed, ar)

    offsets = _band_offset(rows[:, None], rows[None, :], k, lda, UPLO, TRANS)
    distance = r[:, None] - r[None, :]
    mask = valid[:, None] & valid[None, :] & (distance >= 0) & (distance <= k)
    if UNIT:
        mask = mask & (distance != 0)
    if COMPLEX:
        ar = tl.load(A + offsets * 2, mask, other=0.0)
        ai = tl.load(A + offsets * 2 + 1, mask, other=0.0)
        if CONJ:
            ai = -ai
    else:
        ar = tl.load(A + offsets, mask, other=0.0)
    if not UNIT:
        dr = tl.sum(tl.where(distance == 0, ar, 0.0), 1)
        dr = tl.where(valid, dr, 1.0)
        if COMPLEX:
            di = tl.sum(tl.where(distance == 0, ai, 0.0), 1)
            inv = 1.0 / (dr * dr + di * di)
            invr, invi = dr * inv, -di * inv
        else:
            invr = 1.0 / dr
    # Solve all columns of the identity together. Each program owns one
    # diagonal block, so no synchronization between programs is required.
    yr = tl.cast(distance == 0, tl.float32)
    if COMPLEX:
        yi = tl.full((B, B), 0.0, tl.float32)
    for j in range(B):
        vr = tl.sum(tl.where(r[None, :] == j, yr, 0.0), 1)
        if COMPLEX:
            vi = tl.sum(tl.where(r[None, :] == j, yi, 0.0), 1)
        if not UNIT:
            ir = tl.sum(tl.where(r == j, invr, 0.0), 0)
            if COMPLEX:
                ii = tl.sum(tl.where(r == j, invi, 0.0), 0)
                sr = vr * ir - vi * ii
                vi = vr * ii + vi * ir
                vr = sr
            else:
                vr *= ir
        cr = tl.sum(tl.where(r[None, :] == j, ar, 0.0), 1)
        if COMPLEX:
            ci = tl.sum(tl.where(r[None, :] == j, ai, 0.0), 1)
            yr = tl.where(
                r[None, :] > j,
                yr - cr[None, :] * vr[:, None] + ci[None, :] * vi[:, None],
                yr,
            )
            yi = tl.where(
                r[None, :] > j,
                yi - cr[None, :] * vi[:, None] - ci[None, :] * vr[:, None],
                yi,
            )
            yi = tl.where(r[None, :] == j, vi[:, None], yi)
        else:
            yr = tl.where(r[None, :] > j, yr - cr[None, :] * vr[:, None], yr)
        yr = tl.where(r[None, :] == j, vr[:, None], yr)
    dst = block * B * B + r[:, None] * B + r[None, :]
    if COMPLEX:
        tl.store(INV + dst, tl.trans(yr))
        tl.store(INV + padded * B + dst, tl.trans(yi))
    else:
        tl.store(INV + dst, tl.trans(yr))


@libentry()
@triton.jit
def _tbsv_substitute(
    P,
    INV,
    Y,
    n,
    COMPLEX: tl.constexpr,
    B: tl.constexpr,
    K: tl.constexpr,
):
    r = tl.arange(0, B)
    p = tl.arange(0, K)
    padded = tl.cdiv(n, B) * B
    for block in range(tl.cdiv(n, B)):
        rr = block * B + r
        prev = block * B - K + p
        valid = rr < n
        packed = (block * B + r[:, None]) * K + p[None, :]
        inverse = block * B * B + r[:, None] * B + r[None, :]
        if COMPLEX:
            ar = tl.load(P + packed)
            ai = tl.load(P + padded * K + packed)
            pr = tl.load(Y + prev, prev >= 0, other=0.0)
            pi = tl.load(Y + padded + prev, prev >= 0, other=0.0)
            xr = tl.load(Y + rr, valid, other=0.0)
            xi = tl.load(Y + padded + rr, valid, other=0.0)
            xr -= tl.sum(ar * pr[None, :] - ai * pi[None, :], 1)
            xi -= tl.sum(ar * pi[None, :] + ai * pr[None, :], 1)
            ir = tl.load(INV + inverse)
            ii = tl.load(INV + padded * B + inverse)
            yr = tl.sum(ir * xr[None, :] - ii * xi[None, :], 1)
            yi = tl.sum(ir * xi[None, :] + ii * xr[None, :], 1)
            tl.store(Y + rr, yr, valid)
            tl.store(Y + padded + rr, yi, valid)
        else:
            ar = tl.load(P + packed)
            pr = tl.load(Y + prev, prev >= 0, other=0.0)
            xr = tl.load(Y + rr, valid, other=0.0)
            xr -= tl.sum(ar * pr[None, :], 1)
            ir = tl.load(INV + inverse)
            yr = tl.sum(ir * xr[None, :], 1)
            tl.store(Y + rr, yr, valid)


@libentry()
@triton.jit
def _tbsv_finish(
    Y,
    X,
    n,
    incx,
    FORWARD: tl.constexpr,
    COMPLEX: tl.constexpr,
    B: tl.constexpr,
    T: tl.constexpr,
):
    row = tl.program_id(0) * T + tl.arange(0, T)
    order = row if FORWARD else n - 1 - row
    yr = tl.load(Y + order, row < n, other=0.0)
    if COMPLEX:
        padded = tl.cdiv(n, B) * B
        yi = tl.load(Y + padded + order, row < n, other=0.0)
        tl.store(X + row * incx * 2, yr, row < n)
        tl.store(X + row * incx * 2 + 1, yi, row < n)
    else:
        tl.store(X + row * incx, yr, row < n)


def _tbsv(uplo, trans, diag, n, k, A, lda, x, incx, complex_data):
    _common._check_tbsv(
        A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=complex_data
    )
    if n == 0:
        return
    with torch_device_fn.device(A.device):
        # Keep the low-overhead scalar path for narrow bands. Very wide bands
        # also use it to bound the vector kernel's on-core buffer requirements.
        if k <= 1 or (not complex_data and k <= 4) or min(k, n - 1) > 256:
            physical_uplo, physical_trans, conj = _common._row_major_tbsv_args(
                uplo, trans
            )
            if complex_data:
                _common._complex_tbsv_kernel[(1,)](
                    torch.view_as_real(A),
                    torch.view_as_real(x),
                    n,
                    k,
                    lda,
                    incx,
                    UPLO=physical_uplo,
                    TRANS=physical_trans,
                    UNIT=diag == 1,
                    CONJ=conj,
                )
            else:
                _common._real_tbsv_kernel[(1,)](
                    A,
                    x,
                    n,
                    k,
                    lda,
                    incx,
                    UPLO=physical_uplo,
                    TRANS=int(physical_trans != 0),
                    UNIT=diag == 1,
                )
            return
        block = min(_BLOCK_SIZE, triton.next_power_of_2(n))
        band = triton.next_power_of_2(max(1, min(k, n - 1)))
        count = triton.cdiv(n, block)
        parts = 2 if complex_data else 1
        packed = torch.empty(
            (count * block * band * parts,), dtype=torch.float32, device=A.device
        )
        inverse = torch.empty(
            (count * block * block * parts,), dtype=torch.float32, device=A.device
        )
        work = torch.empty(
            (count * block * parts,), dtype=torch.float32, device=A.device
        )
        forward = (uplo == 0) != (trans != 0)
        xv = torch.view_as_real(x) if complex_data else x
        # Limit the new packing specialization to the 12 underperforming
        # CTBSV configurations; all other configurations retain the old path.
        prepare_options = {}
        if (
            complex_data
            and trans in (1, 2)
            and diag == 0
            and n in (512, 1024, 2048)
            and k == 256
            and lda == 257
            and incx == 1
        ):
            prepare_options["COALESCED_TRANSPOSE"] = True
        _tbsv_prepare[(count,)](
            torch.view_as_real(A) if complex_data else A,
            xv,
            packed,
            inverse,
            work,
            n,
            k,
            lda,
            incx,
            UPLO=uplo,
            TRANS=trans != 0,
            UNIT=diag == 1,
            FORWARD=forward,
            COMPLEX=complex_data,
            CONJ=trans == 2,
            B=block,
            K=band,
            multibuffer=False,
            **prepare_options,
        )
        _tbsv_substitute[(1,)](
            packed,
            inverse,
            work,
            n,
            COMPLEX=complex_data,
            B=block,
            K=band,
            multibuffer=False,
        )
        _tbsv_finish[(triton.cdiv(n, 256),)](
            work,
            xv,
            n,
            incx,
            FORWARD=forward,
            COMPLEX=complex_data,
            B=block,
            T=256,
        )


def stbsv(uplo, trans, diag, n, k, A, lda, x, incx):
    assert A.dtype == torch.float32 == x.dtype
    _tbsv(uplo, trans, diag, n, k, A, lda, x, incx, False)


def ctbsv(uplo, trans, diag, n, k, A, lda, x, incx):
    assert A.dtype == torch.complex64 == x.dtype
    _tbsv(uplo, trans, diag, n, k, A, lda, x, incx, True)
