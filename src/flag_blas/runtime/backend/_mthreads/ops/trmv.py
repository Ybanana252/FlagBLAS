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

from flag_blas import runtime
from flag_blas.ops.level2.trmv import _check_trmv, _mode_key
from flag_blas.ops.level2.trmv import ztrmv_kernel as _public_ztrmv_kernel
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner


def _prune(configs, named_args, trans, **kwargs):
    args = {**named_args, **kwargs}
    if args["N"] <= 64:
        bm, bk = 1, 32 if args["N"] <= 32 else 64
    elif trans and not args["COMPLEX"] and args["N"] >= 2048 and args["LDA"] % 16:
        bm, bk = 2, 128
    elif args["N"] < 4096:
        bm, bk = 1, 128
    elif args["COMPLEX"]:
        bm, bk = 4, 64
    elif trans:
        bm, bk = 8, 128
    else:
        bm, bk = 4, 256
    return [c for c in configs if c.kwargs["BM"] == bm and c.kwargs["BK"] == bk]


def _prune_n(configs, named_args, **kwargs):
    return _prune(configs, named_args, False, **kwargs)


def _prune_t(configs, named_args, **kwargs):
    return _prune(configs, named_args, True, **kwargs)


_CONFIGS = runtime.get_tuned_config("mthreads_trmv")
_TINY_CONFIGS = {
    c.kwargs["BK"]: dict(c.kwargs, num_warps=c.num_warps, num_stages=c.num_stages)
    for c in _CONFIGS
    if c.kwargs["BM"] == 1 and c.kwargs["BK"] <= 64
}


_TRMV_KEY = ["N", "LDA", "SX", "SY", "UPPER", "CONJ", "UNIT", "COMPLEX"]


@libentry()
@triton.jit
def mthreads_trmv_copy_kernel(X, Y, N, SX, SY, B: tl.constexpr):
    r = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    tl.store(Y + r * SY, tl.load(X + r * SX, r < N, 0), r < N)


@libentry()
@libtuner(
    configs=_CONFIGS,
    key=_TRMV_KEY,
    prune_configs_by={"early_config_prune": _prune_t},
)
@triton.jit
def mthreads_trmv_trans_kernel(
    A,
    X,
    Y,
    N,
    LDA,
    SX,
    SY,
    UPPER: tl.constexpr,
    CONJ: tl.constexpr,
    UNIT: tl.constexpr,
    COMPLEX: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    ks = tl.arange(0, BK)
    begin = 0 if UPPER else tl.program_id(0) * BM // BK * BK
    end = tl.minimum(N, (tl.program_id(0) + 1) * BM) if UPPER else N
    sr = tl.full((BK, BM), 0, tl.float32)
    si = tl.full((BK, BM), 0, tl.float32)
    for start in range(begin, end, BK):
        k = start + ks
        tri = k[:, None] <= r[None, :] if UPPER else k[:, None] >= r[None, :]
        mask = (k[:, None] < N) & (r[None, :] < N) & tri
        if UNIT:
            mask &= k[:, None] != r[None, :]
        off = k[:, None].to(tl.int64) * LDA + r[None, :]
        if COMPLEX:
            av = tl.load(A + off, mask, 0)
            ar = av.to(tl.int32).to(tl.float32, bitcast=True)
            ai = (av >> 32).to(tl.int32).to(tl.float32, bitcast=True)
            xv = tl.load(X + k.to(tl.int64) * SX, k < N, 0)
            xr = xv.to(tl.int32).to(tl.float32, bitcast=True)
            xi = (xv >> 32).to(tl.int32).to(tl.float32, bitcast=True)
            if CONJ:
                ai = -ai
            sr += ar * xr[:, None] - ai * xi[:, None]
            si += ar * xi[:, None] + ai * xr[:, None]
        else:
            av = tl.load(A + off, mask, 0)
            xv = tl.load(X + k.to(tl.int64) * SX, k < N, 0)
            sr += av * xv[:, None]
    rr, ri = tl.sum(sr, 0), tl.sum(si, 0)
    if UNIT:
        xv = tl.load(X + r.to(tl.int64) * SX, r < N, 0)
        if COMPLEX:
            rr += xv.to(tl.int32).to(tl.float32, bitcast=True)
            ri += (xv >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        else:
            rr += xv
    if COMPLEX:
        result = rr.to(tl.uint32, bitcast=True).to(tl.uint64) | (
            ri.to(tl.uint32, bitcast=True).to(tl.uint64) << 32
        )
    else:
        result = rr
    tl.store(Y + r.to(tl.int64) * SY, result, r < N)


@libentry()
@libtuner(
    configs=_CONFIGS,
    key=_TRMV_KEY,
    prune_configs_by={"early_config_prune": _prune_n},
)
@triton.jit
def mthreads_trmv_kernel(
    A,
    X,
    Y,
    N,
    LDA,
    SX,
    SY,
    UPPER: tl.constexpr,
    TRANS: tl.constexpr,
    CONJ: tl.constexpr,
    UNIT: tl.constexpr,
    COMPLEX: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    ks = tl.arange(0, BK)
    lower: tl.constexpr = UPPER == TRANS
    begin = 0 if lower else tl.program_id(0) * BM // BK * BK
    end = tl.minimum(N, (tl.program_id(0) + 1) * BM) if lower else N
    sr = tl.full((BM, BK), 0, tl.float32)
    si = tl.full((BM, BK), 0, tl.float32)
    for start in range(begin, end, BK):
        k = start + ks
        tri = k[None, :] <= r[:, None] if lower else k[None, :] >= r[:, None]
        mask = (r[:, None] < N) & (k[None, :] < N) & tri
        if UNIT:
            mask &= k[None, :] != r[:, None]
        off = (
            k[None, :].to(tl.int64) * LDA + r[:, None]
            if TRANS
            else r[:, None].to(tl.int64) * LDA + k[None, :]
        )
        if COMPLEX:
            av = tl.load(A + off, mask, 0)
            ar = av.to(tl.int32).to(tl.float32, bitcast=True)
            ai = (av >> 32).to(tl.int32).to(tl.float32, bitcast=True)
            xv = tl.load(X + k.to(tl.int64) * SX, k < N, 0)
            xr = xv.to(tl.int32).to(tl.float32, bitcast=True)
            xi = (xv >> 32).to(tl.int32).to(tl.float32, bitcast=True)
            if CONJ:
                ai = -ai
            sr += ar * xr[None, :] - ai * xi[None, :]
            si += ar * xi[None, :] + ai * xr[None, :]
        else:
            av = tl.load(A + off, mask, 0)
            xv = tl.load(X + k.to(tl.int64) * SX, k < N, 0)
            sr += av * xv[None, :]
    rr, ri = tl.sum(sr, 1), tl.sum(si, 1)
    if UNIT:
        xv = tl.load(X + r.to(tl.int64) * SX, r < N, 0)
        if COMPLEX:
            rr += xv.to(tl.int32).to(tl.float32, bitcast=True)
            ri += (xv >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        else:
            rr += xv
    if COMPLEX:
        result = rr.to(tl.uint32, bitcast=True).to(tl.uint64) | (
            ri.to(tl.uint32, bitcast=True).to(tl.uint64) << 32
        )
    else:
        result = rr
    tl.store(Y + r.to(tl.int64) * SY, result, r < N)


def _trmv(uplo, trans, diag, n, A, lda, x, incx, is_complex):
    assert A.dtype == x.dtype == (torch.complex64 if is_complex else torch.float32)
    _check_trmv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=is_complex)
    assert A.device.type == "musa"
    if n == 0:
        return
    with torch_device_fn.device(A.device):
        temp = torch.empty((n,), device=x.device, dtype=x.dtype)
        av, xv, tv = (
            (A.view(torch.int64), x.view(torch.int64), temp.view(torch.int64))
            if is_complex
            else (A, x, temp)
        )
        config = _TINY_CONFIGS[32 if n <= 32 else 64] if n <= 64 else {}
        kernel = mthreads_trmv_trans_kernel if trans else mthreads_trmv_kernel
        copy_kernel = mthreads_trmv_copy_kernel
        if n <= 64:
            # Use the validated tiny configuration without autotuner dispatch.
            kernel = kernel.jit_function
            copy_kernel = copy_kernel.jit_function
        grid = (n,) if n <= 64 else lambda meta: (triton.cdiv(n, meta["BM"]),)
        # Preserve the original logical x; autotuning only overwrites the output.
        # The snapshot stays unchanged across all program and tuning launches.
        copy_kernel[(triton.cdiv(n, 256),)](xv, tv, n, incx, 1, 256, num_warps=4)
        if trans:
            kernel[grid](
                av,
                tv,
                xv,
                n,
                lda,
                1,
                incx,
                bool(uplo),
                trans == 2,
                bool(diag),
                is_complex,
                **config,
            )
        else:
            kernel[grid](
                av,
                tv,
                xv,
                n,
                lda,
                1,
                incx,
                bool(uplo),
                False,
                False,
                bool(diag),
                is_complex,
                **config,
            )


def strmv(uplo, trans, diag, n, A, lda, x, incx):
    return _trmv(uplo, trans, diag, n, A, lda, x, incx, False)


def ctrmv(uplo, trans, diag, n, A, lda, x, incx):
    return _trmv(uplo, trans, diag, n, A, lda, x, incx, True)


@triton.jit
def mthreads_dtrmv_small_kernel(
    A,
    X,
    N,
    LDA,
    INCX,
    UPPER: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    B: tl.constexpr,
):
    r = tl.arange(0, B)
    k = tl.arange(0, B)
    lower: tl.constexpr = UPPER == TRANS
    tri = k[None, :] <= r[:, None] if lower else k[None, :] >= r[:, None]
    mask = (r[:, None] < N) & (k[None, :] < N) & tri
    if UNIT:
        mask &= r[:, None] != k[None, :]
    off = (
        k[None, :].to(tl.int64) * LDA + r[:, None]
        if TRANS
        else r[:, None].to(tl.int64) * LDA + k[None, :]
    )
    av = tl.load(A + off, mask, 0)
    xv = tl.load(X + k * INCX, k < N, 0)
    y = tl.sum(av * xv[None, :], 1)
    if UNIT:
        y += tl.load(X + r * INCX, r < N, 0)
    tl.debug_barrier()
    tl.store(X + r * INCX, y, r < N)


@triton.jit
def mthreads_ztrmv_copy_kernel(X, Y, N, SX, B: tl.constexpr):
    r = tl.program_id(0) * B + tl.arange(0, B)
    source = 2 * r.to(tl.int64) * SX
    target = 2 * r.to(tl.int64)
    tl.store(Y + target, tl.load(X + source, r < N, 0), r < N)
    tl.store(Y + target + 1, tl.load(X + source + 1, r < N, 0), r < N)


@triton.jit
def mthreads_trmv_fp64_kernel(
    A,
    X,
    Y,
    N,
    LDA,
    INCX,
    UPPER: tl.constexpr,
    TRANS: tl.constexpr,
    CONJ: tl.constexpr,
    UNIT: tl.constexpr,
    COMPLEX: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    ks = tl.arange(0, BK)
    lower: tl.constexpr = UPPER == TRANS
    begin = 0 if lower else tl.program_id(0) * BM // BK * BK
    end = tl.minimum(N, (tl.program_id(0) + 1) * BM) if lower else N
    if TRANS:
        sr = tl.full((BK, BM), 0, tl.float64)
        si = tl.full((BK, BM), 0, tl.float64)
    else:
        sr = tl.full((BM, BK), 0, tl.float64)
        si = tl.full((BM, BK), 0, tl.float64)
    for start in range(begin, end, BK):
        k = start + ks
        if TRANS:
            tri = k[:, None] <= r[None, :] if lower else k[:, None] >= r[None, :]
            mask = (k[:, None] < N) & (r[None, :] < N) & tri
            off = k[:, None].to(tl.int64) * LDA + r[None, :]
        else:
            tri = k[None, :] <= r[:, None] if lower else k[None, :] >= r[:, None]
            mask = (r[:, None] < N) & (k[None, :] < N) & tri
            off = r[:, None].to(tl.int64) * LDA + k[None, :]
        if UNIT:
            mask &= (k[:, None] != r[None, :]) if TRANS else (k[None, :] != r[:, None])
        if COMPLEX:
            ar = tl.load(A + 2 * off, mask, 0)
            ai = tl.load(A + 2 * off + 1, mask, 0)
            xr = tl.load(X + 2 * k, k < N, 0)
            xi = tl.load(X + 2 * k + 1, k < N, 0)
            if CONJ:
                ai = -ai
            if TRANS:
                sr += ar * xr[:, None] - ai * xi[:, None]
                si += ar * xi[:, None] + ai * xr[:, None]
            else:
                sr += ar * xr[None, :] - ai * xi[None, :]
                si += ar * xi[None, :] + ai * xr[None, :]
        else:
            av = tl.load(A + off, mask, 0)
            xv = tl.load(X + k, k < N, 0)
            sr += av * (xv[:, None] if TRANS else xv[None, :])
    if TRANS:
        rr, ri = tl.sum(sr, 0), tl.sum(si, 0)
    else:
        rr, ri = tl.sum(sr, 1), tl.sum(si, 1)
    if UNIT:
        if COMPLEX:
            rr += tl.load(X + 2 * r, r < N, 0)
            ri += tl.load(X + 2 * r + 1, r < N, 0)
        else:
            rr += tl.load(X + r, r < N, 0)
    if COMPLEX:
        tl.store(Y + 2 * r * INCX, rr, r < N)
        tl.store(Y + 2 * r * INCX + 1, ri, r < N)
    else:
        tl.store(Y + r * INCX, rr, r < N)


def _trmv_fp64(uplo, trans, diag, n, A, lda, x, incx, is_complex):
    assert A.dtype == x.dtype == (torch.complex128 if is_complex else torch.float64)
    _check_trmv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=is_complex)
    assert A.device.type == "musa"
    if n == 0:
        return
    if not is_complex and n <= 32:
        with torch_device_fn.device(A.device):
            mthreads_dtrmv_small_kernel[(1,)](
                A,
                x,
                n,
                lda,
                incx,
                UPPER=bool(uplo),
                TRANS=bool(trans),
                UNIT=bool(diag),
                B=32,
                num_warps=16,
                num_stages=1,
            )
        return
    with torch_device_fn.device(A.device):
        temp = torch.empty((n,), device=x.device, dtype=x.dtype)
        if is_complex:
            av, xv, tv = (torch.view_as_real(v) for v in (A, x, temp))
            mthreads_ztrmv_copy_kernel[(triton.cdiv(n, 256),)](
                xv, tv, n, incx, 256, num_warps=4
            )
            if n <= 32:
                trans_flag = int(trans != 0)
                conj = int(trans == 2)
                unit = int(bool(diag))
                _public_ztrmv_kernel.jit_function[(triton.cdiv(n, 16),)](
                    av,
                    tv,
                    xv,
                    n,
                    lda,
                    incx,
                    _mode_key(uplo, trans_flag, unit) | (conj << 8),
                    UPLO=uplo,
                    TRANS=trans_flag,
                    UNIT=unit,
                    CONJ=conj,
                    BLOCK_SIZE_M=16,
                    BLOCK_K=16,
                    num_warps=8,
                    num_stages=1,
                )
                return
            if n >= 8192 and not uplo and trans:
                bm, bk, warps = 4, 32, 4
            else:
                bm, bk, warps = (2, 128, 4) if n <= 2048 else (2, 64, 4)
        else:
            av, xv, tv = A, x, temp
            mthreads_trmv_copy_kernel.jit_function[(triton.cdiv(n, 256),)](
                xv, tv, n, incx, 1, 256, num_warps=4
            )
            bm, bk, warps = (2, 128, 4) if n <= 2048 else (4, 128, 4)
        mthreads_trmv_fp64_kernel[(triton.cdiv(n, bm),)](
            av,
            tv,
            xv,
            n,
            lda,
            incx,
            UPPER=bool(uplo),
            TRANS=bool(trans),
            CONJ=trans == 2,
            UNIT=bool(diag),
            COMPLEX=is_complex,
            BM=bm,
            BK=bk,
            num_warps=warps,
            num_stages=1,
        )


def dtrmv(uplo, trans, diag, n, A, lda, x, incx):
    return _trmv_fp64(uplo, trans, diag, n, A, lda, x, incx, False)


def ztrmv(uplo, trans, diag, n, A, lda, x, incx):
    return _trmv_fp64(uplo, trans, diag, n, A, lda, x, incx, True)
