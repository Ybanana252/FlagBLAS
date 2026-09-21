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
from flag_blas.ops.level2.trmv import _check_trmv
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
