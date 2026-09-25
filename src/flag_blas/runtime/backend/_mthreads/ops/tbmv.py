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
from flag_blas.ops.level2.tbmv import _check_tbmv
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

_CONFIGS = {
    (c.kwargs["BM"], c.kwargs["BK"], c.num_warps, c.num_stages): dict(
        c.kwargs, num_warps=c.num_warps, num_stages=c.num_stages
    )
    for c in runtime.get_tuned_config("mthreads_tbmv")
}


def _config(n, k, is_complex):
    if n <= 1024:
        key = (1, 8 if k <= 8 else 64, 1, 1)
    elif k >= 128:
        key = (2, 128, 4, 1)
    elif k <= 8:
        key = (32, 4, 4, 1)
    elif is_complex and n >= 16384 and k <= 64:
        key = (8, 32, 4, 1)
    else:
        key = (16, 32, 4, 1)
    return _CONFIGS[key]


@libentry()
@triton.jit
def mthreads_tbmv_copy_kernel(X, Y, N, SX, B: tl.constexpr):
    r = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    tl.store(Y + r, tl.load(X + r * SX, r < N, 0), r < N)


@libentry()
@triton.jit
def mthreads_tbmv_kernel(
    A,
    X,
    Y,
    N,
    K,
    LDA,
    SY,
    UPPER: tl.constexpr,
    TRANS: tl.constexpr,
    CONJ: tl.constexpr,
    UNIT: tl.constexpr,
    COMPLEX: tl.constexpr,
    SKEW: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.arange(0, BM)
    ks = tl.arange(0, BK)
    base = tl.program_id(0) * BM
    # Skew band coordinates so neighboring lanes read neighboring band slots.
    if TRANS and SKEW:
        rr = (
            (m[:, None] + ks[None, :]) & (BM - 1)
            if UPPER
            else (m[:, None] - ks[None, :]) & (BM - 1)
        )
    else:
        rr = m[:, None] + tl.full((1, BK), 0, tl.int32)
    rows = base + rr
    sr = tl.full((BM, BK), 0, tl.float32)
    si = tl.full((BM, BK), 0, tl.float32)
    for start in range(0, tl.minimum(K, N - 1) + 1, BK):
        d = start + ks[None, :]
        if TRANS:
            j = rows - d if UPPER else rows + d
            ar = j
        else:
            j = rows + d if UPPER else rows - d
            ar = rows
        slot = d if UPPER else K - d
        mask = (rows < N) & (j >= 0) & (j < N) & (d <= K)
        if UNIT:
            mask &= d != 0
        offset = ar.to(tl.int64) * LDA + slot
        av = tl.load(A + offset, mask, 0)
        xv = tl.load(X + j.to(tl.int64), mask, 0)
        if COMPLEX:
            vr = av.to(tl.int32).to(tl.float32, bitcast=True)
            vi = (av >> 32).to(tl.int32).to(tl.float32, bitcast=True)
            xr = xv.to(tl.int32).to(tl.float32, bitcast=True)
            xi = (xv >> 32).to(tl.int32).to(tl.float32, bitcast=True)
            if CONJ:
                vi = -vi
            sr += vr * xr - vi * xi
            si += vr * xi + vi * xr
        else:
            sr += av * xv
    # Map the register accumulators back to the output rows owned by this CTA.
    if TRANS and SKEW:
        idx = (
            (m[:, None] - ks[None, :]) & (BM - 1)
            if UPPER
            else (m[:, None] + ks[None, :]) & (BM - 1)
        )
        sr = tl.gather(sr, idx, axis=0)
        si = tl.gather(si, idx, axis=0)
    yr = tl.sum(sr, 1)
    yi = tl.sum(si, 1)
    r = base + m
    if UNIT:
        xv = tl.load(X + r.to(tl.int64), r < N, 0)
        if COMPLEX:
            yr += xv.to(tl.int32).to(tl.float32, bitcast=True)
            yi += (xv >> 32).to(tl.int32).to(tl.float32, bitcast=True)
        else:
            yr += xv
    if COMPLEX:
        result = yr.to(tl.uint32, bitcast=True).to(tl.uint64) | (
            yi.to(tl.uint32, bitcast=True).to(tl.uint64) << 32
        )
    else:
        result = yr
    tl.store(Y + r.to(tl.int64) * SY, result, r < N)


def _tbmv(uplo, trans, diag, n, k, A, lda, x, incx, is_complex):
    assert A.dtype == x.dtype == (torch.complex64 if is_complex else torch.float32)
    _check_tbmv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=is_complex)
    assert A.device.type == "musa"
    if n == 0:
        return
    with torch_device_fn.device(A.device):
        # Preserve only the logical input vector, including for large strides.
        temp = torch.empty((n,), device=x.device, dtype=x.dtype)
        av, xv, tv = (
            (A.view(torch.int64), x.view(torch.int64), temp.view(torch.int64))
            if is_complex
            else (A, x, temp)
        )
        config = _config(n, k, is_complex)
        copy_kernel, kernel = mthreads_tbmv_copy_kernel, mthreads_tbmv_kernel
        if n <= 1024:
            copy_kernel, kernel = copy_kernel.jit_function, kernel.jit_function
        copy_kernel[(triton.cdiv(n, 256),)](xv, tv, n, incx, 256, num_warps=4)
        kernel[(triton.cdiv(n, config["BM"]),)](
            av,
            tv,
            xv,
            n,
            k,
            lda,
            incx,
            bool(uplo),
            bool(trans),
            trans == 2,
            bool(diag),
            is_complex,
            True,
            **config,
        )


def stbmv(uplo, trans, diag, n, k, A, lda, x, incx):
    return _tbmv(uplo, trans, diag, n, k, A, lda, x, incx, False)


def ctbmv(uplo, trans, diag, n, k, A, lda, x, incx):
    return _tbmv(uplo, trans, diag, n, k, A, lda, x, incx, True)
