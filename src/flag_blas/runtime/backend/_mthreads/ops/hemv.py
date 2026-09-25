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
from flag_blas.ops.level2.hemv import ScalarType, _check_common, _complex_scalars
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mthreads_hemv"),
    key=["N", "LDA", "INCX", "INCY", "UPLO", "ZERO_ALPHA", "ZERO_BETA", "WIDE"],
    restore_value=["TUNE_Y"],
)
@triton.jit
def mthreads_hemv_kernel(
    A,
    X,
    Y,
    TUNE_Y,
    N,
    LDA,
    INCX,
    INCY,
    ar,
    ai,
    br,
    bi,
    UPLO: tl.constexpr,
    ZERO_ALPHA: tl.constexpr,
    ZERO_BETA: tl.constexpr,
    WIDE: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    # Complete the reduction in the output-owning program, without partials.
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    if WIDE:
        N, LDA = tl.cast(N, tl.int64), tl.cast(LDA, tl.int64)
        INCX, INCY = tl.cast(INCX, tl.int64), tl.cast(INCY, tl.int64)
    ks = tl.arange(0, BK)
    sr = tl.full((BM, BK), 0, tl.float32)
    si = tl.full((BM, BK), 0, tl.float32)
    if not ZERO_ALPHA:
        for start in range(tl.cdiv(N, BK)):
            k = start * BK + ks
            direct = r[:, None] >= k[None, :] if UPLO == 0 else r[:, None] <= k[None, :]
            off = tl.where(
                direct,
                r[:, None].to(tl.int64) * LDA + k[None, :],
                k[None, :].to(tl.int64) * LDA + r[:, None],
            )
            mask = (r[:, None] < N) & (k[None, :] < N)
            vr = tl.load(A + 2 * off, mask, 0)
            # Clear ignored diagonal bits before any floating-point arithmetic.
            imag_bits = tl.load(A + 2 * off + 1, mask, 0).to(tl.int32, bitcast=True)
            imag_bits = imag_bits & tl.where(r[:, None] != k[None, :], -1, 0)
            vi = imag_bits.to(tl.float32, bitcast=True)
            vi = tl.where(direct, vi, -vi)
            xr = tl.load(X + 2 * k.to(tl.int64) * INCX, k < N, 0)
            xi = tl.load(X + 2 * k.to(tl.int64) * INCX + 1, k < N, 0)
            sr += vr * xr[None, :] - vi * xi[None, :]
            si += vr * xi[None, :] + vi * xr[None, :]
    rr, ri = tl.sum(sr, 1), tl.sum(si, 1)
    vr, vi = ar * rr - ai * ri, ar * ri + ai * rr
    yo = 2 * r.to(tl.int64) * INCY
    if not ZERO_BETA:
        yr = tl.load(Y + yo, r < N, 0)
        yi = tl.load(Y + yo + 1, r < N, 0)
        vr += br * yr - bi * yi
        vi += br * yi + bi * yr
    tl.store(Y + yo, vr, r < N)
    tl.store(Y + yo + 1, vi, r < N)


def chemv(
    uplo: int,
    n: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    assert A.dtype == x.dtype == y.dtype == torch.complex64
    _check_common(A, x, y, uplo, n, lda, incx, incy)
    assert A.device.type == "musa"
    if n == 0:
        return
    ar, ai, br, bi = _complex_scalars(alpha, beta)
    zero_alpha, zero_beta = ar == 0.0 and ai == 0.0, br == 0.0 and bi == 0.0
    if zero_alpha and br == 1.0 and bi == 0.0:
        return
    wide = 2 * max(n * lda, n * incx, n * incy) >= 2**31
    with torch_device_fn.device(A.device):
        av, xv, yv = (torch.view_as_real(t) for t in (A, x, y))
        kernel, config = mthreads_hemv_kernel, {}
        if wide:
            kernel = kernel.jit_function
            config = dict(BM=4, BK=128, num_warps=4, num_stages=1)
            tune_y = yv
        else:
            tune_y = torch.view_as_real(y.view(-1)[: n * incy : incy])
        kernel[lambda meta: (triton.cdiv(n, meta["BM"]),)](
            av,
            xv,
            yv,
            tune_y,
            n,
            lda,
            incx,
            incy,
            ar,
            ai,
            br,
            bi,
            UPLO=uplo,
            ZERO_ALPHA=zero_alpha,
            ZERO_BETA=zero_beta,
            WIDE=wide,
            **config,
        )
