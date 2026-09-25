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
from flag_blas.ops.level2.symv import ScalarType, _check_common, _complex_scalars
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner


def _prune_symv_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    bm, bk = 4, 128
    if not args["WIDE"] and not args["ZERO_ALPHA"]:
        if args["PACKED"]:
            bk = 64
        elif not args["COMPLEX"] and args["N"] >= 8192:
            bm, bk = 16, 64 if args["LDA"] % 16 else 128
    return [c for c in configs if c.kwargs["BM"] == bm and c.kwargs["BK"] == bk]


@triton.jit
def _unpack_complex(value):
    real = value.to(tl.int32).to(tl.float32, bitcast=True)
    imag = (value >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    return real, imag


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mthreads_symv"),
    key=[
        "N",
        "LDA",
        "INCX",
        "INCY",
        "UPLO",
        "COMPLEX",
        "PACKED",
        "ZERO_ALPHA",
        "ZERO_BETA",
        "WIDE",
    ],
    restore_value=["TUNE_Y"],
    prune_configs_by={"early_config_prune": _prune_symv_configs},
)
@triton.jit
def mthreads_symv_kernel(
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
    COMPLEX: tl.constexpr,
    PACKED: tl.constexpr,
    ZERO_ALPHA: tl.constexpr,
    ZERO_BETA: tl.constexpr,
    WIDE: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    # A program owns its outputs throughout the reduction: no global scratch.
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
            if COMPLEX:
                if PACKED:
                    vr, vi = _unpack_complex(tl.load(A + off, mask, 0))
                    xr, xi = _unpack_complex(
                        tl.load(X + k.to(tl.int64) * INCX, k < N, 0)
                    )
                else:
                    vr = tl.load(A + 2 * off, mask, 0)
                    vi = tl.load(A + 2 * off + 1, mask, 0)
                    xr = tl.load(X + 2 * k.to(tl.int64) * INCX, k < N, 0)
                    xi = tl.load(X + 2 * k.to(tl.int64) * INCX + 1, k < N, 0)
                sr += vr * xr[None, :] - vi * xi[None, :]
                si += vr * xi[None, :] + vi * xr[None, :]
            else:
                av = tl.load(A + off, mask, 0)
                xv = tl.load(X + k.to(tl.int64) * INCX, k < N, 0)
                sr += av * xv[None, :]
    rr, ri = tl.sum(sr, 1), tl.sum(si, 1)
    if COMPLEX:
        vr, vi = ar * rr - ai * ri, ar * ri + ai * rr
        yo = 2 * r.to(tl.int64) * INCY
        if not ZERO_BETA:
            yr = tl.load(Y + yo, r < N, 0)
            yi = tl.load(Y + yo + 1, r < N, 0)
            vr += br * yr - bi * yi
            vi += br * yi + bi * yr
        tl.store(Y + yo, vr, r < N)
        tl.store(Y + yo + 1, vi, r < N)
    else:
        vr = ar * rr
        if not ZERO_BETA:
            vr += br * tl.load(Y + r.to(tl.int64) * INCY, r < N, 0)
        tl.store(Y + r.to(tl.int64) * INCY, vr, r < N)


def _symv(uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    _check_common(A, x, y, uplo, n, lda, incx, incy)
    assert A.device.type == "musa"
    if n == 0:
        return
    is_complex = A.dtype == torch.complex64
    ar, ai, br, bi = (
        _complex_scalars(alpha, beta)
        if is_complex
        else (float(alpha), 0.0, float(beta), 0.0)
    )
    zero_alpha, zero_beta = ar == 0.0 and ai == 0.0, br == 0.0 and bi == 0.0
    if zero_alpha and br == 1.0 and bi == 0.0:
        return
    factor = 2 if is_complex else 1
    wide = max(n * lda, n * incx, n * incy) * factor >= 2**31
    packed = is_complex and n >= 8192 and lda % 16 != 0 and not wide and not zero_alpha
    with torch_device_fn.device(A.device):
        if is_complex:
            av, xv = (
                (A.view(torch.int64), x.view(torch.int64))
                if packed
                else (torch.view_as_real(A), torch.view_as_real(x))
            )
            yv = torch.view_as_real(y)
        else:
            av, xv, yv = A, x, y
        kernel, config = mthreads_symv_kernel, {}
        if wide:
            # Do not clone a huge strided view during cold-call autotuning.
            kernel = kernel.jit_function
            config = dict(BM=4, BK=128, num_warps=4, num_stages=1)
            tune_y = yv
        else:
            tune_y = y.view(-1)[: n * incy : incy]
            if is_complex:
                tune_y = torch.view_as_real(tune_y)
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
            COMPLEX=is_complex,
            PACKED=packed,
            ZERO_ALPHA=zero_alpha,
            ZERO_BETA=zero_beta,
            WIDE=wide,
            **config,
        )


def ssymv(
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
    assert A.dtype == x.dtype == y.dtype == torch.float32
    _symv(uplo, n, alpha, A, lda, x, incx, beta, y, incy)


def csymv(
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
    _symv(uplo, n, alpha, A, lda, x, incx, beta, y, incy)
