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

from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level2._constants import CUBLAS_OP_C, CUBLAS_OP_N, CUBLAS_OP_T
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

ScalarType = Union[float, int, complex, torch.Tensor]
_GBMV_KEY = [
    "M",
    "N",
    "KL",
    "KU",
    "LDA",
    "INCX",
    "INCY",
    "TRANS",
    "CONJ",
    "ZERO_BETA",
    "WIDE",
]


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mthreads_sgbmv"),
    key=_GBMV_KEY,
    restore_value=["Y"],
)
@triton.jit
def mthreads_sgbmv_kernel(
    A,
    X,
    Y,
    alpha,
    beta,
    M,
    N,
    KL,
    KU,
    LDA,
    INCX,
    INCY,
    TRANS: tl.constexpr,
    CONJ: tl.constexpr,
    ZERO_BETA: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    WIDE: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
        M, N = tl.cast(M, tl.int64), tl.cast(N, tl.int64)
        KL, KU = tl.cast(KL, tl.int64), tl.cast(KU, tl.int64)
        LDA = tl.cast(LDA, tl.int64)
        INCX, INCY = tl.cast(INCX, tl.int64), tl.cast(INCY, tl.int64)
    rows = pid * BM + tl.arange(0, BM)
    out = N if TRANS else M
    band = KL + KU + 1
    acc = tl.zeros((BM,), tl.float32)
    for start in range(0, band, BK):
        t = start + tl.arange(0, BK)
        if TRANS:
            j = rows[:, None] + t[None, :] - KU
            mask = (rows[:, None] < out) & (t[None, :] < band) & (j >= 0) & (j < M)
            safe_j = tl.where(mask, j, 0)
            aoff = safe_j * LDA + tl.where(t < band, band - 1 - t, 0)[None, :]
        else:
            j = rows[:, None] + t[None, :] - KL
            mask = (rows[:, None] < out) & (t[None, :] < band) & (j >= 0) & (j < N)
            safe_j = tl.where(mask, j, 0)
            aoff = (
                tl.where(rows < out, rows, 0)[:, None] * LDA
                + tl.where(t < band, t, 0)[None, :]
            )
        av = tl.load(A + aoff, mask, 0.0)
        xv = tl.load(X + safe_j * INCX, mask, 0.0)
        acc += tl.sum(av * xv, 1)
    result = alpha * acc
    if not ZERO_BETA:
        result += beta * tl.load(Y + rows * INCY, rows < out, 0.0)
    tl.store(Y + rows * INCY, result, rows < out)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mthreads_cgbmv"),
    key=_GBMV_KEY,
    restore_value=["Y"],
)
@triton.jit
def mthreads_cgbmv_kernel(
    A,
    X,
    Y,
    ar,
    ai,
    br,
    bi,
    M,
    N,
    KL,
    KU,
    LDA,
    INCX,
    INCY,
    TRANS: tl.constexpr,
    CONJ: tl.constexpr,
    ZERO_BETA: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    WIDE: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
        M, N = tl.cast(M, tl.int64), tl.cast(N, tl.int64)
        KL, KU = tl.cast(KL, tl.int64), tl.cast(KU, tl.int64)
        LDA = tl.cast(LDA, tl.int64)
        INCX, INCY = tl.cast(INCX, tl.int64), tl.cast(INCY, tl.int64)
    rows = pid * BM + tl.arange(0, BM)
    out = N if TRANS else M
    band = KL + KU + 1
    acc_r = tl.zeros((BM,), tl.float32)
    acc_i = tl.zeros((BM,), tl.float32)
    for start in range(0, band, BK):
        t = start + tl.arange(0, BK)
        if TRANS:
            j = rows[:, None] + t[None, :] - KU
            mask = (rows[:, None] < out) & (t[None, :] < band) & (j >= 0) & (j < M)
            safe_j = tl.where(mask, j, 0)
            aoff = safe_j * LDA + tl.where(t < band, band - 1 - t, 0)[None, :]
        else:
            j = rows[:, None] + t[None, :] - KL
            mask = (rows[:, None] < out) & (t[None, :] < band) & (j >= 0) & (j < N)
            safe_j = tl.where(mask, j, 0)
            aoff = (
                tl.where(rows < out, rows, 0)[:, None] * LDA
                + tl.where(t < band, t, 0)[None, :]
            )
        avr = tl.load(A + 2 * aoff, mask, 0.0)
        avi = tl.load(A + 2 * aoff + 1, mask, 0.0)
        xr = tl.load(X + 2 * safe_j * INCX, mask, 0.0)
        xi = tl.load(X + 2 * safe_j * INCX + 1, mask, 0.0)
        if CONJ:
            avi = -avi
        acc_r += tl.sum(avr * xr - avi * xi, 1)
        acc_i += tl.sum(avr * xi + avi * xr, 1)
    rr, ri = ar * acc_r - ai * acc_i, ar * acc_i + ai * acc_r
    if not ZERO_BETA:
        yr = tl.load(Y + 2 * rows * INCY, rows < out, 0.0)
        yi = tl.load(Y + 2 * rows * INCY + 1, rows < out, 0.0)
        rr += br * yr - bi * yi
        ri += br * yi + bi * yr
    tl.store(Y + 2 * rows * INCY, rr, rows < out)
    tl.store(Y + 2 * rows * INCY + 1, ri, rows < out)


@triton.jit
def mthreads_gbmv_scale_kernel(
    Y, br, bi, OUT, INCY, ZERO: tl.constexpr, COMPLEX: tl.constexpr, BLOCK: tl.constexpr
):
    r = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    off = r * INCY
    if COMPLEX:
        vr = tl.full((BLOCK,), 0.0, tl.float32)
        vi = tl.full((BLOCK,), 0.0, tl.float32)
        if not ZERO:
            yr = tl.load(Y + 2 * off, r < OUT, 0.0)
            yi = tl.load(Y + 2 * off + 1, r < OUT, 0.0)
            vr, vi = br * yr - bi * yi, br * yi + bi * yr
        tl.store(Y + 2 * off, vr, r < OUT)
        tl.store(Y + 2 * off + 1, vi, r < OUT)
    else:
        v = tl.full((BLOCK,), 0.0, tl.float32)
        if not ZERO:
            v = br * tl.load(Y + off, r < OUT, 0.0)
        tl.store(Y + off, v, r < OUT)


def _gbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy, is_complex):
    dtype = torch.complex64 if is_complex else torch.float32
    assert A.dtype == x.dtype == y.dtype == dtype
    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    assert A.device == x.device == y.device and A.device.type == "musa"
    assert trans in (
        (CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C)
        if is_complex
        else (CUBLAS_OP_N, CUBLAS_OP_T)
    )
    assert m >= 0 and n >= 0 and kl >= 0 and ku >= 0
    assert lda >= kl + ku + 1 and incx > 0 and incy > 0
    if m == 0 or n == 0:
        return
    out, red = (m, n) if trans == CUBLAS_OP_N else (n, m)
    assert (
        A.numel() >= m * lda
        and x.numel() >= (red - 1) * incx + 1
        and y.numel() >= (out - 1) * incy + 1
    )
    cast = complex if is_complex else float
    alpha = cast(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta = cast(beta.item() if isinstance(beta, torch.Tensor) else beta)
    with torch_device_fn.device(A.device):
        if alpha == 0:
            if beta != 1:
                yp = torch.view_as_real(y) if is_complex else y
                mthreads_gbmv_scale_kernel[(triton.cdiv(out, 256),)](
                    yp,
                    beta.real,
                    beta.imag if is_complex else 0.0,
                    out,
                    incy,
                    beta == 0,
                    is_complex,
                    256,
                    num_warps=4,
                )
            return
        logical_y = y.view(-1)[: out * incy : incy]
        width = 2 if is_complex else 1
        wide = (
            width * max(m * lda, red * incx, out * incy, m + n + kl + ku) > 2**31 - 1
        )
        kernel = mthreads_cgbmv_kernel if is_complex else mthreads_sgbmv_kernel
        launch_config = {"WIDE": wide}
        if wide:
            # A single conservative tile avoids copying very large-stride views
            # during tuning, including MUSA's unsupported complex real-view copy.
            kernel = kernel.jit_function
            launch_config.update(BM=4, BK=16, num_warps=4, num_stages=1)
            logical_y = y
        grid = lambda meta: (triton.cdiv(out, meta["BM"]),)
        common = (
            m,
            n,
            kl,
            ku,
            lda,
            incx,
            incy,
            trans != CUBLAS_OP_N,
            trans == CUBLAS_OP_C,
            beta == 0,
        )
        if is_complex:
            kernel[grid](
                torch.view_as_real(A),
                torch.view_as_real(x),
                torch.view_as_real(logical_y),
                alpha.real,
                alpha.imag,
                beta.real,
                beta.imag,
                *common,
                **launch_config,
            )
        else:
            kernel[grid](A, x, logical_y, alpha, beta, *common, **launch_config)


def sgbmv(
    trans: int,
    m: int,
    n: int,
    kl: int,
    ku: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    _gbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy, False)


def cgbmv(
    trans: int,
    m: int,
    n: int,
    kl: int,
    ku: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    _gbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy, True)


@triton.jit
def mthreads_dgbmv_2d_kernel(
    A,
    X,
    Y,
    alpha_int: tl.int64,
    beta_int: tl.int64,
    M,
    N,
    LDA,
    INCX,
    INCY,
    KL,
    KU,
    BAND,
    TRANS: tl.constexpr,
    ZERO_BETA: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    t = tl.arange(0, BK)
    alpha = alpha_int.to(tl.float64, bitcast=True)
    beta = beta_int.to(tl.float64, bitcast=True)
    acc = tl.zeros((BM,), tl.float64)
    if TRANS:
        out = N
        for start in tl.range(0, BAND, BK):
            e_idx = start + t
            i = rows[:, None] + e_idx[None, :] - KU
            mask = (rows[:, None] < out) & (e_idx[None, :] < BAND) & (i >= 0) & (i < M)
            safe_i = tl.where(mask, i, 0)
            av = tl.load(A + rows[:, None] * LDA + e_idx[None, :], mask, 0.0)
            xv = tl.load(X + safe_i * INCX, mask, 0.0)
            acc += tl.sum(av * xv, 1)
    else:
        out = M
        for start in tl.range(0, BAND, BK):
            d_idx = start + t
            d = d_idx - KL
            j = rows[:, None] + d[None, :]
            mask = (rows[:, None] < out) & (d_idx[None, :] < BAND) & (j >= 0) & (j < N)
            safe_j = tl.where(mask, j, 0)
            av = tl.load(A + safe_j * LDA + (KU - d)[None, :], mask, 0.0)
            xv = tl.load(X + safe_j * INCX, mask, 0.0)
            acc += tl.sum(av * xv, 1)
    result = alpha * acc
    if not ZERO_BETA:
        result += beta * tl.load(Y + rows * INCY, rows < out, 0.0)
    tl.store(Y + rows * INCY, result, rows < out)


def _use_common_gbmv(m, n, kl, ku, lda, incx, incy, trans, is_complex):
    out = m if trans == CUBLAS_OP_N else n
    red = n if trans == CUBLAS_OP_N else m
    width = 2 if is_complex else 1
    wide = width * max(m * lda, red * incx, out * incy) >= 2**31
    return kl + ku + 1 < (5 if is_complex else 11) or wide


def dgbmv(
    trans: int,
    m: int,
    n: int,
    kl: int,
    ku: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    from flag_blas.ops.level2 import gbmv as common

    assert A.dtype == x.dtype == y.dtype == torch.float64
    assert A.device.type == "musa"
    common._check_common(A, x, y, trans, m, n, kl, ku, lda, incx, incy, False)
    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta_value = float(beta.item() if isinstance(beta, torch.Tensor) else beta)
    if (
        m == 0
        or n == 0
        or alpha_value == 0
        or _use_common_gbmv(m, n, kl, ku, lda, incx, incy, trans, False)
    ):
        return common.dgbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)

    physical, pm, pn, pkl, pku, _ = common._row_major_gbmv_args(trans, m, n, kl, ku)
    out = pm if physical == CUBLAS_OP_N else pn
    bm, bk = (4, 32) if out <= 2048 else (8, 16)
    with torch_device_fn.device(A.device):
        mthreads_dgbmv_2d_kernel[(triton.cdiv(out, bm),)](
            A,
            x,
            y,
            common._f64_to_i64(alpha_value),
            common._f64_to_i64(beta_value),
            pm,
            pn,
            lda,
            incx,
            incy,
            pkl,
            pku,
            pkl + pku + 1,
            TRANS=physical != CUBLAS_OP_N,
            ZERO_BETA=beta_value == 0,
            BM=bm,
            BK=bk,
            num_warps=4,
            num_stages=1,
        )


def zgbmv(
    trans: int,
    m: int,
    n: int,
    kl: int,
    ku: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    from flag_blas.ops.level2 import gbmv as common

    assert A.dtype == x.dtype == y.dtype == torch.complex128
    assert A.device.type == "musa"
    common._check_common(A, x, y, trans, m, n, kl, ku, lda, incx, incy, True)
    ar, ai, br, bi = common._complex_scalars(alpha, beta)
    if (
        m == 0
        or n == 0
        or (ar == 0 and ai == 0)
        or _use_common_gbmv(m, n, kl, ku, lda, incx, incy, trans, True)
    ):
        return common.zgbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)

    physical, pm, pn, pkl, pku, conj = common._row_major_gbmv_args(trans, m, n, kl, ku)
    band = pkl + pku + 1
    out = pm if physical == CUBLAS_OP_N else pn
    if band >= 32 and out <= 2048 and (incy != 1 or y.numel() != out):
        return common.zgbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    a_real = triton.reinterpret(A, tl.float64)
    x_real = triton.reinterpret(x, tl.float64)
    y_real = triton.reinterpret(y, tl.float64)
    alpha_r, alpha_i, beta_r, beta_i = (common._f64_to_i64(v) for v in (ar, ai, br, bi))
    with torch_device_fn.device(A.device):
        if band >= 32 and out <= 2048:
            if br == 0 and bi == 0:
                y.zero_()
            elif br != 1 or bi != 0:
                y.mul_(complex(br, bi))
            kernel = (
                common.zgbmv_n_split_band_kernel.fn
                if physical == CUBLAS_OP_N
                else common.zgbmv_t_split_band_kernel.fn
            )
            kernel[(triton.cdiv(out, 8), 8)](
                a_real,
                x_real,
                y_real,
                alpha_r,
                alpha_i,
                pm,
                pn,
                lda,
                incx,
                incy,
                pkl,
                pku,
                band,
                8,
                out,
                common._band_bucket(band),
                CONJ=conj,
                BLOCK_SIZE_M=8,
                BAND_TILE=8,
                num_warps=2,
                num_stages=1,
            )
        else:
            if band < 32:
                bm, bt = 4, triton.next_power_of_2(band)
            else:
                bm, bt = (8, 8) if out >= 8192 else (4, 16)
            kernel = (
                common.zgbmv_n_kernel.fn
                if physical == CUBLAS_OP_N
                else common.zgbmv_t_kernel.fn
            )
            kernel[(triton.cdiv(out, bm),)](
                a_real,
                x_real,
                y_real,
                alpha_r,
                alpha_i,
                beta_r,
                beta_i,
                pm,
                pn,
                lda,
                incx,
                incy,
                pkl,
                pku,
                band,
                out,
                common._band_bucket(band),
                CONJ=conj,
                BETA_IS_ZERO=br == 0 and bi == 0,
                BLOCK_SIZE_M=bm,
                BAND_TILE=bt,
                num_warps=2,
                num_stages=1,
            )
