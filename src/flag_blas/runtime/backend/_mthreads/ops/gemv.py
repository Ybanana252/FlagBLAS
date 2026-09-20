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
from flag_blas.utils import triton_lang_extension as tle

ScalarType = Union[float, int, complex, torch.Tensor]


_GEMV_KEY = [
    "OUT",
    "RED",
    "LDA",
    "INCX",
    "INCY",
    "TRANS",
    "ZERO_BETA",
    "SPLITS",
    "WIDE",
]
_TINY_GEMV_KEY = ["OUT", "RED", "LDA", "INCX", "SPLITS", "WIDE"]


def _prune_gemv_configs(configs, named_args, **kwargs):
    args = {**named_args, **kwargs}
    if args["WIDE"]:
        # Wide addressing is compiled only with the verified conservative tile.
        return [
            c
            for c in configs
            if c.kwargs.get("BM", 4) == 4
            and c.kwargs["BK"] == 128
            and c.num_warps == 4
            and c.num_stages == 1
        ]
    # The wide-column tile is useful only with enough parallel reduction chunks.
    return [
        c
        for c in configs
        if c.kwargs["BK"] != 1
        or (args.get("TRANS", False) and args["OUT"] >= 8192 and args["SPLITS"] >= 32)
    ]


def _prune_bfgemv_configs(configs, named_args, **kwargs):
    configs = _prune_gemv_configs(configs, named_args, **kwargs)
    args = {**named_args, **kwargs}
    if (
        not args["WIDE"]
        and not args["TRANS"]
        and args["OUT"] <= 64
        and args["RED"] <= 128
    ):
        # Avoid large masked reduction tiles on small matrix-vector products.
        return [c for c in configs if c.kwargs["BM"] <= 8 and c.kwargs["BK"] == 128]
    return configs


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mthreads_sgemv"),
    key=_GEMV_KEY,
    restore_value=["Y"],
    prune_configs_by={"early_config_prune": _prune_gemv_configs},
)
@triton.jit
def mthreads_sgemv_kernel(
    A,
    X,
    Y,
    P,
    alpha,
    beta,
    OUT,
    RED,
    LDA,
    INCX,
    INCY,
    TRANS: tl.constexpr,
    ZERO_BETA: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
    WIDE: tl.constexpr = False,
):
    # Keep the common path in int32; widen before products for large addresses.
    row_program = tl.program_id(0)
    part = tl.program_id(1)
    if WIDE:
        row_program = row_program.to(tl.int64)
        part = part.to(tl.int64)
        OUT = tl.cast(OUT, tl.int64)
        RED = tl.cast(RED, tl.int64)
        LDA = tl.cast(LDA, tl.int64)
        INCX = tl.cast(INCX, tl.int64)
        INCY = tl.cast(INCY, tl.int64)
    rows = row_program * BM + tl.arange(0, BM)
    chunk = tl.cdiv(RED, SPLITS)
    begin = part * chunk
    end = tl.minimum(begin + chunk, RED)
    k = begin + tl.arange(0, BK)
    if TRANS:
        ap = A + rows[:, None] + k[None, :] * LDA
        step = BK * LDA
    else:
        ap = A + rows[:, None] * LDA + k[None, :]
        step = BK
    xp = X + k * INCX
    acc = tl.zeros((BM, BK), tl.float32)
    for offset in range(0, chunk, BK):
        mask_k = k + offset < end
        av = tl.load(ap, (rows[:, None] < OUT) & mask_k[None, :], 0.0)
        xv = tl.load(xp, mask_k, 0.0)
        acc += av * xv[None, :]
        ap += step
        xp += BK * INCX
    result = tl.sum(acc, 1)
    if SPLITS == 1:
        result *= alpha
        if not ZERO_BETA:
            result += beta * tl.load(Y + rows * INCY, rows < OUT, 0.0)
        tl.store(Y + rows * INCY, result, rows < OUT)
    else:
        tl.store(P + part * OUT + rows, result, rows < OUT)


@libentry()
@triton.jit
def mthreads_gemv_reduce_kernel(
    P,
    Y,
    alpha,
    beta,
    OUT,
    INCY,
    SPLITS: tl.constexpr,
    ZERO_BETA: tl.constexpr,
    BM: tl.constexpr,
    BS: tl.constexpr,
    WIDE: tl.constexpr = False,
):
    if WIDE:
        OUT = tl.cast(OUT, tl.int64)
        INCY = tl.cast(INCY, tl.int64)
    rows = tle.program_id(0) * BM + tl.arange(0, BM)
    parts = tl.arange(0, BS)
    value = tl.load(
        P + parts[:, None] * OUT + rows[None, :],
        (parts[:, None] < SPLITS) & (rows[None, :] < OUT),
        0.0,
    )
    result = alpha * tl.sum(value, 0)
    if not ZERO_BETA:
        result += beta * tl.load(Y + rows * INCY, rows < OUT, 0.0)
    tl.store(Y + rows * INCY, result, rows < OUT)


@libentry()
@triton.jit
def mthreads_gemv_scale_kernel(Y, beta, OUT, INCY, ZERO: tl.constexpr, B: tl.constexpr):
    rows = tle.program_id(0) * B + tl.arange(0, B)
    if ZERO:
        result = tl.full((B,), 0.0, tl.float32)
    else:
        result = beta * tl.load(Y + rows * INCY, rows < OUT, 0.0)
    tl.store(Y + rows * INCY, result, rows < OUT)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mthreads_sgemv_tiny_t"),
    key=_TINY_GEMV_KEY,
    prune_configs_by={"early_config_prune": _prune_gemv_configs},
)
@triton.jit
def mthreads_sgemv_tiny_t_kernel(
    A,
    X,
    P,
    RED: tl.constexpr,
    LDA: tl.constexpr,
    INCX,
    OUT: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
    WIDE: tl.constexpr = False,
):
    part = tl.program_id(0)
    if WIDE:
        part = part.to(tl.int64)
        INCX = tl.cast(INCX, tl.int64)
    chunk = tl.cdiv(RED, SPLITS)
    begin = part * chunk
    end = tl.minimum(begin + chunk, RED)
    k = begin + tl.arange(0, BK)
    for col in tl.static_range(OUT):
        acc = tl.full((BK,), 0.0, tl.float32)
        for offset in range(0, chunk, BK):
            ki = k + offset
            av = tl.load(A + ki * LDA + col, ki < end, 0.0)
            xv = tl.load(X + ki * INCX, ki < end, 0.0)
            acc += av * xv
        value = tl.sum(acc, 0)
        tl.store(P + part * OUT + col, value)


def sgemv(
    trans: int,
    m: int,
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
    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    assert A.device == x.device == y.device and A.device.type == "musa"
    assert trans in (CUBLAS_OP_N, CUBLAS_OP_T) and m >= 0 and n >= 0
    assert lda >= n and incx > 0 and incy > 0
    if m == 0 or n == 0:
        return
    out, red = (m, n) if trans == CUBLAS_OP_N else (n, m)
    assert A.numel() >= (m - 1) * lda + n
    assert x.numel() >= (red - 1) * incx + 1
    assert y.numel() >= (out - 1) * incy + 1
    alpha, beta = float(alpha), float(beta)
    with torch_device_fn.device(A.device):
        if alpha == 0:
            if beta != 1:
                mthreads_gemv_scale_kernel[(triton.cdiv(out, 256),)](
                    y, beta, out, incy, beta == 0, 256, num_warps=4
                )
            return
        if out <= 64 and red >= 4096:
            splits = 128 if out <= 4 else 64
        elif trans == CUBLAS_OP_T and max(m, n) > 1024 and lda % 16 != 0 and red > 64:
            splits = 8
        else:
            splits = 1
        wide = (
            max(m * lda, red * incx, out * incy, out * splits, red + splits)
            > 2**31 - 1
        )
        partial = (
            torch.empty((splits, out), dtype=y.dtype, device=y.device)
            if splits > 1
            else y
        )
        if trans == CUBLAS_OP_T and out <= 4 and splits > 1:
            mthreads_sgemv_tiny_t_kernel[(splits,)](
                A,
                x,
                partial,
                red,
                lda,
                incx,
                out,
                SPLITS=splits,
                WIDE=wide,
            )
        else:
            # Restore logical output only; split kernels do not write Y.
            tune_y = y.view(-1)[: out * incy : incy] if splits == 1 else y.view(-1)[:0]
            mthreads_sgemv_kernel[lambda meta: (triton.cdiv(out, meta["BM"]), splits)](
                A,
                x,
                tune_y,
                partial,
                alpha,
                beta,
                out,
                red,
                lda,
                incx,
                incy,
                trans == CUBLAS_OP_T,
                beta == 0,
                SPLITS=splits,
                WIDE=wide,
            )
        if splits > 1:
            finish_bm = 1 if out <= 4 and not wide else 32
            finish_warps = 1 if finish_bm == 1 else 4
            mthreads_gemv_reduce_kernel[(triton.cdiv(out, finish_bm),)](
                partial,
                y,
                alpha,
                beta,
                out,
                incy,
                splits,
                beta == 0,
                finish_bm,
                triton.next_power_of_2(splits),
                num_warps=finish_warps,
                num_stages=1,
                WIDE=wide,
            )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mthreads_hgemv"),
    key=_GEMV_KEY,
    restore_value=["Y"],
    prune_configs_by={"early_config_prune": _prune_gemv_configs},
)
@triton.jit
def mthreads_hgemv_kernel(
    A,
    X,
    Y,
    P,
    alpha,
    beta,
    OUT,
    RED,
    LDA,
    INCX,
    INCY,
    TRANS: tl.constexpr,
    ZERO_BETA: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
    WIDE: tl.constexpr = False,
):
    # Keep the common path in int32; widen before products for large addresses.
    row_program = tl.program_id(0)
    part = tl.program_id(1)
    if WIDE:
        row_program = row_program.to(tl.int64)
        part = part.to(tl.int64)
        OUT = tl.cast(OUT, tl.int64)
        RED = tl.cast(RED, tl.int64)
        LDA = tl.cast(LDA, tl.int64)
        INCX = tl.cast(INCX, tl.int64)
        INCY = tl.cast(INCY, tl.int64)
    rows = row_program * BM + tl.arange(0, BM)
    chunk = tl.cdiv(RED, SPLITS)
    begin = part * chunk
    end = tl.minimum(begin + chunk, RED)
    k = begin + tl.arange(0, BK)
    if TRANS:
        ap = A + rows[:, None] + k[None, :] * LDA
        step = BK * LDA
    else:
        ap = A + rows[:, None] * LDA + k[None, :]
        step = BK
    xp = X + k * INCX
    acc = tl.zeros((BM, BK), tl.float32)
    for offset in range(0, chunk, BK):
        mask_k = k + offset < end
        av = tl.load(ap, (rows[:, None] < OUT) & mask_k[None, :], 0.0).to(tl.float32)
        xv = tl.load(xp, mask_k, 0.0).to(tl.float32)
        acc += av * xv[None, :]
        ap += step
        xp += BK * INCX
    result = tl.sum(acc, 1)
    if SPLITS == 1:
        result *= alpha
        if not ZERO_BETA:
            result += beta * tl.load(Y + rows * INCY, rows < OUT, 0.0)
        tl.store(Y + rows * INCY, result, rows < OUT)
    else:
        tl.store(P + part * OUT + rows, result, rows < OUT)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mthreads_hgemv_tiny_t"),
    key=_TINY_GEMV_KEY,
    prune_configs_by={"early_config_prune": _prune_gemv_configs},
)
@triton.jit
def mthreads_hgemv_tiny_t_kernel(
    A,
    X,
    P,
    RED: tl.constexpr,
    LDA: tl.constexpr,
    INCX,
    OUT: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
    WIDE: tl.constexpr = False,
):
    part = tl.program_id(0)
    if WIDE:
        part = part.to(tl.int64)
        INCX = tl.cast(INCX, tl.int64)
    chunk = tl.cdiv(RED, SPLITS)
    begin = part * chunk
    end = tl.minimum(begin + chunk, RED)
    k = begin + tl.arange(0, BK)
    for col in tl.static_range(OUT):
        acc = tl.full((BK,), 0.0, tl.float32)
        for offset in range(0, chunk, BK):
            ki = k + offset
            av = tl.load(A + ki * LDA + col, ki < end, 0.0).to(tl.float32)
            xv = tl.load(X + ki * INCX, ki < end, 0.0).to(tl.float32)
            acc += av * xv
        value = tl.sum(acc, 0)
        tl.store(P + part * OUT + col, value)


def hgemv(
    trans: int,
    m: int,
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
    assert A.dtype == x.dtype == y.dtype == torch.float16
    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    assert A.device == x.device == y.device and A.device.type == "musa"
    assert trans in (CUBLAS_OP_N, CUBLAS_OP_T) and m >= 0 and n >= 0
    assert lda >= n and incx > 0 and incy > 0
    if m == 0 or n == 0:
        return
    out, red = (m, n) if trans == CUBLAS_OP_N else (n, m)
    assert A.numel() >= (m - 1) * lda + n
    assert x.numel() >= (red - 1) * incx + 1
    assert y.numel() >= (out - 1) * incy + 1
    alpha, beta = float(alpha), float(beta)
    with torch_device_fn.device(A.device):
        if alpha == 0:
            if beta != 1:
                mthreads_gemv_scale_kernel[(triton.cdiv(out, 256),)](
                    y, beta, out, incy, beta == 0, 256, num_warps=4
                )
            return
        if out <= 64 and red >= 4096:
            splits = 128 if trans == CUBLAS_OP_T and out <= 4 else 64
        elif trans == CUBLAS_OP_T and max(m, n) > 1024 and lda % 16 != 0 and red > 64:
            splits = 32 if out >= 8192 and red >= 2048 else 4
        else:
            splits = 1
        wide = (
            max(m * lda, red * incx, out * incy, out * splits, red + splits)
            > 2**31 - 1
        )
        partial = (
            torch.empty((splits, out), dtype=torch.float32, device=y.device)
            if splits > 1
            else y
        )
        if trans == CUBLAS_OP_T and out <= 4 and splits > 1:
            mthreads_hgemv_tiny_t_kernel[(splits,)](
                A,
                x,
                partial,
                red,
                lda,
                incx,
                out,
                SPLITS=splits,
                WIDE=wide,
            )
        else:
            # Restore logical output only; split kernels do not write Y.
            tune_y = y.view(-1)[: out * incy : incy] if splits == 1 else y.view(-1)[:0]
            mthreads_hgemv_kernel[lambda meta: (triton.cdiv(out, meta["BM"]), splits)](
                A,
                x,
                tune_y,
                partial,
                alpha,
                beta,
                out,
                red,
                lda,
                incx,
                incy,
                trans == CUBLAS_OP_T,
                beta == 0,
                SPLITS=splits,
                WIDE=wide,
            )
        if splits > 1:
            finish_bm = 1 if trans == CUBLAS_OP_T and out <= 4 and not wide else 32
            finish_warps = 1 if finish_bm == 1 else 4
            mthreads_gemv_reduce_kernel[(triton.cdiv(out, finish_bm),)](
                partial,
                y,
                alpha,
                beta,
                out,
                incy,
                splits,
                beta == 0,
                finish_bm,
                triton.next_power_of_2(splits),
                num_warps=finish_warps,
                num_stages=1,
                WIDE=wide,
            )


@triton.jit
def _cgemv_unpack(value):
    real = value.to(tl.int32).to(tl.float32, bitcast=True)
    imag = (value >> 32).to(tl.int32).to(tl.float32, bitcast=True)
    return real, imag


@triton.jit
def _cgemv_store_result(
    Y, rows, OUT, INCY, real, imag, AR, AI, BR, BI, ZERO_BETA: tl.constexpr
):
    yr = AR * real - AI * imag
    yi = AR * imag + AI * real
    offsets = rows * INCY * 2
    if not ZERO_BETA:
        oldr = tl.load(Y + offsets, rows < OUT, 0.0)
        oldi = tl.load(Y + offsets + 1, rows < OUT, 0.0)
        yr += BR * oldr - BI * oldi
        yi += BR * oldi + BI * oldr
    tl.store(Y + offsets, yr, rows < OUT)
    tl.store(Y + offsets + 1, yi, rows < OUT)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mthreads_cgemv"),
    key=_GEMV_KEY + ["CONJ"],
    restore_value=["Y"],
    prune_configs_by={"early_config_prune": _prune_gemv_configs},
)
@triton.jit
def mthreads_cgemv_kernel(
    A,
    X,
    Y,
    P,
    AR,
    AI,
    BR,
    BI,
    OUT,
    RED,
    LDA,
    INCX,
    INCY,
    TRANS: tl.constexpr,
    CONJ: tl.constexpr,
    ZERO_BETA: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
    WIDE: tl.constexpr,
):
    # Narrow indices avoid MUSA codegen faults; widen before products when needed.
    row_program = tl.program_id(0)
    part = tl.program_id(1)
    if WIDE:
        row_program = row_program.to(tl.int64)
        part = part.to(tl.int64)
        OUT = tl.cast(OUT, tl.int64)
        RED = tl.cast(RED, tl.int64)
        LDA = tl.cast(LDA, tl.int64)
        INCX = tl.cast(INCX, tl.int64)
        INCY = tl.cast(INCY, tl.int64)
    A = A.to(tl.pointer_type(tl.int64))
    X = X.to(tl.pointer_type(tl.int64))
    rows = row_program * BM + tl.arange(0, BM)
    chunk = tl.cdiv(RED, SPLITS)
    begin = part * chunk
    end = tl.minimum(begin + chunk, RED)
    k = begin + tl.arange(0, BK)
    if TRANS:
        ap = A + rows[:, None] + k[None, :] * LDA
        step = BK * LDA
    else:
        ap = A + rows[:, None] * LDA + k[None, :]
        step = BK
    xp = X + k * INCX
    acc_r = tl.zeros((BM, BK), tl.float32)
    acc_i = tl.zeros((BM, BK), tl.float32)
    for offset in range(0, chunk, BK):
        mask_k = k + offset < end
        av = tl.load(ap, (rows[:, None] < OUT) & mask_k[None, :], 0)
        xv = tl.load(xp, mask_k, 0)
        ar, ai = _cgemv_unpack(av)
        xr, xi = _cgemv_unpack(xv)
        if CONJ:
            ai = -ai
        acc_r += ar * xr[None, :] - ai * xi[None, :]
        acc_i += ar * xi[None, :] + ai * xr[None, :]
        ap += step
        xp += BK * INCX
    real = tl.sum(acc_r, 1)
    imag = tl.sum(acc_i, 1)
    if SPLITS == 1:
        _cgemv_store_result(Y, rows, OUT, INCY, real, imag, AR, AI, BR, BI, ZERO_BETA)
    else:
        po = (part * OUT + rows) * 2
        tl.store(P + po, real, rows < OUT)
        tl.store(P + po + 1, imag, rows < OUT)


@libentry()
@triton.jit
def mthreads_cgemv_reduce_kernel(
    P,
    Y,
    AR,
    AI,
    BR,
    BI,
    OUT,
    INCY,
    SPLITS: tl.constexpr,
    ZERO_BETA: tl.constexpr,
    BM: tl.constexpr,
    BS: tl.constexpr,
    WIDE: tl.constexpr,
):
    pid = tl.program_id(0)
    if WIDE:
        pid = pid.to(tl.int64)
        OUT = tl.cast(OUT, tl.int64)
        INCY = tl.cast(INCY, tl.int64)
    rows = pid * BM + tl.arange(0, BM)
    parts = tl.arange(0, BS)
    offsets = (parts[:, None] * OUT + rows[None, :]) * 2
    mask = (parts[:, None] < SPLITS) & (rows[None, :] < OUT)
    real = tl.sum(tl.load(P + offsets, mask, 0.0), 0)
    imag = tl.sum(tl.load(P + offsets + 1, mask, 0.0), 0)
    _cgemv_store_result(Y, rows, OUT, INCY, real, imag, AR, AI, BR, BI, ZERO_BETA)


@libentry()
@triton.jit
def mthreads_cgemv_scale_kernel(
    Y, BR, BI, OUT, INCY, ZERO: tl.constexpr, B: tl.constexpr
):
    rows = tl.program_id(0).to(tl.int64) * B + tl.arange(0, B)
    offsets = rows * INCY * 2
    real = tl.full((B,), 0.0, tl.float32)
    imag = tl.full((B,), 0.0, tl.float32)
    if not ZERO:
        yr = tl.load(Y + offsets, rows < OUT, 0.0)
        yi = tl.load(Y + offsets + 1, rows < OUT, 0.0)
        real = BR * yr - BI * yi
        imag = BR * yi + BI * yr
    tl.store(Y + offsets, real, rows < OUT)
    tl.store(Y + offsets + 1, imag, rows < OUT)


def cgemv(
    trans: int,
    m: int,
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
    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    assert A.device == x.device == y.device and A.device.type == "musa"
    assert trans in (CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C) and m >= 0 and n >= 0
    assert lda >= n and incx > 0 and incy > 0
    if m == 0 or n == 0:
        return
    out, red = (m, n) if trans == CUBLAS_OP_N else (n, m)
    assert A.numel() >= (m - 1) * lda + n
    assert x.numel() >= (red - 1) * incx + 1
    assert y.numel() >= (out - 1) * incy + 1
    alpha = complex(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta = complex(beta.item() if isinstance(beta, torch.Tensor) else beta)
    with torch_device_fn.device(A.device):
        yf = torch.view_as_real(y)
        if alpha == 0:
            if beta != 1:
                mthreads_cgemv_scale_kernel[(triton.cdiv(out, 256),)](
                    yf, beta.real, beta.imag, out, incy, beta == 0, 256, num_warps=4
                )
            return
        if out <= 64 and red >= 4096:
            splits = 128 if trans != CUBLAS_OP_N and out <= 4 else 64
        elif trans != CUBLAS_OP_N and max(m, n) > 1024 and red > 64:
            splits = 8 if lda % 16 else (16 if out > 1024 else 1)
        else:
            splits = 1
        wide = (
            max(m * lda, red * incx, 2 * out * incy, 2 * out * splits, red + splits)
            > 2**31 - 1
        )
        partial = (
            torch.empty((splits, out, 2), dtype=torch.float32, device=y.device)
            if splits > 1
            else yf
        )
        if wide:
            # One verified wide-address tile; avoid MUSA's large-stride real-view copy.
            kernel = mthreads_cgemv_kernel.jit_function
            launch_config = dict(BM=4, BK=128, num_warps=4, num_stages=1)
            tune_y = yf
        else:
            kernel = mthreads_cgemv_kernel
            launch_config = {}
            tune_y = torch.view_as_real(
                y.view(-1)[: out * incy : incy] if splits == 1 else y.view(-1)[:0]
            )
        kernel[lambda meta: (triton.cdiv(out, meta["BM"]), splits)](
            torch.view_as_real(A),
            torch.view_as_real(x),
            tune_y,
            partial,
            alpha.real,
            alpha.imag,
            beta.real,
            beta.imag,
            out,
            red,
            lda,
            incx,
            incy,
            trans != CUBLAS_OP_N,
            trans == CUBLAS_OP_C,
            beta == 0,
            SPLITS=splits,
            WIDE=wide,
            **launch_config,
        )
        if splits > 1:
            fbm = 1 if out <= 4 and not wide else 32
            fw = 1 if fbm == 1 else 4
            mthreads_cgemv_reduce_kernel[(triton.cdiv(out, fbm),)](
                partial,
                yf,
                alpha.real,
                alpha.imag,
                beta.real,
                beta.imag,
                out,
                incy,
                splits,
                beta == 0,
                fbm,
                triton.next_power_of_2(splits),
                wide,
                num_warps=fw,
                num_stages=1,
            )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mthreads_bfgemv"),
    key=_GEMV_KEY,
    restore_value=["Y"],
    prune_configs_by={"early_config_prune": _prune_bfgemv_configs},
)
@triton.jit
def mthreads_bfgemv_kernel(
    A,
    X,
    Y,
    P,
    alpha,
    beta,
    OUT,
    RED,
    LDA,
    INCX,
    INCY,
    TRANS: tl.constexpr,
    ZERO_BETA: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
    WIDE: tl.constexpr = False,
):
    # Keep the common path in int32; widen before products for large addresses.
    row_program = tl.program_id(0)
    part = tl.program_id(1)
    if WIDE:
        row_program = row_program.to(tl.int64)
        part = part.to(tl.int64)
        OUT = tl.cast(OUT, tl.int64)
        RED = tl.cast(RED, tl.int64)
        LDA = tl.cast(LDA, tl.int64)
        INCX = tl.cast(INCX, tl.int64)
        INCY = tl.cast(INCY, tl.int64)
    rows = row_program * BM + tl.arange(0, BM)
    chunk = tl.cdiv(RED, SPLITS)
    begin = part * chunk
    end = tl.minimum(begin + chunk, RED)
    k = begin + tl.arange(0, BK)
    if TRANS:
        ap = A + rows[:, None] + k[None, :] * LDA
        step = BK * LDA
    else:
        ap = A + rows[:, None] * LDA + k[None, :]
        step = BK
    xp = X + k * INCX
    acc = tl.zeros((BM, BK), tl.float32)
    for offset in range(0, chunk, BK):
        mask_k = k + offset < end
        av = tl.load(ap, (rows[:, None] < OUT) & mask_k[None, :], 0.0).to(tl.float32)
        xv = tl.load(xp, mask_k, 0.0).to(tl.float32)
        acc += av * xv[None, :]
        ap += step
        xp += BK * INCX
    result = tl.sum(acc, 1)
    if SPLITS == 1:
        result *= alpha
        if not ZERO_BETA:
            result += beta * tl.load(Y + rows * INCY, rows < OUT, 0.0)
        tl.store(Y + rows * INCY, result, rows < OUT)
    else:
        tl.store(P + part * OUT + rows, result, rows < OUT)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("mthreads_bfgemv_tiny_t"),
    key=_TINY_GEMV_KEY,
    prune_configs_by={"early_config_prune": _prune_gemv_configs},
)
@triton.jit
def mthreads_bfgemv_tiny_t_kernel(
    A,
    X,
    P,
    RED: tl.constexpr,
    LDA: tl.constexpr,
    INCX,
    OUT: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
    WIDE: tl.constexpr = False,
):
    part = tl.program_id(0)
    if WIDE:
        part = part.to(tl.int64)
        INCX = tl.cast(INCX, tl.int64)
    chunk = tl.cdiv(RED, SPLITS)
    begin = part * chunk
    end = tl.minimum(begin + chunk, RED)
    k = begin + tl.arange(0, BK)
    for col in tl.static_range(OUT):
        acc = tl.full((BK,), 0.0, tl.float32)
        for offset in range(0, chunk, BK):
            ki = k + offset
            av = tl.load(A + ki * LDA + col, ki < end, 0.0).to(tl.float32)
            xv = tl.load(X + ki * INCX, ki < end, 0.0).to(tl.float32)
            acc += av * xv
        value = tl.sum(acc, 0)
        tl.store(P + part * OUT + col, value)


def bfgemv(
    trans: int,
    m: int,
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
    assert A.dtype == x.dtype == y.dtype == torch.bfloat16
    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    assert A.device == x.device == y.device and A.device.type == "musa"
    assert trans in (CUBLAS_OP_N, CUBLAS_OP_T) and m >= 0 and n >= 0
    assert lda >= n and incx > 0 and incy > 0
    if m == 0 or n == 0:
        return
    out, red = (m, n) if trans == CUBLAS_OP_N else (n, m)
    assert A.numel() >= (m - 1) * lda + n
    assert x.numel() >= (red - 1) * incx + 1
    assert y.numel() >= (out - 1) * incy + 1
    alpha, beta = float(alpha), float(beta)
    with torch_device_fn.device(A.device):
        if alpha == 0:
            if beta != 1:
                mthreads_gemv_scale_kernel[(triton.cdiv(out, 256),)](
                    y, beta, out, incy, beta == 0, 256, num_warps=4
                )
            return
        if out <= 64 and red >= 4096:
            splits = 128 if trans == CUBLAS_OP_T and out <= 4 else 64
        elif trans == CUBLAS_OP_T and max(m, n) > 1024 and lda % 16 != 0 and red > 64:
            splits = 32 if out >= 8192 and red >= 2048 else 4
        else:
            splits = 1
        wide = (
            max(m * lda, red * incx, out * incy, out * splits, red + splits)
            > 2**31 - 1
        )
        partial = (
            torch.empty((splits, out), dtype=torch.float32, device=y.device)
            if splits > 1
            else y
        )
        if trans == CUBLAS_OP_T and out <= 4 and splits > 1:
            mthreads_bfgemv_tiny_t_kernel[(splits,)](
                A,
                x,
                partial,
                red,
                lda,
                incx,
                out,
                SPLITS=splits,
                WIDE=wide,
            )
        else:
            # Restore logical output only; split kernels do not write Y.
            tune_y = y.view(-1)[: out * incy : incy] if splits == 1 else y.view(-1)[:0]
            mthreads_bfgemv_kernel[lambda meta: (triton.cdiv(out, meta["BM"]), splits)](
                A,
                x,
                tune_y,
                partial,
                alpha,
                beta,
                out,
                red,
                lda,
                incx,
                incy,
                trans == CUBLAS_OP_T,
                beta == 0,
                SPLITS=splits,
                WIDE=wide,
            )
        if splits > 1:
            finish_bm = 1 if trans == CUBLAS_OP_T and out <= 4 and not wide else 32
            finish_warps = 1 if finish_bm == 1 else 4
            mthreads_gemv_reduce_kernel[(triton.cdiv(out, finish_bm),)](
                partial,
                y,
                alpha,
                beta,
                out,
                incy,
                splits,
                beta == 0,
                finish_bm,
                triton.next_power_of_2(splits),
                num_warps=finish_warps,
                num_stages=1,
                WIDE=wide,
            )
