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
import struct
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner
from flag_blas.utils import triton_lang_extension as tle


ScalarType = Union[float, int, torch.Tensor]

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1
_NARROW_T_MIN_K = 65536
_NARROW_T_MAX_OUTPUT = 4
_NARROW_T_SPLITS = 32
_LARGE_LOWP_T_MIN_K = 8192
_LARGE_LOWP_T_MIN_OUTPUT = 3584
_NARROW_LOWP_T_MIN_K = 65536
_NARROW_LOWP_T_MAX_OUTPUT = 2
_THEAD_SM_COUNT = 64
_K1_T_MIN_OUTPUT = 65536

_common = importlib.import_module("flag_blas.ops.level2.gemv")


# Small, verified regions only; keep the shared autotuner configs untouched.
_sg_small_t = runtime.get_tuned_config("thead_sgemv_t_fullk")
_SMALL_T_CONFIGS = {
    torch.float32: {c.kwargs["BLOCK_SIZE_K"]: c for c in _sg_small_t},
    torch.float64: {
        256: runtime.get_tuned_config("thead_dgemv_t_fullk")[0],
        # Same tile as the verified FP32 1024 region; no duplicate YAML config.
        1024: next(c for c in _sg_small_t if c.kwargs["BLOCK_SIZE_K"] == 1024),
    },
}
_LOWP_N_CONFIG = runtime.get_tuned_config("thead_lowp_gemv_n_medium")[0]
_LOWP_T_K4_CONFIG = runtime.get_tuned_config("thead_lowp_gemv_t_k4")[0]
_LOWP_DIRECT_KERNELS = {
    (dtype, trans): libentry()(getattr(_common, name + suffix).jit_function)
    for dtype, name in ((torch.float16, "hgemv"), (torch.bfloat16, "bfgemv"))
    for trans, suffix in ((CUBLAS_OP_N, "_n_kernel"), (CUBLAS_OP_T, "_t_kernel"))
}


@libentry()
@triton.jit
def sgemv_t_runtime_k_thead_kernel(
    A, X, Y, alpha, beta, M, K, LDA, INCX, INCY,
    BETA_ZERO: tl.constexpr, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    # Dispatch guarantees K <= BLOCK_SIZE_K; the actual K remains a runtime
    # argument, with masked loads (also valid for non-power-of-two lengths).
    cols = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    ks = tl.arange(0, BLOCK_SIZE_K)
    a = tl.load(
        A + ks[:, None] * LDA + cols[None, :],
        (ks[:, None] < K) & (cols[None, :] < M), other=0,
    )
    x = tl.load(X + ks * INCX, ks < K, other=0)
    result = alpha * tl.sum(a * x[:, None], 0)
    if not BETA_ZERO:
        result += beta * tl.load(Y + cols * INCY, cols < M, other=0)
    tl.store(Y + cols * INCY, result, cols < M)


@libentry()
@triton.jit
def gemv_t_fullk_thead_kernel(
    A, X, Y, alpha, beta, M: tl.constexpr, K: tl.constexpr,
    LDA, INCX, INCY, BETA_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, FP64: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    ks = tl.arange(0, BLOCK_SIZE_K)
    if FP64:
        acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), tl.float64)
        # Zero bit patterns can be inferred as int32 at the Python boundary.
        aa = alpha.to(tl.int64).to(tl.float64, bitcast=True)
        bb = beta.to(tl.int64).to(tl.float64, bitcast=True)
    else:
        acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), tl.float32)
        aa = alpha
        bb = beta
    for start in range(0, K, BLOCK_SIZE_K):
        k = start + ks
        a = tl.load(
            A + k[:, None] * LDA + cols[None, :],
            (k[:, None] < K) & (cols[None, :] < M), other=0,
        )
        x = tl.load(X + k * INCX, k < K, other=0)
        acc += a * x[:, None]
    result = aa * tl.sum(acc, 0)
    if not BETA_ZERO:
        result += bb * tl.load(Y + cols * INCY, cols < M, other=0)
    tl.store(Y + cols * INCY, result, cols < M)


def _try_direct_gemv(dtype, trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
    if incx != 1 or incy != 1:
        return False
    real = dtype in _SMALL_T_CONFIGS
    if real:
        config = _SMALL_T_CONFIGS[dtype].get(m) if trans == CUBLAS_OP_T and m == n else None
        kernel = (
            sgemv_t_runtime_k_thead_kernel
            if dtype == torch.float32 else gemv_t_fullk_thead_kernel
        )
    else:
        if trans == CUBLAS_OP_N and m == n == 1024:
            config = _LOWP_N_CONFIG
        elif trans == CUBLAS_OP_T and m == 4 and n == 131072:
            config = _LOWP_T_K4_CONFIG
        else:
            return False
        kernel = _LOWP_DIRECT_KERNELS[(dtype, trans)]
    if config is None:
        return False
    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    assert A.dtype == x.dtype == y.dtype == dtype
    assert A.device == x.device == y.device
    assert lda >= n
    out_m, reduce_k = (m, n) if trans == CUBLAS_OP_N else (n, m)
    assert x.numel() >= reduce_k and y.numel() >= out_m
    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta_value = float(beta.item() if isinstance(beta, torch.Tensor) else beta)
    if alpha_value == 0.0:
        _scale_y(y, beta_value)
        return True
    aa, bb = alpha_value, beta_value
    if dtype == torch.float64:
        aa = struct.unpack("=q", struct.pack("=d", aa))[0]
        bb = struct.unpack("=q", struct.pack("=d", bb))[0]
    extra = {"FP64": True} if dtype == torch.float64 else {}
    with torch_device_fn.device(A.device):
        kernel[(triton.cdiv(out_m, config.kwargs["BLOCK_SIZE_M"]),)](
            A, x, y, aa, bb, out_m, reduce_k, lda, incx, incy,
            beta_value == 0.0, **config.all_kwargs(), **extra,
        )
    return True


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemv_t_narrow_thead"),
    key=["m", "n", "STRIDE_AK", "INCX", "INCY", "num_k_splits"],
    restore_value=["y_ptr"],
)
@triton.jit
def sgemv_t_narrow_thead_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    m,
    n,
    STRIDE_AK,
    INCX,
    INCY,
    alpha: tl.float32,
    num_k_splits,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    chunk_k = (n + num_k_splits - 1) // num_k_splits
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, n)
    ks0 = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + rows[:, None] + (k_begin + ks0)[None, :] * STRIDE_AK
    x_ptrs = x_ptr + (k_begin + ks0) * INCX
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    for k_offset in range(0, chunk_k, BLOCK_SIZE_K):
        ks = k_begin + k_offset + ks0
        k_mask = ks < k_end
        a = tl.load(
            a_ptrs,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        x = tl.load(x_ptrs, mask=k_mask, other=0.0, eviction_policy="evict_last")
        acc += tl.sum(a * x[None, :], axis=1)
        a_ptrs += BLOCK_SIZE_K * STRIDE_AK
        x_ptrs += BLOCK_SIZE_K * INCX
    tl.atomic_add(
        y_ptr + rows * INCY,
        acc * alpha,
        mask=row_mask,
        sem="relaxed",
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("dgemv_t_narrow_thead"),
    key=["m", "n", "STRIDE_AK", "INCX", "INCY", "num_k_splits"],
    restore_value=["y_ptr"],
)
@triton.jit
def dgemv_t_narrow_thead_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    m,
    n,
    STRIDE_AK,
    INCX,
    INCY,
    alpha_int: tl.int64,
    num_k_splits,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    alpha = alpha_int.to(tl.float64, bitcast=True)
    chunk_k = (n + num_k_splits - 1) // num_k_splits
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, n)
    ks0 = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + rows[:, None] + (k_begin + ks0)[None, :] * STRIDE_AK
    x_ptrs = x_ptr + (k_begin + ks0) * INCX
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float64)
    for k_offset in range(0, chunk_k, BLOCK_SIZE_K):
        ks = k_begin + k_offset + ks0
        k_mask = ks < k_end
        a = tl.load(
            a_ptrs,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        )
        x = tl.load(x_ptrs, mask=k_mask, other=0.0, eviction_policy="evict_last")
        acc += tl.sum(a * x[None, :], axis=1)
        a_ptrs += BLOCK_SIZE_K * STRIDE_AK
        x_ptrs += BLOCK_SIZE_K * INCX
    tl.atomic_add(
        y_ptr + rows * INCY,
        acc * alpha,
        mask=row_mask,
        sem="relaxed",
    )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("lowp_gemv_t_thead"),
    key=["out_m", "reduce_k", "STRIDE_AK", "INCX", "INCY", "BETA_IS_ZERO"],
    restore_value=["y_ptr"],
)
@triton.jit
def lowp_gemv_t_thead_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    out_m,
    reduce_k,
    STRIDE_AK,
    INCX,
    INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    rows = tle.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < out_m
    ks0 = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + rows[:, None] + ks0[None, :] * STRIDE_AK
    x_ptrs = x_ptr + ks0 * INCX
    acc_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for k_start in range(0, reduce_k, BLOCK_SIZE_K):
        ks = k_start + ks0
        k_mask = ks < reduce_k
        a = tl.load(
            a_ptrs,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = tl.load(
            x_ptrs,
            mask=k_mask,
            other=0.0,
            eviction_policy="evict_last",
        ).to(tl.float32)
        acc_2d += a * x[None, :]
        a_ptrs += BLOCK_SIZE_K * STRIDE_AK
        x_ptrs += BLOCK_SIZE_K * INCX
    acc = tl.sum(acc_2d, axis=1)
    y_ptrs = y_ptr + rows * INCY
    if BETA_IS_ZERO:
        result = alpha * acc
    else:
        old_y = tl.load(y_ptrs, mask=row_mask, other=0.0).to(tl.float32)
        result = alpha * acc + beta * old_y
    tl.store(y_ptrs, result, mask=row_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("lowp_gemv_t_narrow_split_thead"),
    key=["out_m", "reduce_k", "STRIDE_AK", "num_k_splits"],
)
@triton.jit
def lowp_gemv_t_narrow_split_thead_kernel(
    a_ptr,
    x_ptr,
    partial_ptr,
    out_m,
    reduce_k,
    STRIDE_AK,
    num_k_splits,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tle.program_id(0)
    pid_k = tle.program_id(1)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < out_m
    chunk_k = (reduce_k + num_k_splits - 1) // num_k_splits
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, reduce_k)
    ks0 = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + rows[:, None] + (k_begin + ks0)[None, :] * STRIDE_AK
    x_ptrs = x_ptr + k_begin + ks0
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for k_offset in range(0, chunk_k, BLOCK_SIZE_K):
        ks = k_begin + k_offset + ks0
        k_mask = ks < k_end
        a = tl.load(
            a_ptrs,
            mask=row_mask[:, None] & k_mask[None, :],
            other=0.0,
            eviction_policy="evict_first",
        ).to(tl.float32)
        x = tl.load(
            x_ptrs, mask=k_mask, other=0.0, eviction_policy="evict_last"
        ).to(tl.float32)
        acc += a * x[None, :]
        a_ptrs += BLOCK_SIZE_K * STRIDE_AK
        x_ptrs += BLOCK_SIZE_K
    sums = tl.sum(acc, axis=1)
    tl.store(partial_ptr + pid_k * out_m + rows, sums, mask=row_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemv_t_k1_thead"),
    key=["out_m", "INCY", "BETA_IS_ZERO"],
    restore_value=["y_ptr"],
)
@triton.jit
def sgemv_t_k1_thead_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    out_m,
    INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tle.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < out_m
    scale = tl.load(x_ptr).to(tl.float32)
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y_ptrs = y_ptr + offsets * INCY
    if BETA_IS_ZERO:
        result = alpha * a * scale
    else:
        old_y = tl.load(y_ptrs, mask=mask, other=0.0).to(tl.float32)
        result = alpha * a * scale + beta * old_y
    tl.store(y_ptrs, result, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("lowp_gemv_t_k1_thead"),
    key=["out_m", "INCY", "BETA_IS_ZERO"],
    restore_value=["y_ptr"],
)
@triton.jit
def lowp_gemv_t_k1_thead_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    out_m,
    INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tle.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < out_m
    scale = tl.load(x_ptr).to(tl.float32)
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y_ptrs = y_ptr + offsets * INCY
    if BETA_IS_ZERO:
        result = alpha * a * scale
    else:
        old_y = tl.load(y_ptrs, mask=mask, other=0.0).to(tl.float32)
        result = alpha * a * scale + beta * old_y
    tl.store(y_ptrs, result, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("dgemv_n3_thead"),
    key=["m", "STRIDE_AM", "INCX", "INCY", "BETA_IS_ZERO"],
    restore_value=["y_ptr"],
)
@triton.jit
def dgemv_n3_thead_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_int: tl.int64,
    beta_int: tl.int64,
    m,
    STRIDE_AM,
    INCX,
    INCY,
    BETA_IS_ZERO: tl.constexpr,
    REDUCE_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    rows = tle.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    ks = tl.arange(0, BLOCK_SIZE_K)
    k_mask = ks < REDUCE_K
    a = tl.load(
        a_ptr + rows[:, None] * STRIDE_AM + ks[None, :],
        mask=row_mask[:, None] & k_mask[None, :],
        other=0.0,
    )
    x = tl.load(x_ptr + ks * INCX, mask=k_mask, other=0.0)
    acc = tl.sum(a * x[None, :], axis=1)
    alpha = alpha_int.to(tl.float64, bitcast=True)
    beta = beta_int.to(tl.float64, bitcast=True)
    y_ptrs = y_ptr + rows * INCY
    if BETA_IS_ZERO:
        result = alpha * acc
    else:
        old_y = tl.load(y_ptrs, mask=row_mask, other=0.0)
        result = alpha * acc + beta * old_y
    tl.store(y_ptrs, result, mask=row_mask)


def _use_narrow_t_path(trans: int, m: int, n: int, incx: int, incy: int) -> bool:
    return (
        trans == CUBLAS_OP_T
        and m >= _NARROW_T_MIN_K
        and 0 < n <= _NARROW_T_MAX_OUTPUT
        and incx == 1
        and incy == 1
    )


def _use_large_lowp_t_path(
    trans: int, m: int, n: int, incx: int, incy: int
) -> bool:
    return (
        trans == CUBLAS_OP_T
        and m >= _LARGE_LOWP_T_MIN_K
        and n >= _LARGE_LOWP_T_MIN_OUTPUT
        and incx == 1
        and incy == 1
    )


def _use_narrow_lowp_t_path(
    trans: int, m: int, n: int, incx: int, incy: int
) -> bool:
    return (
        trans == CUBLAS_OP_T
        and m >= _NARROW_LOWP_T_MIN_K
        and 0 < n <= _NARROW_LOWP_T_MAX_OUTPUT
        and incx == 1
        and incy == 1
    )


def _use_t_k1_path(trans: int, m: int, n: int, incx: int, incy: int) -> bool:
    return (
        trans == CUBLAS_OP_T
        and m == 1
        and n >= _K1_T_MIN_OUTPUT
        and incx == 1
        and incy == 1
    )


def _use_dgemv_n3_path(trans: int, m: int, n: int, incx: int, incy: int) -> bool:
    return (
        trans == CUBLAS_OP_N
        and m >= 65536
        and n == 3
        and incx == 1
        and incy == 1
    )


def _prepare_narrow_t(
    trans: int,
    m: int,
    n: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    dtype: torch.dtype,
) -> None:
    assert A.is_contiguous()
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert A.dtype == dtype
    assert x.dtype == dtype
    assert y.dtype == dtype
    assert A.device == x.device == y.device
    assert trans == CUBLAS_OP_T
    assert incx > 0 and incy > 0
    assert lda >= n
    assert x.numel() >= 1 + (m - 1) * incx
    assert y.numel() >= 1 + (n - 1) * incy


def _scale_y(y: torch.Tensor, beta: float) -> None:
    if beta == 0.0:
        y.zero_()
    elif beta != 1.0:
        y.mul_(beta)


def _gemv_t_k1(
    kernel,
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
    dtype: torch.dtype,
) -> None:
    assert A.is_contiguous()
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert A.dtype == dtype
    assert x.dtype == dtype
    assert y.dtype == dtype
    assert A.device == x.device == y.device
    assert trans == CUBLAS_OP_T and m == 1
    assert incx == 1 and incy == 1
    assert lda >= n
    assert x.numel() >= 1
    assert y.numel() >= n

    alpha_value = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta_value = beta.item() if isinstance(beta, torch.Tensor) else float(beta)
    if alpha_value == 0.0:
        _scale_y(y, beta_value)
        return

    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    with torch_device_fn.device(A.device):
        kernel[grid](
            A,
            x,
            y,
            alpha_value,
            beta_value,
            n,
            incy,
            beta_value == 0.0,
        )


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
    if _try_direct_gemv(torch.float32, trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
        return
    if _use_t_k1_path(trans, m, n, incx, incy):
        return _gemv_t_k1(
            sgemv_t_k1_thead_kernel,
            trans,
            m,
            n,
            alpha,
            A,
            lda,
            x,
            incx,
            beta,
            y,
            incy,
            torch.float32,
        )
    if not _use_narrow_t_path(trans, m, n, incx, incy):
        return _common.sgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)

    _prepare_narrow_t(trans, m, n, A, lda, x, incx, y, incy, torch.float32)
    alpha_value = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta_value = beta.item() if isinstance(beta, torch.Tensor) else float(beta)
    if alpha_value == 0.0:
        _scale_y(y, beta_value)
        return

    _scale_y(y, beta_value)
    grid = lambda meta: (
        triton.cdiv(n, meta["BLOCK_SIZE_M"]),
        _NARROW_T_SPLITS,
    )
    with torch_device_fn.device(A.device):
        sgemv_t_narrow_thead_kernel[grid](
            A,
            x,
            y,
            n,
            m,
            lda,
            incx,
            incy,
            alpha_value,
            _NARROW_T_SPLITS,
        )


def dgemv(
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
    if _try_direct_gemv(torch.float64, trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
        return
    if _use_dgemv_n3_path(trans, m, n, incx, incy):
        assert A.is_contiguous()
        assert x.is_contiguous()
        assert y.is_contiguous()
        assert A.dtype == torch.float64
        assert x.dtype == torch.float64
        assert y.dtype == torch.float64
        assert A.device == x.device == y.device
        assert lda >= n
        assert x.numel() >= 3
        assert y.numel() >= m
        alpha_value = float(
            alpha.item() if isinstance(alpha, torch.Tensor) else alpha
        )
        beta_value = float(beta.item() if isinstance(beta, torch.Tensor) else beta)
        if alpha_value == 0.0:
            _scale_y(y, beta_value)
            return
        alpha_int = struct.unpack("=q", struct.pack("=d", alpha_value))[0]
        beta_int = struct.unpack("=q", struct.pack("=d", beta_value))[0]
        grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE_M"]),)
        with torch_device_fn.device(A.device):
            dgemv_n3_thead_kernel[grid](
                A,
                x,
                y,
                alpha_int,
                beta_int,
                m,
                lda,
                incx,
                incy,
                beta_value == 0.0,
                3,
            )
        return
    if not _use_narrow_t_path(trans, m, n, incx, incy):
        return _common.dgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)

    _prepare_narrow_t(trans, m, n, A, lda, x, incx, y, incy, torch.float64)
    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta_value = float(beta.item() if isinstance(beta, torch.Tensor) else beta)
    if alpha_value == 0.0:
        _scale_y(y, beta_value)
        return

    alpha_int = struct.unpack("=q", struct.pack("=d", alpha_value))[0]
    _scale_y(y, beta_value)
    grid = lambda meta: (
        triton.cdiv(n, meta["BLOCK_SIZE_M"]),
        _NARROW_T_SPLITS,
    )
    with torch_device_fn.device(A.device):
        dgemv_t_narrow_thead_kernel[grid](
            A,
            x,
            y,
            n,
            m,
            lda,
            incx,
            incy,
            alpha_int,
            _NARROW_T_SPLITS,
        )


def _lowp_gemv(
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
    dtype: torch.dtype,
) -> None:
    assert A.is_contiguous()
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert A.dtype == dtype
    assert x.dtype == dtype
    assert y.dtype == dtype
    assert A.device == x.device == y.device
    assert trans == CUBLAS_OP_T
    assert incx > 0 and incy > 0
    assert lda >= n
    assert x.numel() >= 1 + (m - 1) * incx
    assert y.numel() >= 1 + (n - 1) * incy

    alpha_value = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta_value = beta.item() if isinstance(beta, torch.Tensor) else float(beta)
    if alpha_value == 0.0:
        _scale_y(y, beta_value)
        return

    beta_is_zero = beta_value == 0.0
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
    with torch_device_fn.device(A.device):
        lowp_gemv_t_thead_kernel[grid](
            A,
            x,
            y,
            alpha_value,
            beta_value,
            n,
            m,
            lda,
            incx,
            incy,
            beta_is_zero,
        )


def _lowp_gemv_narrow_split(
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
    dtype: torch.dtype,
) -> None:
    assert A.is_contiguous()
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert A.dtype == dtype
    assert x.dtype == dtype
    assert y.dtype == dtype
    assert A.device == x.device == y.device
    assert trans == CUBLAS_OP_T
    assert incx == 1 and incy == 1
    assert lda >= n
    assert x.numel() >= m
    assert y.numel() >= n

    alpha_value = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta_value = beta.item() if isinstance(beta, torch.Tensor) else float(beta)
    if alpha_value == 0.0:
        _scale_y(y, beta_value)
        return

    num_k_splits = min(triton.cdiv(m, 1024), _THEAD_SM_COUNT // n)
    partial = torch.empty(num_k_splits, n, dtype=torch.float32, device=A.device)
    split_grid = lambda meta: (
        triton.cdiv(n, meta["BLOCK_SIZE_M"]),
        num_k_splits,
    )
    reduce_grid = (triton.cdiv(n, 32),)
    reduce_kernel = (
        _common.hgemv_splitk_reduce_kernel
        if dtype == torch.float16
        else _common.bfgemv_splitk_reduce_kernel
    )
    with torch_device_fn.device(A.device):
        lowp_gemv_t_narrow_split_thead_kernel[split_grid](
            A,
            x,
            partial,
            n,
            m,
            lda,
            num_k_splits,
        )
        reduce_kernel[reduce_grid](
            partial,
            y,
            alpha_value,
            beta_value,
            n,
            num_k_splits,
            incy,
            beta_value == 0.0,
            BLOCK_SIZE_M=32,
        )


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
    if _try_direct_gemv(torch.float16, trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
        return
    if _use_t_k1_path(trans, m, n, incx, incy):
        return _gemv_t_k1(
            lowp_gemv_t_k1_thead_kernel,
            trans,
            m,
            n,
            alpha,
            A,
            lda,
            x,
            incx,
            beta,
            y,
            incy,
            torch.float16,
        )
    if _use_narrow_lowp_t_path(trans, m, n, incx, incy):
        return _lowp_gemv_narrow_split(
            trans, m, n, alpha, A, lda, x, incx, beta, y, incy, torch.float16
        )
    if not _use_large_lowp_t_path(trans, m, n, incx, incy):
        return _common.hgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    return _lowp_gemv(
        trans, m, n, alpha, A, lda, x, incx, beta, y, incy, torch.float16
    )


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
    if _try_direct_gemv(torch.bfloat16, trans, m, n, alpha, A, lda, x, incx, beta, y, incy):
        return
    if _use_t_k1_path(trans, m, n, incx, incy):
        return _gemv_t_k1(
            lowp_gemv_t_k1_thead_kernel,
            trans,
            m,
            n,
            alpha,
            A,
            lda,
            x,
            incx,
            beta,
            y,
            incy,
            torch.bfloat16,
        )
    if _use_narrow_lowp_t_path(trans, m, n, incx, incy):
        return _lowp_gemv_narrow_split(
            trans, m, n, alpha, A, lda, x, incx, beta, y, incy, torch.bfloat16
        )
    if not _use_large_lowp_t_path(trans, m, n, incx, incy):
        return _common.bfgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    return _lowp_gemv(
        trans, m, n, alpha, A, lda, x, incx, beta, y, incy, torch.bfloat16
    )
