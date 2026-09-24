import ctypes
import importlib
from functools import lru_cache
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level2._constants import CUBLAS_OP_C, CUBLAS_OP_N, CUBLAS_OP_T
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

ScalarType = Union[float, int, complex, torch.Tensor]

_common = importlib.import_module("flag_blas.ops.level2.gemv")


@triton.jit
def _sgemv_n_short_k1_jit(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    STRIDE_AM,
    INCX,
    INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    a_values = tl.load(a_ptr + rows * STRIDE_AM, mask=row_mask, other=0.0)
    x_value = tl.load(x_ptr)
    y_ptrs = y_ptr + rows * INCY
    if BETA_IS_ZERO:
        result = alpha * a_values * x_value
    else:
        y_values = tl.load(y_ptrs, mask=row_mask, other=0.0)
        result = alpha * a_values * x_value + beta * y_values
    tl.store(y_ptrs, result, mask=row_mask)


sgemv_n_wide_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("sgemv_n_wide_ascend"),
        key=_common._GEMV_N_KEY,
        restore_value=["y_ptr"],
    )(_common.sgemv_n_kernel.jit_function)
)

sgemv_n_four_to_one_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("sgemv_n_four_to_one_ascend"),
        key=_common._GEMV_N_KEY,
        restore_value=["y_ptr"],
    )(_common.sgemv_n_kernel.jit_function)
)

sgemv_n_small_square_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("sgemv_n_small_square_ascend"),
        key=_common._GEMV_N_KEY,
        restore_value=["y_ptr"],
    )(_common.sgemv_n_kernel.jit_function)
)

sgemv_n_short_k1_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("sgemv_n_short_k1_ascend"),
        key=_common._GEMV_N_KEY,
        restore_value=["y_ptr"],
    )(_sgemv_n_short_k1_jit)
)


def _select_sgemv_n_path(m, n):
    if m <= 4 and n >= 65536:
        return "wide_regular"
    if 4095 <= m <= 4096 and 1023 <= n <= 1024:
        return "four_to_one"
    if m == n and 63 <= m <= 1024:
        return "small_square"
    if m >= 65536 and n == 1:
        return "short_k1"
    if m <= 64 and n >= 4096:
        return "splitk"
    return "regular"


def _launch_sgemv_n(
    entry_tag,
    kernel,
    A,
    x,
    y,
    alpha,
    beta,
    m,
    n,
    lda,
    incx,
    incy,
    beta_is_zero,
):
    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE_M"]),)
    kernel[grid](A, x, y, alpha, beta, m, n, lda, incx, incy, beta_is_zero)


def _launch_on_tensor_device(tensor, launch):
    device_index = tensor.device.index
    if device_index is None or device_index == torch_device_fn.current_device():
        launch()
        return
    with torch_device_fn.device(tensor.device):
        launch()


def _scale_complex_y(y, length, stride, beta_real, beta_imag):
    logical_y = torch.view_as_real(y).as_strided((length, 2), (stride * 2, 1))
    if beta_real == 0.0 and beta_imag == 0.0:
        logical_y.zero_()
    elif beta_real != 1.0 or beta_imag != 0.0:
        values = logical_y.clone()
        logical_y[:, 0].copy_(beta_real * values[:, 0] - beta_imag * values[:, 1])
        logical_y[:, 1].copy_(beta_real * values[:, 1] + beta_imag * values[:, 0])


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cgemv_ascend"),
    key=[
        "m",
        "n",
        "STRIDE_AM",
        "STRIDE_AN",
        "INCX",
        "INCY",
        "CONJ",
        "BETA_IS_ZERO",
    ],
    restore_value=["y_ptr"],
)
@triton.jit
def cgemv_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_real: tl.float32,
    alpha_imag: tl.float32,
    beta_real: tl.float32,
    beta_imag: tl.float32,
    m,
    n,
    STRIDE_AM,
    STRIDE_AN,
    INCX,
    INCY,
    CONJ: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    k_init = tl.arange(0, BLOCK_SIZE_K)
    acc_real_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    acc_imag_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for k_start in range(0, n, BLOCK_SIZE_K):
        ks = k_start + k_init
        k_mask = ks < n
        mask = row_mask[:, None] & k_mask[None, :]
        a_elem = rows[:, None] * STRIDE_AM + ks[None, :] * STRIDE_AN
        a_off = a_elem * 2
        x_off = ks * INCX * 2
        a_real = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        a_imag = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
        x_real = tl.load(x_ptr + x_off, mask=k_mask, other=0.0)
        x_imag = tl.load(x_ptr + x_off + 1, mask=k_mask, other=0.0)
        if CONJ:
            a_imag = -a_imag
        acc_real_2d += a_real * x_real[None, :] - a_imag * x_imag[None, :]
        acc_imag_2d += a_real * x_imag[None, :] + a_imag * x_real[None, :]

    acc_real = tl.sum(acc_real_2d, axis=1)
    acc_imag = tl.sum(acc_imag_2d, axis=1)
    out_real = alpha_real * acc_real - alpha_imag * acc_imag
    out_imag = alpha_real * acc_imag + alpha_imag * acc_real
    y_off = rows * INCY * 2
    if not BETA_IS_ZERO:
        y_real = tl.load(y_ptr + y_off, mask=row_mask, other=0.0)
        y_imag = tl.load(y_ptr + y_off + 1, mask=row_mask, other=0.0)
        out_real += beta_real * y_real - beta_imag * y_imag
        out_imag += beta_real * y_imag + beta_imag * y_real
    tl.store(y_ptr + y_off, out_real, mask=row_mask)
    tl.store(y_ptr + y_off + 1, out_imag, mask=row_mask)


@triton.jit
def _cgemv_ktranspose_splitk_jit(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_real: tl.float32,
    alpha_imag: tl.float32,
    m,
    n,
    STRIDE_AM,
    STRIDE_AN,
    CONJ: tl.constexpr,
    SPLIT_COUNT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    chunk_k = (n + SPLIT_COUNT - 1) // SPLIT_COUNT
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, n)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    acc_real_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    acc_imag_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for k_offset in range(0, chunk_k, BLOCK_SIZE_K):
        ks = k_begin + k_offset + k_offsets
        k_mask = ks < k_end
        load_mask = k_mask[:, None] & row_mask[None, :]
        a_elem = ks[:, None] * STRIDE_AN + rows[None, :] * STRIDE_AM
        a_off = a_elem * 2
        a_real = tl.trans(tl.load(a_ptr + a_off, mask=load_mask, other=0.0))
        a_imag = tl.trans(tl.load(a_ptr + a_off + 1, mask=load_mask, other=0.0))
        x_real = tl.load(x_ptr + ks * 2, mask=k_mask, other=0.0)
        x_imag = tl.load(x_ptr + ks * 2 + 1, mask=k_mask, other=0.0)
        if CONJ:
            a_imag = -a_imag
        acc_real_2d += a_real * x_real[None, :] - a_imag * x_imag[None, :]
        acc_imag_2d += a_real * x_imag[None, :] + a_imag * x_real[None, :]

    acc_real = tl.sum(acc_real_2d, axis=1)
    acc_imag = tl.sum(acc_imag_2d, axis=1)
    result_real = alpha_real * acc_real - alpha_imag * acc_imag
    result_imag = alpha_real * acc_imag + alpha_imag * acc_real
    y_off = rows * 2
    tl.atomic_add(y_ptr + y_off, result_real, mask=row_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_off + 1, result_imag, mask=row_mask, sem="relaxed")


@libentry()
@triton.jit
def cgemv_t_partial_kernel_ascend(
    a_ptr,
    x_ptr,
    partial_ptr,
    m,
    n,
    STRIDE_AM,
    STRIDE_AN,
    CONJ: tl.constexpr,
    SPLIT_COUNT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    chunk_k = (n + SPLIT_COUNT - 1) // SPLIT_COUNT
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, n)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    acc_real_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    acc_imag_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for k_offset in range(0, chunk_k, BLOCK_SIZE_K):
        ks = k_begin + k_offset + k_offsets
        k_mask = ks < k_end
        load_mask = k_mask[:, None] & row_mask[None, :]
        a_elem = ks[:, None] * STRIDE_AN + rows[None, :] * STRIDE_AM
        a_off = a_elem * 2
        a_real = tl.trans(tl.load(a_ptr + a_off, mask=load_mask, other=0.0))
        a_imag = tl.trans(tl.load(a_ptr + a_off + 1, mask=load_mask, other=0.0))
        x_real = tl.load(x_ptr + ks * 2, mask=k_mask, other=0.0)
        x_imag = tl.load(x_ptr + ks * 2 + 1, mask=k_mask, other=0.0)
        if CONJ:
            a_imag = -a_imag
        acc_real_2d += a_real * x_real[None, :] - a_imag * x_imag[None, :]
        acc_imag_2d += a_real * x_imag[None, :] + a_imag * x_real[None, :]

    partial_elem = pid_k * m + rows
    partial_off = partial_elem * 2
    tl.store(
        partial_ptr + partial_off,
        tl.sum(acc_real_2d, axis=1),
        mask=row_mask,
    )
    tl.store(
        partial_ptr + partial_off + 1,
        tl.sum(acc_imag_2d, axis=1),
        mask=row_mask,
    )


@libentry()
@triton.jit
def cgemv_t_reduce_kernel_ascend(
    partial_ptr,
    y_ptr,
    alpha_real: tl.float32,
    alpha_imag: tl.float32,
    beta_real: tl.float32,
    beta_imag: tl.float32,
    m,
    BETA_IS_ZERO: tl.constexpr,
    SPLIT_COUNT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    splits = tl.arange(0, BLOCK_SPLITS)
    row_mask = rows < m
    mask = row_mask[:, None] & (splits[None, :] < SPLIT_COUNT)
    partial_elem = splits[None, :] * m + rows[:, None]
    partial_off = partial_elem * 2
    sum_real = tl.sum(tl.load(partial_ptr + partial_off, mask=mask, other=0.0), axis=1)
    sum_imag = tl.sum(
        tl.load(partial_ptr + partial_off + 1, mask=mask, other=0.0), axis=1
    )
    out_real = alpha_real * sum_real - alpha_imag * sum_imag
    out_imag = alpha_real * sum_imag + alpha_imag * sum_real
    y_off = rows * 2
    if not BETA_IS_ZERO:
        y_real = tl.load(y_ptr + y_off, mask=row_mask, other=0.0)
        y_imag = tl.load(y_ptr + y_off + 1, mask=row_mask, other=0.0)
        out_real += beta_real * y_real - beta_imag * y_imag
        out_imag += beta_real * y_imag + beta_imag * y_real
    tl.store(y_ptr + y_off, out_real, mask=row_mask)
    tl.store(y_ptr + y_off + 1, out_imag, mask=row_mask)


@libentry()
@triton.jit
def cgemv_t_small_kernel_ascend(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_real: tl.float32,
    alpha_imag: tl.float32,
    beta_real: tl.float32,
    beta_imag: tl.float32,
    m,
    n,
    STRIDE_AM,
    STRIDE_AN,
    CONJ: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    acc_real_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    acc_imag_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for k_start in range(0, n, BLOCK_SIZE_K):
        ks = k_start + k_offsets
        k_mask = ks < n
        load_mask = k_mask[:, None] & row_mask[None, :]
        a_elem = ks[:, None] * STRIDE_AN + rows[None, :] * STRIDE_AM
        a_off = a_elem * 2
        a_real = tl.trans(tl.load(a_ptr + a_off, mask=load_mask, other=0.0))
        a_imag = tl.trans(tl.load(a_ptr + a_off + 1, mask=load_mask, other=0.0))
        x_real = tl.load(x_ptr + ks * 2, mask=k_mask, other=0.0)
        x_imag = tl.load(x_ptr + ks * 2 + 1, mask=k_mask, other=0.0)
        if CONJ:
            a_imag = -a_imag
        acc_real_2d += a_real * x_real[None, :] - a_imag * x_imag[None, :]
        acc_imag_2d += a_real * x_imag[None, :] + a_imag * x_real[None, :]

    sum_real = tl.sum(acc_real_2d, axis=1)
    sum_imag = tl.sum(acc_imag_2d, axis=1)
    out_real = alpha_real * sum_real - alpha_imag * sum_imag
    out_imag = alpha_real * sum_imag + alpha_imag * sum_real
    y_off = rows * 2
    if not BETA_IS_ZERO:
        y_real = tl.load(y_ptr + y_off, mask=row_mask, other=0.0)
        y_imag = tl.load(y_ptr + y_off + 1, mask=row_mask, other=0.0)
        out_real += beta_real * y_real - beta_imag * y_imag
        out_imag += beta_real * y_imag + beta_imag * y_real
    tl.store(y_ptr + y_off, out_real, mask=row_mask)
    tl.store(y_ptr + y_off + 1, out_imag, mask=row_mask)


cgemv_n_k1_kernel_ascend = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("cgemv_n_k1_ascend"),
        key=["m", "n", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(cgemv_kernel.jit_function)
)

cgemv_n_shortk_kernel_ascend = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("cgemv_n_shortk_ascend"),
        key=["m", "n", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(cgemv_kernel.jit_function)
)

cgemv_n_small_tile_kernel_ascend = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("cgemv_n_small_tile_ascend"),
        key=["m", "n", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(cgemv_kernel.jit_function)
)

cgemv_n_k1024_kernel_ascend = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("cgemv_n_k1024_ascend"),
        key=["m", "n", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(cgemv_kernel.jit_function)
)

cgemv_t_splitk_kernel_ascend = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("cgemv_t_splitk_ascend"),
        key=[
            "m",
            "n",
            "STRIDE_AM",
            "STRIDE_AN",
            "CONJ",
            "SPLIT_COUNT",
        ],
        restore_value=["y_ptr"],
    )(_cgemv_ktranspose_splitk_jit)
)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("cgemv_splitk_ascend"),
    key=[
        "m",
        "n",
        "STRIDE_AM",
        "STRIDE_AN",
        "CONJ",
        "SPLIT_COUNT",
    ],
    restore_value=["y_ptr"],
)
@triton.jit
def cgemv_splitk_kernel_ascend(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha_real: tl.float32,
    alpha_imag: tl.float32,
    m,
    n,
    STRIDE_AM,
    STRIDE_AN,
    CONJ: tl.constexpr,
    SPLIT_COUNT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    row_mask = rows < m
    chunk_k = (n + SPLIT_COUNT - 1) // SPLIT_COUNT
    k_begin = pid_k * chunk_k
    k_end = tl.minimum(k_begin + chunk_k, n)
    k_init = tl.arange(0, BLOCK_SIZE_K)
    acc_real_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    acc_imag_2d = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for k_offset in range(0, chunk_k, BLOCK_SIZE_K):
        ks = k_begin + k_offset + k_init
        k_mask = ks < k_end
        mask = row_mask[:, None] & k_mask[None, :]
        a_elem = rows[:, None] * STRIDE_AM + ks[None, :] * STRIDE_AN
        a_off = a_elem * 2
        x_off = ks * 2
        a_real = tl.load(a_ptr + a_off, mask=mask, other=0.0)
        a_imag = tl.load(a_ptr + a_off + 1, mask=mask, other=0.0)
        x_real = tl.load(x_ptr + x_off, mask=k_mask, other=0.0)
        x_imag = tl.load(x_ptr + x_off + 1, mask=k_mask, other=0.0)
        if CONJ:
            a_imag = -a_imag
        acc_real_2d += a_real * x_real[None, :] - a_imag * x_imag[None, :]
        acc_imag_2d += a_real * x_imag[None, :] + a_imag * x_real[None, :]

    acc_real = tl.sum(acc_real_2d, axis=1)
    acc_imag = tl.sum(acc_imag_2d, axis=1)
    result_real = alpha_real * acc_real - alpha_imag * acc_imag
    result_imag = alpha_real * acc_imag + alpha_imag * acc_real
    y_off = rows * 2
    tl.atomic_add(y_ptr + y_off, result_real, mask=row_mask, sem="relaxed")
    tl.atomic_add(y_ptr + y_off + 1, result_imag, mask=row_mask, sem="relaxed")


def _cgemv_split_count(eff_m, eff_n, trans):
    if trans == CUBLAS_OP_N and 3 <= eff_m <= 4 and eff_n >= 131071:
        return 128
    if eff_n <= 512:
        return 8
    if eff_n <= 4096:
        return 16
    if eff_m >= 32:
        return 64
    return min(triton.cdiv(eff_n, 2048), 64)


def _cgemv_t_split_count(eff_n):
    if eff_n <= 512:
        return 8
    if eff_n <= 1024:
        return 16
    if 2049 <= eff_n <= 3584:
        return 32
    if 4095 <= eff_n <= 4096:
        return 24
    if 7167 <= eff_n <= 7168:
        return 24
    if eff_n == 18944:
        return 28
    return 16


def _use_cgemv_t_workspace(eff_m, eff_n, split_count):
    if eff_m <= 1024 or split_count == 32:
        return True
    if eff_m == 4095 and eff_n == 4095:
        return True
    if 1023 <= eff_n <= 1024 and eff_m >= 4095:
        return True
    if 4095 <= eff_n <= 4096 and eff_m >= 8192:
        return True
    if 7167 <= eff_n <= 7168 and 7167 <= eff_m <= 7168:
        return True
    return eff_n == 18944 and eff_m == 3584


def _use_cgemv_t_small(eff_m, eff_n):
    return 255 <= eff_m <= 256 and 255 <= eff_n <= 256


@triton.jit
def _lowp_gemv_t_coalesced_jit(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    STRIDE_AK,
    INCX,
    INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    cols = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    ki = tl.arange(0, BLOCK_SIZE_K)
    acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), tl.float32)
    for start in range(0, n, BLOCK_SIZE_K):
        ks = start + ki
        av = tl.load(
            a_ptr + ks[:, None] * STRIDE_AK + cols[None, :],
            mask=(ks[:, None] < n) & (cols[None, :] < m),
            other=0,
        ).to(tl.float32)
        xv = tl.load(x_ptr + ks * INCX, mask=ks < n, other=0).to(tl.float32)
        acc += av * xv[:, None]
    result = alpha * tl.sum(acc, 0)
    if not BETA_IS_ZERO:
        result += beta * tl.load(y_ptr + cols * INCY, mask=cols < m, other=0).to(
            tl.float32
        )
    tl.store(y_ptr + cols * INCY, result, mask=cols < m)


@triton.jit
def _cgemv_packed_jit(
    a_ptr,
    x_ptr,
    y_ptr,
    ar: tl.float32,
    ai: tl.float32,
    br: tl.float32,
    bi: tl.float32,
    OUT,
    REDUCE,
    LDA,
    TRANSPOSE: tl.constexpr,
    CONJ: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    oi = tl.arange(0, BLOCK_SIZE_M)
    cols = tl.program_id(0) * BLOCK_SIZE_M + oi
    ks0 = tl.arange(0, BLOCK_SIZE_K)
    pair = tl.arange(0, 2)
    if TRANSPOSE:
        acc_r = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), tl.float32)
        acc_i = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), tl.float32)
    else:
        acc_r = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float32)
        acc_i = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float32)
    for start in range(0, REDUCE, BLOCK_SIZE_K):
        ks = start + ks0
        xp = tl.load(
            x_ptr + ks[:, None] * 2 + pair[None, :], mask=ks[:, None] < REDUCE, other=0
        )
        xr, xi = tl.split(xp)
        if TRANSPOSE:
            cp = tl.program_id(0) * BLOCK_SIZE_M * 2 + tl.arange(0, BLOCK_SIZE_M * 2)
            ap = tl.load(
                a_ptr + ks[:, None] * (LDA * 2) + cp[None, :],
                mask=(ks[:, None] < REDUCE) & (cp[None, :] < OUT * 2),
                other=0,
            )
            av_r, av_i = tl.split(tl.reshape(ap, (BLOCK_SIZE_K, BLOCK_SIZE_M, 2)))
            if CONJ:
                av_i = -av_i
            acc_r += av_r * xr[:, None] - av_i * xi[:, None]
            acc_i += av_r * xi[:, None] + av_i * xr[:, None]
        else:
            kp = start * 2 + tl.arange(0, BLOCK_SIZE_K * 2)
            ap = tl.load(
                a_ptr + cols[:, None] * (LDA * 2) + kp[None, :],
                mask=(cols[:, None] < OUT) & (kp[None, :] < REDUCE * 2),
                other=0,
            )
            av_r, av_i = tl.split(tl.reshape(ap, (BLOCK_SIZE_M, BLOCK_SIZE_K, 2)))
            acc_r += av_r * xr[None, :] - av_i * xi[None, :]
            acc_i += av_r * xi[None, :] + av_i * xr[None, :]
    if TRANSPOSE:
        sr, si = tl.sum(acc_r, 0), tl.sum(acc_i, 0)
    else:
        sr, si = tl.sum(acc_r, 1), tl.sum(acc_i, 1)
    rr = ar * sr - ai * si
    ri = ar * si + ai * sr
    if not BETA_IS_ZERO:
        yp = tl.load(
            y_ptr + cols[:, None] * 2 + pair[None, :], mask=cols[:, None] < OUT, other=0
        )
        yr, yi = tl.split(yp)
        rr += br * yr - bi * yi
        ri += br * yi + bi * yr
    packed = tl.reshape(tl.join(rr, ri), (BLOCK_SIZE_M * 2,))
    cp = tl.program_id(0) * BLOCK_SIZE_M * 2 + tl.arange(0, BLOCK_SIZE_M * 2)
    tl.store(y_ptr + cp, packed, mask=cp < OUT * 2)


hgemv_t_coalesced_kernel_ascend = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("hgemv_t_coalesced_ascend"),
        key=_common._GEMV_T_KEY,
        restore_value=["y_ptr"],
    )(_lowp_gemv_t_coalesced_jit)
)


bfgemv_t_coalesced_kernel_ascend = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("bfgemv_t_coalesced_ascend"),
        key=_common._GEMV_T_KEY,
        restore_value=["y_ptr"],
    )(_lowp_gemv_t_coalesced_jit)
)


cgemv_n_packed_kernel_ascend = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("cgemv_n_packed_ascend"),
        key=["OUT", "REDUCE", "LDA", "TRANSPOSE", "CONJ", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(_cgemv_packed_jit)
)


cgemv_t_packed_kernel_ascend = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("cgemv_t_packed_ascend"),
        key=["OUT", "REDUCE", "LDA", "TRANSPOSE", "CONJ", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(_cgemv_packed_jit)
)


def _use_large_gemv_t(m, n, element_size):
    # Wider tiles amortize loop overhead on streaming matrices. Smaller inputs
    # retain the existing config pools to avoid reducing their parallelism.
    return min(m, n) >= 1024 and m * n * element_size >= 256 * 1024 * 1024


@lru_cache(maxsize=None)
def _cgemv_matrix_l2_offset(device_index):
    # AscendC L2CacheAlter uses this runtime-provided alias on A2. Query only
    # after entering the tensor device context; unsupported runtimes fall back.
    if not torch.npu.get_device_name(device_index).startswith("Ascend910B"):
        return 0
    try:
        library = ctypes.CDLL("libruntime.so")
        get_offset = library.rtGetL2CacheOffset
        get_offset.argtypes = [ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
        get_offset.restype = ctypes.c_int32
        offset = ctypes.c_uint64(0)
        if get_offset(device_index, ctypes.byref(offset)) != 0:
            return 0
        return offset.value
    except (AttributeError, OSError):
        return 0


@triton.jit
def _cgemv_t_packed_split_jit(
    a_ptr,
    x_ptr,
    y_ptr,
    ar: tl.float32,
    ai: tl.float32,
    br: tl.float32,
    bi: tl.float32,
    OUT,
    REDUCE,
    LDA,
    TRANSPOSE: tl.constexpr,
    CONJ: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_COUNT: tl.constexpr,
    L2_BYPASS_OFFSET: tl.constexpr = 0,
):
    # A is streamed once; x/y keep their normal cache policy.
    if L2_BYPASS_OFFSET:
        a_ptr = (a_ptr.to(tl.uint64) + L2_BYPASS_OFFSET).to(tl.pointer_type(tl.float32))
    oi = tl.arange(0, BLOCK_SIZE_M)
    cols = tl.program_id(0) * BLOCK_SIZE_M + oi
    ks0 = tl.arange(0, BLOCK_SIZE_K)
    pair = tl.arange(0, 2)
    if TRANSPOSE:
        acc_r = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), tl.float32)
        acc_i = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), tl.float32)
    else:
        acc_r = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float32)
        acc_i = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float32)
    chunk = tl.cdiv(REDUCE, SPLIT_COUNT * BLOCK_SIZE_K) * BLOCK_SIZE_K
    begin = tl.program_id(1) * chunk
    end = tl.minimum(begin + chunk, REDUCE)
    for start in range(begin, end, BLOCK_SIZE_K):
        ks = start + ks0
        xp = tl.load(
            x_ptr + ks[:, None] * 2 + pair[None, :], mask=ks[:, None] < end, other=0
        )
        xr, xi = tl.split(xp)
        # conj(A).T @ x = conj(A.T @ conj(x)); conjugate before alpha.
        # Negate the small x tile instead of every matrix imaginary tile.
        if TRANSPOSE and CONJ:
            xi = -xi
        if TRANSPOSE:
            cp = tl.program_id(0) * BLOCK_SIZE_M * 2 + tl.arange(0, BLOCK_SIZE_M * 2)
            ap = tl.load(
                a_ptr + ks[:, None] * (LDA * 2) + cp[None, :],
                mask=(ks[:, None] < end) & (cp[None, :] < OUT * 2),
                other=0,
            )
            av_r, av_i = tl.split(tl.reshape(ap, (BLOCK_SIZE_K, BLOCK_SIZE_M, 2)))
            acc_r += av_r * xr[:, None] - av_i * xi[:, None]
            acc_i += av_r * xi[:, None] + av_i * xr[:, None]
        else:
            kp = start * 2 + tl.arange(0, BLOCK_SIZE_K * 2)
            ap = tl.load(
                a_ptr + cols[:, None] * (LDA * 2) + kp[None, :],
                mask=(cols[:, None] < OUT) & (kp[None, :] < end * 2),
                other=0,
            )
            av_r, av_i = tl.split(tl.reshape(ap, (BLOCK_SIZE_M, BLOCK_SIZE_K, 2)))
            acc_r += av_r * xr[None, :] - av_i * xi[None, :]
            acc_i += av_r * xi[None, :] + av_i * xr[None, :]
    if TRANSPOSE:
        sr, si = tl.sum(acc_r, 0), tl.sum(acc_i, 0)
    else:
        sr, si = tl.sum(acc_r, 1), tl.sum(acc_i, 1)
    if TRANSPOSE and CONJ:
        si = -si
    rr = ar * sr - ai * si
    ri = ar * si + ai * sr
    packed = tl.reshape(tl.join(rr, ri), (BLOCK_SIZE_M * 2,))
    cp = tl.program_id(0) * BLOCK_SIZE_M * 2 + tl.arange(0, BLOCK_SIZE_M * 2)
    tl.atomic_add(y_ptr + cp, packed, mask=cp < OUT * 2, sem="relaxed")

@triton.jit
def _cgemv_scale_output_jit(y_ptr, length, br: tl.float32, bi: tl.float32,
                         ZERO: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * (BLOCK * 2) + tl.arange(0, BLOCK * 2)
    if ZERO:
        result = tl.full((BLOCK * 2,), 0, tl.float32)
    else:
        values = tl.load(y_ptr + offsets, mask=offsets < length * 2, other=0)
        yr, yi = tl.split(tl.reshape(values, (BLOCK, 2)))
        rr = br * yr - bi * yi
        ri = br * yi + bi * yr
        result = tl.reshape(tl.join(rr, ri), (BLOCK * 2,))
    tl.store(y_ptr + offsets, result, mask=offsets < length * 2)


hgemv_t_large_kernel_ascend = libentry()(
    libtuner(configs=runtime.get_tuned_config("hgemv_t_large_ascend"),
             key=_common._GEMV_T_KEY, restore_value=["y_ptr"])(_lowp_gemv_t_coalesced_jit)
)


bfgemv_t_large_kernel_ascend = libentry()(
    libtuner(configs=runtime.get_tuned_config("bfgemv_t_large_ascend"),
             key=_common._GEMV_T_KEY, restore_value=["y_ptr"])(_lowp_gemv_t_coalesced_jit)
)


cgemv_t_large_kernel_ascend = libentry()(
    libtuner(configs=runtime.get_tuned_config("cgemv_t_large_ascend"),
             key=["OUT", "REDUCE", "LDA", "TRANSPOSE", "CONJ", "BETA_IS_ZERO"], restore_value=["y_ptr"])(_cgemv_packed_jit)
)


cgemv_t_packed_split_kernel_ascend = libentry()(
    libtuner(configs=runtime.get_tuned_config("cgemv_t_packed_split_ascend"),
             key=["OUT", "REDUCE", "LDA", "TRANSPOSE", "CONJ", "BETA_IS_ZERO", "L2_BYPASS_OFFSET"],
             restore_value=["y_ptr"])(_cgemv_t_packed_split_jit)
)

cgemv_scale_output_kernel_ascend = libentry()(_cgemv_scale_output_jit)


cgemv_n_large_kernel_ascend = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("cgemv_n_large_ascend"),
        key=["OUT", "REDUCE", "LDA", "TRANSPOSE", "CONJ", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(_cgemv_packed_jit)
)


# Keep the normal-cache JIT unchanged for small inputs and unsupported SoCs.
@triton.jit
def _cgemv_n_l2_stream_jit(
    a_ptr,
    x_ptr,
    y_ptr,
    ar: tl.float32,
    ai: tl.float32,
    br: tl.float32,
    bi: tl.float32,
    OUT,
    REDUCE,
    LDA,
    TRANSPOSE: tl.constexpr,
    CONJ: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    L2_BYPASS_OFFSET: tl.constexpr,
):
    a_ptr = (a_ptr.to(tl.uint64) + L2_BYPASS_OFFSET).to(tl.pointer_type(tl.float32))
    oi = tl.arange(0, BLOCK_SIZE_M)
    cols = tl.program_id(0) * BLOCK_SIZE_M + oi
    ks0 = tl.arange(0, BLOCK_SIZE_K)
    pair = tl.arange(0, 2)
    if TRANSPOSE:
        acc_r = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), tl.float32)
        acc_i = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), tl.float32)
    else:
        acc_r = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float32)
        acc_i = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float32)
    for start in range(0, REDUCE, BLOCK_SIZE_K):
        ks = start + ks0
        xp = tl.load(
            x_ptr + ks[:, None] * 2 + pair[None, :], mask=ks[:, None] < REDUCE, other=0
        )
        xr, xi = tl.split(xp)
        if TRANSPOSE:
            cp = tl.program_id(0) * BLOCK_SIZE_M * 2 + tl.arange(0, BLOCK_SIZE_M * 2)
            ap = tl.load(
                a_ptr + ks[:, None] * (LDA * 2) + cp[None, :],
                mask=(ks[:, None] < REDUCE) & (cp[None, :] < OUT * 2),
                other=0,
            )
            av_r, av_i = tl.split(tl.reshape(ap, (BLOCK_SIZE_K, BLOCK_SIZE_M, 2)))
            if CONJ:
                av_i = -av_i
            acc_r += av_r * xr[:, None] - av_i * xi[:, None]
            acc_i += av_r * xi[:, None] + av_i * xr[:, None]
        else:
            kp = start * 2 + tl.arange(0, BLOCK_SIZE_K * 2)
            ap = tl.load(
                a_ptr + cols[:, None] * (LDA * 2) + kp[None, :],
                mask=(cols[:, None] < OUT) & (kp[None, :] < REDUCE * 2),
                other=0,
            )
            av_r, av_i = tl.split(tl.reshape(ap, (BLOCK_SIZE_M, BLOCK_SIZE_K, 2)))
            acc_r += av_r * xr[None, :] - av_i * xi[None, :]
            acc_i += av_r * xi[None, :] + av_i * xr[None, :]
    if TRANSPOSE:
        sr, si = tl.sum(acc_r, 0), tl.sum(acc_i, 0)
    else:
        sr, si = tl.sum(acc_r, 1), tl.sum(acc_i, 1)
    rr = ar * sr - ai * si
    ri = ar * si + ai * sr
    if not BETA_IS_ZERO:
        yp = tl.load(
            y_ptr + cols[:, None] * 2 + pair[None, :], mask=cols[:, None] < OUT, other=0
        )
        yr, yi = tl.split(yp)
        rr += br * yr - bi * yi
        ri += br * yi + bi * yr
    packed = tl.reshape(tl.join(rr, ri), (BLOCK_SIZE_M * 2,))
    cp = tl.program_id(0) * BLOCK_SIZE_M * 2 + tl.arange(0, BLOCK_SIZE_M * 2)
    tl.store(y_ptr + cp, packed, mask=cp < OUT * 2)


@lru_cache(maxsize=None)
def _cgemv_n_matrix_kernel(device_index):
    offset = _cgemv_matrix_l2_offset(device_index)
    if not offset:
        return cgemv_n_large_kernel_ascend
    configs = [
        triton.Config(
            dict(cfg.kwargs, L2_BYPASS_OFFSET=offset),
            num_warps=cfg.num_warps,
            num_stages=cfg.num_stages,
            num_ctas=cfg.num_ctas,
        )
        for cfg in runtime.get_tuned_config("cgemv_n_large_ascend")
    ]
    return libentry()(
        libtuner(
            configs=configs,
            key=["OUT", "REDUCE", "LDA", "TRANSPOSE", "CONJ", "BETA_IS_ZERO"],
            restore_value=["y_ptr"],
        )(_cgemv_n_l2_stream_jit)
    )


def _cgemv_standard(
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
    assert A.is_contiguous()
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert A.dtype == torch.complex64
    assert x.dtype == torch.complex64
    assert y.dtype == torch.complex64
    assert A.device == x.device == y.device
    assert trans in [CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C]
    assert incx > 0 and incy > 0
    assert lda >= n
    if m == 0 or n == 0:
        return

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    beta = beta.item() if isinstance(beta, torch.Tensor) else beta
    alpha_real = float(alpha.real) if isinstance(alpha, complex) else float(alpha)
    alpha_imag = float(alpha.imag) if isinstance(alpha, complex) else 0.0
    beta_real = float(beta.real) if isinstance(beta, complex) else float(beta)
    beta_imag = float(beta.imag) if isinstance(beta, complex) else 0.0

    if trans == CUBLAS_OP_N:
        len_x, len_y = n, m
        eff_m, eff_n = m, n
        stride_am, stride_an = lda, 1
    else:
        len_x, len_y = m, n
        eff_m, eff_n = n, m
        stride_am, stride_an = 1, lda
    assert x.numel() >= 1 + (len_x - 1) * incx
    assert y.numel() >= 1 + (len_y - 1) * incy
    if alpha_real == 0.0 and alpha_imag == 0.0:
        _scale_complex_y(y, len_y, incy, beta_real, beta_imag)
        return

    beta_is_zero = beta_real == 0.0 and beta_imag == 0.0
    with torch_device_fn.device(A.device):
        # Adjacent real/imaginary lanes are loaded together, with no staging
        # workspace. Retain existing strided and narrow-reduction paths.
        if (
            incx == 1
            and incy == 1
            and not (trans != CUBLAS_OP_N and n <= 64 and m >= 4096)
        ):
            large_transpose = trans != CUBLAS_OP_N and _use_large_gemv_t(m, n, 8)
            if large_transpose:
                # Split long reductions across cores for both aligned and
                # non-aligned rows; accumulate directly into y.
                a_real = torch.view_as_real(A)
                x_real = torch.view_as_real(x)
                y_real = torch.view_as_real(y)
                cgemv_scale_output_kernel_ascend[(triton.cdiv(eff_m, 256),)](
                    y_real, eff_m, beta_real, beta_imag, beta_is_zero,
                    256, num_warps=1, num_stages=1,
                )
                grid = lambda meta: (triton.cdiv(eff_m, meta["BLOCK_SIZE_M"]), meta["SPLIT_COUNT"])
                cgemv_t_packed_split_kernel_ascend[grid](
                    a_real, x_real, y_real, alpha_real, alpha_imag,
                    beta_real, beta_imag, eff_m, eff_n, lda,
                    True, trans == CUBLAS_OP_C, beta_is_zero,
                    L2_BYPASS_OFFSET=_cgemv_matrix_l2_offset(A.device.index),
                )
                return
            kernel = (
                (
                    _cgemv_n_matrix_kernel(A.device.index)
                    if _use_large_gemv_t(m, n, 8)
                    else cgemv_n_packed_kernel_ascend
                )
                if trans == CUBLAS_OP_N
                else cgemv_t_large_kernel_ascend if large_transpose
                else cgemv_t_packed_kernel_ascend
            )
            grid = lambda meta: (triton.cdiv(eff_m, meta["BLOCK_SIZE_M"]),)
            kernel[grid](
                torch.view_as_real(A),
                torch.view_as_real(x),
                torch.view_as_real(y),
                alpha_real,
                alpha_imag,
                beta_real,
                beta_imag,
                eff_m,
                eff_n,
                lda,
                trans != CUBLAS_OP_N,
                trans == CUBLAS_OP_C,
                beta_is_zero,
            )
            return

        use_n_shortk = (
            trans == CUBLAS_OP_N and m >= 65536 and n <= 64 and incx == 1 and incy == 1
        )
        if use_n_shortk:
            A_real = torch.view_as_real(A)
            x_real = torch.view_as_real(x)
            y_real = torch.view_as_real(y)
            kernel = (
                cgemv_n_k1_kernel_ascend if n == 1 else cgemv_n_shortk_kernel_ascend
            )
            grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE_M"]),)
            kernel[grid](
                A_real,
                x_real,
                y_real,
                alpha_real,
                alpha_imag,
                beta_real,
                beta_imag,
                m,
                n,
                stride_am,
                stride_an,
                1,
                1,
                CONJ=False,
                BETA_IS_ZERO=beta_is_zero,
            )
            return

        use_n_small_tile = (
            trans == CUBLAS_OP_N
            and 511 <= m <= 512
            and 511 <= n <= 512
            and incx == 1
            and incy == 1
        )
        use_n_k1024 = (
            trans == CUBLAS_OP_N
            and 1023 <= n <= 1024
            and (1023 <= m <= 1024 or 4095 <= m <= 4096)
            and incx == 1
            and incy == 1
        )
        if use_n_small_tile or use_n_k1024:
            A_real = torch.view_as_real(A)
            x_real = torch.view_as_real(x)
            y_real = torch.view_as_real(y)
            kernel = (
                cgemv_n_small_tile_kernel_ascend
                if use_n_small_tile
                else cgemv_n_k1024_kernel_ascend
            )
            grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE_M"]),)
            kernel[grid](
                A_real,
                x_real,
                y_real,
                alpha_real,
                alpha_imag,
                beta_real,
                beta_imag,
                m,
                n,
                stride_am,
                stride_an,
                1,
                1,
                CONJ=False,
                BETA_IS_ZERO=beta_is_zero,
            )
            return

        use_t_small = (
            trans != CUBLAS_OP_N
            and _use_cgemv_t_small(eff_m, eff_n)
            and incx == 1
            and incy == 1
        )
        if use_t_small:
            A_real = torch.view_as_real(A)
            x_real = torch.view_as_real(x)
            y_real = torch.view_as_real(y)
            grid = (triton.cdiv(eff_m, 16),)
            cgemv_t_small_kernel_ascend[grid](
                A_real,
                x_real,
                y_real,
                alpha_real,
                alpha_imag,
                beta_real,
                beta_imag,
                eff_m,
                eff_n,
                stride_am,
                stride_an,
                CONJ=trans == CUBLAS_OP_C,
                BETA_IS_ZERO=beta_is_zero,
                BLOCK_SIZE_M=16,
                BLOCK_SIZE_K=256,
                num_warps=1,
                num_stages=1,
            )
            return

        use_long_splitk = eff_m <= 64 and eff_n >= 65536 and incx == 1 and incy == 1
        use_t_contiguous_splitk = (
            trans != CUBLAS_OP_N
            and 511 <= eff_m
            and 511 <= eff_n
            and incx == 1
            and incy == 1
        )
        use_splitk = use_long_splitk or use_t_contiguous_splitk
        if use_splitk:
            A_real = torch.view_as_real(A)
            x_real = torch.view_as_real(x)
            y_real = torch.view_as_real(y)
            if use_t_contiguous_splitk:
                split_count = _cgemv_t_split_count(eff_n)
                if _use_cgemv_t_workspace(eff_m, eff_n, split_count):
                    partial = torch.empty(
                        (split_count, eff_m, 2),
                        dtype=torch.float32,
                        device=A.device,
                    )
                    partial_grid = (triton.cdiv(eff_m, 32), split_count)
                    cgemv_t_partial_kernel_ascend[partial_grid](
                        A_real,
                        x_real,
                        partial,
                        eff_m,
                        eff_n,
                        stride_am,
                        stride_an,
                        CONJ=trans == CUBLAS_OP_C,
                        SPLIT_COUNT=split_count,
                        BLOCK_SIZE_M=32,
                        BLOCK_SIZE_K=64,
                        num_warps=1,
                        num_stages=1,
                    )
                    reduce_grid = (triton.cdiv(eff_m, 32),)
                    cgemv_t_reduce_kernel_ascend[reduce_grid](
                        partial,
                        y_real,
                        alpha_real,
                        alpha_imag,
                        beta_real,
                        beta_imag,
                        eff_m,
                        BETA_IS_ZERO=beta_is_zero,
                        SPLIT_COUNT=split_count,
                        BLOCK_SIZE_M=32,
                        BLOCK_SPLITS=32,
                        num_warps=1,
                        num_stages=1,
                    )
                    return
                kernel = cgemv_t_splitk_kernel_ascend
            else:
                split_count = _cgemv_split_count(eff_m, eff_n, trans)
                kernel = cgemv_splitk_kernel_ascend
            _scale_complex_y(y, len_y, incy, beta_real, beta_imag)
            grid = lambda meta: (
                triton.cdiv(eff_m, meta["BLOCK_SIZE_M"]),
                split_count,
            )
            kernel[grid](
                A_real,
                x_real,
                y_real,
                alpha_real,
                alpha_imag,
                eff_m,
                eff_n,
                stride_am,
                stride_an,
                CONJ=trans == CUBLAS_OP_C,
                SPLIT_COUNT=split_count,
            )
            return

        A_real = torch.view_as_real(A)
        x_real = torch.view_as_real(x)
        y_real = torch.view_as_real(y)
        kernel_incx = incx
        kernel_incy = incy
        if incx != 1:
            x_real = x_real.as_strided((len_x, 2), (incx * 2, 1)).clone()
            kernel_incx = 1
        logical_y_real = None
        if incy != 1:
            logical_y_real = y_real.as_strided((len_y, 2), (incy * 2, 1))
            y_real = logical_y_real.clone()
            kernel_incy = 1
        grid = lambda meta: (triton.cdiv(eff_m, meta["BLOCK_SIZE_M"]),)
        cgemv_kernel[grid](
            A_real,
            x_real,
            y_real,
            alpha_real,
            alpha_imag,
            beta_real,
            beta_imag,
            eff_m,
            eff_n,
            stride_am,
            stride_an,
            kernel_incx,
            kernel_incy,
            CONJ=trans == CUBLAS_OP_C,
            BETA_IS_ZERO=beta_is_zero,
        )
        if logical_y_real is not None:
            logical_y_real.copy_(y_real)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgemv_t_coalesced_ascend"),
    key=_common._GEMV_T_KEY,
    restore_value=["y_ptr"],
)
@triton.jit
def sgemv_t_coalesced_kernel(
    a_ptr,
    x_ptr,
    y_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    STRIDE_AK,
    INCX,
    INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Load contiguous physical columns and reduce along the matrix rows.
    # The Ascend entry normalizes vector strides before calling this kernel.
    rows = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    ks0 = tl.arange(0, BLOCK_SIZE_K)
    acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), tl.float32)
    for k_start in range(0, n, BLOCK_SIZE_K):
        ks = k_start + ks0
        a = tl.load(
            a_ptr + ks[:, None] * STRIDE_AK + rows[None, :],
            mask=(ks[:, None] < n) & (rows[None, :] < m),
            other=0.0,
        )
        x = tl.load(x_ptr + ks * INCX, mask=ks < n, other=0.0)
        acc += a * x[:, None]
    total = tl.sum(acc, 0)
    result = alpha * total
    if not BETA_IS_ZERO:
        result += beta * tl.load(y_ptr + rows * INCY, mask=rows < m, other=0.0)
    tl.store(y_ptr + rows * INCY, result, mask=rows < m)


sgemv_t_large_kernel_ascend = libentry()(
    libtuner(configs=runtime.get_tuned_config("sgemv_t_large_ascend"),
             key=_common._GEMV_T_KEY, restore_value=["y_ptr"])(
        sgemv_t_coalesced_kernel.jit_function
    )
)


def _sgemv_standard(
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
    if m == 0 or n == 0:
        _common.sgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
        return

    assert A.is_contiguous()
    assert x.is_contiguous()
    assert y.is_contiguous()
    assert A.dtype == torch.float32
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
    assert A.device == x.device == y.device
    assert trans in [CUBLAS_OP_N, CUBLAS_OP_T]
    assert incx > 0 and incy > 0
    assert lda >= n

    len_x, len_y = (n, m) if trans == CUBLAS_OP_N else (m, n)
    assert x.numel() >= 1 + (len_x - 1) * incx
    assert y.numel() >= 1 + (len_y - 1) * incy
    if incy != 1:
        logical_y = y.as_strided((len_y,), (incy,))
        y_contiguous = logical_y.clone()
        sgemv(trans, m, n, alpha, A, lda, x, incx, beta, y_contiguous, 1)
        logical_y.copy_(y_contiguous)
        return
    if incx != 1:
        x_contiguous = x.as_strided((len_x,), (incx,)).clone()
        sgemv(trans, m, n, alpha, A, lda, x_contiguous, 1, beta, y, incy)
        return

    # Existing split-K scaling must not touch extra storage after logical y.
    if y.numel() > len_y:
        y = y[:len_y]

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else float(beta)
    if alpha == 0.0:
        if beta == 0.0:
            y.zero_()
        elif beta != 1.0:
            y.mul_(beta)
        return

    if trans != CUBLAS_OP_N:
        # Keep the existing split-K path for very narrow outputs.
        if n <= _common.SPLITK_M_THRESHOLD and m >= _common.SPLITK_K_THRESHOLD:
            _common.sgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
        else:
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)

            kernel = sgemv_t_large_kernel_ascend if _use_large_gemv_t(m, n, 4) else sgemv_t_coalesced_kernel

            def launch_t():
                kernel[grid](
                    A, x, y, alpha, beta, n, m, lda, incx, incy, beta == 0.0
                )

            _launch_on_tensor_device(A, launch_t)
        return

    path = _select_sgemv_n_path(m, n)
    if path == "splitk":
        num_k_splits = min(triton.cdiv(n, 2048), 128)
        if beta == 0.0:
            y.zero_()
        elif beta != 1.0:
            y.mul_(beta)
        grid = lambda meta: (triton.cdiv(m, meta["BLOCK_SIZE_M"]), num_k_splits)

        def launch():
            _common.sgemv_n_splitk_kernel[grid](
                A, x, y, m, n, lda, incx, incy, alpha, num_k_splits
            )

    else:
        if path == "wide_regular":
            kernel = sgemv_n_wide_kernel
        elif path == "four_to_one":
            kernel = sgemv_n_four_to_one_kernel
        elif path == "small_square":
            kernel = sgemv_n_small_square_kernel
        elif path == "short_k1":
            kernel = sgemv_n_short_k1_kernel
        else:
            kernel = _common.sgemv_n_kernel
        beta_is_zero = beta == 0.0

        def launch():
            _launch_sgemv_n(
                path,
                kernel,
                A,
                x,
                y,
                alpha,
                beta,
                m,
                n,
                lda,
                incx,
                incy,
                beta_is_zero,
            )

    _launch_on_tensor_device(A, launch)


def _lowp_gemv_standard(
    original, dtype, kernel, trans, m, n, alpha, A, lda, x, incx, beta, y, incy
):
    # Existing paths cover strided vectors and narrow split reductions.
    if (
        trans != CUBLAS_OP_T
        or incx != 1
        or incy != 1
        or (n <= 64 and m >= 4096)
        or m == 0
        or n == 0
    ):
        return original(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)

    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    assert A.dtype == dtype and x.dtype == dtype and y.dtype == dtype
    assert A.device == x.device == y.device
    assert lda >= n
    assert x.numel() >= m and y.numel() >= n
    alpha = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta = float(beta.item() if isinstance(beta, torch.Tensor) else beta)
    if alpha == 0.0:
        return original(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)

    if _use_large_gemv_t(m, n, 2):
        kernel = hgemv_t_large_kernel_ascend if dtype == torch.float16 else bfgemv_t_large_kernel_ascend
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
    with torch_device_fn.device(A.device):
        kernel[grid](A, x, y, alpha, beta, n, m, lda, 1, 1, beta == 0.0)


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
    return _lowp_gemv_ascend(
        _common.hgemv,
        torch.float16,
        hgemv_t_coalesced_kernel_ascend,
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
    return _lowp_gemv_ascend(
        _common.bfgemv,
        torch.bfloat16,
        bfgemv_t_coalesced_kernel_ascend,
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
    )


# Keep launch-only additions after the kernels: Triton includes their source
# line numbers in JIT/autotune cache keys, so moving kernels would retune them.
# Match the Ascend SPR launch path. Unsupported launchers keep the normal path.
try:
    from triton.backends.ascend import utils as _ascend_driver_utils  # noqa: E402
    from triton.backends.ascend.driver import NPULauncher as _NPU_LAUNCHER  # noqa: E402
except ImportError:
    _NPU_LAUNCHER = None

try:
    from torch_npu._C import _npu_getCurrentRawStreamNoWait as _current_raw_stream  # noqa: E402
except ImportError:

    def _current_raw_stream(device):
        return triton.runtime.driver.active.get_current_stream(device)


try:
    from torch_npu._C import _npu_getDevice as _current_device  # noqa: E402
except ImportError:
    _current_device = torch_device_fn.current_device

_CGEMV_LAUNCH_CACHE = {}
_HOOK_CHAIN_TYPE = getattr(triton.knobs, "HookChain", None)


class _CGemvDevicePointer(int):
    """Current complex storage address with float-view shape for profiling."""

    def size(self):
        return (*self.tensor.size(), 2)


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
    # The packed N kernel also supports contiguous odd dimensions.
    if (
        _NPU_LAUNCHER is None
        or min(m, n) < 64
        or ((m % 2 or n % 2) and (
            trans != CUBLAS_OP_N
            or not (A.is_contiguous() and x.is_contiguous() and y.is_contiguous())
        ))
        or max(m, n) >= 16 * min(m, n)
        or incx != 1 or incy != 1
        or m * n * 8 >= 256 * 1024 * 1024
    ):
        return _cgemv_standard(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    assert A.dtype == x.dtype == y.dtype == torch.complex64
    assert A.device == x.device == y.device
    assert trans in (CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C) and lda >= n
    out, reduce = (m, n) if trans == CUBLAS_OP_N else (n, m)
    assert x.numel() >= reduce and y.numel() >= out
    if A.is_conj() or x.is_conj() or y.is_conj():
        return _cgemv_standard(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
    beta = beta.item() if isinstance(beta, torch.Tensor) else beta
    ar = float(alpha.real) if isinstance(alpha, complex) else float(alpha)
    ai = float(alpha.imag) if isinstance(alpha, complex) else 0.0
    br = float(beta.real) if isinstance(beta, complex) else float(beta)
    bi = float(beta.imag) if isinstance(beta, complex) else 0.0
    if ar == ai == 0:
        return _cgemv_standard(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    assert A.device.type == "npu"
    device = A.device.index
    if device != _current_device():
        with torch_device_fn.device(A.device):
            return cgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    pointers = (A.data_ptr(), x.data_ptr(), y.data_ptr())
    scalars = (
        ar, ai, br, bi, out, reduce, lda,
        trans != CUBLAS_OP_N, trans == CUBLAS_OP_C, br == bi == 0,
    )
    key = (device, scalars, tuple(pointer % 16 for pointer in pointers))
    cached = _CGEMV_LAUNCH_CACHE.get(key)
    if cached is None:
        entry = (
            cgemv_n_packed_kernel_ascend
            if trans == CUBLAS_OP_N else cgemv_t_packed_kernel_ascend
        )
        args = (torch.view_as_real(A), torch.view_as_real(x), torch.view_as_real(y), *scalars)
        grid = lambda meta: (triton.cdiv(out, meta["BLOCK_SIZE_M"]),)
        # Preserve the original libentry/libtuner first call and y restoration.
        compiled, meta = entry[grid](*args)
        constants = tuple(meta[name] for name in entry.jit_function.arg_names[len(args):])
        resolved = (grid(meta)[0], 1, 1)
        run = compiled.run
        special = (
            getattr(run, "compile_only", False)
            or getattr(run, "enable_msprof_register_tensor", False)
            or getattr(compiled.metadata, "debug_enabled", False)
        )
        if len(_CGEMV_LAUNCH_CACHE) >= 512:
            _CGEMV_LAUNCH_CACHE.clear()
        _CGEMV_LAUNCH_CACHE[key] = (compiled, run, resolved, constants, special)
        return
    compiled, run, grid, constants, special = cached
    stream = _current_raw_stream(device)
    enter = triton.knobs.runtime.launch_enter_hook
    exit_hook = triton.knobs.runtime.launch_exit_hook
    has_enter = enter is not None and (
        type(enter) is not _HOOK_CHAIN_TYPE or bool(enter.calls)
    )
    has_exit = exit_hook is not None and (
        type(exit_hook) is not _HOOK_CHAIN_TYPE or bool(exit_hook.calls)
    )
    if special or has_enter or has_exit or type(run) is not _NPU_LAUNCHER:
        compiled[grid](
            torch.view_as_real(A), torch.view_as_real(x), torch.view_as_real(y),
            *scalars, *constants, stream=stream,
        )
        return
    pointer_args = []
    for pointer, tensor in zip(pointers, (A, x, y)):
        arg = _CGemvDevicePointer(pointer)
        arg.tensor = tensor
        pointer_args.append(arg)
    direct = (
        not run.compile_only
        and not run.enable_msprof_register_tensor
        and not getattr(run.metadata, "debug_enabled", False)
    )
    launch = run.launch if direct else run
    registered = launch(
        *grid, stream, compiled.function, compiled.packed_metadata, None, None, None,
        *pointer_args, *scalars, *constants,
    )
    if direct:
        _ascend_driver_utils.TRITON_PROFILER_REGISTERED = registered == 1


# Keep new code below existing kernels to preserve their Triton cache identities.
# SGEMV kernels are defined below; preserve existing kernel source positions.


@lru_cache(maxsize=None)
def _sgemv_stream_entry(transpose, device):
    offset = _cgemv_matrix_l2_offset(device)
    name = "sgemv_t_stream_ascend" if transpose else "sgemv_n_stream_ascend"
    configs = []
    for config in runtime.get_tuned_config(name):
        kwargs = dict(config.kwargs, TRANSPOSE=transpose, L2_BYPASS_OFFSET=offset)
        configs.append(triton.Config(
            kwargs, num_warps=config.num_warps, num_stages=config.num_stages,
        ))
    return libentry()(libtuner(
        configs=configs,
        key=["OUT", "REDUCE", "LDA", "INCX", "INCY", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(sgemv_stream_kernel))


_SGEMV_LAUNCH_CACHE = {}


class _SGemvDevicePointer(int):
    def size(self):
        return self.tensor.size()


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
    # Retain existing paths for odd, extreme, strided, and multi-GB matrices.
    if (
        _NPU_LAUNCHER is None
        or min(m, n) < 64
        or m % 2 or n % 2
        or max(m, n) >= 16 * min(m, n)
        or incx != 1 or incy != 1
        or m * n * 4 >= 2 * 1024**3
    ):
        return _sgemv_standard(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    assert A.dtype == x.dtype == y.dtype == torch.float32
    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    assert A.device == x.device == y.device and A.device.type == "npu"
    assert trans in (CUBLAS_OP_N, CUBLAS_OP_T) and lda >= n
    out, reduce = (m, n) if trans == CUBLAS_OP_N else (n, m)
    assert x.numel() >= reduce and y.numel() >= out
    alpha = float(alpha.item()) if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = float(beta.item()) if isinstance(beta, torch.Tensor) else float(beta)
    if alpha == 0:
        return _sgemv_standard(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    device = A.device.index
    if device != _current_device():
        with torch_device_fn.device(A.device):
            return sgemv(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    pointers = (A.data_ptr(), x.data_ptr(), y.data_ptr())
    scalars = (alpha, beta, out, reduce, lda, incx, incy, beta == 0)
    key = (device, trans, scalars, tuple(pointer % 16 for pointer in pointers))
    cached = _SGEMV_LAUNCH_CACHE.get(key)
    if cached is None:
        if _use_sgemv_medium_t(trans, m, n) or _use_sgemv_tile_reduce(trans, m, n):
            entry = _sgemv_medium_t_entry(device) if _use_sgemv_medium_t(trans, m, n) else _sgemv_reduce_entry(trans == CUBLAS_OP_T, device)
        elif (min(m, n) >= 1024 and m * n * 4 >= 64 * 1024**2) or (
            trans == CUBLAS_OP_T and min(m, n) >= 2048 and m * n * 4 >= 32 * 1024**2
        ):
            entry = _sgemv_stream_entry(trans == CUBLAS_OP_T, device)
        elif trans == CUBLAS_OP_T:
            entry = sgemv_t_coalesced_kernel
        else:
            path = _select_sgemv_n_path(m, n)
            entry = {
                "small_square": sgemv_n_small_square_kernel,
                "four_to_one": sgemv_n_four_to_one_kernel,
            }.get(path, _common.sgemv_n_kernel)
        args = (A, x, y, *scalars)
        grid = lambda meta: (triton.cdiv(out, meta["BLOCK_SIZE_M"]),)
        # First call retains libentry/libtuner and its y restoration.
        compiled, meta = entry[grid](*args)
        constants = tuple(meta[name] for name in entry.jit_function.arg_names[len(args):])
        run = compiled.run
        special = (
            getattr(run, "compile_only", False)
            or getattr(run, "enable_msprof_register_tensor", False)
            or getattr(compiled.metadata, "debug_enabled", False)
        )
        if len(_SGEMV_LAUNCH_CACHE) >= 512:
            _SGEMV_LAUNCH_CACHE.clear()
        _SGEMV_LAUNCH_CACHE[key] = (
            compiled, run, (grid(meta)[0], 1, 1), constants, special,
        )
        return
    compiled, run, grid, constants, special = cached
    stream = _current_raw_stream(device)
    enter = triton.knobs.runtime.launch_enter_hook
    exit_hook = triton.knobs.runtime.launch_exit_hook
    has_enter = enter is not None and (
        type(enter) is not _HOOK_CHAIN_TYPE or bool(enter.calls)
    )
    has_exit = exit_hook is not None and (
        type(exit_hook) is not _HOOK_CHAIN_TYPE or bool(exit_hook.calls)
    )
    if special or has_enter or has_exit or type(run) is not _NPU_LAUNCHER:
        compiled[grid](A, x, y, *scalars, *constants, stream=stream)
        return
    pointer_args = []
    for pointer, tensor in zip(pointers, (A, x, y)):
        arg = _SGemvDevicePointer(pointer)
        arg.tensor = tensor
        pointer_args.append(arg)
    direct = (
        not run.compile_only
        and not run.enable_msprof_register_tensor
        and not getattr(run.metadata, "debug_enabled", False)
    )
    launch = run.launch if direct else run
    registered = launch(
        *grid, stream, compiled.function, compiled.packed_metadata, None, None, None,
        *pointer_args, *scalars, *constants,
    )
    if direct:
        _ascend_driver_utils.TRITON_PROFILER_REGISTERED = registered == 1


@triton.jit
def sgemv_stream_kernel(
    a_ptr, x_ptr, y_ptr, alpha: tl.float32, beta: tl.float32,
    OUT, REDUCE, LDA, INCX, INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    L2_BYPASS_OFFSET: tl.constexpr,
):
    if L2_BYPASS_OFFSET:
        a_ptr = (a_ptr.to(tl.uint64) + L2_BYPASS_OFFSET).to(tl.pointer_type(tl.float32))
    outputs = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    ki = tl.arange(0, BLOCK_SIZE_K)
    if TRANSPOSE:
        acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), tl.float32)
    else:
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float32)
    for start in range(0, REDUCE, BLOCK_SIZE_K):
        ks = start + ki
        xv = tl.load(x_ptr + ks * INCX, mask=ks < REDUCE, other=0)
        if TRANSPOSE:
            av = tl.load(a_ptr + ks[:, None] * LDA + outputs[None, :],
                         mask=(ks[:, None] < REDUCE) & (outputs[None, :] < OUT), other=0)
            acc += av * xv[:, None]
        else:
            av = tl.load(a_ptr + outputs[:, None] * LDA + ks[None, :],
                         mask=(outputs[:, None] < OUT) & (ks[None, :] < REDUCE), other=0)
            acc += av * xv[None, :]
    if TRANSPOSE:
        total = tl.sum(acc, 0)
    else:
        total = tl.sum(acc, 1)
    value = alpha * total
    if not BETA_IS_ZERO:
        value += beta * tl.load(y_ptr + outputs * INCY, mask=outputs < OUT, other=0)
    tl.store(y_ptr + outputs * INCY, value, mask=outputs < OUT)


@triton.jit
def sgemv_reduce_each_tile(
    a_ptr, x_ptr, y_ptr, alpha: tl.float32, beta: tl.float32,
    OUT, REDUCE, LDA, INCX, INCY, BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    TRANSPOSE: tl.constexpr, L2_BYPASS_OFFSET: tl.constexpr,
):
    if L2_BYPASS_OFFSET:
        a_ptr = (a_ptr.to(tl.uint64) + L2_BYPASS_OFFSET).to(tl.pointer_type(tl.float32))
    outputs = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    ki = tl.arange(0, BLOCK_SIZE_K)
    acc = tl.zeros((BLOCK_SIZE_M,), tl.float32)
    for start in range(0, REDUCE, BLOCK_SIZE_K):
        ks = start + ki
        xv = tl.load(x_ptr + ks * INCX, mask=ks < REDUCE, other=0)
        if TRANSPOSE:
            av = tl.load(a_ptr + ks[:, None] * LDA + outputs[None, :],
                         mask=(ks[:, None] < REDUCE) & (outputs[None, :] < OUT), other=0)
            acc += tl.sum(av * xv[:, None], 0)
        else:
            av = tl.load(a_ptr + outputs[:, None] * LDA + ks[None, :],
                         mask=(outputs[:, None] < OUT) & (ks[None, :] < REDUCE), other=0)
            acc += tl.sum(av * xv[None, :], 1)
    value = alpha * acc
    if not BETA_IS_ZERO:
        value += beta * tl.load(y_ptr + outputs * INCY, mask=outputs < OUT, other=0)
    tl.store(y_ptr + outputs * INCY, value, mask=outputs < OUT)


def _use_sgemv_tile_reduce(trans, m, n):
    size = m * n * 4
    if trans == CUBLAS_OP_N:
        return min(m, n) >= 1024 and m <= n <= 4096 and 16 * 1024**2 <= size < 64 * 1024**2
    return min(m, n) >= 1024 and m >= 4 * n and 64 * 1024**2 <= size < 512 * 1024**2


@lru_cache(maxsize=None)
def _sgemv_reduce_entry(transpose, device):
    offset = _cgemv_matrix_l2_offset(device)
    name = "sgemv_t_reduce_tiles_ascend" if transpose else "sgemv_n_reduce_tiles_ascend"
    configs = []
    for config in runtime.get_tuned_config(name):
        kwargs = dict(config.kwargs, TRANSPOSE=transpose, L2_BYPASS_OFFSET=offset)
        configs.append(triton.Config(
            kwargs, num_warps=config.num_warps, num_stages=config.num_stages,
        ))
    return libentry()(libtuner(
        configs=configs,
        key=["OUT", "REDUCE", "LDA", "INCX", "INCY", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(sgemv_reduce_each_tile))


@triton.jit
def _lowp_gemv_stream_jit(
    a_ptr, x_ptr, y_ptr, alpha: tl.float32, beta: tl.float32,
    OUT, REDUCE, LDA, INCX, INCY,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    L2_BYPASS_OFFSET: tl.constexpr,
):
    if L2_BYPASS_OFFSET:
        a_ptr = (a_ptr.to(tl.uint64) + L2_BYPASS_OFFSET).to(tl.pointer_type(a_ptr.dtype.element_ty))
    outputs = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    ki = tl.arange(0, BLOCK_SIZE_K)
    if TRANSPOSE:
        acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), tl.float32)
    else:
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float32)
    for start in range(0, REDUCE, BLOCK_SIZE_K):
        ks = start + ki
        xv = tl.load(x_ptr + ks * INCX, mask=ks < REDUCE, other=0).to(tl.float32)
        if TRANSPOSE:
            av = tl.load(a_ptr + ks[:, None] * LDA + outputs[None, :],
                         mask=(ks[:, None] < REDUCE) & (outputs[None, :] < OUT), other=0).to(tl.float32)
            acc += av * xv[:, None]
        else:
            av = tl.load(a_ptr + outputs[:, None] * LDA + ks[None, :],
                         mask=(outputs[:, None] < OUT) & (ks[None, :] < REDUCE), other=0).to(tl.float32)
            acc += av * xv[None, :]
    if TRANSPOSE:
        total = tl.sum(acc, 0)
    else:
        total = tl.sum(acc, 1)
    value = alpha * total
    if not BETA_IS_ZERO:
        value += beta * tl.load(y_ptr + outputs * INCY, mask=outputs < OUT, other=0).to(tl.float32)
    tl.store(y_ptr + outputs * INCY, value, mask=outputs < OUT)



def _use_lowp_gemv_stream(dtype, trans, m, n, incx, incy):
    # Retain existing small, odd, strided, extreme and multi-GB paths.
    return (dtype in (torch.float16, torch.bfloat16)
            and trans in (CUBLAS_OP_N, CUBLAS_OP_T) and incx == incy == 1
            and min(m, n) >= 1024 and m % 2 == n % 2 == 0
            and max(m, n) < 16 * min(m, n)
            and 64 * 1024**2 <= m * n * 2 < 1024**3)


@lru_cache(maxsize=None)
def _lowp_gemv_stream_entry(dtype, transpose, device):
    offset = _cgemv_matrix_l2_offset(device)
    name = ("hgemv" if dtype == torch.float16 else "bfgemv") + (
        "_t_stream_ascend" if transpose else "_n_stream_ascend"
    )
    configs = [triton.Config(
        dict(config.kwargs, TRANSPOSE=transpose, L2_BYPASS_OFFSET=offset),
        num_warps=config.num_warps, num_stages=config.num_stages,
    ) for config in runtime.get_tuned_config(name)]
    return libentry()(libtuner(
        configs=configs,
        key=["OUT", "REDUCE", "LDA", "INCX", "INCY", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(_lowp_gemv_stream_jit))


def _lowp_gemv_launch_standard(
    original, dtype, kernel, trans, m, n, alpha, A, lda, x, incx, beta, y, incy
):
    if not _use_lowp_gemv_stream(dtype, trans, m, n, incx, incy):
        return _lowp_gemv_standard(
            original, dtype, kernel, trans, m, n, alpha, A, lda, x, incx, beta, y, incy
        )
    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    assert A.dtype == x.dtype == y.dtype == dtype
    assert A.device == x.device == y.device and lda >= n
    out, reduce = (m, n) if trans == CUBLAS_OP_N else (n, m)
    assert x.numel() >= reduce and y.numel() >= out
    alpha = float(alpha.item()) if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = float(beta.item()) if isinstance(beta, torch.Tensor) else float(beta)
    if alpha == 0:
        return original(trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    entry = _lowp_gemv_stream_entry(dtype, trans == CUBLAS_OP_T, A.device.index)
    grid = lambda meta: (triton.cdiv(out, meta["BLOCK_SIZE_M"]),)
    with torch_device_fn.device(A.device):
        entry[grid](A, x, y, alpha, beta, out, reduce, lda, incx, incy, beta == 0)


_LOWP_GEMV_LAUNCH_CACHE = {}


def _lowp_gemv_ascend(
    original, dtype, kernel, trans, m, n, alpha, A, lda, x, incx, beta, y, incy
):
    # Retain existing paths for odd, extreme, strided, and multi-GB matrices.
    if (
        _NPU_LAUNCHER is None
        or min(m, n) < 64
        or m % 2 or n % 2
        or max(m, n) >= 16 * min(m, n)
        or incx != 1 or incy != 1
        or m * n * 2 >= 1024**3
    ):
        return _lowp_gemv_launch_standard(original, dtype, kernel, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    assert A.dtype == x.dtype == y.dtype == dtype
    assert A.is_contiguous() and x.is_contiguous() and y.is_contiguous()
    assert A.device == x.device == y.device and A.device.type == "npu"
    assert trans in (CUBLAS_OP_N, CUBLAS_OP_T) and lda >= n
    out, reduce = (m, n) if trans == CUBLAS_OP_N else (n, m)
    assert x.numel() >= reduce and y.numel() >= out
    alpha = float(alpha.item()) if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = float(beta.item()) if isinstance(beta, torch.Tensor) else float(beta)
    if alpha == 0:
        return _lowp_gemv_launch_standard(original, dtype, kernel, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    device = A.device.index
    if device != _current_device():
        with torch_device_fn.device(A.device):
            return _lowp_gemv_ascend(original, dtype, kernel, trans, m, n, alpha, A, lda, x, incx, beta, y, incy)
    pointers = (A.data_ptr(), x.data_ptr(), y.data_ptr())
    small_n = trans == CUBLAS_OP_N and _common._is_small_pow2_n(n)
    scalars = (alpha, beta, m, lda, incx, incy, beta == 0, n) if small_n else (
        alpha, beta, out, reduce, lda, incx, incy, beta == 0
    )
    key = (dtype, device, trans, small_n, scalars, tuple(pointer % 16 for pointer in pointers))
    cached = _LOWP_GEMV_LAUNCH_CACHE.get(key)
    if cached is None:
        if _use_lowp_gemv_block(dtype, trans, m, n) or _use_lowp_large_full(dtype, trans, m, n):
            entry = _lowp_large_full_entry(trans == CUBLAS_OP_T, device) if _use_lowp_large_full(dtype, trans, m, n) else _lowp_gemv_block_entry(dtype, trans == CUBLAS_OP_T, device)
        elif _use_lowp_gemv_stream(dtype, trans, m, n, incx, incy):
            entry = _lowp_gemv_stream_entry(dtype, trans == CUBLAS_OP_T, device)
        elif trans == CUBLAS_OP_T:
            entry = _bfgemv_t_medium_skinny_entry() if dtype == torch.bfloat16 and _use_bfgemv_t_medium_skinny(m, n) else kernel
            if _use_large_gemv_t(m, n, 2):
                entry = hgemv_t_large_kernel_ascend if dtype == torch.float16 else bfgemv_t_large_kernel_ascend
        elif small_n:
            entry = _common.hgemv_n_small_kernel if dtype == torch.float16 else _common.bfgemv_n_small_kernel
        else:
            entry = _common.hgemv_n_kernel if dtype == torch.float16 else _common.bfgemv_n_kernel
        args = (A, x, y, *scalars)
        grid = lambda meta: (triton.cdiv(out, meta["BLOCK_SIZE_M"]),)
        # First call retains libentry/libtuner and its y restoration.
        compiled, meta = entry[grid](*args)
        constants = tuple(meta[name] for name in entry.jit_function.arg_names[len(args):])
        run = compiled.run
        special = (
            getattr(run, "compile_only", False)
            or getattr(run, "enable_msprof_register_tensor", False)
            or getattr(compiled.metadata, "debug_enabled", False)
        )
        if len(_LOWP_GEMV_LAUNCH_CACHE) >= 512:
            _LOWP_GEMV_LAUNCH_CACHE.clear()
        _LOWP_GEMV_LAUNCH_CACHE[key] = (
            compiled, run, (grid(meta)[0], 1, 1), constants, special,
        )
        return
    compiled, run, grid, constants, special = cached
    stream = _current_raw_stream(device)
    enter = triton.knobs.runtime.launch_enter_hook
    exit_hook = triton.knobs.runtime.launch_exit_hook
    has_enter = enter is not None and (
        type(enter) is not _HOOK_CHAIN_TYPE or bool(enter.calls)
    )
    has_exit = exit_hook is not None and (
        type(exit_hook) is not _HOOK_CHAIN_TYPE or bool(exit_hook.calls)
    )
    if special or has_enter or has_exit or type(run) is not _NPU_LAUNCHER:
        compiled[grid](A, x, y, *scalars, *constants, stream=stream)
        return
    pointer_args = []
    for pointer, tensor in zip(pointers, (A, x, y)):
        arg = _SGemvDevicePointer(pointer)
        arg.tensor = tensor
        pointer_args.append(arg)
    direct = (
        not run.compile_only
        and not run.enable_msprof_register_tensor
        and not getattr(run.metadata, "debug_enabled", False)
    )
    launch = run.launch if direct else run
    registered = launch(
        *grid, stream, compiled.function, compiled.packed_metadata, None, None, None,
        *pointer_args, *scalars, *constants,
    )
    if direct:
        _ascend_driver_utils.TRITON_PROFILER_REGISTERED = registered == 1


@triton.jit
def _bfgemv_block_kernel_ascend(
    a_ptr, x_ptr, y_ptr, alpha: tl.float32, beta: tl.float32,
    OUT, REDUCE, LDA, INCX, INCY, BETA_IS_ZERO: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    TRANSPOSE: tl.constexpr, L2_BYPASS_OFFSET: tl.constexpr,
    FULL_TILES: tl.constexpr,
):
    if L2_BYPASS_OFFSET:
        a_ptr = (a_ptr.to(tl.uint64) + L2_BYPASS_OFFSET).to(tl.pointer_type(a_ptr.dtype.element_ty))
    outputs = tl.program_id(0) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    ki = tl.arange(0, BLOCK_SIZE_K)
    if TRANSPOSE:
        acc = tl.zeros((BLOCK_SIZE_K, BLOCK_SIZE_M), tl.float32)
    else:
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), tl.float32)
    for start in range(0, REDUCE, BLOCK_SIZE_K):
        ks = start + ki
        if FULL_TILES:
            xv = tl.load(x_ptr + ks * INCX).to(tl.float32)
        else:
            xv = tl.load(x_ptr + ks * INCX, mask=ks < REDUCE, other=0).to(tl.float32)
        if TRANSPOSE:
            offsets = ks[:, None] * LDA + outputs[None, :]
            if FULL_TILES:
                av = tl.load(a_ptr + offsets).to(tl.float32)
            else:
                av = tl.load(a_ptr + offsets, mask=(ks[:, None] < REDUCE) & (outputs[None, :] < OUT), other=0).to(tl.float32)
            product = av * xv[:, None]
            acc += product
        else:
            offsets = outputs[:, None] * LDA + ks[None, :]
            if FULL_TILES:
                av = tl.load(a_ptr + offsets).to(tl.float32)
            else:
                av = tl.load(a_ptr + offsets, mask=(outputs[:, None] < OUT) & (ks[None, :] < REDUCE), other=0).to(tl.float32)
            product = av * xv[None, :]
            acc += product
    if TRANSPOSE:
        total = tl.sum(acc, 0)
    else:
        total = tl.sum(acc, 1)
    value = alpha * total
    if not BETA_IS_ZERO:
        value += beta * tl.load(y_ptr + outputs * INCY, mask=outputs < OUT, other=0).to(tl.float32)
    tl.store(y_ptr + outputs * INCY, value, mask=outputs < OUT)


def _use_bfgemv_block(dtype, trans, m, n):
    # Output-parallel, aligned medium matrices; keep other shape families unchanged.
    return (dtype == torch.bfloat16 and 2048 <= min(m, n) <= 4096
            and m % 512 == n % 512 == 0
            and 16 * 1024**2 <= m * n * 2 < 128 * 1024**2
            and (m >= n if trans == CUBLAS_OP_N else n >= m))


@lru_cache(maxsize=None)
def _bfgemv_block_entry(transpose, device):
    name = "bfgemv_t_block_ascend" if transpose else "bfgemv_n_block_ascend"
    offset = _cgemv_matrix_l2_offset(device)
    configs = [triton.Config(
        dict(config.kwargs, TRANSPOSE=transpose, L2_BYPASS_OFFSET=offset),
        num_warps=config.num_warps, num_stages=config.num_stages,
    ) for config in runtime.get_tuned_config(name)]
    return libentry()(libtuner(
        configs=configs,
        key=["OUT", "REDUCE", "LDA", "INCX", "INCY", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(_bfgemv_block_kernel_ascend))


def _use_lowp_gemv_block(dtype, trans, m, n):
    return (_use_bfgemv_block(torch.bfloat16, trans, m, n)
            and (dtype == torch.bfloat16 or (dtype == torch.float16
                 and (trans == CUBLAS_OP_N or m * n * 2 < 32 * 1024**2))))


@lru_cache(maxsize=None)
def _lowp_gemv_block_entry(dtype, transpose, device):
    if dtype == torch.bfloat16:
        return _bfgemv_block_entry(transpose, device)
    name = "hgemv_t_block_ascend" if transpose else "hgemv_n_block_ascend"
    offset = _cgemv_matrix_l2_offset(device)
    configs = [triton.Config(
        dict(config.kwargs, TRANSPOSE=transpose, L2_BYPASS_OFFSET=offset),
        num_warps=config.num_warps, num_stages=config.num_stages,
    ) for config in runtime.get_tuned_config(name)]
    return libentry()(libtuner(
        configs=configs,
        key=["OUT", "REDUCE", "LDA", "INCX", "INCY", "BETA_IS_ZERO"],
        restore_value=["y_ptr"],
    )(_bfgemv_block_kernel_ascend))


def _use_sgemv_medium_t(trans, m, n):
    return trans == CUBLAS_OP_T and min(m,n) >= 1024 and m >= 4*n and 16*1024**2 <= m*n*4 < 64*1024**2


@lru_cache(maxsize=None)
def _sgemv_medium_t_entry(device):
    offset = _cgemv_matrix_l2_offset(device)
    configs = [triton.Config(dict(c.kwargs, TRANSPOSE=True,L2_BYPASS_OFFSET=offset),
                            num_warps=c.num_warps,num_stages=c.num_stages)
               for c in runtime.get_tuned_config("sgemv_t_medium_tiles_ascend")]
    return libentry()(libtuner(configs=configs,
        key=["OUT", "REDUCE", "LDA", "INCX", "INCY", "BETA_IS_ZERO"],
        restore_value=["y_ptr"])(sgemv_reduce_each_tile))


def _use_lowp_large_full(dtype, trans, m, n):
    return dtype == torch.float16 and m % 512 == n % 512 == 0 and 128*1024**2 <= m*n*2 < 1024**3 and (trans == CUBLAS_OP_N or (m <= 6144 and n >= 2*m))


@lru_cache(maxsize=None)
def _lowp_large_full_entry(transpose, device):
    name = "hgemv_t_large_block_ascend" if transpose else "hgemv_n_large_block_ascend"
    offset = _cgemv_matrix_l2_offset(device)
    configs = [triton.Config(dict(c.kwargs, TRANSPOSE=transpose,L2_BYPASS_OFFSET=offset,FULL_TILES=True),
                            num_warps=c.num_warps,num_stages=c.num_stages)
               for c in runtime.get_tuned_config(name)]
    return libentry()(libtuner(configs=configs,
        key=["OUT", "REDUCE", "LDA", "INCX", "INCY", "BETA_IS_ZERO"],
        restore_value=["y_ptr"])(_bfgemv_block_kernel_ascend))


def _use_bfgemv_t_medium_skinny(m, n):
    size = m * n * 2
    return (m % 512 == n % 512 == 0 and 1024 <= n <= 4096
            and m >= 4 * n and 8 * 1024**2 <= size < 64 * 1024**2)


@lru_cache(maxsize=None)
def _bfgemv_t_medium_skinny_entry():
    return libentry()(libtuner(
        configs=runtime.get_tuned_config("bfgemv_t_medium_skinny_ascend"),
        key=_common._GEMV_T_KEY,
        restore_value=["y_ptr"],
    )(_lowp_gemv_t_coalesced_jit))
