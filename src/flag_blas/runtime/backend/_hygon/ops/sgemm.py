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
from triton.runtime import heuristics

from flag_blas import runtime
from flag_blas.ops.level3.sgemm import CUBLAS_OP_N, CUBLAS_OP_T, ScalarType
from flag_blas.ops.level3.sgemm import (
    _sgemm_nn_kernel as common_sgemm_nn_kernel,
    _sgemm_nt_kernel as common_sgemm_nt_kernel,
    _sgemm_tn_kernel as common_sgemm_tn_kernel,
    _sgemm_tt_kernel as common_sgemm_tt_kernel,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.runtime.dispatch import StaticDispatch
from flag_blas.utils import libentry, libtuner

_SGEMM_KEY = ["m", "n", "k", "BETA_IS_ZERO"]


@triton.jit
def _sgemm_nn_opt_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)

    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * lda + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * ldb + offs_bn[None, :])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if EVEN_M:
        mask_m = None
    else:
        mask_m = offs_am < m
    if EVEN_N:
        mask_n = None
    else:
        mask_n = offs_bn < n

    if EVEN_K:
        for i in range(k // BLOCK_K):
            if EVEN_M:
                a = tl.load(a_ptrs)
            else:
                a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            if EVEN_N:
                b = tl.load(b_ptrs)
            else:
                b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb
    else:
        k_full_iters = k // BLOCK_K
        k_remainder = k % BLOCK_K
        for i in range(k_full_iters):
            if EVEN_M:
                a = tl.load(a_ptrs)
            else:
                a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
            if EVEN_N:
                b = tl.load(b_ptrs)
            else:
                b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
            a_ptrs += BLOCK_K
            b_ptrs += BLOCK_K * ldb
        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            if EVEN_M:
                a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
            else:
                a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            if EVEN_N:
                b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)
            else:
                b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])

    if EVEN_M and EVEN_N:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc)
        else:
            c_vals = tl.load(c_ptrs)
            tl.store(c_ptrs, alpha * acc + beta * c_vals)
    else:
        cm = offs_cm < m
        cn = offs_cn < n
        c_mask = cm[:, None] & cn[None, :]
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc, mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask)
            tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


sgemm_nn_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("sgemm_nn_hygon"),
        key=_SGEMM_KEY,
        restore_value=["c_ptr"],
    )(heuristics(runtime.get_heuristic_config("sgemm_nn"))(_sgemm_nn_opt_kernel))
)

# ---- 大 K shape 专用: 两阶段 split-k (FlagTree tle 方式, 无原子加) ----
# 内核1: 每个 block 计算部分 K 段写入 scratch(SPLIT_K, m, n)
# 内核2: reduce 合并 SPLIT_K 段累加到 C
# 实测 (py310 venv + flagtree 0.6.1): 仅对 k >= 4*max(m,n) 的 shape 有效,
# 2048x2048x16384 sp 0.778 -> 0.791 (+1.7%); 其他大 K shape 无效或变差


@libentry()
@triton.jit
def _sgemm_nn_splitk_kernel(
    a_ptr,
    b_ptr,
    scratch_ptr,
    m,
    n,
    k,
    lda,
    ldb,
    SPLIT_K: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_k = pid % SPLIT_K
    pid_mn = pid // SPLIT_K
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid_mn // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid_mn % group_size)
    pid_n = (pid_mn % width) // group_size

    k_per = k // SPLIT_K
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * k_per + tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * lda + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * ldb + offs_bn[None, :])
    mask_m = offs_am < m
    mask_n = offs_bn < n
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(k_per // BLOCK_K):
        if EVEN_M:
            a = tl.load(a_ptrs)
        else:
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
        if EVEN_N:
            b = tl.load(b_ptrs)
        else:
            b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * ldb
    if EVEN_K is False:
        kr = k_per % BLOCK_K
        if kr > 0:
            mk = tl.arange(0, BLOCK_K) < kr
            a = tl.load(
                a_ptrs,
                mask=(mask_m[:, None] if not EVEN_M else True) & mk[None, :],
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=mk[:, None] & (mask_n[None, :] if not EVEN_N else True),
                other=0.0,
            )
            acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)

    s_ptrs = (
        scratch_ptr
        + pid_k.to(tl.int64) * (m.to(tl.int64) * n)
        + offs_am[:, None].to(tl.int64) * n
        + offs_bn[None, :].to(tl.int64)
    )
    if EVEN_M and EVEN_N:
        tl.store(s_ptrs, acc)
    else:
        tl.store(s_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@libentry()
@triton.jit
def _sgemm_nn_splitk_reduce_kernel(
    scratch_ptr,
    c_ptr,
    m,
    n,
    alpha: tl.float32,
    beta: tl.float32,
    SPLIT_K: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        s_ptrs = (
            scratch_ptr
            + s * (m.to(tl.int64) * n)
            + offs_am[:, None].to(tl.int64) * n
            + offs_bn[None, :].to(tl.int64)
        )
        acc += tl.load(s_ptrs)
    c_ptrs = (
        c_ptr + offs_am[:, None].to(tl.int64) * n + offs_bn[None, :].to(tl.int64)
    )
    mask = (
        (offs_am[:, None] < m) & (offs_bn[None, :] < n)
        if not (EVEN_M and EVEN_N)
        else None
    )
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc, mask=mask)
    else:
        c_vals = tl.load(c_ptrs, mask=mask)
        tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=mask)


sgemm_tn_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("sgemm_hygon"),
        key=_SGEMM_KEY,
        restore_value=["c_ptr"],
    )(common_sgemm_tn_kernel.fn.fn)
)

sgemm_nt_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("sgemm_hygon"),
        key=_SGEMM_KEY,
        restore_value=["c_ptr"],
    )(common_sgemm_nt_kernel.fn.fn)
)

sgemm_tt_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("sgemm_hygon"),
        key=_SGEMM_KEY,
        restore_value=["c_ptr"],
    )(common_sgemm_tt_kernel.fn.fn)
)


# ---- StaticDispatch: 按 shape 情况分发不同 sgemm_nn kernel ----
# 大 K shape (k >= 4*max(m,n)): 两阶段 split-k, 实测 2048x2048x16384 有效
# 其余: 现网 libtuner opt kernel (保持原行为)

_SPLITK_BLOCK_M = 64
_SPLITK_BLOCK_N = 64
_SPLITK_BLOCK_K = 32
_SPLITK_GROUP_M = 8
_SPLITK_WARPS = 4
_SPLITK_STAGES = 2


def _sgemm_nn_is_large_k(m, n, k, **_kw):
    return k >= 4 * max(m, n)


def _sgemm_nn_is_default(**_kw):
    return True


def _make_sgemm_nn_splitk_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, **_kw
):
    scratch = torch.zeros(2, m, n, dtype=torch.float32, device=C.device)
    ev_m = m % _SPLITK_BLOCK_M == 0
    ev_n = n % _SPLITK_BLOCK_N == 0
    ev_k = k % _SPLITK_BLOCK_K == 0
    grid_mn = triton.cdiv(m, _SPLITK_BLOCK_M) * triton.cdiv(n, _SPLITK_BLOCK_N)
    grid_main = (grid_mn * 2,)
    grid_reduce = (grid_mn,)

    def run():
        _sgemm_nn_splitk_kernel[grid_main](
            A,
            B,
            scratch,
            m,
            n,
            k,
            lda,
            ldb,
            SPLIT_K=2,
            EVEN_M=ev_m,
            EVEN_N=ev_n,
            EVEN_K=ev_k,
            BLOCK_M=_SPLITK_BLOCK_M,
            BLOCK_N=_SPLITK_BLOCK_N,
            BLOCK_K=_SPLITK_BLOCK_K,
            GROUP_M=_SPLITK_GROUP_M,
            num_warps=_SPLITK_WARPS,
            num_stages=_SPLITK_STAGES,
        )
        _sgemm_nn_splitk_reduce_kernel[grid_reduce](
            scratch,
            C,
            m,
            n,
            alpha,
            beta,
            SPLIT_K=2,
            BETA_IS_ZERO=beta_is_zero,
            EVEN_M=ev_m,
            EVEN_N=ev_n,
            BLOCK_M=_SPLITK_BLOCK_M,
            BLOCK_N=_SPLITK_BLOCK_N,
            GROUP_M=_SPLITK_GROUP_M,
            num_warps=_SPLITK_WARPS,
        )

    return run


def _make_sgemm_nn_opt_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid, **_kw
):
    def run():
        sgemm_nn_hygon_kernel[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
        )

    return run


_SGEMM_NN_DISPATCH = StaticDispatch(
    [
        (_sgemm_nn_is_large_k, _make_sgemm_nn_splitk_runner),
        (_sgemm_nn_is_default, _make_sgemm_nn_opt_runner),
    ]
)


def sgemm(
    transa: int,
    transb: int,
    m: int,
    n: int,
    k: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    beta: ScalarType,
    C: torch.Tensor,
    ldc: int,
) -> None:
    assert A.is_contiguous()
    assert B.is_contiguous()
    assert C.is_contiguous()
    assert A.dtype == torch.float32
    assert B.dtype == torch.float32
    assert C.dtype == torch.float32
    assert A.device == B.device == C.device
    assert transa in [CUBLAS_OP_N, CUBLAS_OP_T]
    assert transb in [CUBLAS_OP_N, CUBLAS_OP_T]

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else float(beta)

    if m == 0 or n == 0 or k == 0 or alpha == 0.0:
        if beta == 0.0:
            C.zero_()
        elif beta != 1.0:
            C.mul_(beta)
        return

    beta_is_zero = beta == 0.0
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )

    with torch_device_fn.device(A.device):
        if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
            runner = _SGEMM_NN_DISPATCH.lookup_and_build(
                m,
                n,
                k,
                True,
                context=dict(
                    A=A,
                    lda=lda,
                    B=B,
                    ldb=ldb,
                    C=C,
                    ldc=ldc,
                    m=m,
                    n=n,
                    k=k,
                    alpha=alpha,
                    beta=beta,
                    beta_is_zero=beta_is_zero,
                    grid=grid,
                ),
            )
            runner()
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            sgemm_tn_hygon_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            sgemm_nt_hygon_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        else:
            sgemm_tt_hygon_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )


__all__ = ["sgemm"]
