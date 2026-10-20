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

import os

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

# Runtime injection of per-kernel ITT compile options (without modifying site-packages Triton source).
# Must be executed once before any sgemm kernel launch; if injection fails, only the per-kernel ITT
# capability is lost (falls back to the global default), with no impact on other functionality.
try:
    from flag_blas.runtime.backend._hygon._triton_itt import install_triton_itt_patch

    install_triton_itt_patch()
except Exception:  # pragma: no cover - silently skip when the environment has no Triton HCU backend
    pass

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

# ---- Large-K shape specific: two-stage split-k (FlagTree tle style, no atomic add) ----
# Kernel 1: each block computes a partial K segment and writes scratch(SPLIT_K, m, n)
# Kernel 2: reduce merges the SPLIT_K segments and accumulates into C
# Measured (py310 venv + flagtree 0.6.1): only effective for shapes with k >= 4*max(m,n),
# 2048x2048x16384 sp 0.778 -> 0.791 (+1.7%); other large-K shapes show no gain or regress


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


# ---- Class-A unaligned shape specific: pre-pad + all-EVEN kernel ----
# Idea: pad A(m x k) to A'(mp x kp), B(k x n) to B'(kp x np),
# where mp/kp/np are integer multiples of BLOCK; the main kernel is all EVEN (loads without mask), only stores mask against real m/n.
# Cost: one fused A/B pad-copy (memory-bandwidth bound), in exchange for main-loop mask overhead ~11% + k remainder ~13%.
# Measured (py310 venv + flagtree 0.6.1, compared with the current opt path in the report):
#   8191³ 0.732 -> 0.860, 4095³ 0.618 -> 0.739, 4097x8191x4095 0.705 -> 0.805,
#   1023³ 0.608 -> 0.623 (marginal); 511³ is actually worse (0.845 -> 0.638), excluded by the dispatch condition.


@libentry()
@triton.jit
def _sgemm_nn_pad_copy_kernel(
    a_ptr,
    b_ptr,
    a_pad_ptr,
    b_pad_ptr,
    m,
    n,
    k,
    lda,
    ldb,
    mp,
    np,
    kp,
    CM: tl.constexpr,
    CK: tl.constexpr,
):
    pid = tl.program_id(0)
    nblk_a = tl.cdiv(mp, CM) * tl.cdiv(kp, CK)
    if pid < nblk_a:
        # A tile: real m x k -> pad (mp x kp), zero-fill out of bounds
        bi = pid // tl.cdiv(kp, CK)
        bj = pid % tl.cdiv(kp, CK)
        offs_i = bi * CM + tl.arange(0, CM)
        offs_j = bj * CK + tl.arange(0, CK)
        mask = (offs_i[:, None] < m) & (offs_j[None, :] < k)
        vals = tl.load(
            a_ptr + offs_i[:, None] * lda + offs_j[None, :],
            mask=mask, other=0.0,
        )
        tl.store(a_pad_ptr + offs_i[:, None] * kp + offs_j[None, :], vals)
    else:
        # B tile: real k x n -> pad (kp x np), zero-fill out of bounds
        # Note: B's grid is partitioned by (kp/CM) x (np/CK); offs_i is the k dimension (CM rows),
        # offs_j is the n dimension (CK columns), so tile/mask are always (CM,CK), consistent with the A branch,
        # thereby supporting non-square tiles with CM != CK (the original implementation only matched when CM==CK).
        q = pid - nblk_a
        bi = q // tl.cdiv(np, CK)
        bj = q % tl.cdiv(np, CK)
        offs_i = bi * CM + tl.arange(0, CM)
        offs_j = bj * CK + tl.arange(0, CK)
        mask = (offs_i[:, None] < k) & (offs_j[None, :] < n)
        vals = tl.load(
            b_ptr + offs_i[:, None] * ldb + offs_j[None, :],
            mask=mask, other=0.0,
        )
        tl.store(b_pad_ptr + offs_i[:, None] * np + offs_j[None, :], vals)


@libentry()
@triton.jit
def _sgemm_nn_pad_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    mp,
    np,
    kp,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(mp, BLOCK_M)
    grid_n = tl.cdiv(np, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_am[:, None] * kp + offs_k[None, :])
    b_ptrs = b_ptr + (offs_k[:, None] * np + offs_bn[None, :])

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(kp // BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32, allow_tf32=False)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * np

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])
    c_mask = (offs_cm < m)[:, None] & (offs_cn < n)[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask)
        tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


# ---- tn/nt/tt per-shape compile option decisions (v2.2, 2026-08-30) ----
# ITT: when the dot layout of all three trans modes is MFMA-friendly, ITT=1 benefits across the board (all 9 shapes choose ITT1,
# 4096^3 -4~-6%, 8192^3 -14~-18%, etc.), so uniformly take True.
# buffer: aligned large/medium outputs (m*n>=2M) and 32-unaligned shapes use no-buffer (-1~-7% measured);
# (2048,2048,16384) large-K opt degrades severely under no-buffer (+43% officially confirmed on nn), keep the default.
def _sgemm_xt_use_itt(m, n, k, **_kw):
    return True


def _sgemm_xt_use_buffer_ops(m, n, k, **_kw):
    if (m, n, k) == (2048, 2048, 16384):
        return True
    if m % 32 != 0 or n % 32 != 0 or k % 32 != 0:
        return False
    if m * n >= 2 * 1024 * 1024:
        return False
    return True


# no-buffer variant: explicitly pass use_buffer_ops=False into compile options (distinguished in cache key);
# when buffer=True, do not pass it, keeping the global default artifact (same convention as the nn path).
_XT_NOBUF_KWARGS = {"use_buffer_ops": False}


# ---- tn (C = A^T B) kernel: common tn + EVEN_M/N/K constexpr (v2.2) ----
# A(k,m) lda=m, B(k,n) ldb=n; a(BM,BK) is the A^T slice (last-dim stride lda, ITT transpose),
# b(BK,BN) is contiguous. The EVEN branch eliminates mask overhead for aligned shapes (-5~-10% measured on 4096^3).
@triton.jit
def _sgemm_tn_even_kernel(
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
    a_ptrs = a_ptr + (offs_am[:, None] + offs_k[None, :] * lda)  # (BM,BK)
    b_ptrs = b_ptr + (offs_k[:, None] * ldb + offs_bn[None, :])  # (BK,BN)
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
            a_ptrs += BLOCK_K * lda
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
            a_ptrs += BLOCK_K * lda
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
        c_mask = mask_m[:, None] & mask_n[None, :]
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc, mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask)
            tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


sgemm_tn_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("sgemm_hygon"),
        key=_SGEMM_KEY,
        restore_value=["c_ptr"],
    )(heuristics(runtime.get_heuristic_config("sgemm_tn"))(_sgemm_tn_even_kernel))
)

# ---- tn composite path: pre-transpose A^T + nn dispatch (v2.3, 2026-08-30) ----
# Why tn is slow / nt is fast: MFMA (v_mmac 16x16x8_f32) requires the k dimension of the dot LHS to be contiguous in memory.
# nt's C=A B^T is written directly as acc_t=B A^T (both operands k-contiguous), hence fast; for tn's A(k,m)
# and B(k,n), the k-dimension stride is the row stride (strided), so the LHS is not k-contiguous in any direction,
# and Triton can only perform a layout conversion per iteration; this is tn's structural upper bound.
# The composite scheme follows nt's approach: use a bandwidth-bound transpose kernel to turn A^T into a contiguous (m,k) buffer,
# then go through nn dispatch (opt/pad/splitk full path), with both operands k-contiguous. The transpose kernel already reaches device
# peak (~1.1 TB/s, 128MB in 120us), with no room for further optimization.
# Measured (37-shape): unaligned large shapes 4095³ 0.657->0.758, 8191³ 0.771->0.860,
# 4097x8191x4095 0.759->0.842; aligned shapes 4096³ 0.791->0.849,
# 512x16384x4096 0.814->0.898, 2048x4096x11008 0.826->0.892,
# 2048x11008x4096 0.846->0.896, 2048x16384x2048 0.836->0.857, etc.
_TN_TP_BM = 64
_TN_TP_BK = 64
_TN_TP_GM = 4
# Minimum FLOPs for unaligned shapes to use composite: for medium/small shapes such as 1023³(1.1e9)/511³ the transpose cost
# accounts for a high proportion, and direct is measured better; from 4095³(6.9e10) the transpose takes only 4% and the mask-elimination gain is significant.
_TN_COMP_MIN_ELEMS = 10**10
# Aligned shapes are driven by an exact shape set confirmed by experiment (under the shared config table the nn kernel on this batch of
# shapes is significantly better than tn direct; for other aligned shapes the two are nearly on par, and composite only adds transpose cost).
# Exclude the microsecond-level shape (64,2048,1024): composite = tp+nn two kernels, and measurement is affected by GPU history state
# (32~62us drift); full benchmark measured 62.8us, actually far worse than direct 36.3us;
# (64,4096,2048) composite is stable at 57-60us over 5 independent measurements (direct 62-66us), so keep it.
# Exclude (2048,2048,16384): with a 263us transpose cost the gain is only ~1%, and in two full re-runs it fluctuated
# 3418/3485us (direct 3445us), with comparable positive and negative tails; keeping direct is more stable.
# v2.4 (2026-08-30): exact set expanded after a 37-shape full path scan (warmup=200/rep=50).
# - thin-m shapes (64/128/256): A^T transpose cost m*k is tiny, and the nn kernel is significantly faster than tn direct
#   (128x1024x4096 119->59us -50%, 64x1024x2048 63->35us -45%, etc.), so composite is necessarily better;
# - 16384x4096x512 (-3%), 2048x4096x12288 (-7.5%), 2048x2048x16384 (-1%), etc.
#   aligned medium/large shapes measured compA also better than direct (the original min(m,n)>=512 rule only covered unaligned).
_TN_COMP_ALIGNED_SHAPES = {
    (4096, 4096, 4096),
    (512, 16384, 4096),
    (2048, 4096, 11008),
    (2048, 11008, 4096),
    (2048, 16384, 2048),
    (64, 4096, 2048),
    (64, 512, 512),
    (64, 512, 1024),
    (64, 1024, 2048),
    (64, 2048, 4096),
    (128, 1024, 4096),
    (256, 2048, 8192),
    (512, 4096, 16384),
    (16384, 4096, 512),
    (2048, 4096, 12288),
    (2048, 2048, 16384),
}


@triton.jit
def _sgemm_tn_tp_kernel(
    src,
    dst,
    m,
    k,
    lda,
    BM: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
):
    # src(k,m) lda=m -> dst(m,k) contiguous (A^T transpose); bandwidth-bound kernel, 64x64 reaches device peak.
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BM)
    grid_k = tl.cdiv(k, BK)
    width = GM * grid_k
    gid = pid // width
    gs = min(grid_m - gid * GM, GM)
    pid_m = gid * GM + (pid % gs)
    pid_k = (pid % width) // gs
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_k = pid_k * BK + tl.arange(0, BK)
    s = tl.load(src + (offs_m[:, None] + offs_k[None, :] * lda),
                mask=(offs_k[None, :] < k) & (offs_m[:, None] < m), other=0.0)
    tl.store(dst + (offs_m[:, None] * k + offs_k[None, :]), s,
             mask=(offs_m[:, None] < m) & (offs_k[None, :] < k))


def _sgemm_tn_use_composite(m, n, k, **_kw):
    # Unaligned large shapes: transpose is equivalent to the nn pad path (eliminates mask/fold-back overhead).
    if m % 32 != 0 or n % 32 != 0 or k % 32 != 0:
        return min(m, n) >= 512 and m * n * k >= _TN_COMP_MIN_ELEMS
    return (m, n, k) in _TN_COMP_ALIGNED_SHAPES


def _sgemm_tn_use_composite_y(m, n, k, **_kw):
    # Variant Y (v2.3, 2026-08-30): transpose only B^T (cost n*k) + tt kernel.
    # The tt kernel acc_t=dot(b_t,a_t) naturally puts the strided A(k,m) on the dot RHS (MFMA does not require
    # RHS k-contiguity), so only B needs one transpose; when n≪m, n*k is far smaller than scheme A's m*k, making transpose cost negligible.
    # Measured (exp_tn_comp_tt 37-shape): (16384,512,4096) 0.815->0.853,
    # (32768,1024,1024) 0.798->0.816, (16384,2048,2048) 0.828->0.830,
    # (8192,256,2048) on par with direct; for medium/small shapes (m*n*k<1e9) transpose + two-kernel overhead is a high proportion,
    # so direct is better (e.g., (4096,128,1024) Y 0.690 vs direct 0.738), hence the lower bound on FLOPs.
    return m >= 8 * n and m * n * k >= 10**9


# ---- nt (C = A B^T) direct-layout kernel + EVEN_M/N/K (v2.2, 2026-08-30) ----
# The common _sgemm_nt_kernel does tl.trans(a) then dot(b, a_t) at every k iteration; the explicit transpose
# is extremely expensive on gfx936 (4096^3 6394us). This kernel instead loads a_t (BK,BM) and
# b_t (BN,BK) directly (both MFMA-friendly row-major), and after dot(b_t, a_t) does only one trans at the end:
# 4096^3 6394->3026us, 8192^3 50740->21478us. The EVEN branch further eliminates masks.
@triton.jit
def _sgemm_nt_even_kernel(
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
    # C = A B^T ; A(m,k) lda=k, B(n,k) ldb=k ; acc_t = C^T (N,M) = B A^T
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    gid = pid // width
    gs = min(grid_m - gid * GROUP_M, GROUP_M)
    pid_m = gid * GROUP_M + (pid % gs)
    pid_n = (pid % width) // gs
    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    b_t_ptrs = b_ptr + (offs_bn[:, None] * ldb + offs_k[None, :])  # (BN,BK) contig
    a_t_ptrs = a_ptr + (offs_k[:, None] + offs_am[None, :] * lda)  # (BK,BM)
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
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
            if EVEN_N:
                b_t = tl.load(b_t_ptrs)
            else:
                b_t = tl.load(b_t_ptrs, mask=mask_n[:, None], other=0.0)
            if EVEN_M:
                a_t = tl.load(a_t_ptrs)
            else:
                a_t = tl.load(a_t_ptrs, mask=mask_m[None, :], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
            b_t_ptrs += BLOCK_K
            a_t_ptrs += BLOCK_K
    else:
        k_full_iters = k // BLOCK_K
        k_remainder = k % BLOCK_K
        for i in range(k_full_iters):
            if EVEN_N:
                b_t = tl.load(b_t_ptrs)
            else:
                b_t = tl.load(b_t_ptrs, mask=mask_n[:, None], other=0.0)
            if EVEN_M:
                a_t = tl.load(a_t_ptrs)
            else:
                a_t = tl.load(a_t_ptrs, mask=mask_m[None, :], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
            b_t_ptrs += BLOCK_K
            a_t_ptrs += BLOCK_K
        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            if EVEN_N:
                b_t = tl.load(b_t_ptrs, mask=mask_k[None, :], other=0.0)
            else:
                b_t = tl.load(b_t_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            if EVEN_M:
                a_t = tl.load(a_t_ptrs, mask=mask_k[:, None], other=0.0)
            else:
                a_t = tl.load(a_t_ptrs, mask=mask_m[None, :] & mask_k[:, None], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
    acc = tl.trans(acc_t)
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
        c_mask = mask_m[:, None] & mask_n[None, :]
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc, mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask)
            tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


sgemm_nt_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("sgemm_hygon"),
        key=_SGEMM_KEY,
        restore_value=["c_ptr"],
    )(heuristics(runtime.get_heuristic_config("sgemm_nt"))(_sgemm_nt_even_kernel))
)

# ---- nt pad path (v2.6, 2026-08-30) ----
# The direct kernel for unaligned large shapes has high mask/k remainder overhead (nt 4095³ 0.755/8191³ 0.767/
# 4097x8191x4095 0.769). Following the nn pad idea: pre-pad A(m,k)/B(n,k) to (mp,kp)/(np,kp)
# aligned buffers, the main kernel is all EVEN (loads without mask), only stores mask against real m/n.
# Measured (warmup=100/rep=100, hipBLAS ref): 4095³ 0.755->0.840, 8191³ 0.767->0.866,
# 4097x8191x4095 0.769->0.874 (main 128x128 w8 s2 optimal).
_nt_pad_cache = {}


def _get_nt_pad_buffers(mp, np, kp, device):
    key = (mp, np, kp, str(device))
    bufs = _nt_pad_cache.get(key)
    if bufs is None:
        a = torch.zeros(mp, kp, dtype=torch.float32, device=device)
        b = torch.zeros(np, kp, dtype=torch.float32, device=device)
        bufs = (a, b)
        _nt_pad_cache[key] = bufs
    return bufs


@triton.jit
def _sgemm_nt_pad_copy_kernel(
    a_ptr,
    b_ptr,
    a_pad_ptr,
    b_pad_ptr,
    m,
    n,
    k,
    lda,
    ldb,
    mp,
    np,
    kp,
    CM: tl.constexpr,
    CK: tl.constexpr,
):
    pid = tl.program_id(0)
    nblk_a = tl.cdiv(mp, CM) * tl.cdiv(kp, CK)
    grid_j = tl.cdiv(kp, CK)
    if pid < nblk_a:
        # A(m,k) lda -> A_pad(mp,kp)
        bi = pid // grid_j
        bj = pid % grid_j
        offs_i = bi * CM + tl.arange(0, CM)
        offs_j = bj * CK + tl.arange(0, CK)
        mask = (offs_i[:, None] < m) & (offs_j[None, :] < k)
        vals = tl.load(
            a_ptr + offs_i[:, None] * lda + offs_j[None, :],
            mask=mask, other=0.0,
        )
        tl.store(a_pad_ptr + offs_i[:, None] * kp + offs_j[None, :], vals)
    else:
        # B(n,k) ldb -> B_pad(np,kp)
        q = pid - nblk_a
        bi = q // grid_j
        bj = q % grid_j
        offs_i = bi * CM + tl.arange(0, CM)
        offs_j = bj * CK + tl.arange(0, CK)
        mask = (offs_i[:, None] < n) & (offs_j[None, :] < k)
        vals = tl.load(
            b_ptr + offs_i[:, None] * ldb + offs_j[None, :],
            mask=mask, other=0.0,
        )
        tl.store(b_pad_ptr + offs_i[:, None] * kp + offs_j[None, :], vals)


@triton.jit
def _sgemm_nt_pad_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    mp,
    np,
    kp,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # A_pad(mp,kp) lda=kp, B_pad(np,kp) ldb=kp; all-EVEN load, stores mask against real m/n.
    pid = tl.program_id(0)
    grid_m = tl.cdiv(mp, BLOCK_M)
    grid_n = tl.cdiv(np, BLOCK_N)
    width = GROUP_M * grid_n
    gid = pid // width
    gs = min(grid_m - gid * GROUP_M, GROUP_M)
    pid_m = gid * GROUP_M + (pid % gs)
    pid_n = (pid % width) // gs
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    b_t_ptrs = b_ptr + (offs_bn[:, None] * kp + offs_k[None, :])  # (BN,BK)
    a_t_ptrs = a_ptr + (offs_k[:, None] + offs_am[None, :] * kp)  # (BK,BM)
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for i in range(kp // BLOCK_K):
        b_t = tl.load(b_t_ptrs)
        a_t = tl.load(a_t_ptrs)
        acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
        b_t_ptrs += BLOCK_K
        a_t_ptrs += BLOCK_K
    acc = tl.trans(acc_t)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])
    c_mask = (offs_cm < m)[:, None] & (offs_cn < n)[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask)
        tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


def _sgemm_xt_is_pad_shape(m, n, k, **_kw):
    # Only unaligned and non-small/thin shapes go through pre-pad (shared by nt/tt; nn has its own decision and tiered config)
    if (
        m % _PAD_BLOCK_M == 0
        and n % _PAD_BLOCK_N == 0
        and k % _PAD_BLOCK_K == 0
    ):
        return False
    if min(m, n) < 512:
        return False
    # nt/tt main only has a large-tile config (128x128), without nn's small-output tiers; measured 1023³ (1.07e9)
    # pad 0.45 vs direct 1.08, so the threshold must be significantly higher than nn's 5e8 to cover only 4095³ and above
    if m * n * k < _NT_PAD_MIN_ELEMS:
        return False
    return True


# nt pad specific: only large unaligned shapes of 4095³ (6.9e10) and above go through pre-pad (see _sgemm_nt_is_pad_shape)
_NT_PAD_MIN_ELEMS = 2 * 10**10


def _make_sgemm_nt_pad_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, **_kw
):
    # Unaligned large shapes: main 128x128 w8 s2 is optimal (full scan of 4095³/8191³/4097x8191x4095)
    BM, BN, BK, GM, nw, ns = 128, 128, 32, 8, 8, 2
    mp = triton.cdiv(m, BM) * BM
    np = triton.cdiv(n, BN) * BN
    kp = triton.cdiv(k, _PAD_COPY_CK) * _PAD_COPY_CK
    A_pad, B_pad = _get_nt_pad_buffers(mp, np, kp, A.device)
    grid_pad = (
        triton.cdiv(mp, _PAD_COPY_CM) * triton.cdiv(kp, _PAD_COPY_CK)
        + triton.cdiv(np, _PAD_COPY_CM) * triton.cdiv(kp, _PAD_COPY_CK),
    )
    grid_main = (triton.cdiv(mp, BM) * triton.cdiv(np, BN),)

    def run():
        _sgemm_nt_pad_copy_kernel[grid_pad](
            A,
            B,
            A_pad,
            B_pad,
            m,
            n,
            k,
            lda,
            ldb,
            mp,
            np,
            kp,
            CM=_PAD_COPY_CM,
            CK=_PAD_COPY_CK,
            num_warps=4,
            use_in_thread_transpose=False,
        )
        main_kwargs = dict(
            BETA_IS_ZERO=beta_is_zero,
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_K=BK,
            GROUP_M=GM,
            num_warps=nw,
            num_stages=ns,
            use_in_thread_transpose=True,
            use_buffer_ops=False,
        )
        _sgemm_nt_pad_kernel[grid_main](
            A_pad,
            B_pad,
            C,
            alpha,
            beta,
            m,
            n,
            mp,
            np,
            kp,
            ldc,
            **main_kwargs,
        )

    return run


# ---- tt pad path (v2.7, 2026-08-30) ----
# tt (C=A^T B^T): A(k,m) lda=m, B(n,k) ldb=k, both row-major. For unaligned large shapes the direct kernel
# incurs high mask/k remainder overhead (tt 8191³ 0.883/4097x8191x4095 0.826). Following nt pad:
# A(k,m)->A_pad(kp,mp), B(n,k)->B_pad(np,kp) pre-pad, the main kernel loads all EVEN.
# Measured (warmup=100/rep=100, hipBLAS ref): 8191³ 0.883->1.037, 4097x8191x4095 0.826->0.940,
# 4095³ 1.115->1.239 (main 128x128 w8 s2 optimal).
_tt_pad_cache = {}


def _get_tt_pad_buffers(kp, mp, np, device):
    key = (kp, mp, np, str(device))
    bufs = _tt_pad_cache.get(key)
    if bufs is None:
        a = torch.zeros(kp, mp, dtype=torch.float32, device=device)
        b = torch.zeros(np, kp, dtype=torch.float32, device=device)
        bufs = (a, b)
        _tt_pad_cache[key] = bufs
    return bufs


@triton.jit
def _sgemm_tt_pad_copy_kernel(
    a_ptr,
    b_ptr,
    a_pad_ptr,
    b_pad_ptr,
    k,
    m,
    n,
    lda,
    ldb,
    kp,
    mp,
    np,
    CM: tl.constexpr,
    CK: tl.constexpr,
):
    # A(k,m) lda -> A_pad(kp,mp), B(n,k) ldb -> B_pad(np,kp); the two have different column counts,
    # so grid_j is computed separately.
    pid = tl.program_id(0)
    nblk_a = tl.cdiv(kp, CM) * tl.cdiv(mp, CK)
    if pid < nblk_a:
        bi = pid // tl.cdiv(mp, CK)
        bj = pid % tl.cdiv(mp, CK)
        offs_i = bi * CM + tl.arange(0, CM)
        offs_j = bj * CK + tl.arange(0, CK)
        mask = (offs_i[:, None] < k) & (offs_j[None, :] < m)
        vals = tl.load(a_ptr + offs_i[:, None] * lda + offs_j[None, :], mask=mask, other=0.0)
        tl.store(a_pad_ptr + offs_i[:, None] * mp + offs_j[None, :], vals)
    else:
        q = pid - nblk_a
        bi = q // tl.cdiv(kp, CK)
        bj = q % tl.cdiv(kp, CK)
        offs_i = bi * CM + tl.arange(0, CM)
        offs_j = bj * CK + tl.arange(0, CK)
        mask = (offs_i[:, None] < n) & (offs_j[None, :] < k)
        vals = tl.load(b_ptr + offs_i[:, None] * ldb + offs_j[None, :], mask=mask, other=0.0)
        tl.store(b_pad_ptr + offs_i[:, None] * kp + offs_j[None, :], vals)


@triton.jit
def _sgemm_tt_pad_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    mp,
    np,
    kp,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # A_pad(kp,mp) lda=mp, B_pad(np,kp) ldb=kp; all EVEN loads, stores masked against real m/n.
    pid = tl.program_id(0)
    grid_m = tl.cdiv(mp, BLOCK_M)
    grid_n = tl.cdiv(np, BLOCK_N)
    width = GROUP_M * grid_n
    gid = pid // width
    gs = min(grid_m - gid * GROUP_M, GROUP_M)
    pid_m = gid * GROUP_M + (pid % gs)
    pid_n = (pid % width) // gs
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_t_ptrs = a_ptr + (offs_k[:, None] * mp + offs_am[None, :])  # (BK,BM) contiguous
    b_t_ptrs = b_ptr + (offs_bn[:, None] * kp + offs_k[None, :])  # (BN,BK) contiguous
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for i in range(kp // BLOCK_K):
        a_t = tl.load(a_t_ptrs)
        b_t = tl.load(b_t_ptrs)
        acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
        a_t_ptrs += BLOCK_K * mp
        b_t_ptrs += BLOCK_K
    acc = tl.trans(acc_t)
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + (offs_cm[:, None] * ldc + offs_cn[None, :])
    c_mask = (offs_cm < m)[:, None] & (offs_cn < n)[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc, mask=c_mask)
    else:
        c_vals = tl.load(c_ptrs, mask=c_mask)
        tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


def _make_sgemm_tt_pad_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, **_kw
):
    # Unaligned large shapes: main 128x128 w8 s2 is optimal (full scan of 4095³/8191³/4097x8191x4095)
    BM, BN, BK, GM, nw, ns = 128, 128, 32, 8, 8, 2
    mp = triton.cdiv(m, BM) * BM
    np = triton.cdiv(n, BN) * BN
    kp = triton.cdiv(k, _TT_PAD_COPY_CK) * _TT_PAD_COPY_CK
    A_pad, B_pad = _get_tt_pad_buffers(kp, mp, np, A.device)
    grid_pad = (
        triton.cdiv(kp, _TT_PAD_COPY_CM) * triton.cdiv(mp, _TT_PAD_COPY_CK)
        + triton.cdiv(np, _TT_PAD_COPY_CM) * triton.cdiv(kp, _TT_PAD_COPY_CK),
    )
    grid_main = (triton.cdiv(mp, BM) * triton.cdiv(np, BN),)

    def run():
        _sgemm_tt_pad_copy_kernel[grid_pad](
            A,
            B,
            A_pad,
            B_pad,
            k,
            m,
            n,
            lda,
            ldb,
            kp,
            mp,
            np,
            CM=_TT_PAD_COPY_CM,
            CK=_TT_PAD_COPY_CK,
            num_warps=4,
            use_in_thread_transpose=False,
        )
        main_kwargs = dict(
            BETA_IS_ZERO=beta_is_zero,
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_K=BK,
            GROUP_M=GM,
            num_warps=nw,
            num_stages=ns,
            use_in_thread_transpose=True,
            use_buffer_ops=False,
        )
        _sgemm_tt_pad_kernel[grid_main](
            A_pad,
            B_pad,
            C,
            alpha,
            beta,
            m,
            n,
            mp,
            np,
            kp,
            ldc,
            **main_kwargs,
        )

    return run


# ---- tn pad path (v2.8, 2026-08-30) ----
# tn (C=A^T B): A(k,m) lda=m matches tt's A layout, but B(k,n) ldb=n is k row-major,
# and the direct/composite paths (A^T->nn pad) both lag on large unaligned shapes (tn 4095³ 0.880/8191³ 0.918/
# 4097x8191x4095 0.934). Following tt pad: A(k,m)->A_pad(kp,mp), B(k,n) transpose+pad->
# B_pad(np,kp), reusing _sgemm_tt_pad_kernel (all-EVEN main). tt pad already reaches 1.24/
# 1.01/0.94 on the same shapes; after tn reuse, expected 4095³~1.0/8191³~0.96/4097x8191x4095~1.0.
_tn_pad_cache = {}


def _get_tn_pad_buffers(kp, mp, np, device):
    key = (kp, mp, np, str(device))
    bufs = _tn_pad_cache.get(key)
    if bufs is None:
        a = torch.zeros(kp, mp, dtype=torch.float32, device=device)
        b = torch.zeros(np, kp, dtype=torch.float32, device=device)
        bufs = (a, b)
        _tn_pad_cache[key] = bufs
    return bufs


@triton.jit
def _sgemm_tn_pad_copy_kernel(
    a_ptr,
    b_ptr,
    a_pad_ptr,
    b_pad_ptr,
    k,
    m,
    n,
    lda,
    ldb,
    kp,
    mp,
    np,
    CM: tl.constexpr,
    CK: tl.constexpr,
):
    # A(k,m) lda -> A_pad(kp,mp) (same as tt); B(k,n) ldb -> B_pad(np,kp) transpose
    # (b_pad[n,k] = B[k,n]). The grid partition differs for A/B (A by (kp,mp) tile, B by (np,kp) tile).
    pid = tl.program_id(0)
    nblk_a = tl.cdiv(kp, CM) * tl.cdiv(mp, CK)
    if pid < nblk_a:
        bi = pid // tl.cdiv(mp, CK)
        bj = pid % tl.cdiv(mp, CK)
        offs_i = bi * CM + tl.arange(0, CM)
        offs_j = bj * CK + tl.arange(0, CK)
        mask = (offs_i[:, None] < k) & (offs_j[None, :] < m)
        vals = tl.load(a_ptr + offs_i[:, None] * lda + offs_j[None, :], mask=mask, other=0.0)
        tl.store(a_pad_ptr + offs_i[:, None] * mp + offs_j[None, :], vals)
    else:
        q = pid - nblk_a
        bi = q // tl.cdiv(kp, CK)
        bj = q % tl.cdiv(kp, CK)
        offs_i = bi * CM + tl.arange(0, CM)  # n dimension
        offs_j = bj * CK + tl.arange(0, CK)  # k dimension
        mask = (offs_i[:, None] < n) & (offs_j[None, :] < k)
        vals = tl.load(b_ptr + offs_i[:, None] + offs_j[None, :] * ldb, mask=mask, other=0.0)
        tl.store(b_pad_ptr + offs_i[:, None] * kp + offs_j[None, :], vals)


def _make_sgemm_tn_pad_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, **_kw
):
    # main 128x128 w8 s2 same as tt pad (optimal in the full tt scan); the two A/B copies have the same volume as tt pad.
    BM, BN, BK, GM, nw, ns = 128, 128, 32, 8, 8, 2
    mp = triton.cdiv(m, BM) * BM
    np = triton.cdiv(n, BN) * BN
    kp = triton.cdiv(k, _PAD_COPY_CK) * _PAD_COPY_CK
    A_pad, B_pad = _get_tn_pad_buffers(kp, mp, np, A.device)
    grid_pad = (
        triton.cdiv(kp, _PAD_COPY_CM) * triton.cdiv(mp, _PAD_COPY_CK)
        + triton.cdiv(np, _PAD_COPY_CM) * triton.cdiv(kp, _PAD_COPY_CK),
    )
    grid_main = (triton.cdiv(mp, BM) * triton.cdiv(np, BN),)

    def run():
        _sgemm_tn_pad_copy_kernel[grid_pad](
            A,
            B,
            A_pad,
            B_pad,
            k,
            m,
            n,
            lda,
            ldb,
            kp,
            mp,
            np,
            CM=_PAD_COPY_CM,
            CK=_PAD_COPY_CK,
            num_warps=4,
            use_in_thread_transpose=False,
        )
        main_kwargs = dict(
            BETA_IS_ZERO=beta_is_zero,
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_K=BK,
            GROUP_M=GM,
            num_warps=nw,
            num_stages=ns,
            use_in_thread_transpose=True,
            use_buffer_ops=False,
        )
        _sgemm_tt_pad_kernel[grid_main](
            A_pad,
            B_pad,
            C,
            alpha,
            beta,
            m,
            n,
            mp,
            np,
            kp,
            ldc,
            **main_kwargs,
        )

    return run


# ---- tt (C = A^T B^T) kernel: common tt + EVEN_M/N/K (v2.2) ----
# A(k,m) lda=m, B(n,k) ldb=k; a_t(BK,BM) and b_t(BN,BK) are both contiguous row-major,
# dot(b_t, a_t) followed by a single trans writes C. Naturally MFMA-friendly layout, baseline already reaches sp~1.0,
# the EVEN branch gives another -5.5% at 8192^3.
@triton.jit
def _sgemm_tt_even_kernel(
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
    # C = A^T B^T ; acc_t = C^T (N,M) = B^T A^T
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    gid = pid // width
    gs = min(grid_m - gid * GROUP_M, GROUP_M)
    pid_m = gid * GROUP_M + (pid % gs)
    pid_n = (pid % width) // gs
    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_t_ptrs = a_ptr + (offs_k[:, None] * lda + offs_am[None, :])  # (BK,BM) contig
    b_t_ptrs = b_ptr + (offs_bn[:, None] * ldb + offs_k[None, :])  # (BN,BK) contig
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
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
                a_t = tl.load(a_t_ptrs)
            else:
                a_t = tl.load(a_t_ptrs, mask=mask_m[None, :], other=0.0)
            if EVEN_N:
                b_t = tl.load(b_t_ptrs)
            else:
                b_t = tl.load(b_t_ptrs, mask=mask_n[:, None], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
            a_t_ptrs += BLOCK_K * lda
            b_t_ptrs += BLOCK_K
    else:
        k_full_iters = k // BLOCK_K
        k_remainder = k % BLOCK_K
        for i in range(k_full_iters):
            if EVEN_M:
                a_t = tl.load(a_t_ptrs)
            else:
                a_t = tl.load(a_t_ptrs, mask=mask_m[None, :], other=0.0)
            if EVEN_N:
                b_t = tl.load(b_t_ptrs)
            else:
                b_t = tl.load(b_t_ptrs, mask=mask_n[:, None], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
            a_t_ptrs += BLOCK_K * lda
            b_t_ptrs += BLOCK_K
        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            if EVEN_M:
                a_t = tl.load(a_t_ptrs, mask=mask_k[:, None], other=0.0)
            else:
                a_t = tl.load(a_t_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)
            if EVEN_N:
                b_t = tl.load(b_t_ptrs, mask=mask_k[None, :], other=0.0)
            else:
                b_t = tl.load(b_t_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
    acc = tl.trans(acc_t)
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
        c_mask = mask_m[:, None] & mask_n[None, :]
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc, mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask)
            tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


sgemm_tt_hygon_kernel = libentry()(
    libtuner(
        configs=runtime.get_tuned_config("sgemm_hygon"),
        key=_SGEMM_KEY,
        restore_value=["c_ptr"],
    )(heuristics(runtime.get_heuristic_config("sgemm_tt"))(_sgemm_tt_even_kernel))
)


# ---- nt thin shape specific: two-stage split-k (v2.13, 2026-08-31) ----
# With the nt direct kernel's (64,64) tile, blocks are only 8-32, insufficient parallelism (gfx936 has dozens of CUs).
# split-k evenly divides k among SPLIT_K segments in parallel (kernel 1 writes scratch(SPLIT_K,n,m) in C^T layout,
# kernel 2 reduces into C), multiplying blocks by SPLIT_K. Official protocol do_bench(warmup=1000,
# rep=100, median) measured (kpack=2, DB production config):
#   (64,512,512) SP4 20.6->15.7us (+31%), (512,64,512) SP4 19.7->15.0 (+31%),
#   (64,2048,1024) SP4 31.0->25.8 (+20%), (2048,64,1024) SP4 31.0->25.8 (+20%),
#   (64,1024,512) SP4 18.4->16.3 (+13%), (1024,64,512) SP4 18.6->16.6 (+12%),
#   (64,4096,2048) SP2 57.3->53.9 (+6%).
# Shapes already at 64+ blocks (4096x128x1024/128x4096x1024) instead regress with SP (0.59-0.90),
# and (2048,2048,16384) SP 0.92; none enter the whitelist. k must be divisible by SPLIT_K*BLOCK_K,
# and all whitelist shapes are multiples of 64 (all EVEN, no k remainder).
_NT_SPLITK_BLOCK_M = 64
_NT_SPLITK_BLOCK_N = 64
_NT_SPLITK_BLOCK_K = 32
_NT_SPLITK_GROUP_M = 8
_NT_SPLITK_STAGES = 2
# shape -> (SPLIT_K, num_warps) (num_warps takes the w of the DB production direct config)
# v2.26: removed (64,512,512)/(512,64,512)/(1024,64,512) -> routed to direct override
# (BK64+kp2 full config scan official A/B: 17.28/16.80/17.76us -> 12.80/13.76/12.32us,
# significantly better than split-k).
# v2.27: removed the remaining 4 (64,1024,512)/(64,2048,1024)/(2048,64,1024)/
# (64,4096,2048) -> routed to direct override (BK64+kp2 full config scan + official A/B three rounds
# all stable, see _NT_DIRECT_OVERRIDES for details). Whitelist cleared, split-k mechanism kept for spare use.
_NT_SPLITK_SHAPES = {}


def _sgemm_nt_splitk(m, n, k, **_kw):
    # Returns SPLIT_K or 0 (do not use split-k). Only exact whitelist shapes, to avoid unintended hits.
    cfg = _NT_SPLITK_SHAPES.get((m, n, k))
    if cfg is None:
        return 0
    spk, _ = cfg
    if k % (spk * _NT_SPLITK_BLOCK_K) != 0:
        return 0
    return spk


@libentry()
@triton.jit
def _sgemm_nt_splitk_kernel(
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
    # Kernel 1: each block computes a partial K segment, writes scratch(SPLIT_K, n, m) (C^T layout).
    pid = tl.program_id(0)
    pid_k = pid % SPLIT_K
    pid_mn = pid // SPLIT_K
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    gid = pid_mn // width
    gs = min(grid_m - gid * GROUP_M, GROUP_M)
    pid_m = gid * GROUP_M + (pid_mn % gs)
    pid_n = (pid_mn % width) // gs
    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return
    k_per = k // SPLIT_K
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * k_per + tl.arange(0, BLOCK_K)
    b_t_ptrs = b_ptr + (offs_bn[:, None] * ldb + offs_k[None, :])  # (BN,BK)
    a_t_ptrs = a_ptr + (offs_k[:, None] + offs_am[None, :] * lda)  # (BK,BM)
    if EVEN_M:
        mask_m = None
    else:
        mask_m = offs_am < m
    if EVEN_N:
        mask_n = None
    else:
        mask_n = offs_bn < n
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for i in range(k_per // BLOCK_K):
        if EVEN_N:
            b_t = tl.load(b_t_ptrs)
        else:
            b_t = tl.load(b_t_ptrs, mask=mask_n[:, None], other=0.0)
        if EVEN_M:
            a_t = tl.load(a_t_ptrs)
        else:
            a_t = tl.load(a_t_ptrs, mask=mask_m[None, :], other=0.0)
        acc_t = tl.dot(b_t, a_t, acc_t, out_dtype=tl.float32, allow_tf32=False)
        b_t_ptrs += BLOCK_K
        a_t_ptrs += BLOCK_K
    s_ptrs = (
        scratch_ptr
        + pid_k.to(tl.int64) * (n.to(tl.int64) * m)
        + offs_bn[:, None].to(tl.int64) * m
        + offs_am[None, :].to(tl.int64)
    )
    if EVEN_M and EVEN_N:
        tl.store(s_ptrs, acc_t)
    else:
        tl.store(
            s_ptrs,
            acc_t,
            mask=(mask_n[:, None] if not EVEN_N else True)
            & (mask_m[None, :] if not EVEN_M else True),
        )


@libentry()
@triton.jit
def _sgemm_nt_splitk_reduce_kernel(
    scratch_ptr,
    c_ptr,
    m,
    n,
    ldc,
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
    # Kernel 2: reduce SPLIT_K segments -> C (m,n) row-major.
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_M)
    grid_n = tl.cdiv(n, BLOCK_N)
    width = GROUP_M * grid_n
    gid = pid // width
    gs = min(grid_m - gid * GROUP_M, GROUP_M)
    pid_m = gid * GROUP_M + (pid % gs)
    pid_n = (pid % width) // gs
    if pid_m * BLOCK_M >= m or pid_n * BLOCK_N >= n:
        return
    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        s_ptrs = (
            scratch_ptr
            + s * (n.to(tl.int64) * m)
            + offs_bn[:, None].to(tl.int64) * m
            + offs_am[None, :].to(tl.int64)
        )
        acc_t += tl.load(s_ptrs)
    acc = tl.trans(acc_t)
    mask_m = offs_am < m
    mask_n = offs_bn < n
    c_ptrs = c_ptr + (offs_am[:, None].to(tl.int64) * ldc + offs_bn[None, :].to(tl.int64))
    if EVEN_M and EVEN_N:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc)
        else:
            c_vals = tl.load(c_ptrs)
            tl.store(c_ptrs, alpha * acc + beta * c_vals)
    else:
        c_mask = mask_m[:, None] & mask_n[None, :]
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc, mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask)
            tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)


_nt_splitk_scratch_cache = {}


def _get_nt_splitk_scratch(sp, n, m, device):
    key = (sp, n, m, str(device))
    buf = _nt_splitk_scratch_cache.get(key)
    if buf is None:
        buf = torch.zeros(sp, n, m, dtype=torch.float32, device=device)
        _nt_splitk_scratch_cache[key] = buf
    return buf


def _make_sgemm_nt_splitk_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, SPLIT_K, num_warps, **_kw
):
    # Whitelist shapes are all (64,64) tile aligned (m/n multiples of 64) and k divisible by SPLIT_K*BK,
    # so the kernel can be statically all EVEN; the transpose kernel is not involved, no config table dependency.
    BM, BN, BK, GM = (
        _NT_SPLITK_BLOCK_M,
        _NT_SPLITK_BLOCK_N,
        _NT_SPLITK_BLOCK_K,
        _NT_SPLITK_GROUP_M,
    )
    scratch = _get_nt_splitk_scratch(SPLIT_K, n, m, A.device)
    grid1 = (triton.cdiv(m, BM) * triton.cdiv(n, BN) * SPLIT_K,)
    grid2 = (triton.cdiv(m, BM) * triton.cdiv(n, BN),)
    itt = _sgemm_xt_use_itt(m, n, k)
    nobuf_kw = _XT_NOBUF_KWARGS if not _sgemm_xt_use_buffer_ops(m, n, k) else {}

    def run():
        _sgemm_nt_splitk_kernel[grid1](
            A, B, scratch, m, n, k, lda, ldb,
            SPLIT_K, True, True, True, BM, BN, BK, GM,
            num_warps=num_warps,
            num_stages=_NT_SPLITK_STAGES,
            use_in_thread_transpose=itt,
            kpack=2,
            **nobuf_kw,
        )
        _sgemm_nt_splitk_reduce_kernel[grid2](
            scratch, C, m, n, ldc, alpha, beta, SPLIT_K, beta_is_zero,
            True, True, BM, BN, GM,
            num_warps=num_warps,
            use_in_thread_transpose=itt,
            kpack=2,
            **nobuf_kw,
        )

    return run


# ---- nt direct config override (v2.13, 2026-08-31) ----
# libtuner selects configs for the direct kernel from a shared config pool; on some unaligned small shapes the production config
# (num_warps, etc.) is not optimal. Official-protocol raw scan measured 511³ (64,64) tile w16 is 20% faster than production
# w4 (29.4->23.7us, ratio 0.745->0.93), with itt0+kp1 optimal.
# The runner always manually launches the raw kernel.
_NT_DIRECT_OVERRIDES = {
    (511, 511, 511): dict(
        BLOCK_M=64, BLOCK_N=64, GROUP_M=8, num_warps=16,
        num_stages=2, use_in_thread_transpose=False, kpack=1,
    ),
    # v2.14 (2026-08-31): official-protocol alternating A/B measurement (3 do_bench each in the same process)
    # (256,256,256) production (64,64)w4 13.44us -> (64,64)w8 kp1 13.28us (+1.2%, stable)
    # (128,4096,1024)/(4096,128,1024) show no gain or are slower with another config, not landed
    # v2.15 (2026-09-01): small tile scan, (32,32)w4 beats (64,64)w8 by a large margin:
    #   A/B 3 do_bench each in the same process: production 14.56us stable -> (32,32)w4 kp1 11.09us
    #   (MED, min 10.72), 24-36% faster. Parallelism is the bottleneck for small shapes, the 32-block grid becomes 128 programs.
    (256, 256, 256): dict(
        BLOCK_M=32, BLOCK_N=32, GROUP_M=8, num_warps=4,
        num_stages=2, use_in_thread_transpose=False, kpack=1,
    ),
    # v2.15b (2026-09-01): 512^3 same tile, more parallelism: production (64,64)w4kp1 21.60us ->
    #   (64,64)w8kp2 20.43us (-5.4%, confirmed by alternating A/B 3 times, min 20.00us).
    (512, 512, 512): dict(
        BLOCK_M=64, BLOCK_N=64, GROUP_M=8, num_warps=8,
        num_stages=2, use_in_thread_transpose=False, kpack=2,
    ),
    # v2.16 (2026-09-01): (256,8192,2048) same (512,16384,4096) pattern w4->w8 kp2:
    #   production 211.33us -> 201.12us (-4.8%, all 3 A/B runs stable), ratio 0.889->0.941.
    (256, 8192, 2048): dict(
        BLOCK_M=64, BLOCK_N=64, GROUP_M=8, num_warps=8,
        num_stages=2, use_in_thread_transpose=False, kpack=2,
    ),
    # v2.14b: (512,16384,4096) w4->w8 kp2 1356->1343us ratio 0.9495->0.9587 meets target
    #         (8192,256,2048) w4->w8 kp2 205.6->199.7us ratio 0.922->0.949 (+2.9%)
    (512, 16384, 4096): dict(
        BLOCK_M=64, BLOCK_N=64, GROUP_M=8, num_warps=8,
        num_stages=2, use_in_thread_transpose=False, kpack=2,
    ),
    (8192, 256, 2048): dict(
        BLOCK_M=64, BLOCK_N=64, GROUP_M=8, num_warps=8,
        num_stages=2, use_in_thread_transpose=False, kpack=2,
    ),
    # v2.17 (2026-09-01): 1023³ weakest (0.7051 in the 37-shape profile). Direct scan
    # (128,128)w16kp2 large tile improves A cross-block re-read utilization; the key knob is GROUP_M=1
    # (the old scan fixed GM8 and only found a marginal -2% for 128x64w8kp2). Official-protocol alternating A/B 3 rounds:
    # production (64,128)w8gm8 91.68us stable -> 88.80us stable (-3.1%, ratio 0.7051->0.7279).
    # pad path 97.9us (copy 37us eats the gain), composite excluded by min<4096, neither applicable.
    (1023, 1023, 1023): dict(
        BLOCK_M=128, BLOCK_N=128, GROUP_M=1, num_warps=16,
        num_stages=2, use_in_thread_transpose=False, kpack=2,
    ),
    # v2.18 (2026-09-01): 1024³ even-cube outlier (0.8160, 2048³/4096³ both >0.92).
    # Direct scan of 176 configs (11 tile x w8/16 x itt0/1 x kp1/2 x gm1/8): production
    # (128,128)w8gm8kp1 73.92us stable -> (64,128)w8gm1kp2 71.84us stable (-2.7%,
    # ratio 0.8225->0.845). kp2+GM1 outperforms the production tile.
    (1024, 1024, 1024): dict(
        BLOCK_M=64, BLOCK_N=128, GROUP_M=1, num_warps=8,
        num_stages=2, use_in_thread_transpose=False, kpack=2,
    ),
    # v2.19 (2026-09-01): thin shape scan (4096,128,1024)/(128,4096,1024) 128
    # configs each: (4096,128,1024) A/B no gain (scan 45.92us is noise, A/B both 46.56us),
    # not landed. (128,4096,1024) production (64,128)w8gm8 46.72us stable -> (64,128)w16gm1
    # kp2 45.76us stable (-2.1%, ratio 0.8082->0.8252).
    # v2.26b (2026-09-01): BK64 scan improves another -4.5%: (64,64)BK64w16gm8kp2itt1
    # official-protocol A/B three rounds stable 43.20us (r 0.83->0.874, production 44.96-45.60us).
    (128, 4096, 1024): dict(
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, GROUP_M=8, num_warps=16,
        num_stages=2, use_in_thread_transpose=True, kpack=2,
    ),
    # v2.20 (2026-09-01): 64³ ultra-small shape launch-bound (production (64,64)w4gm8 grid=1).
    # 384-config small tile scan (8 tile x w1/2/4 x itt x kp x gm x BK16/32): 54 configs
    # reach 8.16us, far better than production 10.24us (r=0.8125). Official A/B three rounds stably confirm
    # (32,16)w4itt0kp1gm1 8.16/8.00/8.00us (r=0.98-1.0) vs production 9.92-10.08us
    # (r=0.79-0.81). Raising parallelism with small tiles (grid 1 -> 8) is the core gain.
    (64, 64, 64): dict(
        BLOCK_M=32, BLOCK_N=16, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=False, kpack=1,
    ),
    # v2.26 (2026-09-01): 4 thin/small shapes, production config not optimal (BK=64+kp2 key).
    # Full config scan (11 tile x w4/8/16 x itt x kp x gm x BK32/64) + official-protocol A/B three rounds:
    #   (64,512,512):   production (64,64)w8kp2 17.12us r=0.83 -> (16,32)w4itt1kp2gm8
    #                   BK64 12.80/12.80/11.36us r=1.11-1.25. Small tile raises parallelism
    #                   (grid 8 -> 64 blocks).
    #   (1024,64,512):  production 17.76us r=0.85 -> (32,32)w4itt1kp2gm1 BK64
    #                   13.44/13.60/13.60us r=1.11. grid 16 -> 32 blocks.
    #   (512,64,512):   production 15.68-16.80us r=0.79-0.85 -> (16,16)w4itt0kp2gm1
    #                   BK64 13.76us stable r=0.97. grid 16 -> 128 blocks.
    #   (4096,64,2048): production 54.24-54.40us r=0.87 -> (64,64)w16itt0kp2gm1
    #                   BK64 48.32-48.48us r=0.98. grid 64 -> 64 blocks, w16 raises ILP.
    (64, 512, 512): dict(
        BLOCK_M=16, BLOCK_N=32, BLOCK_K=64, GROUP_M=8, num_warps=4,
        num_stages=2, use_in_thread_transpose=True, kpack=2,
    ),
    (1024, 64, 512): dict(
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=64, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=True, kpack=2,
    ),
    (512, 64, 512): dict(
        BLOCK_M=16, BLOCK_N=16, BLOCK_K=64, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=False, kpack=2,
    ),
    (4096, 64, 2048): dict(
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, GROUP_M=1, num_warps=16,
        num_stages=2, use_in_thread_transpose=False, kpack=2,
    ),
    # v2.27 (2026-09-02): the original 4 split-k whitelist shapes, BK64+kp2 full config scan +
    # official-protocol alternating A/B 3 rounds (all stable, see /tmp/nt_v227_ab.py for details):
    #   (64,1024,512):  split-k production 17.60/16.16/17.60us r=0.82-0.89 ->
    #                   (32,32)w4itt0kp2gm1BK64 12.16-13.28us r=1.08-1.18.
    #                   grid 4 -> 64 blocks to raise parallelism.
    #   (64,2048,1024): split-k production 24.80-25.92us r=0.88-0.92 ->
    #                   (32,64)w8itt0kp2gm8BK64 21.12-21.76us r=1.04-1.08.
    #                   grid 8 -> 64 blocks.
    #   (2048,64,1024): split-k production 24.16-25.60us r=0.93-0.98 ->
    #                   (32,64)w8itt0kp2gm8BK64 20.96-22.08us r=1.07-1.13.
    #                   grid 8 -> 64 blocks.
    #   (64,4096,2048): split-k production 53.92us r=0.89 -> (64,64)w16itt0kp2gm8BK64
    #                   48.32us r=1.00 stable (-10.4%).
    (64, 1024, 512): dict(
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=64, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=False, kpack=2,
    ),
    (64, 2048, 1024): dict(
        BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, GROUP_M=8, num_warps=8,
        num_stages=2, use_in_thread_transpose=False, kpack=2,
    ),
    (2048, 64, 1024): dict(
        BLOCK_M=32, BLOCK_N=64, BLOCK_K=64, GROUP_M=8, num_warps=8,
        num_stages=2, use_in_thread_transpose=False, kpack=2,
    ),
    (64, 4096, 2048): dict(
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, GROUP_M=8, num_warps=16,
        num_stages=2, use_in_thread_transpose=False, kpack=2,
    ),
    # v2.28 (2026-09-02): (4096,128,1024) NT largest outlier (r=0.767). BK64 full config
    # scan (12 tile x w4/8/16 x itt x kp x gm): production (64,64)w16 46.56us stable ->
    # (64,64)BK64w16itt1kp2gm1 42.40us stable. Official-protocol alternating A/B 3 rounds:
    # 45.92-46.40us (r=0.755-0.763) -> 42.88-43.04us (r=0.814-0.817, -7.4%).
    # Same (128,4096,1024) v2.26b pattern (BK64+w16+kp2). Still <0.90 but clearly improved.
    (4096, 128, 1024): dict(
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, GROUP_M=1, num_warps=16,
        num_stages=2, use_in_thread_transpose=True, kpack=2,
    ),
    # v2.29 (2026-09-02): (2048,2048,16384) large-K shape (<0.8 outlier). direct
    # config + split-k scan (10 direct + 8 split-k): production DB direct (kp1) 3289.3us
    # (r=0.7855) stable -> (64,64)BK64w16kp2gm8itt1 3190us stable (r=0.809-0.810).
    # Official-protocol alternating A/B 3 rounds: 3289.3us -> 3189.6-3192.8us (-3.0%). BK64+w16+kp2
    # raises k-loop throughput (k=16384 long loop); split-k 4 segments (128,128) 3201us is slightly worse, not taken.
    (2048, 2048, 16384): dict(
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=64, GROUP_M=8, num_warps=16,
        num_stages=2, use_in_thread_transpose=True, kpack=2,
    ),
}


def _make_sgemm_nt_direct_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, ov, **_kw
):
    BM = ov["BLOCK_M"]
    BN = ov["BLOCK_N"]
    BK = ov.get("BLOCK_K", 32)
    GM = ov["GROUP_M"]
    ev = (m % BM == 0, n % BN == 0, k % BK == 0)
    grid = (((m + BM - 1) // BM) * ((n + BN - 1) // BN),)
    nobuf_kw = _XT_NOBUF_KWARGS if not _sgemm_xt_use_buffer_ops(m, n, k) else {}

    def run():
        _sgemm_nt_even_kernel[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
            ev[0], ev[1], ev[2], BM, BN, BK, GM,
            num_warps=ov["num_warps"],
            num_stages=ov.get("num_stages", 2),
            use_in_thread_transpose=ov.get("use_in_thread_transpose", False),
            kpack=ov.get("kpack", 1),
            **nobuf_kw,
        )

    return run


# ---- tn direct config override (v2.22, 2026-09-01) ----
# The 5 underperforming shapes (core_shape profile speedup<0.90) all suffer from insufficient parallelism/efficiency of the production config:
# shrinking tiles for small shapes raises parallelism (same strategy as tt), official-protocol do_bench(1000/100/median)
# scan (scan /tmp/sgemm_tn_scan_results.json + /tmp/sgemm_tn_scan2_results.json)
# all reach speedup>=1.0. The runner always manually launches the raw kernel.
# Note: tn layout A(k,m) lda=m / B(k,n) ldb=n / C(m,n) ldc=n.
_TN_DIRECT_OVERRIDES = {
    (256, 256, 256): dict(
        BLOCK_M=32, BLOCK_N=32, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=False,
    ),
    (64, 512, 512): dict(
        BLOCK_M=32, BLOCK_N=32, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=False,
    ),
    (64, 1024, 512): dict(
        BLOCK_M=32, BLOCK_N=32, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=False,
    ),
    (512, 64, 512): dict(
        BLOCK_M=32, BLOCK_N=32, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=False,
    ),
    (1024, 64, 512): dict(
        BLOCK_M=32, BLOCK_N=32, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=False,
    ),
    # v2.30 (2026-09-02): 64³ ultra-small shape launch-bound (grid=1, production 10.56us
    # r=0.788 official protocol). Small tile scan of 54 configs: w4 across tiles all reach the 8.16us floor
    # (r=1.0-1.16). monkeypatch A/B three alternating rounds: prod 10.56us stable -> (32,32)w4gm1
    # 8.16-9.44us (r=0.87-1.02, worst round still >=0.8). Same as the other tn small shape overrides
    # (32,32)w4gm1. Small tiles raise parallelism (grid 1 -> 4).
    (64, 64, 64): dict(
        BLOCK_M=32, BLOCK_N=32, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=False,
    ),
}


def _make_sgemm_tn_direct_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, ov, **_kw
):
    BM = ov["BLOCK_M"]
    BN = ov["BLOCK_N"]
    BK = ov.get("BLOCK_K", 32)
    GM = ov["GROUP_M"]
    ev = (m % BM == 0, n % BN == 0, k % BK == 0)
    grid = (((m + BM - 1) // BM) * ((n + BN - 1) // BN),)
    nobuf_kw = _XT_NOBUF_KWARGS if not _sgemm_xt_use_buffer_ops(m, n, k) else {}

    def run():
        _sgemm_tn_even_kernel[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
            ev[0], ev[1], ev[2], BM, BN, BK, GM,
            num_warps=ov["num_warps"],
            num_stages=ov.get("num_stages", 2),
            use_in_thread_transpose=ov.get("use_in_thread_transpose", False),
            **nobuf_kw,
        )

    return run


# ---- tn composite path nn kernel config override (v2.23, 2026-09-01) ----
# comp path = tp(A^T) + nn kernel. Production uses _SGEMM_NN_DISPATCH tuner config0
# (64,64,w4,gm8,BK32,s2), which is not optimal for 4096³: experimental scan (warmup=200/rep=30, 362+ configs)
# found (64,128,w8,itt1,gm1,BK32,s2,nobuf) faster (prod 2912->2898us, r 0.886->0.890).
# The tp kernel has reached the bandwidth peak (~1.1TB/s, 120us); direct/compY structures are both worse, this is the structural floor.
_TN_COMP_OVERRIDES = {
    (4096, 4096, 4096): dict(
        BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, GROUP_M=1, num_warps=8,
        num_stages=2, use_in_thread_transpose=True,
    ),
}


def _make_sgemm_tn_comp_override_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid, ov, **_kw
):
    Ac = _get_tn_ac(m, k, A.device)
    grid_tp = ((m + _TN_TP_BM - 1) // _TN_TP_BM) * ((k + _TN_TP_BK - 1) // _TN_TP_BK)
    grid_tp = (grid_tp,)
    BM = ov["BLOCK_M"]
    BN = ov["BLOCK_N"]
    BK = ov.get("BLOCK_K", 32)
    GM = ov["GROUP_M"]
    grid_nn = (((m + BM - 1) // BM) * ((n + BN - 1) // BN),)
    nobuf_kw = _XT_NOBUF_KWARGS if not _sgemm_xt_use_buffer_ops(m, n, k) else {}

    def run():
        _sgemm_tn_tp_kernel[grid_tp](A, Ac, m, k, lda, _TN_TP_BM, _TN_TP_BK, _TN_TP_GM)
        _sgemm_nn_opt_kernel[grid_nn](
            Ac, B, C, alpha, beta, m, n, k, k, ldb, ldc, beta_is_zero,
            m % BM == 0, n % BN == 0, k % BK == 0, BM, BN, BK, GM,
            num_warps=ov["num_warps"],
            num_stages=ov.get("num_stages", 2),
            use_in_thread_transpose=ov.get("use_in_thread_transpose", False),
            **nobuf_kw,
        )

    return run


# ---- tn composite path Y (compY) tt kernel config override (v2.24, 2026-09-01) ----
# compY = tp(B^T) + tt even kernel. The production tuner config is not optimal for these 2 large shapes:
# official-protocol do_bench(1000/100/median) verification (32768,1024,1024) 1556->1540us
# (r 0.892->0.901), (16384,2048,2048) 2953->2921us (r 0.880->0.890, structural ceiling
# not reaching 0.90). The runner always manually launches the raw kernel.
_TN_COMPY_OVERRIDES = {
    (32768, 1024, 1024): dict(
        BLOCK_M=128, BLOCK_N=64, BLOCK_K=32, GROUP_M=1, num_warps=8,
        num_stages=2, kpack=2, use_in_thread_transpose=True,
    ),
    (16384, 2048, 2048): dict(
        BLOCK_M=256, BLOCK_N=128, BLOCK_K=32, GROUP_M=4, num_warps=16,
        num_stages=2, kpack=1, use_in_thread_transpose=True,
    ),
}


def _make_sgemm_tn_compy_override_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid, ov, **_kw
):
    Bc = _get_tn_bc(n, k, B.device)
    grid_tp = ((n + _TN_TP_BM - 1) // _TN_TP_BM) * ((k + _TN_TP_BK - 1) // _TN_TP_BK)
    grid_tp = (grid_tp,)
    BM = ov["BLOCK_M"]
    BN = ov["BLOCK_N"]
    BK = ov.get("BLOCK_K", 32)
    GM = ov["GROUP_M"]
    grid_tt = (((m + BM - 1) // BM) * ((n + BN - 1) // BN),)
    nobuf_kw = _XT_NOBUF_KWARGS if not _sgemm_xt_use_buffer_ops(m, n, k) else {}

    def run():
        _sgemm_tn_tp_kernel[grid_tp](B, Bc, n, k, ldb, _TN_TP_BM, _TN_TP_BK, _TN_TP_GM)
        _sgemm_tt_even_kernel[grid_tt](
            A, Bc, C, alpha, beta, m, n, k, lda, k, ldc, beta_is_zero,
            m % BM == 0, n % BN == 0, k % BK == 0, BM, BN, BK, GM,
            num_warps=ov["num_warps"],
            num_stages=ov.get("num_stages", 2),
            kpack=ov.get("kpack", 1),
            use_in_thread_transpose=ov.get("use_in_thread_transpose", False),
            **nobuf_kw,
        )

    return run


# ---- tt direct config override (v2.21, 2026-09-01) ----
# All 7 underperforming shapes (core_shape profile speedup<0.90) are caused by insufficient grid parallelism:
# the production config's large tiles leave few blocks (64x512x512 (64,64)tile 64 blocks -> 16x16 1024 blocks),
# small tiles raise parallelism, official-protocol do_bench(warmup=1000, rep=100, median) alternating A/B verification
# (scan /tmp/sgemm_tt_scan_results.json + A/B /tmp/verify_sgemm_tt_ab.py) all
# speed up 15-46% reaching speedup>=1.0. The runner always manually launches the raw kernel.
# Note: tt layout A(k,m) lda=m / B(n,k) ldb=k / C(m,n) ldc=n, EVEN determined by m,n,k.
_TT_DIRECT_OVERRIDES = {
    (64, 512, 512): dict(
        BLOCK_M=16, BLOCK_N=16, GROUP_M=1, num_warps=2,
        num_stages=2, use_in_thread_transpose=False,
    ),
    (512, 64, 512): dict(
        BLOCK_M=32, BLOCK_N=32, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=True,
    ),
    (64, 64, 64): dict(
        BLOCK_M=16, BLOCK_N=16, GROUP_M=1, num_warps=1,
        num_stages=2, use_in_thread_transpose=False,
    ),
    (1024, 64, 512): dict(
        BLOCK_M=16, BLOCK_N=16, GROUP_M=1, num_warps=2,
        num_stages=2, use_in_thread_transpose=True,
    ),
    (64, 1024, 512): dict(
        BLOCK_M=32, BLOCK_N=32, GROUP_M=8, num_warps=4,
        num_stages=2, use_in_thread_transpose=False,
    ),
    (511, 511, 511): dict(
        BLOCK_M=64, BLOCK_N=64, GROUP_M=1, num_warps=8,
        num_stages=2, use_in_thread_transpose=False,
    ),
    (256, 256, 256): dict(
        BLOCK_M=32, BLOCK_N=32, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=False,
    ),
}


def _make_sgemm_tt_direct_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, ov, **_kw
):
    BM = ov["BLOCK_M"]
    BN = ov["BLOCK_N"]
    BK = ov.get("BLOCK_K", 32)
    GM = ov["GROUP_M"]
    ev = (m % BM == 0, n % BN == 0, k % BK == 0)
    grid = (((m + BM - 1) // BM) * ((n + BN - 1) // BN),)
    nobuf_kw = _XT_NOBUF_KWARGS if not _sgemm_xt_use_buffer_ops(m, n, k) else {}

    def run():
        _sgemm_tt_even_kernel[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
            ev[0], ev[1], ev[2], BM, BN, BK, GM,
            num_warps=ov["num_warps"],
            num_stages=ov.get("num_stages", 2),
            use_in_thread_transpose=ov.get("use_in_thread_transpose", False),
            **nobuf_kw,
        )

    return run


# ---- StaticDispatch: dispatch different sgemm_nn kernels by shape ----
# Large K shape (k >= 4*max(m,n)): two-stage split-k, measured effective for 2048x2048x16384
# Unaligned and non-small shape: pre-pad + all-EVEN kernel (eliminate mask/k remainder overhead)
# Otherwise: current libtuner opt kernel (keep original behavior)

_SPLITK_BLOCK_M = 64
_SPLITK_BLOCK_N = 64
_SPLITK_BLOCK_K = 32
_SPLITK_GROUP_M = 8
_SPLITK_WARPS = 4
_SPLITK_STAGES = 2

# pad path parameters (measured optimal config)
_PAD_BLOCK_M = 128
_PAD_BLOCK_N = 256
_PAD_BLOCK_K = 32
# v1.8 (2026-08-26): copy tile changed from 64x64 to 32x32 (bandwidth-bound copy, smaller tiles give a larger grid
# and better load balance): 8191³ copy 1214->908us (-25%), 1023³ copy 50->20us, 4097x8191x4095
# copy 356us unchanged; full path 8191³ -1.6%, 1023³ -16%. 32 always divides mp/np/kp
# (mp/np are multiples of BM/BN, kp is a multiple of 128), no alignment risk.
_PAD_COPY_CM = 32
_PAD_COPY_CK = 128
# tt pad-copy dedicated tile (v2.9): measured on unaligned large shapes, a small square tile (32,32) is far better than
# (32,128) -- 4097x8191x4095 copy 487us -> 331us (1.08TB/s -> 1.2TB/s),
# 4095³/8191³ benefit similarly. Small tiles double the program count, better for latency hiding/bandwidth saturation.
_TT_PAD_COPY_CM = 32
_TT_PAD_COPY_CK = 32
# Higher-parallelism candidate for small-output shapes: main (32,64) + copy tile 32
# (1023³ full path 116->92us, -20%; copy 50->20us).
# But main BM=32 makes mp possibly not a multiple of 64, while the copy tile must divide mp/np/kp;
# when not satisfied the runner falls back to (128,128) main (mp/np are always multiples of 128, safe).
# v2.9 once tried (64,32): with buf1 it ties (32,64) (75.4us), no gain, keep (32,64).
_PAD_CFG_SMALL_TILE = (32, 64, 32, 8, 4, 2)
# Large-output shape tiers (v1.8 measured, 2026-08-26):
# - m*n >= 32M (large output, e.g. 8191³/4097x8191x4095): (256,256,32,G1,w16,s1) + ITT1 + no-buffer
#   8191³ 0.862->0.925 meets target (main is the same as opt 8192³ production config, 20652us),
#   4097x8191x4095 0.806->0.869
# - 4M <= m*n < 32M (medium output, e.g. 4095³): (256,128,32,G8,w8,s1) + ITT0 + no-buffer
#   4095³ 0.738->0.788 ((256,256,w16) instead 0.734, because large tiles are inefficient when the grid is only 16x16)
# - m*n < 4M (small output, e.g. 1023³): (32,64) avoids tile waste, keeps ITT0+buf default (0.625->0.724)
_PAD_CFG_LARGE = (256, 256, 32, 1, 16, 2)
# v1.10: MID changed from (256,128,32,G8,w8,s1)+ITT0 to (64,64,32,G8,w4,s2)+ITT1
# (same as opt 4096^3 production config). 4095^3 pad main counts as all-EVEN 4096^3:
# (64,64,ITT1) 2927us vs (256,128,ITT0) 3013us (-2.9%), ITT1 required (ITT0 3201).
# 4095^3 full path 0.788 -> ~0.809. mp/np=cdiv(4095,64)=4096 still satisfies copy tile alignment.
_PAD_CFG_MID = (64, 64, 32, 8, 4, 2)
_PAD_LARGE_OUTPUT = 32 * 1024 * 1024
_PAD_CFG_SMALL = (128, 128, 32, 8, 8, 2)
_PAD_SMALL_OUTPUT = 4 * 1024 * 1024
# Exclude small shapes like 511³ (1.3e8): pad-copy overhead dominates, measured worse instead
_PAD_MIN_ELEMS = 5 * 10**8

# pad buffer cache: avoid reallocating large buffers on every call (cached by padded size)
_pad_cache = {}


def _get_pad_buffers(mp, kp, np, device):
    key = (mp, kp, np, str(device))
    bufs = _pad_cache.get(key)
    if bufs is None:
        a = torch.zeros(mp, kp, dtype=torch.float32, device=device)
        b = torch.zeros(kp, np, dtype=torch.float32, device=device)
        bufs = (a, b)
        _pad_cache[key] = bufs
    return bufs


_LARGE_K_OPT_SHAPES = {(2048, 2048, 16384)}
# v1.10: SPLIT_K=4 is significantly better for thin large-K shapes (measured):
#   128x1024x4096 64.2->52.5us (-18.2%), 256x2048x8192 288.5->251.7us (-12.8%);
#   512x4096x16384 instead regresses with SPK=4 (1700->2825), keep SPK=2.
# Use SPK=4 only when k is divisible by SPLIT_K*BLOCK_K (ensure all EVEN within a segment, no k remainder).
_SPLITK4_SHAPES = {(128, 1024, 4096), (256, 2048, 8192)}


def _sgemm_nn_is_large_k(m, n, k, **_kw):
    # Large K shapes use two-stage split-k; the special case (2048,2048,16384) measured opt+ITT1 (0.852)
    # better than split-k+ITT0 (0.792) and opt+ITT0 (0.778), excluded by _LARGE_K_OPT_SHAPES
    if (m, n, k) in _LARGE_K_OPT_SHAPES:
        return False
    # split-k evenly divides by k_per = k // SPLIT_K; when k is odd the last element is dropped (silent error),
    # and EVEN_K must be determined by the per-segment remainder; odd k always falls back to opt (which has its own k-remainder branch)
    if k % 2 != 0:
        return False
    return k >= 4 * max(m, n)


# ITT (in-thread-transpose) per-shape decision: based on full regression measurements of ITT=0/1
# - ITT=1 (True): large shapes generally benefit (2048³ 0.828->0.904, 256x8192x2048 0.839->0.908, etc.)
# - ITT=0 (False): thin shapes (min(m,n)<=128) and small cubes are hurt by the in-thread-transpose layout
#   (64x512x512 1.054->0.774, 256³ 1.103->0.918, 511³ 0.854->0.790, etc.), the config cannot recover
# - Explicit sets override the rule exceptions (64³ ultra-small shape ITT=1 instead +0.136 to 0.983)
# v1.6 lesson (2026-08-26): the 4-combination (ITT x buffer) scan used manual launch + DB TunedConfig
# table configs, but the production libtuner actually uses benchmark table configs (the two differ, especially for thin shapes)
# -> under manual config ITT0 is better for 16384x512x4096/512x16384x4096 (-2.3%), but monkeypatching the
#    production path confirms ITT1 is the production optimum (1465us vs ITT0 1592us, the combination conclusion is flipped by config).
#    The only retained change: 2048x16384x2048 removed from _ITT0_SHAPES -> production path ITT1+nobuf 2867.6us
#    better than ITT0+nobuf 2901us (+1.2%).
_ITT0_SHAPES = {
    (256, 256, 256),
    (511, 511, 511),
}
_ITT1_SHAPES = {
    (64, 64, 64),
}


def _sgemm_nn_use_itt(m, n, k, **_kw):
    if (m, n, k) in _ITT1_SHAPES:
        return True
    if (m, n, k) in _ITT0_SHAPES:
        return False
    if min(m, n) <= 128:
        return False
    return True


# use_buffer_ops (AMDGCN_USE_BUFFER_OPS) per-shape decision: based on 33 opt shapes
# full quick scan + formal protocol (warmup=1000/rep=100) interleaved confirmation (2026-08-26)
# - no-buffer (False): large/medium output shapes (m*n>=2M) generally benefit -1~-4%
#   (4096³ -3.9%, 512x16384x4096 -4.0%, 2048x11008x4096 -4.1%, etc.);
#   shapes on the unaligned mask path benefit significantly (511³ -10.0%)
# - buffer (True, default): small output shapes keep it (1024³ +3.6%, 128x4096x1024 +3.8%
#   [formally confirmed] etc. become worse with no-buffer); the (2048,2048,16384) large K opt under no-buffer
#   becomes severely worse (+43% formally confirmed), must keep buffer
def _sgemm_nn_use_buffer_ops(m, n, k, **_kw):
    if (m, n, k) == (2048, 2048, 16384):
        return True
    if m % 32 != 0 or n % 32 != 0 or k % 32 != 0:
        return False
    if m * n >= 2 * 1024 * 1024:
        return False
    return True


def _sgemm_nn_is_pad_shape(m, n, k, **_kw):
    # Only unaligned (relative to pad tile size) and non-small/thin shapes take the pre-pad path
    if (
        m % _PAD_BLOCK_M == 0
        and n % _PAD_BLOCK_N == 0
        and k % _PAD_BLOCK_K == 0
    ):
        return False
    # Thin shapes (either output dimension < 512) use opt: measured pad-copy overhead dominates and is actually worse,
    # also covering 511³, 64³ and 4096x64x2048 etc. (n/m=64/128 not aligned to BN=256)
    if min(m, n) < 512:
        return False
    if m * n * k < _PAD_MIN_ELEMS:
        return False
    return True


def _sgemm_nn_is_default(**_kw):
    return True


def _make_sgemm_nn_pad_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, **_kw
):
    if m * n < _PAD_SMALL_OUTPUT:
        # Small output: main (32,64) raises parallelism (1023³ full path 116->92us, -20%);
        # main BM=32 makes mp possibly not a multiple of 128, falls back to (128,128) main when not satisfied.
        # copy uniformly 32x32 (v1.8: optimal for bandwidth-bound copy, 1023³ copy 50->20us).
        BM, BN, BK, GM, nw, ns = _PAD_CFG_SMALL_TILE
        mp = triton.cdiv(m, BM) * BM
        np = triton.cdiv(n, BN) * BN
        if mp % 128 != 0 or np % 128 != 0:
            BM, BN, BK, GM, nw, ns = _PAD_CFG_SMALL
            mp = triton.cdiv(m, BM) * BM
            np = triton.cdiv(n, BN) * BN
        use_buf = True   # small output keeps buffer ops default (1023³ buf0 is actually slightly worse)
        use_itt = False
    elif m * n >= _PAD_LARGE_OUTPUT:
        # Large output: (256,256,32,G1,w16,s2) + ITT1 + no-buffer
        # (s2 is 13% faster than s1: 8191³ 0.795->0.924, 4097x8191x4095 0.755->0.869)
        BM, BN, BK, GM, nw, ns = _PAD_CFG_LARGE
        mp = triton.cdiv(m, BM) * BM
        np = triton.cdiv(n, BN) * BN
        use_buf = False
        use_itt = True
    else:
        # Medium output: (64,64,32,G8,w4,s2) + ITT1 + no-buffer (v1.10, 4095³ 0.809)
        BM, BN, BK, GM, nw, ns = _PAD_CFG_MID
        mp = triton.cdiv(m, BM) * BM
        np = triton.cdiv(n, BN) * BN
        use_buf = False
        use_itt = True
    CCM = CCK = _PAD_COPY_CM
    # kp must be a multiple of both the main kernel BLOCK_K (32) and the pad-copy tile (32):
    # the copy kernel tiles the k dimension by CK and stores without mask; if kp is not a multiple of CK
    # the last block writes beyond kp (e.g. 512x12800x5210, kp=5216 -> writes to row 5247, out-of-bounds VMFault).
    # 128 is a multiple of 32, aligning to _PAD_COPY_CK (128) satisfies both.
    kp = triton.cdiv(k, _PAD_COPY_CK) * _PAD_COPY_CK
    A_pad, B_pad = _get_pad_buffers(mp, kp, np, A.device)
    grid_pad = (
        triton.cdiv(mp, CCM) * triton.cdiv(kp, CCK)
        + triton.cdiv(kp, CCM) * triton.cdiv(np, CCK),
    )
    grid_main = (triton.cdiv(mp, BM) * triton.cdiv(np, BN),)

    def run():
        _sgemm_nn_pad_copy_kernel[grid_pad](
            A,
            B,
            A_pad,
            B_pad,
            m,
            n,
            k,
            lda,
            ldb,
            mp,
            np,
            kp,
            CM=CCM,
            CK=CCK,
            num_warps=4,  # nw4 is better than nw8 with tile32: 8191³ copy 948->908, 4095³ 229->223, 4097 359->345
            use_in_thread_transpose=False,  # pad path regresses with ITT=1 (8191³ 0.862->0.823)
        )
        main_kwargs = dict(
            BETA_IS_ZERO=beta_is_zero,
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_K=BK,
            GROUP_M=GM,
            num_warps=nw,
            num_stages=ns,
            use_in_thread_transpose=use_itt,  # large/medium output ITT1, small output ITT0
        )
        if not use_buf:
            # no-buffer variant (large/medium output): explicitly pass use_buffer_ops=False into the compile options;
            # when use_buf=True do not pass it, keeping the global default artifact
            main_kwargs["use_buffer_ops"] = False
        _sgemm_nn_pad_kernel[grid_main](
            A_pad,
            B_pad,
            C,
            alpha,
            beta,
            m,
            n,
            mp,
            np,
            kp,
            ldc,
            **main_kwargs,
        )

    return run


def _make_sgemm_nn_splitk_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, **_kw
):
    # per-shape SPLIT_K: thin large-K shapes (128x1024x4096/256x2048x8192) measured SPK=4
    # significantly better (-18.2%/-12.8%); 512x4096x16384 etc. severely regress with SPK=4, keep SPK=2.
    # SPK=4 is used only when k is divisible by SPLIT_K*BLOCK_K (ensure all EVEN within a segment, no k remainder).
    if (m, n, k) in _SPLITK4_SHAPES and k % (_SPLITK_BLOCK_K * 4) == 0:
        spk = 4
    else:
        spk = 2
    scratch = torch.zeros(spk, m, n, dtype=torch.float32, device=C.device)
    ev_m = m % _SPLITK_BLOCK_M == 0
    ev_n = n % _SPLITK_BLOCK_N == 0
    # EVEN_K determined by the per-segment remainder: k_per = k // SPLIT_K, all EVEN within a segment is equivalent to k_per % BK == 0
    # (the original k % BK == 0 did not account for split-k division, e.g. k=96/SPK=2 -> k_per=48 %32 != 0)
    ev_k = (k // spk) % _SPLITK_BLOCK_K == 0
    grid_mn = triton.cdiv(m, _SPLITK_BLOCK_M) * triton.cdiv(n, _SPLITK_BLOCK_N)
    grid_main = (grid_mn * spk,)
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
            SPLIT_K=spk,
            EVEN_M=ev_m,
            EVEN_N=ev_n,
            EVEN_K=ev_k,
            BLOCK_M=_SPLITK_BLOCK_M,
            BLOCK_N=_SPLITK_BLOCK_N,
            BLOCK_K=_SPLITK_BLOCK_K,
            GROUP_M=_SPLITK_GROUP_M,
            num_warps=_SPLITK_WARPS,
            num_stages=_SPLITK_STAGES,
            use_in_thread_transpose=False,  # split-k severely regresses with ITT=1 (0.792->0.493)
        )
        _sgemm_nn_splitk_reduce_kernel[grid_reduce](
            scratch,
            C,
            m,
            n,
            alpha,
            beta,
            SPLIT_K=spk,
            BETA_IS_ZERO=beta_is_zero,
            EVEN_M=ev_m,
            EVEN_N=ev_n,
            BLOCK_M=_SPLITK_BLOCK_M,
            BLOCK_N=_SPLITK_BLOCK_N,
            GROUP_M=_SPLITK_GROUP_M,
            num_warps=_SPLITK_WARPS,
            use_in_thread_transpose=False,
        )

    return run


def _make_sgemm_nn_opt_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid, **_kw
):
    use_itt = _sgemm_nn_use_itt(m, n, k)
    use_buf = _sgemm_nn_use_buffer_ops(m, n, k)

    def run():
        kwargs = dict(use_in_thread_transpose=use_itt)
        if not use_buf:
            # no-buffer variant: explicitly pass use_buffer_ops=False into the compile options (distinguished by cache key);
            # when use_buf=True do not pass it, keeping the global default artifact, exactly as before injection
            kwargs["use_buffer_ops"] = False
        sgemm_nn_hygon_kernel[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
            **kwargs,
        )

    return run


_SGEMM_NN_DISPATCH = StaticDispatch(
    [
        (_sgemm_nn_is_large_k, _make_sgemm_nn_splitk_runner),
        (_sgemm_nn_is_pad_shape, _make_sgemm_nn_pad_runner),
        (_sgemm_nn_is_default, _make_sgemm_nn_opt_runner),
    ]
)


# ---- nn opt kernel config override (v2.25, 2026-09-01) ----
# Production uses the libtuner DB best_config, but for the following shapes the DB config is not optimal:
# full config scan (26 config x ITT x buffer) + official protocol (warmup=1000/rep=100) confirm
# manually launching the following configs is faster:
#   (256,256,256):     (32,32,BK64,w4,itt0,buf)   12.64->10.40us r 0.911->1.108
#   (1024,1024,1024):  (64,128,BK32,w8,itt1,nobuf) 75.20->70.08us r 0.851->0.913
#   (128,4096,1024):   (64,64,BK16,gm1,w4,itt1,nobuf) 51.04->43.84us r 0.809->0.942
_NN_OPT_OVERRIDES = {
    (256, 256, 256): dict(
        BLOCK_M=32, BLOCK_N=32, BLOCK_K=64, GROUP_M=8, num_warps=4,
        num_stages=2, use_in_thread_transpose=False,
    ),
    (1024, 1024, 1024): dict(
        BLOCK_M=64, BLOCK_N=128, BLOCK_K=32, GROUP_M=8, num_warps=8,
        num_stages=2, use_in_thread_transpose=True, use_buffer_ops=False,
    ),
    (128, 4096, 1024): dict(
        BLOCK_M=64, BLOCK_N=64, BLOCK_K=16, GROUP_M=1, num_warps=4,
        num_stages=2, use_in_thread_transpose=True, use_buffer_ops=False,
    ),
}


def _make_sgemm_nn_opt_override_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, ov, **_kw
):
    BM = ov["BLOCK_M"]
    BN = ov["BLOCK_N"]
    BK = ov["BLOCK_K"]
    GM = ov["GROUP_M"]
    ev = (m % BM == 0, n % BN == 0, k % BK == 0)
    grid = (((m + BM - 1) // BM) * ((n + BN - 1) // BN),)

    def run():
        kwargs = dict(
            num_warps=ov["num_warps"],
            num_stages=ov.get("num_stages", 2),
            use_in_thread_transpose=ov.get("use_in_thread_transpose", True),
        )
        if not ov.get("use_buffer_ops", True):
            # no-buffer variant: explicitly pass use_buffer_ops=False into the compile options
            kwargs["use_buffer_ops"] = False
        _sgemm_nn_opt_kernel[grid](
            A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
            ev[0], ev[1], ev[2], BM, BN, BK, GM, **kwargs,
        )

    return run


# ---- tn composite path: A^T buffer cache + runner (v2.3, 2026-08-30) ----
# Same convention as _get_pad_buffers: cached by (m,k,device), avoiding allocating a large buffer on every call
# (Ac for 8191³ reaches 268MB). The transpose kernel fully overwrites the buffer (no padding), so torch.empty suffices.
_tn_ac_cache = {}


def _get_tn_ac(m, k, device):
    key = (m, k, str(device))
    buf = _tn_ac_cache.get(key)
    if buf is None:
        buf = torch.empty(m, k, dtype=torch.float32, device=device)
        _tn_ac_cache[key] = buf
    return buf


def _make_sgemm_tn_composite_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid, **_kw
):
    Ac = _get_tn_ac(m, k, A.device)
    # Plain int arithmetic instead of triton.cdiv: the constexpr wrapper of cdiv costs ~7.8us each time, and the runner
    # rebuilds it on every call (sgemm entry), which is ~15us of pure Python overhead for small shapes
    grid_tp = (
        (m + _TN_TP_BM - 1) // _TN_TP_BM
    ) * ((k + _TN_TP_BK - 1) // _TN_TP_BK)
    grid_tp = (grid_tp,)
    nn_runner = _SGEMM_NN_DISPATCH.lookup_and_build(
        m,
        n,
        k,
        True,
        context=dict(
            A=Ac,
            lda=k,
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

    def run():
        _sgemm_tn_tp_kernel[grid_tp](
            A, Ac, m, k, lda, _TN_TP_BM, _TN_TP_BK, _TN_TP_GM
        )
        nn_runner()

    return run


# ---- tn composite path variant Y: B^T buffer cache + tt kernel runner (v2.3, 2026-08-30) ----
# Same convention as _get_tn_ac: cached by (n,k,device), the transpose kernel fully overwrites (no padding).
_tn_bc_cache = {}


def _get_tn_bc(n, k, device):
    key = (n, k, str(device))
    buf = _tn_bc_cache.get(key)
    if buf is None:
        buf = torch.empty(n, k, dtype=torch.float32, device=device)
        _tn_bc_cache[key] = buf
    return buf


def _make_sgemm_tn_composite_y_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid, **_kw
):
    # Transpose B(k,n) ldb=n -> Bc(n,k) contiguous, then run the tt kernel (A keeps (k,m) lda=m,
    # strided as the dot RHS, MFMA does not require the RHS to be k-contiguous).
    Bc = _get_tn_bc(n, k, B.device)
    # Same as above: cdiv constexpr wrapper ~7.8us each time, the runner rebuilds on every call, switch to plain int arithmetic
    grid_tp = (
        (n + _TN_TP_BM - 1) // _TN_TP_BM
    ) * ((k + _TN_TP_BK - 1) // _TN_TP_BK)
    grid_tp = (grid_tp,)
    itt = _sgemm_xt_use_itt(m, n, k)
    nobuf = not _sgemm_xt_use_buffer_ops(m, n, k)

    def run():
        _sgemm_tn_tp_kernel[grid_tp](
            B, Bc, n, k, ldb, _TN_TP_BM, _TN_TP_BK, _TN_TP_GM
        )
        sgemm_tt_hygon_kernel[grid](
            A, Bc, C, alpha, beta, m, n, k, lda, k, ldc, beta_is_zero,
            use_in_thread_transpose=itt,
            **(_XT_NOBUF_KWARGS if nobuf else {}),
        )

    return run


# ---- nt composite path (At+TT, v2.8, 2026-08-31) ----
# The direct kernel for C = A B^T reads A column blocks as (BK,BM) (stride-lda fragmentation), which is the main
# bottleneck for large shapes (16384^3 nt direct 184.2ms vs hipBLAS ref 149.9ms). Solution: one transpose
# A(m,k)->At(k,m) (~2ms), then run the tt kernel (A side becomes 1KB contiguous, B side 128B), same
# structure as direct but lower total time. Measured (production tt entry + in-pool 256x256 G8, warmup=25/rep=25):
# 16384^3 0.884->0.914, 8192^3 0.885->0.907, 8192x28672x8192 0.888->0.925,
# 4096x8192x28672 0.830->0.851; for medium/small shapes the transpose + two-kernel overhead dominates, direct is better.
_nt_at_cache = {}


# kpack=2 per-shape exception table (v2.12). Official-protocol do_bench(warmup=1000/rep=100/median)
# measured: on these direct shapes kpack=2 is >=1% slower than kpack=1 (d column), falling back to kpack=1
# eliminates the regression; on other direct shapes kpack=2 is net positive (2048x4096x11008 -4.3% etc.).
#   (64,512,512)  +1.7% | (512,64,512) +2.5% | (512,512,512) +3.1%
#   (511,511,511) +1.1% | (4096,128,1024) +1.0% | (2048,2048,16384) +1.4%
_NT_KPACK1_SHAPES = {
    (64, 512, 512),
    (512, 64, 512),
    (512, 512, 512),
    (511, 511, 511),
    (4096, 128, 1024),
    (2048, 2048, 16384),
}


def _sgemm_nt_direct_kpack(m, n, k):
    return 2 if (m, n, k) not in _NT_KPACK1_SHAPES else 1


def _get_nt_at(k, m, device):
    key = (k, m, str(device))
    buf = _nt_at_cache.get(key)
    if buf is None:
        buf = torch.empty(k, m, dtype=torch.float32, device=device)
        _nt_at_cache[key] = buf
    return buf


def _sgemm_nt_use_composite(m, n, k, **_kw):
    # Aligned large shapes only: both the transpose kernel and the tt kernel require 32 alignment; mutually exclusive with the pad path
    # (unaligned shapes like 8191^3 continue to use pre-pad). Threshold m*n*k>=5e11 and min(m,n)>=4096
    # covers 16384^3/8192^3/8192x28672x8192/4096x8192x28672, excludes 4096^3 (direct 0.917
    # is better) and 2048x2048x16384 (min=2048, the transpose overhead eats the kernel gain).
    if m % 32 != 0 or n % 32 != 0 or k % 32 != 0:
        return False
    if min(m, n) < 4096:
        return False
    return m * n * k >= 5 * 10**11


def _make_sgemm_nt_composite_runner(
    A, lda, B, ldb, C, ldc, m, n, k, alpha, beta, beta_is_zero, grid, **_kw
):
    At = _get_nt_at(k, m, A.device)
    # Transpose parameter semantics: _sgemm_tn_tp_kernel(src, dst, m=dst rows, k=dst cols, lda=src row stride);
    # A(m,k) lda=k -> At(k,m): (A, At, k, m, lda).
    grid_tp = (
        (k + _TN_TP_BM - 1) // _TN_TP_BM
    ) * ((m + _TN_TP_BK - 1) // _TN_TP_BK)
    grid_tp = (grid_tp,)
    itt = _sgemm_xt_use_itt(m, n, k)
    nobuf = not _sgemm_xt_use_buffer_ops(m, n, k)

    def run():
        _sgemm_tn_tp_kernel[grid_tp](
            A, At, k, m, lda, _TN_TP_BM, _TN_TP_BK, _TN_TP_GM
        )
        # kpack=2 (HIPOptions compile option, packing 2 k elements per instruction) measured effective for all composite
        # shapes (v2.11, scan4: 16384^3 0.915->0.925, 8192x28672x8192 0.925
        # ->0.936 etc.); this option is passed through by libentry and included in entry_key, not cross-used with the TT path's
        # kpack=1 artifacts.
        sgemm_tt_hygon_kernel[grid](
            At, B, C, alpha, beta, m, n, k, m, ldb, ldc, beta_is_zero,
            use_in_thread_transpose=itt,
            kpack=2,
            **(_XT_NOBUF_KWARGS if nobuf else {}),
        )

    return run


# ---- sgemm entry-level runner cache (v2.5, 2026-08-30) ----
# The nn dispatch lookup_and_build and composite runner are rebuilt on every call (~7-17us pure Python
# overhead, cdiv constexpr wrapper ~7.8us each time). For microsecond-level shapes (64x512x512 etc.) the overhead dominates,
# so runner closures are cached by (shape, stride, scalars, tensor data_ptr) and reused on hit.
# data_ptr participates in the key: different tensors necessarily rebuild; the closure holds references to prevent false hits from address reuse. An upper bound prevents unbounded growth.
# v2.10 (2026-08-31): closures hold A/B/C references, so indiscriminate caching would pin large-shape tensors long-term (the full
# benchmark 37-shape x 4 variants accumulated to 12GB for a single 16384³ runner, causing OOM). Startup overhead only matters for
# microsecond-level shapes, so gate by total pinned elements: large shapes rebuild on every call (~10-20us, negligible relative to
# millisecond kernels), and tensors can be released as soon as the call ends.
_sgemm_runner_cache = {}
_sgemm_runner_cache_max = 8192
# Per-runner pin limit 16M elements (~64MB fp32): covers regular small/medium shapes up to 2048³.
_SGEMM_RUNNER_CACHE_MAX_PIN = 16 * 2**20


def _cache_sgemm_runner(key, runner):
    # key = (transa, transb, m, n, k, ...); gated by total pinned elements
    m, n, k = key[2], key[3], key[4]
    if m * k + n * k + m * n > _SGEMM_RUNNER_CACHE_MAX_PIN:
        return runner  # large shapes are not cached, avoiding the closure pinning hundreds of MB~GB of tensors
    if len(_sgemm_runner_cache) < _sgemm_runner_cache_max:
        _sgemm_runner_cache[key] = runner
    return runner


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
    # Plain int arithmetic (not triton.cdiv): avoids constexpr wrapper overhead, grid is evaluated on every launch
    grid = lambda meta: (
        ((m + meta["BLOCK_M"] - 1) // meta["BLOCK_M"])
        * ((n + meta["BLOCK_N"] - 1) // meta["BLOCK_N"]),
    )

    # The runner is cached by (shape, stride, scalars, tensor pointers): the nn dispatch/composite runner is
    # rebuilt on every call (lookup_and_build ~7us + grid_tp construction), which is significant overhead for microsecond-level shapes.
    # data_ptr in the key guarantees different tensors necessarily rebuild; old tensors held by the closure are not released and cannot reuse addresses.
    key = (
        transa,
        transb,
        m,
        n,
        k,
        lda,
        ldb,
        ldc,
        alpha,
        beta,
        beta_is_zero,
        A.data_ptr(),
        B.data_ptr(),
        C.data_ptr(),
        str(A.device),
    )

    with torch_device_fn.device(A.device):
        if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
            if (m, n, k) in _NN_OPT_OVERRIDES:
                # nn opt kernel config override (v2.25): DB best_config is not optimal,
                # the optimal config confirmed by manual launch scan (official protocol 256³ 1.108/1024³ 0.913/
                # 128x4096x1024 0.942).
                runner = _sgemm_runner_cache.get(key)
                if runner is None:
                    runner = _cache_sgemm_runner(
                        key,
                        _make_sgemm_nn_opt_override_runner(
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
                            ov=_NN_OPT_OVERRIDES[(m, n, k)],
                        ),
                    )
                runner()
            else:
                runner = _sgemm_runner_cache.get(key)
                if runner is None:
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
                _cache_sgemm_runner(key, runner)
                runner()
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            if (m, n, k) in _TN_DIRECT_OVERRIDES:
                # tn direct config override (v2.22): production config not optimal for small shapes,
                # shrink tiles to raise parallelism (official-protocol A/B all >=1.0).
                runner = _sgemm_runner_cache.get(key)
                if runner is None:
                    runner = _cache_sgemm_runner(
                        key,
                        _make_sgemm_tn_direct_runner(
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
                            ov=_TN_DIRECT_OVERRIDES[(m, n, k)],
                        ),
                    )
                runner()
            elif _sgemm_xt_is_pad_shape(m, n, k):
                # Unaligned large shapes: A(k,m)->A_pad(kp,mp) + B(k,n) transpose->B_pad(np,kp),
                # reuse tt even main (v2.8, 4095³ 0.880->~1.0/8191³ 0.918->~0.96/
                # 4097x8191x4095 0.934->~1.0)
                runner = _sgemm_runner_cache.get(key)
                if runner is None:
                    runner = _cache_sgemm_runner(
                        key,
                        _make_sgemm_tn_pad_runner(
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
            elif _sgemm_tn_use_composite_y(m, n, k):
                if (m, n, k) in _TN_COMPY_OVERRIDES:
                    # compY tt kernel config override (v2.24): manually launch for large shapes
                    runner = _sgemm_runner_cache.get(key)
                    if runner is None:
                        runner = _cache_sgemm_runner(
                            key,
                            _make_sgemm_tn_compy_override_runner(
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
                                ov=_TN_COMPY_OVERRIDES[(m, n, k)],
                            ),
                        )
                    runner()
                else:
                    # Variant Y: when n≪m, transposing only B^T (cost n*k) is cheaper than plan A (m*k), use the tt kernel
                    runner = _sgemm_runner_cache.get(key)
                    if runner is None:
                        runner = _cache_sgemm_runner(
                            key,
                            _make_sgemm_tn_composite_y_runner(
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
            elif _sgemm_tn_use_composite(m, n, k):
                if (m, n, k) in _TN_COMP_OVERRIDES:
                    # comp path nn kernel config override (v2.23): 4096³ uses the scan-optimal
                    # (64,128,w8,itt1,gm1,BK32,s2,nobuf) to replace tuner config0
                    runner = _sgemm_runner_cache.get(key)
                    if runner is None:
                        runner = _cache_sgemm_runner(
                            key,
                            _make_sgemm_tn_comp_override_runner(
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
                                ov=_TN_COMP_OVERRIDES[(m, n, k)],
                            ),
                        )
                    runner()
                else:
                    # Composite path: transpose A^T -> contiguous (m,k) buffer -> nn dispatch
                    runner = _sgemm_runner_cache.get(key)
                    if runner is None:
                        runner = _cache_sgemm_runner(
                            key,
                            _make_sgemm_tn_composite_runner(
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
            else:
                sgemm_tn_hygon_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
                    use_in_thread_transpose=_sgemm_xt_use_itt(m, n, k),
                    **(_XT_NOBUF_KWARGS if not _sgemm_xt_use_buffer_ops(m, n, k) else {}),
                )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            _nt_spk = _sgemm_nt_splitk(m, n, k)
            if _nt_spk:
                # Thin shape split-k (v2.13): when blocks are only 8-32, split k into segments in parallel,
                # 7 shapes such as (64,512,512) are 6-31% faster under the official protocol. The whitelist covers them exactly.
                _nt_spk_w = _NT_SPLITK_SHAPES[(m, n, k)][1]
                runner = _sgemm_runner_cache.get(key)
                if runner is None:
                    runner = _cache_sgemm_runner(
                        key,
                        _make_sgemm_nt_splitk_runner(
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
                            SPLIT_K=_nt_spk,
                            num_warps=_nt_spk_w,
                        ),
                    )
                runner()
            elif (m, n, k) in _NT_DIRECT_OVERRIDES:
                # Direct config override (v2.13): production config not optimal for 511³ etc., always manually
                # launch the raw kernel (w16/itt0/kp1, official protocol 0.745->0.93).
                runner = _sgemm_runner_cache.get(key)
                if runner is None:
                    runner = _cache_sgemm_runner(
                        key,
                        _make_sgemm_nt_direct_runner(
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
                            ov=_NT_DIRECT_OVERRIDES[(m, n, k)],
                        ),
                    )
                runner()
            elif _sgemm_nt_use_composite(m, n, k):
                # Aligned large shapes: A^T transpose + tt kernel (v2.8), avoiding the direct a_t stride-lda
                # fragmented reads (16384^3 nt direct 0.884 -> composite 0.914 etc., see _make_sgemm_nt_composite_runner)
                runner = _sgemm_runner_cache.get(key)
                if runner is None:
                    runner = _cache_sgemm_runner(
                        key,
                        _make_sgemm_nt_composite_runner(
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
            elif _sgemm_xt_is_pad_shape(m, n, k):
                # Unaligned large shapes: pre-pad A/B into aligned buffers, the main kernel loads all EVEN,
                # avoiding mask/k remainder overhead (v2.6, 4095³ 0.755->0.840 etc.)
                runner = _sgemm_runner_cache.get(key)
                if runner is None:
                    runner = _cache_sgemm_runner(
                        key,
                        _make_sgemm_nt_pad_runner(
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
            else:
                # kpack=2 measured effective for most direct shapes (v2.11: 2048x4096x11008 0.918
                # ->0.949-0.970, 16384x512x4096 0.908->0.942, 512x16384x4096 0.938
                # ->0.949 etc.). On a few shapes kpack=2 is instead slower (official protocol
                # do_bench(1000/100/median) measured >=1% slower), use the _NT_KPACK1_SHAPES
                # exception table to fall back to kpack=1, eliminating small-shape regressions (v2.12).
                sgemm_nt_hygon_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
                    use_in_thread_transpose=_sgemm_xt_use_itt(m, n, k),
                    kpack=_sgemm_nt_direct_kpack(m, n, k),
                    **(_XT_NOBUF_KWARGS if not _sgemm_xt_use_buffer_ops(m, n, k) else {}),
                )
        else:
            if (m, n, k) in _TT_DIRECT_OVERRIDES:
                # tt direct config override (v2.21): for small shapes the production config's large tiles
                # leave insufficient grid parallelism, shrink tiles to raise parallelism (official-protocol A/B all >=1.0).
                runner = _sgemm_runner_cache.get(key)
                if runner is None:
                    runner = _cache_sgemm_runner(
                        key,
                        _make_sgemm_tt_direct_runner(
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
                            ov=_TT_DIRECT_OVERRIDES[(m, n, k)],
                        ),
                    )
                runner()
            elif _sgemm_xt_is_pad_shape(m, n, k):
                # Unaligned large shapes: pre-pad A(k,m)->A_pad(kp,mp)/B(n,k)->B_pad(np,kp),
                # all-EVEN main (v2.7, 8191³ 0.883->1.037, 4097x8191x4095 0.826->0.940)
                runner = _sgemm_runner_cache.get(key)
                if runner is None:
                    runner = _cache_sgemm_runner(
                        key,
                        _make_sgemm_tt_pad_runner(
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
            else:
                sgemm_tt_hygon_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
                    use_in_thread_transpose=_sgemm_xt_use_itt(m, n, k),
                    **(_XT_NOBUF_KWARGS if not _sgemm_xt_use_buffer_ops(m, n, k) else {}),
                )


__all__ = ["sgemm"]
