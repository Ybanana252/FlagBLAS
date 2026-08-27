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

# per-kernel ITT 编译选项的运行时注入（不改 site-packages Triton 源码）。
# 必须在任何 sgemm kernel launch 前执行一次；注入失败仅失去 per-kernel ITT
# 能力（回落全局默认），不影响其余功能。
try:
    from flag_blas.runtime.backend._hygon._triton_itt import install_triton_itt_patch

    install_triton_itt_patch()
except Exception:  # pragma: no cover - 环境无 Triton HCU 后端时静默跳过
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


# ---- A 类非对齐 shape 专用: pre-pad + 全 EVEN 内核 ----
# 思路: 将 A(m x k) pad 到 A'(mp x kp), B(k x n) pad 到 B'(kp x np),
# 其中 mp/kp/np 是 BLOCK 的整数倍; 主内核全 EVEN (load 无 mask), 仅 store 对真实 m/n 打 mask.
# 成本: 一次 A/B 合并 pad-copy (内存带宽受限), 换取主循环 mask 开销 ~11% + k remainder ~13%.
# 实测 (py310 venv + flagtree 0.6.1, 对比报告当前 opt 路径):
#   8191³ 0.732 -> 0.860、4095³ 0.618 -> 0.739、4097x8191x4095 0.705 -> 0.805、
#   1023³ 0.608 -> 0.623 (边际); 511³ 反而更差 (0.845 -> 0.638), 由分发条件排除.


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
        # A tile: 真实 m x k -> pad (mp x kp), 越界补零
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
        # B tile: 真实 k x n -> pad (kp x np), 越界补零
        q = pid - nblk_a
        bi = q // tl.cdiv(np, CK)
        bj = q % tl.cdiv(np, CK)
        offs_i = bi * CK + tl.arange(0, CK)
        offs_j = bj * CM + tl.arange(0, CM)
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
# 非对齐且非小 shape: pre-pad + 全 EVEN 内核 (消除 mask/k remainder 开销)
# 其余: 现网 libtuner opt kernel (保持原行为)

_SPLITK_BLOCK_M = 64
_SPLITK_BLOCK_N = 64
_SPLITK_BLOCK_K = 32
_SPLITK_GROUP_M = 8
_SPLITK_WARPS = 4
_SPLITK_STAGES = 2

# pad 路径参数 (实测最优 config)
_PAD_BLOCK_M = 128
_PAD_BLOCK_N = 256
_PAD_BLOCK_K = 32
# v1.8 (2026-08-26): copy tile 由 64x64 改为 32x32 (带宽型 copy, 小 tile 网格更大
# 负载均衡更好): 8191³ copy 1214->908us (-25%)、1023³ copy 50->20us、4097x8191x4095
# copy 356us 持平; 完整路径 8191³ -1.6%、1023³ -16%。32 恒整除 mp/np/kp
# (mp/np 为 BM/BN 倍数, kp 为 128 倍数), 无对齐风险。
_PAD_COPY_CM = 32
_PAD_COPY_CK = 128
# 小输出 shape 的更高并行度候选: main (32,64) + copy tile 32
# (1023³ 完整路径 116->92us, -20%; copy 50->20us)。
# 但 main BM=32 使 mp 可能非 64 倍数, 而 copy tile 必须整除 mp/np/kp,
# 不满足时由 runner 回退 (128,128) main (mp/np 恒为 128 倍数, 安全)。
_PAD_CFG_SMALL_TILE = (32, 64, 32, 8, 4, 2)
# 大输出 shape 分级 (v1.8 实测, 2026-08-26):
# - m*n >= 32M (大输出, 如 8191³/4097x8191x4095): (256,256,32,G1,w16,s1) + ITT1 + no-buffer
#   8191³ 0.862->0.925 达标 (main 与 opt 8192³ 生产 config 同款, 20652us)、
#   4097x8191x4095 0.806->0.869
# - 4M <= m*n < 32M (中输出, 如 4095³): (256,128,32,G8,w8,s1) + ITT0 + no-buffer
#   4095³ 0.738->0.788 ((256,256,w16) 反而 0.734, 因 grid 仅 16x16 时大 tile 低效)
# - m*n < 4M (小输出, 如 1023³): (32,64) 避免 tile 浪费, 保持 ITT0+buf 默认 (0.625->0.724)
_PAD_CFG_LARGE = (256, 256, 32, 1, 16, 2)
# v1.10: MID 由 (256,128,32,G8,w8,s1)+ITT0 改为 (64,64,32,G8,w4,s2)+ITT1
# (与 opt 4096^3 生产 config 同款)。4095^3 pad main 算全 EVEN 4096^3:
# (64,64,ITT1) 2927us vs (256,128,ITT0) 3013us (-2.9%), ITT1 必需 (ITT0 3201)。
# 4095^3 完整路径 0.788 -> ~0.809。mp/np=cdiv(4095,64)=4096 仍满足 copy tile 对齐。
_PAD_CFG_MID = (64, 64, 32, 8, 4, 2)
_PAD_LARGE_OUTPUT = 32 * 1024 * 1024
_PAD_CFG_SMALL = (128, 128, 32, 8, 8, 2)
_PAD_SMALL_OUTPUT = 4 * 1024 * 1024
# 排除 511³ (1.3e8) 等小 shape: pad-copy 开销占比大, 实测反而更差
_PAD_MIN_ELEMS = 5 * 10**8

# pad 缓冲区缓存: 避免每次调用重复分配大 buffer (按 pad 后尺寸缓存)
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
# v1.10: SPLIT_K=4 对薄大 K shape 显著更优 (实测):
#   128x1024x4096 64.2->52.5us (-18.2%)、256x2048x8192 288.5->251.7us (-12.8%);
#   512x4096x16384 反而 SPK=4 退化 (1700->2825), 保持 SPK=2。
# 仅当 k 能被 SPLIT_K*BLOCK_K 整除时走 SPK=4 (保证段内全 EVEN, 无 k remainder)。
_SPLITK4_SHAPES = {(128, 1024, 4096), (256, 2048, 8192)}


def _sgemm_nn_is_large_k(m, n, k, **_kw):
    # 大 K shape 走两阶段 split-k; 特例 (2048,2048,16384) 实测 opt+ITT1 (0.852)
    # 优于 split-k+ITT0 (0.792) 与 opt+ITT0 (0.778), 由 _LARGE_K_OPT_SHAPES 排除
    if (m, n, k) in _LARGE_K_OPT_SHAPES:
        return False
    # split-k 按 k_per = k // SPLIT_K 均分, k 为奇数时末元素被丢弃 (静默错误),
    # 且 EVEN_K 需按段内余数判断; 奇数 k 一律回退 opt (其自带 k-remainder 分支)
    if k % 2 != 0:
        return False
    return k >= 4 * max(m, n)


# ITT (in-thread-transpose) per-shape 决策: 基于 ITT=0/1 全量回归实测
# - ITT=1 (True): 大 shape 普遍受益 (2048³ 0.828->0.904、256x8192x2048 0.839->0.908 等)
# - ITT=0 (False): 瘦 shape (min(m,n)<=128) 与小立方受 in-thread-transpose 布局损害
#   (64x512x512 1.054->0.774、256³ 1.103->0.918、511³ 0.854->0.790 等), config 无法恢复
# - 显式集合覆盖规则例外 (64³ 超小 shape ITT=1 反而 +0.136 至 0.983)
# v1.6 教训 (2026-08-26): 4 组合 (ITT x buffer) 扫描用手动 launch + DB TunedConfig
# 表 config, 但生产 libtuner 实际用 benchmark 表 config (二者不一致, 瘦 shape 尤甚)
# -> 手动 config 下 ITT0 对 16384x512x4096/512x16384x4096 更优 (-2.3%), 但 monkeypatch
#    生产路径确认 ITT1 才是生产最优 (1465us vs ITT0 1592us, 组合结论被 config 翻转).
#    唯一保留: 2048x16384x2048 从 _ITT0_SHAPES 移除 -> 生产路径 ITT1+nobuf 2867.6us
#    优于 ITT0+nobuf 2901us (+1.2%).
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


# use_buffer_ops (AMDGCN_USE_BUFFER_OPS) per-shape 决策: 基于 33 个 opt shape
# 全量快扫 + 正式协议 (warmup=1000/rep=100) 交错确认 (2026-08-26)
# - no-buffer (False): 大/中输出 shape (m*n>=2M) 普遍受益 -1~-4%
#   (4096³ -3.9%、512x16384x4096 -4.0%、2048x11008x4096 -4.1% 等);
#   非对齐 mask 路径 shape 受益显著 (511³ -10.0%)
# - buffer (True, 默认): 小输出 shape 保持 (1024³ +3.6%、128x4096x1024 +3.8%
#   [正式确认] 等 no-buffer 反而变差); (2048,2048,16384) 大 K opt 在 no-buffer 下
#   严重变差 (+43% 正式确认), 必须保持 buffer
def _sgemm_nn_use_buffer_ops(m, n, k, **_kw):
    if (m, n, k) == (2048, 2048, 16384):
        return True
    if m % 32 != 0 or n % 32 != 0 or k % 32 != 0:
        return False
    if m * n >= 2 * 1024 * 1024:
        return False
    return True


def _sgemm_nn_is_pad_shape(m, n, k, **_kw):
    # 非对齐(相对 pad tile 尺寸)且非小/瘦 shape 才走 pre-pad 路径
    if (
        m % _PAD_BLOCK_M == 0
        and n % _PAD_BLOCK_N == 0
        and k % _PAD_BLOCK_K == 0
    ):
        return False
    # 瘦 shape (输出任一维度 < 512) 走 opt: 实测 pad-copy 开销占比大反而更差,
    # 同时覆盖 511³、64³ 及 4096x64x2048 等 (n/m=64/128 对 BN=256 不对齐)
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
        # 小输出: main (32,64) 提升并行度 (1023³ 完整路径 116->92us, -20%);
        # main BM=32 使 mp 可能非 128 倍数, 不满足时回退 (128,128) main。
        # copy 统一 32x32 (v1.8: 带宽型 copy 最优, 1023³ copy 50->20us)。
        BM, BN, BK, GM, nw, ns = _PAD_CFG_SMALL_TILE
        mp = triton.cdiv(m, BM) * BM
        np = triton.cdiv(n, BN) * BN
        if mp % 128 != 0 or np % 128 != 0:
            BM, BN, BK, GM, nw, ns = _PAD_CFG_SMALL
            mp = triton.cdiv(m, BM) * BM
            np = triton.cdiv(n, BN) * BN
        use_buf = True   # 小输出保持 buffer ops 默认 (1023³ buf0 反而略差)
        use_itt = False
    elif m * n >= _PAD_LARGE_OUTPUT:
        # 大输出: (256,256,32,G1,w16,s2) + ITT1 + no-buffer
        # (s2 比 s1 快 13%: 8191³ 0.795->0.924, 4097x8191x4095 0.755->0.869)
        BM, BN, BK, GM, nw, ns = _PAD_CFG_LARGE
        mp = triton.cdiv(m, BM) * BM
        np = triton.cdiv(n, BN) * BN
        use_buf = False
        use_itt = True
    else:
        # 中输出: (64,64,32,G8,w4,s2) + ITT1 + no-buffer (v1.10, 4095³ 0.809)
        BM, BN, BK, GM, nw, ns = _PAD_CFG_MID
        mp = triton.cdiv(m, BM) * BM
        np = triton.cdiv(n, BN) * BN
        use_buf = False
        use_itt = True
    CCM = CCK = _PAD_COPY_CM
    # kp 需同时为主内核 BLOCK_K(32) 与 pad-copy tile (32) 的整数倍:
    # copy kernel 对 k 维按 CK 分块且 store 无 mask, 若 kp 非 CK 倍数
    # 末块会写到 kp 之外 (如 512x12800x5210, kp=5216 -> 写到 5247 行, 越界 VMFault)。
    # 128 是 32 的倍数, 按 _PAD_COPY_CK(128) 对齐可同时满足两者。
    kp = triton.cdiv(k, _PAD_COPY_CK) * _PAD_COPY_CK
    A_pad, B_pad = _get_pad_buffers(mp, kp, np, A.device)
    grid_pad = (
        triton.cdiv(mp, CCM) * triton.cdiv(kp, CCK)
        + triton.cdiv(kp, CCK) * triton.cdiv(np, CCM),
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
            num_warps=4,  # tile32 下 nw4 优于 nw8: 8191³ copy 948->908, 4095³ 229->223, 4097 359->345
            use_in_thread_transpose=False,  # pad 路径在 ITT=1 下变差 (8191³ 0.862->0.823)
        )
        main_kwargs = dict(
            BETA_IS_ZERO=beta_is_zero,
            BLOCK_M=BM,
            BLOCK_N=BN,
            BLOCK_K=BK,
            GROUP_M=GM,
            num_warps=nw,
            num_stages=ns,
            use_in_thread_transpose=use_itt,  # 大/中输出 ITT1, 小输出 ITT0
        )
        if not use_buf:
            # no-buffer 变体 (大/中输出): 显式传 use_buffer_ops=False 进编译选项;
            # use_buf=True 时不传, 保持全局默认产物
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
    # per-shape SPLIT_K: 薄大 K shape (128x1024x4096/256x2048x8192) 实测 SPK=4
    # 显著更优 (-18.2%/-12.8%); 512x4096x16384 等 SPK=4 严重退化, 保持 SPK=2。
    # SPK=4 仅在 k 能被 SPLIT_K*BLOCK_K 整除时采用 (保证段内全 EVEN, 无 k remainder)。
    if (m, n, k) in _SPLITK4_SHAPES and k % (_SPLITK_BLOCK_K * 4) == 0:
        spk = 4
    else:
        spk = 2
    scratch = torch.zeros(spk, m, n, dtype=torch.float32, device=C.device)
    ev_m = m % _SPLITK_BLOCK_M == 0
    ev_n = n % _SPLITK_BLOCK_N == 0
    # EVEN_K 按段内余数判断: k_per = k // SPLIT_K, 段内全 EVEN 等价于 k_per % BK == 0
    # (原 k % BK == 0 未考虑 split-k 均分, 例如 k=96/SPK=2 -> k_per=48 %32 != 0)
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
            use_in_thread_transpose=False,  # split-k 在 ITT=1 下严重退化 (0.792->0.493)
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
            # no-buffer 变体: 显式传 use_buffer_ops=False 进编译选项 (cache key 区分);
            # use_buf=True 时不传, 保持全局默认产物, 与未注入前完全一致
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
