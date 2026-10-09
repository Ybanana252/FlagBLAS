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

"""
Hygon DTK cgemm 专用实现（自研 complex 内核）。

complex64 GEMM: C = alpha * op(A) @ op(B) + beta * C, op ∈ {N, T, C(共轭转置)}。
本模块自带一个 complex Triton 内核（在 kernel 内处理 N/T/C 与共轭），并额外提供一条
hygon 定制的「转置复合」路径：当 transa/transb 含转置且 shape 足够大时，先把物理
(k,m)/(n,k) 的非共轭/共轭转置物化成 (m,k)/(k,n) 连续缓冲，再以 N/N 调用主内核，
避免跨步复数读带来的带宽损失（与 _hygon/ops/bfgemm.py / hgemm.py 的转置复合同思路）。

数据布局: 复数以 real/imag 交错存放, 传入内核前用 torch.view_as_real(...).reshape(-1)
展平成 fp32 视图, 内核按 2*elem / 2*elem+1 取 re/im。

注: 真机精度/性能验证前, hygon 专用 config 表与转置复合阈值为保守取值, 待实测收敛。
"""

import torch
import triton
import triton.language as tl

from flag_blas.ops.level3.cgemm import (
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    ScalarType,
    _complex_scalar_parts,
    _validate_cgemm_args,
)
from flag_blas.runtime import torch_device_fn

# 转置复合拷贝 tile
_TR_BR = 64
_TR_BC = 64
# 转置复合启用阈值: 2*m*n*k 下界 (小 shape 上拷贝开销占比过高, 不复合)。
_TR_MIN_FLOPS = 4 << 24


@triton.jit
def _cgemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha_r: tl.float32,
    alpha_i: tl.float32,
    beta_r: tl.float32,
    beta_i: tl.float32,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    ALPHA_IS_ONE: tl.constexpr,
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

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m
    mask_n = offs_n < n

    acc_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, k, BLOCK_K):
        cur_k = k_start + offs_k
        mask_k = cur_k < k
        # A: N -> 物理 (m,k); T/C -> 物理 (k,m)
        if TRANS_A == 0:
            a_elem = offs_m[:, None] * lda + cur_k[None, :]
        else:
            a_elem = cur_k[None, :] * lda + offs_m[:, None]
        # B: N -> 物理 (k,n); T/C -> 物理 (n,k)
        if TRANS_B == 0:
            b_elem = cur_k[:, None] * ldb + offs_n[None, :]
        else:
            b_elem = offs_n[None, :] * ldb + cur_k[:, None]

        a_mask = mask_m[:, None] & mask_k[None, :]
        b_mask = mask_k[:, None] & mask_n[None, :]
        ar = tl.load(a_ptr + 2 * a_elem, mask=a_mask, other=0.0)
        ai = tl.load(a_ptr + 2 * a_elem + 1, mask=a_mask, other=0.0)
        br = tl.load(b_ptr + 2 * b_elem, mask=b_mask, other=0.0)
        bi = tl.load(b_ptr + 2 * b_elem + 1, mask=b_mask, other=0.0)
        # 共轭转置: 虚部取反
        if TRANS_A == 2:
            ai = -ai
        if TRANS_B == 2:
            bi = -bi

        # (ar + ai i)(br + bi i) = (ar*br - ai*bi) + (ar*bi + ai*br) i
        # gfx936 后端只允许 input_precision="ieee" (无 tf32), 即真 fp32 点积,
        # 精度优于 tf32x3, 与 hipblasCgemm 参考同口径。
        acc_r += tl.dot(ar, br, out_dtype=tl.float32, input_precision="ieee")
        acc_r -= tl.dot(ai, bi, out_dtype=tl.float32, input_precision="ieee")
        acc_i += tl.dot(ar, bi, out_dtype=tl.float32, input_precision="ieee")
        acc_i += tl.dot(ai, br, out_dtype=tl.float32, input_precision="ieee")

    if ALPHA_IS_ONE:
        out_r = acc_r
        out_i = acc_i
    else:
        out_r = alpha_r * acc_r - alpha_i * acc_i
        out_i = alpha_r * acc_i + alpha_i * acc_r

    c_elem = offs_m[:, None] * ldc + offs_n[None, :]
    c_mask = mask_m[:, None] & mask_n[None, :]
    if not BETA_IS_ZERO:
        cr = tl.load(c_ptr + 2 * c_elem, mask=c_mask, other=0.0)
        ci = tl.load(c_ptr + 2 * c_elem + 1, mask=c_mask, other=0.0)
        out_r += beta_r * cr - beta_i * ci
        out_i += beta_r * ci + beta_i * cr

    tl.store(c_ptr + 2 * c_elem, out_r, mask=c_mask)
    tl.store(c_ptr + 2 * c_elem + 1, out_i, mask=c_mask)


@triton.jit
def _cgemm_trans_kernel(
    src,
    dst,
    R,
    Cc,
    sr,
    CONJ: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    # src: 物理 (R, Cc) 复数行主序 (行 stride=sr), 实虚交错 [re,im]
    # dst: (Cc, R) 复数行主序, 行 stride=R; dst = trans(src) (CONJ 时虚部取反)
    pid = tl.program_id(0)
    ngc = tl.cdiv(Cc, BC)
    pr = pid // ngc
    pc = pid % ngc
    offs_r = pr * BR + tl.arange(0, BR)  # 0..R-1
    offs_c = pc * BC + tl.arange(0, BC)  # 0..Cc-1
    msk = (offs_r[:, None] < R) & (offs_c[None, :] < Cc)
    idx = offs_r[:, None] * sr + offs_c[None, :]
    re = tl.load(src + 2 * idx, mask=msk, other=0.0)
    im = tl.load(src + 2 * idx + 1, mask=msk, other=0.0)
    if CONJ:
        im = -im
    didx = offs_c[:, None] * R + offs_r[None, :]
    dmsk = tl.trans(msk)
    tl.store(dst + 2 * didx, tl.trans(re), mask=dmsk)
    tl.store(dst + 2 * didx + 1, tl.trans(im), mask=dmsk)


def _select_cgemm_config_hygon(m, n, k):
    # hygon gfx936 分档 (复数内核有 2 个 fp32 累加器 + 4 个操作数 tile, 寄存器压力
    # 约实数 gemm 的 2x, 故 BK/tile 取小)。为保守起点, 与通用表同族, 待实测收敛。
    max_dim = max(m, n, k)
    if max_dim <= 32:
        return 16, 16, 16, 4, 1
    if max_dim <= 128:
        return 16, 16, 32, 4, 1
    if max_dim <= 512:
        return 32, 32, 32, 4, 1
    if max_dim <= 1024:
        return 32, 64, 32, 4, 4
    return 32, 64, 32, 8, 4


def _use_trans_composite(m, n, k, transa, transb):
    # 仅在含转置且输出规模足够大 (2*m*n*k >= _TR_MIN_FLOPS) 时物化连续操作数,
    # 小 shape 上拷贝开销占比过高, 直接走 kernel 内转置。
    if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
        return False
    return 2.0 * m * n * k >= _TR_MIN_FLOPS and min(m, n) >= 64


def _materialize_transpose(x_real, rows_phys, cols_phys, ld, conj, device):
    # x_real: 物理 (rows_phys, cols_phys) 复数的 f32 交错视图
    # 返回 (cols_phys, rows_phys) 复数的连续 (行主序) 张量, 即 trans(x) (conj 时共轭)
    out_complex = torch.empty(
        (cols_phys, rows_phys), dtype=torch.complex64, device=device
    )
    out_real = torch.view_as_real(out_complex).reshape(-1)
    grid = (
        triton.cdiv(rows_phys, _TR_BR) * triton.cdiv(cols_phys, _TR_BC),
    )
    _cgemm_trans_kernel[grid](
        x_real,
        out_real,
        rows_phys,
        cols_phys,
        ld,
        CONJ=conj,
        BR=_TR_BR,
        BC=_TR_BC,
        num_warps=8,
    )
    return out_complex


def cgemm(
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
    _validate_cgemm_args(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)

    alpha_r, alpha_i = _complex_scalar_parts(alpha)
    beta_r, beta_i = _complex_scalar_parts(beta)

    if m == 0 or n == 0 or k == 0 or (alpha_r == 0.0 and alpha_i == 0.0):
        if beta_r == 0.0 and beta_i == 0.0:
            C.zero_()
        elif not (beta_r == 1.0 and beta_i == 0.0):
            C.mul_(complex(beta_r, beta_i))
        return

    beta_is_zero = beta_r == 0.0 and beta_i == 0.0
    alpha_is_one = alpha_r == 1.0 and alpha_i == 0.0

    block_m, block_n, block_k, num_warps, group_m = _select_cgemm_config_hygon(m, n, k)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

    A_real = torch.view_as_real(A).reshape(-1)
    B_real = torch.view_as_real(B).reshape(-1)
    C_real = torch.view_as_real(C).reshape(-1)

    eff_trans_a, eff_lda = transa, lda
    eff_trans_b, eff_ldb = transb, ldb

    with torch_device_fn.device(A.device):
        if _use_trans_composite(m, n, k, transa, transb):
            # A 物理 (k,m) -> op(A) 连续 (m,k); 共轭转置时虚部取反
            if transa != CUBLAS_OP_N:
                op_a = _materialize_transpose(
                    A_real, k, m, lda, transa == CUBLAS_OP_C, A.device
                )
                A_real = torch.view_as_real(op_a).reshape(-1)
                eff_trans_a, eff_lda = CUBLAS_OP_N, k
            # B 物理 (n,k) -> op(B) 连续 (k,n)
            if transb != CUBLAS_OP_N:
                op_b = _materialize_transpose(
                    B_real, n, k, ldb, transb == CUBLAS_OP_C, B.device
                )
                B_real = torch.view_as_real(op_b).reshape(-1)
                eff_trans_b, eff_ldb = CUBLAS_OP_N, n

        _cgemm_kernel[grid](
            A_real,
            B_real,
            C_real,
            alpha_r,
            alpha_i,
            beta_r,
            beta_i,
            m,
            n,
            k,
            eff_lda,
            eff_ldb,
            ldc,
            eff_trans_a,
            eff_trans_b,
            beta_is_zero,
            alpha_is_one,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=group_m,
            num_warps=num_warps,
        )
