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
Hygon DTK bfgemm 定制路径。

TN (transa=T, transb=N) 下, A 的物理布局是 (k, m) 行主序, 直接 TN 主循环
沿 k 步进时 A tile 的末维 (k) 跨度为 lda, 无法向量化/连续读, 带宽效率差
(core-shape 基准 TN 变体 median 仅 ~0.63)。这里对 TN 采用 sgemm 同款思路:
先用专用转置拷贝内核把 A(k,m) 落成 A^T(m,k) 连续缓冲区, 再走 NN 主内核,
使 A 的加载与 NN 变体一致 (末维连续)。转置内核读写均按连续片段进行,
开销远小于直接 TN 的跨步读损失。

NN / NT / TT 与通用 level3 内核一致 (实测 median >=0.95, 不额外处理)。
"""

import torch
import triton
import triton.language as tl

from flag_blas.ops.level3.bfgemm import (
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    ScalarType,
    _bfgemm_nn_kernel,
    _bfgemm_nt_kernel,
    _bfgemm_tn_kernel,
    _bfgemm_tt_kernel,
)
from flag_blas.runtime import torch_device_fn

# TN 复合转置内核 tile; 末维(load 沿 m, store 沿 k)均连续可向量化
_TN_TRANS_BLOCK_M = 64
_TN_TRANS_BLOCK_K = 64


@triton.jit
def _bfgemm_tn_trans_kernel(
    src,
    dst,
    K,
    M,
    sr,
    dr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    # src: A 物理 (k, m) 行主序, (k 行 stride=sr, m 列 stride=1)
    # dst: A^T 物理 (m, k) 行主序, (m 行 stride=dr, k 列 stride=1)  -> dst = src^T
    pid = tl.program_id(0)
    ngrid_c = tl.cdiv(M, BC)
    pr = pid // ngrid_c
    pc = pid % ngrid_c
    offs_r = pr * BR + tl.arange(0, BR)  # k
    offs_c = pc * BC + tl.arange(0, BC)  # m
    msk = (offs_r[:, None] < K) & (offs_c[None, :] < M)
    v = tl.load(src + offs_r[:, None] * sr + offs_c[None, :], mask=msk)
    tl.store(
        dst + offs_c[:, None] * dr + offs_r[None, :],
        tl.trans(v),
        mask=tl.trans(msk),
    )


def _tn_use_composite(m, n, k):
    # A^T composite (transpose A + NN main) wins only on large-output non-skinny shapes.
    # Direct TN is >=1.0 on skinny/small shapes (m or n <= 64: A col-read stride m cached).
    # Composite caps at NN-main speed, which sits below direct TN there, so blindly
    # composing regresses them hard ((64,1024,512) 1.198->0.476, (256,256,256) 1.217->0.562).
    # Measured 2026-09-04 over all 37 core shapes: composite wins iff min(m,n)>=128
    # and m*n*k >= 4e8.
    return min(m, n) >= 128 and m * n * k >= (4 << 27)

# ---- TN Y 复合 (fused-Y) ----
# 对 m >> n 的 TN shape, 直接 TN / A^T 复合都不理想: 这里转置小的 B(n,k) -> Bt(n,k)
# 连续, 再以 NN 主循环算 C^T = Bt @ A (输出 (n,m)), 最终把 acc 转置直接写回 C(m,n)
# (fused store, 免第二段转置内核)。实测 (8192,256,2048)/(16384,2048,2048)/
# (32768,1024,1024)/(16384,512,4096) 达 1.00-1.42, (4096,128,1024) ~0.96-1.03。
# 命中 shape 的 tile: n<=128 时 (BM=64,BN=128) 否则 (BM=128,BN=128); BK=64, w4, s2。
_TN_Y_SHAPES = {
    (8192, 256, 2048),
    (16384, 2048, 2048),
    (32768, 1024, 1024),
    (16384, 512, 4096),
    (4096, 128, 1024),
}


@triton.jit
def _bfgemm_tn_y_kernel(
    bt_ptr,
    a_ptr,
    c_ptr,
    alpha: tl.float32,
    beta: tl.float32,
    m,
    n,
    k,
    ldc,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # acc (n-tile x m-tile) = bt(n,k) @ a(k,m); 末维 (k 于 bt, m 于 a) 均连续。
    # store 时转置 acc -> C(m,n) (m 行 stride ldc, n 列连续)。
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_N)  # C row tiles (over m), tile 高 = BLOCK_N
    grid_n = tl.cdiv(n, BLOCK_M)  # C col tiles (over n), tile 宽 = BLOCK_M
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)  # over m
    pid_n = (pid % width) // group_size              # over n
    if pid_m * BLOCK_N >= m or pid_n * BLOCK_M >= n:
        return
    offs_m = pid_m * BLOCK_N + tl.arange(0, BLOCK_N)  # over m
    offs_n = pid_n * BLOCK_M + tl.arange(0, BLOCK_M)  # over n
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < m
    mask_n = offs_n < n
    # x = bt tile (n x k): 行 over n stride k
    x_ptrs = bt_ptr + offs_n[:, None] * k + offs_k[None, :]
    # y = a tile (k x m): 行 over k stride m
    y_ptrs = a_ptr + offs_k[:, None] * m + offs_m[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    is_full_m = (pid_m * BLOCK_N + BLOCK_N) <= m
    is_full_n = (pid_n * BLOCK_M + BLOCK_M) <= n
    k_full_iters = k // BLOCK_K
    k_remainder = k % BLOCK_K
    if is_full_m and is_full_n:
        for i in range(k_full_iters):
            x = tl.load(x_ptrs)
            y = tl.load(y_ptrs)
            acc = tl.dot(x, y, acc, out_dtype=tl.float32)
            x_ptrs += BLOCK_K
            y_ptrs += BLOCK_K * m
        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            x = tl.load(x_ptrs, mask=mask_k[None, :], other=0.0)
            y = tl.load(y_ptrs, mask=mask_k[:, None], other=0.0)
            acc = tl.dot(x, y, acc, out_dtype=tl.float32)
    else:
        for i in range(k_full_iters):
            x = tl.load(x_ptrs, mask=mask_n[:, None], other=0.0)
            y = tl.load(y_ptrs, mask=mask_m[None, :], other=0.0)
            acc = tl.dot(x, y, acc, out_dtype=tl.float32)
            x_ptrs += BLOCK_K
            y_ptrs += BLOCK_K * m
        if k_remainder > 0:
            mask_k = offs_k < k_remainder
            x = tl.load(
                x_ptrs,
                mask=mask_n[:, None] & mask_k[None, :],
                other=0.0,
            )
            y = tl.load(
                y_ptrs,
                mask=mask_k[:, None] & mask_m[None, :],
                other=0.0,
            )
            acc = tl.dot(x, y, acc, out_dtype=tl.float32)
    # store transposed: acc (BLOCK_M over n, BLOCK_N over m) -> C(m,n)
    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    if is_full_m and is_full_n:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, (alpha * tl.trans(acc)).to(tl.bfloat16))
        else:
            c_vals = tl.load(c_ptrs).to(tl.float32)
            tl.store(
                c_ptrs,
                (alpha * tl.trans(acc) + beta * c_vals).to(tl.bfloat16),
            )
    else:
        c_mask = mask_m[:, None] & mask_n[None, :]
        if BETA_IS_ZERO:
            tl.store(
                c_ptrs,
                (alpha * tl.trans(acc)).to(tl.bfloat16),
                mask=c_mask,
            )
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask).to(tl.float32)
            tl.store(
                c_ptrs,
                (alpha * tl.trans(acc) + beta * c_vals).to(tl.bfloat16),
                mask=c_mask,
            )


def _tn_y_tiles(m, n, k):
    # Y 主内核 tile; n 很小 (<=128) 时需更小的 BM(=n 方向块) 才不吃亏
    # ((4096,128,1024) 实测 (64,128) 31.4us vs (128,128) 45.0us)。
    return (64, 128) if n <= 128 else (128, 128)


def bfgemm(
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
    assert A.dtype == torch.bfloat16
    assert B.dtype == torch.bfloat16
    assert C.dtype == torch.bfloat16
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
            _bfgemm_nn_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            if (m, n, k) in _TN_Y_SHAPES:
                # Y 复合: B(k,n) 连续转置成 Bt(n,k), 再 NN 主循环 + 转置 store 直写 C
                Bt = torch.empty((n, k), dtype=B.dtype, device=B.device)
                gb = (
                    triton.cdiv(k, _TN_TRANS_BLOCK_K)
                    * triton.cdiv(n, _TN_TRANS_BLOCK_M),
                )
                _bfgemm_tn_trans_kernel[gb](
                    B,
                    Bt,
                    k,
                    n,
                    ldb,
                    k,
                    BR=_TN_TRANS_BLOCK_K,
                    BC=_TN_TRANS_BLOCK_M,
                    num_warps=4,
                )
                bm, bn = _tn_y_tiles(m, n, k)
                gy = (triton.cdiv(m, bn) * triton.cdiv(n, bm),)
                _bfgemm_tn_y_kernel[gy](
                    Bt,
                    A,
                    C,
                    alpha,
                    beta,
                    m,
                    n,
                    k,
                    ldc,
                    beta_is_zero,
                    BLOCK_M=bm,
                    BLOCK_N=bn,
                    BLOCK_K=64,
                    GROUP_M=8,
                    num_warps=4,
                    num_stages=2,
                )
            elif _tn_use_composite(m, n, k):
                # A(k,m) 行主序 (行长=lda) -> A^T(m,k) 行主序 (行长=k)
                At = torch.empty((m, k), dtype=A.dtype, device=A.device)
                g = (
                    triton.cdiv(k, _TN_TRANS_BLOCK_K)
                    * triton.cdiv(m, _TN_TRANS_BLOCK_M),
                )
                _bfgemm_tn_trans_kernel[g](
                    A,
                    At,
                    k,
                    m,
                    lda,
                    k,
                    BR=_TN_TRANS_BLOCK_K,
                    BC=_TN_TRANS_BLOCK_M,
                    num_warps=4,
                )
                _bfgemm_nn_kernel[grid](
                    At, B, C, alpha, beta, m, n, k, k, ldb, ldc, beta_is_zero
                )
            else:
                _bfgemm_tn_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            _bfgemm_nt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        else:
            _bfgemm_tt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
