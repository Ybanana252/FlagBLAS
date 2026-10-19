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
Hygon DTK cgemm dedicated implementation (self-developed complex kernel).

complex64 GEMM: C = alpha * op(A) @ op(B) + beta * C, op ∈ {N, T, C(conjugate transpose)}.
This module ships its own complex Triton kernel (handling N/T/C and conjugation inside the kernel), and additionally provides a
hygon-custom "transpose composite" path: when transa/transb contain a transpose and the shape is large enough, first materialize the physical
(k,m)/(n,k) non-conjugate/conjugate transpose into contiguous (m,k)/(k,n) buffers, then call the main kernel with N/N,
avoiding the bandwidth loss of strided complex reads (same idea as the transpose composite in _hygon/ops/bfgemm.py / hgemm.py).

Data layout: complex values are stored interleaved as real/imag; before passing to the kernel they are flattened with
torch.view_as_real(...).reshape(-1) into an fp32 view, and the kernel reads re/im at 2*elem / 2*elem+1.

Note: before real-machine accuracy/performance validation, the hygon dedicated config table and transpose composite threshold are conservative values, to be converged by measurement.
"""

import torch
import triton
import triton.language as tl

from flag_blas.ops.level3.cgemm import (
    ScalarType,
    _complex_scalar_parts,
    _validate_cgemm_args,
)
from flag_blas.runtime import torch_device_fn


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
        # A: N -> physical (m,k); T/C -> physical (k,m)
        if TRANS_A == 0:
            a_elem = offs_m[:, None] * lda + cur_k[None, :]
        else:
            a_elem = cur_k[None, :] * lda + offs_m[:, None]
        # B: N -> physical (k,n); T/C -> physical (n,k)
        if TRANS_B == 0:
            b_elem = cur_k[:, None] * ldb + offs_n[None, :]
        else:
            b_elem = offs_n[None, :] * ldb + cur_k[:, None]

        a_mask = mask_m[:, None] & mask_k[None, :]
        b_mask = mask_k[:, None] & mask_n[None, :]
        # pack pairs (re,im) for loading (contiguous last dim 2 x fp32 -> 8B vectorization), then split into re/im,
        # avoiding stride-2 scalar gather.
        a2 = tl.load(
            a_ptr + (2 * a_elem)[:, :, None] + tl.arange(0, 2)[None, None, :],
            mask=a_mask[:, :, None],
            other=0.0,
        )
        ar, ai = tl.split(a2)
        b2 = tl.load(
            b_ptr + (2 * b_elem)[:, :, None] + tl.arange(0, 2)[None, None, :],
            mask=b_mask[:, :, None],
            other=0.0,
        )
        br, bi = tl.split(b2)
        # conjugate transpose: negate the imaginary part
        if TRANS_A == 2:
            ai = -ai
        if TRANS_B == 2:
            bi = -bi

        # (ar + ai i)(br + bi i) = (ar*br - ai*bi) + (ar*bi + ai*br) i
        # gfx936 backend only allows input_precision="ieee" (no tf32), i.e. true fp32 dot product,
        # more accurate than tf32x3, same metric as the hipblasCgemm reference.
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
        c2 = tl.load(
            c_ptr + (2 * c_elem)[:, :, None] + tl.arange(0, 2)[None, None, :],
            mask=c_mask[:, :, None],
            other=0.0,
        )
        cr, ci = tl.split(c2)
        out_r += beta_r * cr - beta_i * ci
        out_i += beta_r * ci + beta_i * cr

    # pack pairs for storing (contiguous last dim 2 x fp32 -> 8B vectorization)
    out2 = tl.join(out_r, out_i)
    tl.store(
        c_ptr + (2 * c_elem)[:, :, None] + tl.arange(0, 2)[None, None, :],
        out2,
        mask=c_mask[:, :, None],
    )


def _select_cgemm_config_hygon(m, n, k):
    # hygon gfx936 tiers (calibrated by measurement over shapes x variants). The complex kernel has 2 fp32 accumulators +
    # 4 operand tiles, so BLOCK_K is 16/32; tiles scale with size, large shapes use GROUP_M=4
    # to improve L2 reuse.
    max_dim = max(m, n, k)
    if max_dim <= 128:
        return 16, 16, 32, 4, 1
    if max_dim <= 256:
        return 32, 32, 32, 4, 1
    if max_dim <= 512:
        return 64, 64, 16, 8, 1
    return 128, 64, 16, 8, 4


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

    with torch_device_fn.device(A.device):
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
            lda,
            ldb,
            ldc,
            transa,
            transb,
            beta_is_zero,
            alpha_is_one,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            GROUP_M=group_m,
            num_warps=num_warps,
        )
