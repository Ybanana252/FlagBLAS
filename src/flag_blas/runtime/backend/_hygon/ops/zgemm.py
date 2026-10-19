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
Hygon DTK zgemm dedicated implementation (self-developed complex128 kernel).

complex128 GEMM: C = alpha * op(A) @ op(B) + beta * C, op ∈ {N, T, C(conjugate transpose)}.
Structurally identical to _hygon/ops/cgemm.py (complex multiply-add = 4 real dots), the only difference is the dtype is fp64:

- Complex values are stored interleaved as [re,im]; before passing to the kernel they are flattened with
  torch.view_as_real(...).reshape(-1) into an fp64 view, and the kernel accesses memory in **packed pairs**
  (contiguous last dim 2 x fp64 = 16B, tl.load/tl.store + tl.split/tl.join), avoiding stride-2 scalar gather
  (measured 1.8x-level gain for this item on cgemm).
- fp64 dot uses the MMAC/FMA path on gfx936; allow_tf32=False is used to guarantee fp64 semantics.
- Includes the hygon-custom "transpose composite" path (when transa/transb contain a transpose and the shape is large enough, first materialize the physical
  (k,m)/(n,k) into contiguous (m,k)/(k,n) buffers, negating the imaginary part when conjugating), then call the main kernel with N/N.

Note: before real-machine accuracy/performance validation, the hygon dedicated config table and transpose composite threshold are conservative values, to be converged by measurement.
"""

import torch
import triton
import triton.language as tl

from flag_blas.ops.level3.zgemm import (
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    ScalarType,
    _complex_scalar_parts,
    _validate_zgemm_args,
)
from flag_blas.runtime import torch_device_fn

# transpose composite copy tile
_TR_BR = 64
_TR_BC = 64
# transpose composite enable threshold: 2*m*n*k lower bound (on small shapes copy overhead is too high, no composite).
_TR_MIN_FLOPS = 4 << 24


@triton.jit
def _zgemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha_r: tl.float64,
    alpha_i: tl.float64,
    beta_r: tl.float64,
    beta_i: tl.float64,
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

    acc_r = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    acc_i = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
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
        # pack pairs (re,im) for loading (contiguous last dim 2 x fp64 -> 16B vectorization), then split into re/im,
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
        acc_r += tl.dot(ar, br, out_dtype=tl.float64, allow_tf32=False)
        acc_r -= tl.dot(ai, bi, out_dtype=tl.float64, allow_tf32=False)
        acc_i += tl.dot(ar, bi, out_dtype=tl.float64, allow_tf32=False)
        acc_i += tl.dot(ai, br, out_dtype=tl.float64, allow_tf32=False)

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

    # pack pairs for storing (contiguous last dim 2 x fp64 -> 16B vectorization)
    out2 = tl.join(out_r, out_i)
    tl.store(
        c_ptr + (2 * c_elem)[:, :, None] + tl.arange(0, 2)[None, None, :],
        out2,
        mask=c_mask[:, :, None],
    )


@triton.jit
def _zgemm_trans_kernel(
    src,
    dst,
    R,
    Cc,
    sr,
    CONJ: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    # src: physical (R, Cc) complex row-major (row stride=sr), real/imag interleaved [re,im]
    # dst: (Cc, R) complex row-major, row stride=R; dst = trans(src) (negate imaginary part when CONJ)
    pid = tl.program_id(0)
    ngc = tl.cdiv(Cc, BC)
    pr = pid // ngc
    pc = pid % ngc
    offs_r = pr * BR + tl.arange(0, BR)  # 0..R-1
    offs_c = pc * BC + tl.arange(0, BC)  # 0..Cc-1
    msk = (offs_r[:, None] < R) & (offs_c[None, :] < Cc)
    idx = offs_r[:, None] * sr + offs_c[None, :]
    v2 = tl.load(
        src + (2 * idx)[:, :, None] + tl.arange(0, 2)[None, None, :],
        mask=msk[:, :, None],
        other=0.0,
    )
    re, im = tl.split(v2)
    if CONJ:
        im = -im
    didx = offs_c[:, None] * R + offs_r[None, :]
    dmsk = tl.trans(msk)
    o2 = tl.join(tl.trans(re), tl.trans(im))
    tl.store(
        dst + (2 * didx)[:, :, None] + tl.arange(0, 2)[None, None, :],
        o2,
        mask=dmsk[:, :, None],
    )


def _select_zgemm_config_hygon(m, n, k):
    # hygon gfx936 tiers (calibrated by shape measurement on FlagTree/triton with f64 MMAC).
    # fp64 complex kernel: 2 fp64 accumulators + 4 operand tiles; BLOCK_K=8/16 optimal, small shapes
    # use 16x16x16 w2, large shapes use K width 8 to reduce register/pipeline pressure.
    max_dim = max(m, n, k)
    if max_dim <= 128:
        return 16, 16, 16, 2, 4
    if max_dim <= 512:
        return 32, 64, 16, 4, 4
    if max_dim <= 1024:
        return 128, 64, 8, 8, 4
    return 64, 64, 8, 4, 4


def _use_trans_composite(m, n, k, transa, transb):
    # Materialize contiguous operands only when a transpose is involved and the output is large enough (2*m*n*k >= _TR_MIN_FLOPS);
    # on small shapes copy overhead is too high, so use in-kernel transpose directly.
    if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
        return False
    return 2.0 * m * n * k >= _TR_MIN_FLOPS and min(m, n) >= 64


def _materialize_transpose(x_real, rows_phys, cols_phys, ld, conj, device):
    # x_real: f64 interleaved view of a physical (rows_phys, cols_phys) complex value
    # returns a contiguous (row-major) tensor of (cols_phys, rows_phys) complex values, i.e. trans(x) (conjugate when conj)
    out_complex = torch.empty(
        (cols_phys, rows_phys), dtype=torch.complex128, device=device
    )
    out_real = torch.view_as_real(out_complex).reshape(-1)
    grid = (
        triton.cdiv(rows_phys, _TR_BR) * triton.cdiv(cols_phys, _TR_BC),
    )
    _zgemm_trans_kernel[grid](
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


def zgemm(
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
    _validate_zgemm_args(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)

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

    block_m, block_n, block_k, num_warps, group_m = _select_zgemm_config_hygon(m, n, k)
    grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)

    A_real = torch.view_as_real(A).reshape(-1)
    B_real = torch.view_as_real(B).reshape(-1)
    C_real = torch.view_as_real(C).reshape(-1)

    eff_trans_a, eff_lda = transa, lda
    eff_trans_b, eff_ldb = transb, ldb

    with torch_device_fn.device(A.device):
        if _use_trans_composite(m, n, k, transa, transb):
            # A physical (k,m) -> op(A) contiguous (m,k); negate imaginary part for conjugate transpose
            if transa != CUBLAS_OP_N:
                op_a = _materialize_transpose(
                    A_real, k, m, lda, transa == CUBLAS_OP_C, A.device
                )
                A_real = torch.view_as_real(op_a).reshape(-1)
                eff_trans_a, eff_lda = CUBLAS_OP_N, k
            # B physical (n,k) -> op(B) contiguous (k,n)
            if transb != CUBLAS_OP_N:
                op_b = _materialize_transpose(
                    B_real, n, k, ldb, transb == CUBLAS_OP_C, B.device
                )
                B_real = torch.view_as_real(op_b).reshape(-1)
                eff_trans_b, eff_ldb = CUBLAS_OP_N, n

        _zgemm_kernel[grid](
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
            num_stages=2,
        )
