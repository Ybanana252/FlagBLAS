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
Hygon DTK hgemm custom path.

Previously there was no dedicated `hgemm` implementation on the hygon side; it reused the generic level3 kernels + the NVIDIA default tune pool.
This module follows the existing pattern of `_hygon/ops/bfgemm.py` and adds two hygon custom paths for fp16 gemm,
while the remaining layouts still reuse the generic level3 kernels:

1. TN (transa=T, transb=N): A physical (k, m) row-major; when the direct main loop steps along k, the last
   dimension (k) of the A tile has stride lda, so it cannot be vectorized/read contiguously and bandwidth
   efficiency is poor. Here a dedicated transpose-copy kernel is used first to materialize
   A(k,m) into a contiguous A^T(m,k) buffer, then run the NN main kernel, so that A loading matches the NN variant.
   The transpose kernel reads and writes contiguous segments, so its overhead is far smaller than the strided-read penalty of direct TN.

2. odd / misaligned layouts (any ld % 8 != 0): row start address = row * ld * 2B; when ld is not a multiple of 8,
   even 16B alignment cannot be guaranteed, and Triton narrows 2D tile row-vectorized load/store across the board. For large shapes
   pad inputs to aligned buffers (NN additionally transposes B into Bt to use the NT main kernel), then crop back to real C.

NN / NT / TT match the generic level3 kernels (measured median >= 0.95, no extra handling); only the hygon
dedicated tune pool is added (hgemm_* section of tune_configs.yaml) to avoid falling back to the NVIDIA default pool.

Note: the compiler-level tunings stacked on the bfgemm side by shape whitelist (sched_latency / kpack / NN->NT routing etc.)
are all bf16 measurement conclusions and are not ported to this module; hgemm keeps only the dtype-independent structural paths.
"""

import torch
import triton
import triton.language as tl

from flag_blas.ops.level3.hgemm import (
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    ScalarType,
    _hgemm_nn_kernel,
    _hgemm_nt_kernel,
    _hgemm_tn_kernel,
    _hgemm_tt_kernel,
)
from flag_blas.runtime import torch_device_fn

# TN composite transpose kernel tile; last dimension (load along m, store along k) is contiguous and vectorizable.
# Same values as _hygon/ops/bfgemm.py (both 2B dtypes, same transpose bandwidth characteristics).
_TN_TRANS_BLOCK_M = 128
_TN_TRANS_BLOCK_K = 128


@triton.jit
def _hgemm_tn_trans_kernel(
    src,
    dst,
    K,
    M,
    sr,
    dr,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    # src: A physical (k, m) row-major (k row stride=sr, m col stride=1)
    # dst: A^T physical (m, k) row-major (m row stride=dr, k col stride=1) -> dst = src^T
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
    # A^T composite (transpose A + NN main) has net positive gain only on large-output non-skinny shapes;
    # skinny/small shapes (m or n <= 64) are faster with direct TN, hence the lower bound. Criteria identical to bfgemm.
    return min(m, n) >= 128 and m * n * k >= (4 << 27)


# ---- pad path for odd / misaligned layouts ----
# Criteria / thresholds identical to _hygon/ops/bfgemm.py: fp16 and bf16 are both 2B dtypes, row alignment condition
# (ld % 8 == 0 corresponds to 16B row start alignment) and pad tiles are exactly the same.
_PAD_ALIGN_MN = 256  # max BLOCK_M/BLOCK_N in pool
_PAD_ALIGN_K = 128  # max BLOCK_K in pool
_PAD_BLK = 64  # transpose-copy tile width BC (src n direction)
_PAD_TRANS_BLK = 128  # transpose-copy tile height BR (src k direction; k row-major reads with large stride)
_PAD_MIN_FLOPS = 5.0e9  # lower bound of 2*m*n*k to enable pad (small odd shapes dominated by launch overhead, no pad)


def _padded_size(m, n, k, lda, ldb, ldc):
    # Row start address 16B alignment requires ld*2B % 16 == 0 -> ld % 8 == 0. If any fails it is
    # misaligned -> padding needed (covers all odd shapes and cases where lda/ldb/ldc are not multiples of 8).
    if lda % 8 == 0 and ldb % 8 == 0 and ldc % 8 == 0:
        return None
    return (
        triton.cdiv(m, _PAD_ALIGN_MN) * _PAD_ALIGN_MN,
        triton.cdiv(n, _PAD_ALIGN_MN) * _PAD_ALIGN_MN,
        triton.cdiv(k, _PAD_ALIGN_K) * _PAD_ALIGN_K,
    )


def _row_view(t, ld):
    # physical buffer -> (row, ld) row-major view (no copy), for slice copies to take the logical (m,k) region;
    # compatible with both 2D (row,ld) and 1D flat buffers.
    if t.dim() == 2 and t.shape[1] == ld:
        return t
    return t.view(-1, ld)


@triton.jit
def _hgemm_pad_trans_kernel(
    src,
    dst,
    K,
    N,
    sr,
    dr,
    np,
    BR: tl.constexpr,
    BC: tl.constexpr,
):
    # src: B(k,n) row-major (k row stride=sr, n column contiguous)
    # dst: Bt(n_pad,k_pad) row-major (n row stride=dr, k column contiguous); does pad+transpose in one pass,
    # grid = cdiv(k_pad,BR) x cdiv(n_pad,BC) full coverage -> store without mask; out-of-range source
    # (k>=K or n>=N) reads 0 -> pad region is 0.
    pid = tl.program_id(0)
    ngrid_c = tl.cdiv(np, BC)
    pr = pid // ngrid_c  # k tiles (k_pad dimension)
    pc = pid % ngrid_c  # n tiles
    offs_k = pr * BR + tl.arange(0, BR)
    offs_n = pc * BC + tl.arange(0, BC)
    msk = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    v = tl.load(src + offs_k[:, None] * sr + offs_n[None, :], mask=msk, other=0.0)
    tl.store(dst + offs_n[:, None] * dr + offs_k[None, :], tl.trans(v))


def _launch_pad_nt(
    A,
    Ap,
    Bt,
    Cp,
    alpha,
    beta,
    beta_is_zero,
    mp,
    np,
    kp,
    lda_p,
):
    # NT main kernel (input Ap(m,kp) lda=lda_p, Bt(np,kp) ldb=kp) -> Cp(mp,np);
    # grid lambda is expanded by libtuner.
    grid = lambda meta: (
        triton.cdiv(mp, meta["BLOCK_M"]) * triton.cdiv(np, meta["BLOCK_N"]),
    )
    _hgemm_nt_kernel[grid](
        Ap,
        Bt,
        Cp,
        alpha,
        beta,
        mp,
        np,
        kp,
        lda_p,
        kp,
        np,
        beta_is_zero,
    )


def hgemm(
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
    assert A.dtype == torch.float16
    assert B.dtype == torch.float16
    assert C.dtype == torch.float16
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
        # odd/misalign layout (any ld%8!=0) + large shape: pad to aligned buffers and uniformly use the NT main
        # kernel (B reads contiguous along k columns). For valid outputs (i<m,j<n) the k-direction dot product only spans the k-pad
        # columns of A/B, so only the [:, k:kp] strip of the two buffers needs zeroing; m-pad rows / n-pad rows / the pad region of Cp
        # only affect cropped outputs, no need to zero. alpha/beta semantics preserved.
        ps = _padded_size(m, n, k, lda, ldb, ldc)
        if ps is not None and 2.0 * m * n * k >= _PAD_MIN_FLOPS:
            mp, np, kp = ps
            Ap = torch.empty(mp, kp, dtype=A.dtype, device=A.device)
            Bt = torch.empty(np, kp, dtype=A.dtype, device=A.device)
            Cp = torch.empty(mp, np, dtype=A.dtype, device=A.device)
            if transa == CUBLAS_OP_N:  # A physical (m,k): same-layout pad
                Ap[:m, :k].copy_(_row_view(A, lda)[:m, :k])
            else:  # A physical (k,m): pad+transpose -> Ap(m,k)
                g = (
                    triton.cdiv(kp, _PAD_TRANS_BLK) * triton.cdiv(mp, _PAD_BLK),
                )
                _hgemm_pad_trans_kernel[g](
                    A,
                    Ap,
                    k,
                    m,
                    lda,
                    kp,
                    mp,
                    BR=_PAD_TRANS_BLK,
                    BC=_PAD_BLK,
                    num_warps=8,
                )
            if transb == CUBLAS_OP_N:  # B physical (k,n): pad+transpose -> Bt(n,k)
                b_pad_trans = True
                g = (
                    triton.cdiv(kp, _PAD_TRANS_BLK) * triton.cdiv(np, _PAD_BLK),
                )
                _hgemm_pad_trans_kernel[g](
                    B,
                    Bt,
                    k,
                    n,
                    ldb,
                    kp,
                    np,
                    BR=_PAD_TRANS_BLK,
                    BC=_PAD_BLK,
                    num_warps=8,
                )
            else:  # B physical (n,k): same-layout pad
                b_pad_trans = False
                Bt[:n, :k].copy_(_row_view(B, ldb)[:n, :k])
            if kp > k:
                # zero the k-pad strip: only needed for same-layout copy_ buffers (copy_ only writes
                # [:m,:k]/[:n,:k], k-pad is torch.empty residue); the transpose kernel's
                # store covers the whole buffer and writes 0 at out-of-range source, so k-pad is already 0, skip.
                if transa == CUBLAS_OP_N:  # A same-layout copy, k-pad needs zeroing
                    Ap[:, k:].zero_()
                if not b_pad_trans:
                    Bt[:, k:].zero_()
            if not beta_is_zero:  # pre-set beta*C semantics: copy C into the top-left of the pad buffer
                Cp[:m, :n].copy_(_row_view(C, ldc)[:m, :n])
            _launch_pad_nt(A, Ap, Bt, Cp, alpha, beta, beta_is_zero, mp, np, kp, kp)
            _row_view(C, ldc)[:m, :n].copy_(Cp[:m, :n])
            return

        if transa == CUBLAS_OP_N and transb == CUBLAS_OP_N:
            _hgemm_nn_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            if _tn_use_composite(m, n, k):
                # TN composite: A(k,m) row-major (row length=lda) -> A^T(m,k) row-major (row length=k),
                # then run the NN main kernel, so that A loading matches the NN variant (contiguous last dimension).
                At = torch.empty((m, k), dtype=A.dtype, device=A.device)
                g = (
                    triton.cdiv(k, _TN_TRANS_BLOCK_K)
                    * triton.cdiv(m, _TN_TRANS_BLOCK_M),
                )
                _hgemm_tn_trans_kernel[g](
                    A,
                    At,
                    k,
                    m,
                    lda,
                    k,
                    BR=_TN_TRANS_BLOCK_K,
                    BC=_TN_TRANS_BLOCK_M,
                    num_warps=8,
                )
                _hgemm_nn_kernel[grid](
                    At, B, C, alpha, beta, m, n, k, k, ldb, ldc, beta_is_zero
                )
            else:
                _hgemm_tn_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            _hgemm_nt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
        else:
            _hgemm_tt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
