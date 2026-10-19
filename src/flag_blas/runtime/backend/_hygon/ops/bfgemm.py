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
Hygon DTK bfgemm custom path.

Under TN (transa=T, transb=N), the physical layout of A is (k, m) row-major. When
stepping along k in the direct TN main loop, the last dimension (k) of the A tile
has stride lda, so it cannot be vectorized/read contiguously and bandwidth
efficiency is poor (core-shape benchmark TN variant median only ~0.63). Here the
same idea as sgemm is used for TN: first use a dedicated transpose-copy kernel to
materialize A(k,m) into a contiguous A^T(m,k) buffer, then run the NN main kernel,
so that A loading matches the NN variant (contiguous last dimension). The
transpose kernel reads and writes contiguous segments, so its overhead is far
smaller than the strided-read penalty of direct TN.

NN / NT / TT match the generic level3 kernels (measured median >=0.95, no extra
handling).
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

# TN composite transpose kernel tile; last dimension (load along m, store along k) is contiguous and vectorizable
# dir4 (2026-09-07): transpose tile 64x64 w4 -> 128x128 w8. Swept 5 NN-route B shapes
# (8192/28672/24576/12288/11008 x ...), bandwidth 811-903GB/s -> 1027-1144GB/s
# (+16~+34%, torch copy ceiling only ~610GB/s). Net routing gain NN ~2-5%, TN ~4-7%,
# all still <0.8 (main kernel gap = fork level, see report §10).
_TN_TRANS_BLOCK_M = 128
_TN_TRANS_BLOCK_K = 128

# ---- 16 large shape compiler alignment (2026-09-07): NN main kernel sched_latency per-shape gating ----
# flagtree clang latency layout (-mmac-latency=5 + -ds-load-store-latency) benefit groups by production tile:
#   (256,256,BK64,w8/s2) large tile shapes were originally tiered by mmac5-ds12 (measured -3.6~-3.9%);
#   but mmac5-ds12/-ds14 only exist in locally patched triton builds; the FlagTree upstream backend
#   (third_party/hcu/backend/compiler_hcu.py) sched_latency whitelist only has
#   mmac5-ds6 / mmac5-ds10 -> production side uniformly falls back to mmac5-ds10 (other values upstream directly raise
#   ValueError: Unsupported scheduling latency).
#   (128,256,BK32,w4/s2) tile (4096^3 / 2048x11008x4096) -> mmac5-ds10 -2.0~-2.3%.
# Measured by calling the production config directly (consistent with v8 warm DB), base aligned with v8 official run ±0.3%. Other shapes
# (pad / fused-Y / direct-TN / NT / TT) untouched, keep default scheduling zero risk.
_NN_SCHED_BY_SHAPE = {
    (8192, 8192, 8192): "mmac5-ds10",
    (4096, 24576, 8192): "mmac5-ds10",
    (8192, 28672, 8192): "mmac5-ds10",
    (4096, 8192, 28672): "mmac5-ds10",
    (2048, 4096, 11008): "mmac5-ds10",
    (2048, 12288, 4096): "mmac5-ds10",
    (512, 16384, 4096): "mmac5-ds10",
    (4096, 4096, 4096): "mmac5-ds10",
    (2048, 11008, 4096): "mmac5-ds10",
}


def _nn_sched_kwargs(m, n, k):
    sched = _NN_SCHED_BY_SHAPE.get((m, n, k))
    return {"sched_latency": sched} if sched else {}

# ---- direction ② landed (2026-09-07): kpack=2 gating (official-run quantification see report §9) ----
# HIPOptions.kpack controls how many K values are packed per 32bit for bf16 dot operands (default 1). kpack=2
# (2-K/32bit packing, removes v_perm/reduces register round-trips) official-run full-spectrum quantification:
#   * NT main kernel unconditional: NN-route 7 dims -0.5~-3.5, TN-route 6 dims -0.3~-3.6,
#     NT-native large shapes -2.5~-3.8 (8192x28672x8192 sp 0.821->0.852 meets target),
#     pad-odd (NN/TN/TT 8191^3/4097x8191x4095) -0.4~-3.1, small shapes within noise, no regression;
#   * NN main kernel per-shape: only (2048,11008,4096) (NN/TN residual, k=11008 w4/ds10) -4.2~-4.5%;
#   * excluded: NN 512x16384x4096 (w8/ds12) +4.1%, direct-TN small shapes 1024^3 +22% /
#     128x4096x1024 +34%, direct-TT large shapes 4096x8192x28672 +4.5% / 2048x2048x16384
#     +5.1% / 2048x4096x11008 +4.9% / 2048^3 +3.7% regression (direct-TT excluded per kernel);
#   * other compiler knobs ruled out: matrix_instr_nonkdim=16 no effect (gfx936 bf16 only 16x16x16),
#     mmac_layout_force 0 numerical error (err 1.38) / 3 compile failure (lit/lts unsupported).
def _kp2_kwargs():
    return {"kpack": 2}

# NN main kernel kpack=2 whitelist shapes (residual NN direct path + TN A^T composite share the same main kernel)
_NN_KP2_SHAPES = {(2048, 11008, 4096)}


def _nn_kernel_kwargs(m, n, k):
    """NN main kernel launch kwargs: sched_latency gating + direction ② kpack whitelist."""
    kw = _nn_sched_kwargs(m, n, k)
    if (m, n, k) in _NN_KP2_SHAPES:
        kw["kpack"] = 2
    return kw


# ---- direction ① (2026-09-07): transpose NN large-shape B once -> NT main kernel ----
# NN layout (both A/B k last dim) loses on scheduling/issue in the tl.dot main loop; under the same instructions the NT layout
# (B reads (n,k) with contiguous last dim) has a 16-23% lower flag (report §7 / probe dir1_ntroute 2026-09-07).
# For shapes with positive net gain: each call first uses _bfgemm_tn_trans_kernel to copy B(k,n) into
# a Bt(n,k) contiguous buffer (same NT main kernel path as _launch_pad_nt). Transpose overhead 0.07-0.31ms
# is covered by main-kernel savings (net -2.7~-17.4%). Exceptions: 512x16384x4096 (trans 0.31ms >
# main-kernel saving 0.09ms, net +49.9%) and 2048x11008x4096 (net +1.4%) are net losses, keep the NN
# main kernel + ds gating. NT main kernel config reuses warm DB (v8 NT tune section, official run hits).
_NN_NT_ROUTE_SHAPES = {
    (8192, 8192, 8192),
    (4096, 24576, 8192),
    (8192, 28672, 8192),
    (4096, 8192, 28672),
    (4096, 4096, 4096),
    (2048, 4096, 11008),
    (2048, 12288, 4096),
}

# ---- direction ① TN side (2026-09-07): Aᵀ composite plus transB -> NT main kernel ----
# The TN composite (transA + NN main kernel) switching to the NT main kernel requires an extra transB
# (B(k,n)->Bt(n,k)); the transA composite path already pays. net = main-kernel gap - transB, §7 quantified 6 dims
# -2.3~-16.4% (8192³ -16.42 / 8192x28672 -15.38 / 4096x8192x28672 -10.28 /
# 4096x24576x8192 -9.87 / 4096³ -6.82 / 2048x12288x4096 -2.25). 2048x11008x4096
# (+1.72% net loss) keeps the NN main kernel + ds; 2048x4096x11008 TN sp>=0.85 is not included. Hits go
# through the NT main kernel (warm DB config, no ds gating stacked).
_TN_NT_ROUTE_SHAPES = {
    (8192, 8192, 8192),
    (4096, 24576, 8192),
    (8192, 28672, 8192),
    (4096, 8192, 28672),
    (4096, 4096, 4096),
    (2048, 12288, 4096),
}


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
    # src: A physical (k, m) row-major, (k row stride=sr, m col stride=1)
    # dst: A^T physical (m, k) row-major, (m row stride=dr, k col stride=1)  -> dst = src^T
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

# ---- TN Y composite (fused-Y) ----
# For TN shapes with m >> n, neither direct TN nor the A^T composite is ideal: here transpose the small B(n,k) -> Bt(n,k)
# contiguously, then use the NN main loop to compute C^T = Bt @ A (output (n,m)), and finally transpose acc and write it
# directly back to C(m,n) (fused store, avoiding a second transpose kernel). Measured (8192,256,2048)/(16384,2048,2048)/
# (32768,1024,1024)/(16384,512,4096) reach 1.00-1.42, (4096,128,1024) ~0.96-1.03.
# Tile for hit shapes: (BM=64,BN=128) when n<=128 otherwise (BM=128,BN=128); BK=64, w4, s2.
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
    # acc (n-tile x m-tile) = bt(n,k) @ a(k,m); last dims (k in bt, m in a) are both contiguous.
    # store transposes acc -> C(m,n) (m row stride ldc, n column contiguous).
    pid = tl.program_id(0)
    grid_m = tl.cdiv(m, BLOCK_N)  # C row tiles (over m), tile height = BLOCK_N
    grid_n = tl.cdiv(n, BLOCK_M)  # C col tiles (over n), tile width = BLOCK_M
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
    # x = bt tile (n x k): rows over n, stride k
    x_ptrs = bt_ptr + offs_n[:, None] * k + offs_k[None, :]
    # y = a tile (k x m): rows over k, stride m
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
    # Y main kernel tile; when n is very small (<=128) a smaller BM (=n-direction block) is needed
    # ((4096,128,1024) measured (64,128) 31.4us vs (128,128) 45.0us).
    return (64, 128) if n <= 128 else (128, 128)


# ---- pad path for odd / misaligned layouts ----
# When (m,n,k) in the core-shape table are odd (8191/4095/1023/511/4097x8191x4095 etc.),
# the row start address of A/B/C = row * ld * 2B; odd ld means even 4B/8B alignment of the row base cannot be guaranteed,
# Triton narrows 2D tile row-vectorized load/store across the board, and large-shape latency inflates ~1.5-2x
# (8191^3 NN 11.27ms vs 8192^3 5.54ms); the masked k-loop of boundary tiles loses another notch.
# (Diagnosis: /tmp/diag_kodd.py + diag_padfull.py, 2026-09-05; config sweep of 27 entries no solution,
# confirmed as layout/mask issue rather than tile config issue)
# Fix: pad inputs to (BLOCK_M/N=256, BLOCK_K=128) integer-multiple buffers + mask-free main
# kernel; for NN also transpose B(k,n) into Bt(n,k) to use the NT main kernel (B reads contiguous along k columns, same pool
# measured ~25-30% faster: 8192^3 NN main 5.54ms -> NT main 4.10ms). Output goes to the pad C
# buffer first then cropped back to real C, alpha/beta semantics preserved. After padding call the
# autotuner kernel with (m_pad,n_pad,k_pad), hitting the same tune entry as the already-tuned aligned shape (8192^3 etc.).
_PAD_ALIGN_MN = 256  # max BLOCK_M/BLOCK_N in pool
_PAD_ALIGN_K = 128  # max BLOCK_K in pool
_PAD_BLK = 64  # transpose-copy tile width BC (src n direction; measured odd optimum 64x128/w8)
_PAD_TRANS_BLK = 128  # transpose-copy tile height BR (src k direction; k row-major reads with large stride)
# Small odd shapes (511/1023 level, 2mnk < _PAD_MIN_FLOPS) are not padded: padding needs 4-5 extra
# kernel/copy launches (~25us each), and on small shapes launch overhead dominates the misalign gain
# (experiment 2026-09-06: 1023^3 padded 0.20ms vs direct 0.06ms, regression across the board).
_PAD_MIN_FLOPS = 5.0e9  # lower bound of 2*m*n*k to enable pad

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
def _bfgemm_pad_trans_kernel(
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
    _bfgemm_nt_kernel[grid](
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
        **_kp2_kwargs(),
    )


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
        # odd/misalign layout (any ld%8!=0) + large shape: pad to aligned buffers and uniformly use the NT main
        # kernel, see the pad path comments above. probe_pad_dispatch2 (2026-09-05, official-run e2e metric,
        # numeric beta0/β=1.5 all PASS): NN 8191^3 11.27->6.05ms, NT 9.48->6.23,
        # TN 11.76->5.89, TT 13.19->6.07; sp 0.75-1.54 (relative to same-process torch).
        ps = _padded_size(m, n, k, lda, ldb, ldc)
        if ps is not None and 2.0 * m * n * k >= _PAD_MIN_FLOPS:
            mp, np, kp = ps
            # torch.empty instead of zeros: three large buffers fmem (~400MB at 8191^3 level) per call
            # ~0.8ms, pure waste. For valid outputs (i<m,j<n) the k-direction dot product only spans the k-pad
            # columns of A/B, so only the [:, k:kp] strip of the two buffers needs zeroing (see below); m-pad rows / n-pad rows
            # / the pad region of Cp only affect cropped outputs, no need to zero.
            Ap = torch.empty(mp, kp, dtype=A.dtype, device=A.device)
            Bt = torch.empty(np, kp, dtype=A.dtype, device=A.device)
            Cp = torch.empty(mp, np, dtype=A.dtype, device=A.device)
            if transa == CUBLAS_OP_N:  # A physical (m,k): same-layout pad
                Ap[:m, :k].copy_(_row_view(A, lda)[:m, :k])
            else:  # A physical (k,m): pad+transpose -> Ap(m,k)
                g = (
                    triton.cdiv(kp, _PAD_TRANS_BLK) * triton.cdiv(mp, _PAD_BLK),
                )
                _bfgemm_pad_trans_kernel[g](
                    A, Ap, k, m, lda, kp, mp,
                    BR=_PAD_TRANS_BLK, BC=_PAD_BLK, num_warps=8,
                )
            if transb == CUBLAS_OP_N:  # B physical (k,n): pad+transpose -> Bt(n,k)
                b_pad_trans = True
                g = (
                    triton.cdiv(kp, _PAD_TRANS_BLK) * triton.cdiv(np, _PAD_BLK),
                )
                _bfgemm_pad_trans_kernel[g](
                    B, Bt, k, n, ldb, kp, np,
                    BR=_PAD_TRANS_BLK, BC=_PAD_BLK, num_warps=8,
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
            if (m, n, k) in _NN_NT_ROUTE_SHAPES:
                # direction ①: B(k,n) (k row stride=ldb) -> Bt(n,k) (row stride=k), then run the NT main kernel
                Bt = torch.empty((n, k), dtype=B.dtype, device=B.device)
                gt = (
                    triton.cdiv(k, _TN_TRANS_BLOCK_K)
                    * triton.cdiv(n, _TN_TRANS_BLOCK_M),
                )
                _bfgemm_tn_trans_kernel[gt](
                    B,
                    Bt,
                    k,
                    n,
                    ldb,
                    k,
                    BR=_TN_TRANS_BLOCK_K,
                    BC=_TN_TRANS_BLOCK_M,
                    num_warps=8,
                )
                _bfgemm_nt_kernel[grid](
                    A, Bt, C, alpha, beta, m, n, k, lda, k, ldc, beta_is_zero,
                    **_kp2_kwargs(),
                )
            else:
                _bfgemm_nn_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
                    **_nn_kernel_kwargs(m, n, k),
                )
        elif transa == CUBLAS_OP_T and transb == CUBLAS_OP_N:
            if (m, n, k) in _TN_Y_SHAPES:
                # Y composite: transpose B(k,n) contiguously into Bt(n,k), then NN main loop + transposed store directly writes C
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
                    num_warps=8,
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
                # A(k,m) row-major (row length=lda) -> A^T(m,k) row-major (row length=k)
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
                    num_warps=8,
                )
                if (m, n, k) in _TN_NT_ROUTE_SHAPES:
                    # direction ① TN: additionally B(k,n)->Bt(n,k) (row stride=k), At+Bt run the NT main kernel
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
                        num_warps=8,
                    )
                    _bfgemm_nt_kernel[grid](
                        At, Bt, C, alpha, beta, m, n, k, k, k, ldc, beta_is_zero,
                        **_kp2_kwargs(),
                    )
                else:
                    _bfgemm_nn_kernel[grid](
                        At, B, C, alpha, beta, m, n, k, k, ldb, ldc, beta_is_zero,
                        **_nn_kernel_kwargs(m, n, k),
                    )
            else:
                _bfgemm_tn_kernel[grid](
                    A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
                )
        elif transa == CUBLAS_OP_N and transb == CUBLAS_OP_T:
            _bfgemm_nt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero,
                **_kp2_kwargs(),
            )
        else:
            _bfgemm_tt_kernel[grid](
                A, B, C, alpha, beta, m, n, k, lda, ldb, ldc, beta_is_zero
            )
