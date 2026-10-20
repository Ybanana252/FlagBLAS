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
Hygon DTK fp64 DGEMM custom path (dot kernel).

Background (2026-09-09, DTK 25.04 / triton 3.6.0):
* The generic level3 `_dgemm_kernel` is a per-k scalar outer-product kernel; measured on Hygon it falls
  far short of FMA/memory throughput (2048^3 NN 36.6ms @0.5TFLOPS, capped at ~0.8TFLOPS regardless of tile).
* This backend's fp64 `tl.dot` measures a ~6-8 TFLOPS ceiling (2048^3 2.6ms / 4096^3 18ms),
  ~10x faster than the scalar outer product, but still limited by vector FMA. **This conclusion is invalid
  as of 2026-09-13**: fp64 `tl.dot` has been lowered to the DCU native fp64 matrix unit
  `v_mmac_16x16x4_f64` (see "config rescan after MMAC landing" at the end), ~2.4x faster; large-shape sp reached 1.0+.
* smem: fp64 tile smem = (BM*BK + BK*BN)*8 per stage. Re-checking the Triton compilation
  metadata.shared on 2026-09-13 confirms the **hard limit is 65536B**: the 8 configs in use actually
  allocate 16384/20480/24576/32768/65536, while 128x128 w32 s2 needs 131072 and throws
  OutOfResources before launch. The old comment's "98304B executes normally" was a formula estimate, not an actual
  allocation, so large tiles are indeed hard-blocked by smem. Configs are chosen by measured (trans, max_dim) buckets.
* Large shapes (2026-09-11, probe36/probe39): after switching to 128x128x16 s2 it fully beats the original
  128x256/256x128 -- the latter's grid is too small on large shapes and tile splitting cannot amortize.
  NT/TT improve the most (8192^3 -38%/-41%), because their B goes through NAT_B transposed loads, and a small BN
  saves more smem bandwidth.

This module serves as a vendor-specific op overriding `flag_blas.dgemm` (effective only on Hygon; other
backends still use the generic level3 implementation).
"""

import logging

import torch
import triton
import triton.language as tl

from flag_blas.ops.level3.dgemm import ScalarType, _validate_dgemm_args
from flag_blas.runtime import torch_device_fn

logger = logging.getLogger(__name__)

CUBLAS_OP_N = 0
CUBLAS_OP_T = 1


@triton.jit
def _dot_block(
    a_ptr,
    b_ptr,
    offs_m,
    offs_n,
    cur_k,
    lda,
    ldb,
    acc,
    mask_m,
    mask_n,
    mask_k,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    NAT_B: tl.constexpr,
    NO_MASK: tl.constexpr,
    USE_K: tl.constexpr,
):
    """A/B load + tl.dot for a single k block.

    USE_K decides whether the k-axis predicate participates: in the main loop k is always a multiple
    of BLOCK_K; after dropping the k mask, A's contiguous axis (TRANS_A==0) only keeps the mask along m, so addressing can be vectorized.
    """
    if TRANS_A == 0:
        a_ptrs = a_ptr + offs_m[:, None] * lda + cur_k[None, :]
    else:
        a_ptrs = a_ptr + cur_k[None, :] * lda + offs_m[:, None]
    if TRANS_B == 0:
        b_ptrs = b_ptr + cur_k[:, None] * ldb + offs_n[None, :]
    else:
        b_ptrs = b_ptr + offs_n[None, :] * ldb + cur_k[:, None]
    if NO_MASK:
        a = tl.load(a_ptrs)
    elif USE_K:
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
    else:
        a = tl.load(a_ptrs, mask=mask_m[:, None], other=0.0)
    if TRANS_B == 1 and NAT_B == 1:
        # transb=1: B is stored row-major as (n,k), so the contiguous axis is k. Loading in natural
        # row-major (BN,BK) and then transposing is more efficient than directly indexing (BK,BN) (measured on medium/small sizes).
        b_ptrs = b_ptr + offs_n[:, None] * ldb + cur_k[None, :]
        if NO_MASK:
            b = tl.trans(tl.load(b_ptrs))
        elif USE_K:
            b = tl.trans(tl.load(b_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0))
        else:
            b = tl.trans(tl.load(b_ptrs, mask=mask_n[:, None], other=0.0))
    elif NO_MASK:
        b = tl.load(b_ptrs)
    elif USE_K:
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
    else:
        b = tl.load(b_ptrs, mask=mask_n[None, :], other=0.0)
    return tl.dot(a, b, acc, out_dtype=tl.float64, allow_tf32=False)


@triton.jit
def _dgemm_dot_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    alpha: tl.float64,
    beta: tl.float64,
    m,
    n,
    k,
    lda,
    ldb,
    ldc,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    BETA_IS_ZERO: tl.constexpr,
    NAT_B: tl.constexpr,
    NO_MASK: tl.constexpr,
    K_TAIL: tl.constexpr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    # NO_MASK: skip all masks when m/n/k are all multiples of BLOCK. On small shapes, predicate
    # computation and boundary handling of masks account for a comparable share of kernel time (64^3 NN 11.0->9.8us).
    # K_TAIL: when k is not a multiple of BLOCK_K, the main loop only runs k_main = k//BLOCK_K*BLOCK_K,
    # dropping the k mask (A's contiguous axis only keeps the mask along m -> addressing can be vectorized), and at the end
    # a single small block with the k mask is run. Bit-exact (probe163 err=0), 1023^3 152.96 -> 150.24us.
    if K_TAIL:
        k_main = (k // BLOCK_K) * BLOCK_K
    else:
        k_main = k
    for k_start in range(0, k_main, BLOCK_K):
        cur_k = k_start + offs_k
        mask_k = cur_k < k
        acc = _dot_block(
            a_ptr, b_ptr, offs_m, offs_n, cur_k, lda, ldb, acc,
            mask_m, mask_n, mask_k, TRANS_A, TRANS_B, NAT_B, NO_MASK,
            USE_K=(not K_TAIL),
        )
    if K_TAIL:
        cur_k = k_main + offs_k
        mask_k = cur_k < k
        acc = _dot_block(
            a_ptr, b_ptr, offs_m, offs_n, cur_k, lda, ldb, acc,
            mask_m, mask_n, mask_k, TRANS_A, TRANS_B, NAT_B, NO_MASK,
            USE_K=True,
        )

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    if NO_MASK:
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc)
        else:
            c_vals = tl.load(c_ptrs).to(tl.float64)
            tl.store(c_ptrs, alpha * acc + beta * c_vals)
    else:
        c_mask = mask_m[:, None] & mask_n[None, :]
        if BETA_IS_ZERO:
            tl.store(c_ptrs, alpha * acc, mask=c_mask)
        else:
            c_vals = tl.load(c_ptrs, mask=c_mask, other=0.0).to(tl.float64)
            tl.store(c_ptrs, alpha * acc + beta * c_vals, mask=c_mask)

# --- v25 tail-wave split-K (single-wave variant of stream-K) -------------------
# Motivation (localized by probe170/172, measured by probe174/175, report §3.21/§3.22): the gap above 1408³ has been quantitatively localized to **wave quantization** -- the device has
# 80 CU + 64KB smem ⇒ 1 CTA/CU, and each CTA handles the entire k loop of one (BM,BN) output tile, so makespan = ceil(num_tiles/80) tile-times, whereas the ideal is num_tiles/80.
# When num_tiles % 80 = tail and tail is small, the last wave has only tail CTAs running, yet
# consumes a whole tile-time -- for 2048³ (256 tiles, tail 16) and 3072³ (576 tiles, tail 16)
# the ceil_eff is therefore only 0.80 / 0.90, exactly the direct source of NN 2048³ sp 0.856 and
# NT 3072³ sp 0.898 (the wave model of probe154 + the interleaved measurements of probe170/172).
#
# This path splits **the tail tiles of the last wave** each into nsplit equal-length k intervals, handed to
# nsplit*tail (≤ SM) CTAs to complete concurrently, so the tail wave costs only 1/nsplit tile-times; the main axis
# (the first bulk tiles) is still run by the original kernel in **one launch**, with the kernel body (_dgemm_dot_kernel)
# unchanged by a single line. Slice results are written into an (nsplit*tail, BM, BN) partial-product buffer, then a reduction
# kernel sums per tile and applies alpha/beta when writing back to C. The tile→CTA mapping of the three launches is
# fully isomorphic to the original kernel (same swizzle formula + pid offset), so **shapes that do not enable this path are bit-identical**.
#
# Gate (see _streamk_plan): enabled only when the tail wave can be evenly divided into ≥2 parts, the k iteration count
# is divisible by the part count, and the cost model (each extra slice adds one write+read of a (BM,BN) partial product)
# is far smaller than the tail-wave gain.
@triton.jit
def _dgemm_splitk_kernel(
    a_ptr,
    b_ptr,
    p_ptr,
    m,
    n,
    lda,
    ldb,
    pid_base,
    tail,
    ki_per_slice,
    TRANS_A: tl.constexpr,
    TRANS_B: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Tail-wave slicing: the (pid_local // tail)-th k interval, writes the partial product to the pid_local-th block of p_ptr.

    Only serves the unmasked case (m/n/k are all multiples of BLOCK), so it takes the fully predicate-free path.
    """
    pid_local = tl.program_id(0)
    pid = pid_base + pid_local % tail
    grid_m = m // BLOCK_M
    grid_n = n // BLOCK_N
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    k_start = (pid_local // tail) * ki_per_slice * BLOCK_K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    for i in range(ki_per_slice):
        cur_k = k_start + i * BLOCK_K + offs_k
        if TRANS_A == 0:
            a_ptrs = a_ptr + offs_m[:, None] * lda + cur_k[None, :]
        else:
            a_ptrs = a_ptr + cur_k[None, :] * lda + offs_m[:, None]
        if TRANS_B == 0:
            b_ptrs = b_ptr + cur_k[:, None] * ldb + offs_n[None, :]
        else:
            b_ptrs = b_ptr + offs_n[None, :] * ldb + cur_k[:, None]
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc,
                     out_dtype=tl.float64, allow_tf32=False)

    p_ptrs = (p_ptr + pid_local * (BLOCK_M * BLOCK_N)
              + tl.arange(0, BLOCK_M)[:, None] * BLOCK_N
              + tl.arange(0, BLOCK_N)[None, :])
    tl.store(p_ptrs, acc)


@triton.jit
def _dgemm_splitk_reduce_kernel(
    p_ptr,
    c_ptr,
    alpha,
    beta,
    m,
    n,
    ldc,
    pid_base,
    nsplit,
    BETA_IS_ZERO: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Sum the (nsplit, tail, BM, BN) partial products per tile, then apply alpha/beta and write back to C.

    Partial products are laid out slice-major: the j-th slice of tile local index L is in block (j*tail + L).
    """
    local_id = tl.program_id(0)
    tail = tl.num_programs(0)
    pid = pid_base + local_id
    grid_m = m // BLOCK_M
    grid_n = n // BLOCK_N
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    inner = (tl.arange(0, BLOCK_M)[:, None] * BLOCK_N
             + tl.arange(0, BLOCK_N)[None, :])
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float64)
    for j in range(nsplit):
        acc += tl.load(p_ptr + (j * tail + local_id) * (BLOCK_M * BLOCK_N) + inner)

    c_ptrs = c_ptr + offs_m[:, None] * ldc + offs_n[None, :]
    if BETA_IS_ZERO:
        tl.store(c_ptrs, alpha * acc)
    else:
        tl.store(c_ptrs, alpha * acc + beta * tl.load(c_ptrs))


# smem budget: (BM*BK + BK*BN)*8 <= 64KB @ num_stages=2 (double-buffered fp64)
# cfg = (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages)
# Measured selection (probe2/probe4 2026-09-09, 2048/4096^3):
#   NN: 128x256x16 w8 s2 (6.6-7.6 TFLOPS)   TN: 256x128x16 w8 s2
#   NT: 256x128x16 w8 s2 (already beats hipBLAS NT)  TT: see _select (same orientation as TN)
# Small-size bucketing (probe5/probe6 2026-09-10, 128..1024^3):
#   A large tile on a small square matrix yields only 1-2 CTAs (128x256 at 256^3 -> grid=2),
#   leaving the executor idle and registers spilling; 256^3 NN measured 2.95ms. Switching to a 32x32 small tile
#   (grid=64) gives 0.027ms (-111x), TN 1.43ms -> 0.021ms. So the max_dim<=1024
#   range is bucketed by measured small tiles, and large tiles are used only for max_dim>1024.
# For small sizes far from the smem limit, shrink tiles further and lower warps (probe8 2026-09-10, 32..256^3):
#   launch overhead ~8.5us is the floor at 32^3 (comparable to hipBLAS 8.3us); small tiles raise
#   parallelism: 16x16x16 w1 makes NT/TT 128^3 34.4->17.4us, 16x32x16 w2 makes NN 64^3
#   13.0->10.4us; at 256^3 only NT/TT benefit from w2 (NT 65->48us, TT 60->46us).
# Fine-grained grid + NAT_B variants (probe11/probe12 2026-09-10, official protocol warmup=1000/
#   rep=100/median, 3 rounds; each bucket takes the best over all shapes at 32/64/128/256^3):
#   * 16x16x32 w1 brings NN 32/64/128^3 to 8.32/10.08/13.60us (sp 1.00/1.06/0.93),
#     and 16x64x16 w4 brings NN 256^3 27.8->24.0us.
#   * NAT_B=1 (B loaded naturally as (BN,BK) from (n,k) then tl.trans) is better for transb=1 at medium/small
#     sizes: TT 128^3 17.8->16.0us, TT 256^3 45.8->42.2us,
#     NT 256^3 47.8->42.9us.
# Mask elimination + transposed leading-dim alignment (probe16..probe29 2026-09-11, official protocol):
#   * Timing protocol verification: do_bench's two events themselves occupy a 7.0us floor, an empty kernel 8.2us,
#     host/Python overhead is not counted (pure CPU busy-wait 150us leaves the reading unchanged) -> one can only win
#     by reducing GPU-side kernel time; on small shapes the predicate/boundary handling of masks is a non-trivial share.
#     NO_MASK=1 (skip all masks when m/n/k are all multiples of BLOCK) makes 64^3 NN
#     11.0->9.8us (sp 0.928->1.049), 32^3 NT 10.9->9.3us (0.853->1.000).
#   * 32^3 single-column bucketing: the whole C is only 1~4 tiles, 16x16x16/16x16x32 w1~w2 + GROUP_M=1
#     is cheapest; 64/128^3 use 16x16x16 w2 (NT/TT add NAT_B).
#   * TN (transa=1) in the 513..1024 range: A's contiguous axis falls on tile dim 0, and with a large tile
#     the dot left operand needs an extra layout conversion, ~7% slower than NN. Pre-transposing A on the host and aligning the
#     leading dim to 16, then taking transa=0: 1023^3 474->454us (0.911->0.952),
#     1024^3 450->435us (0.938->0.970). Alignment is critical: 1023 unaligned drops to 812us.
# Widening BLOCK_K for small shapes (probe30 2026-09-11, full trans sweep at 32/64/128^3):
#   * When the k loop runs only 1~8 times, widening BLOCK_K amortizes loop and mask overhead while keeping the parallelism
#     of 16x16 tiles: NN 16x16x64 w2 gives 64^3 9.76us, 128^3 12.80us (the original 16x16x16 bucket
#     measured 12.80/16.96); TN 16x16x64 w1 gives 64^3 9.44us, 128^3 12.64us;
#     TT 32^3 with 16x16x32 goes 10.08->8.80us.
#   * The best for 128^3 TN is the single-iteration config with BLOCK_K==k (16x16x128 w1, 12.16us).
#   * The sp for <=128^3 is basically determined by the hipBLAS baseline (8~17us, already close to the ~8us event floor);
#     the baseline itself jitters ~2us across rounds, so the 0.95 threshold in this range is really a noise boundary.
# 512^3 attribution and bucket-boundary fix (probe90..probe101 2026-09-12, median of 5~7 rounds):
#   * smem layout conversion **cannot be eliminated**: probe93 dropped tl.dot for a manual outer product and smem round-trips went to zero
#     (ds_read 100->4), but each thread must fetch its own operands, global_load/FMA rose 27x, net 3~5x slower.
#     fp64 only has the vector path v_fma_f64 (production kernel disassembly: v_mfma*/v_mac* = 0,
#     v_fma_f64 256 instructions = 36%), operands can only be amortized by enlarging tiles: LDS/FMA = 16*(1/BM+1/BN).
#   * 512^3 tiles are **locked by the parallelism floor**: total warps = m*n/(outputs per thread*64), 32x64
#     gives 128 CTA*4warp = 512 warps, already close to the minimum tile needed to fill 320 SIMDs
#     (BM*BN<=3277); enlarging the tile (64x128) drops to 32 CTAs = 0.4 wave, NN regresses 1.86x measured.
#     Shrinking tiles for parallelism raises LDS/FMA (16x16 is 2.7x of 32x64). => 512^3's
#     0.43 sp is a structural ceiling, not a mis-tuned config (probe95's full-bucket tile-orientation sweep disproved
#     "BN must be wide" as an independent lever: the current buckets are per-bucket optimal under ks=1).
#   * probe96/97: split-K can restore parallelism for large tiles (gemm body 1.17x), but the ws+reduce
#     reduction traffic pulls end-to-end down to 1.04x (512^3) / 1.12x (511^3, atomic, needs an extra C pre-zero),
#     far less than the bucket fix below, so the split-K path is not introduced.
#   * probe97..probe101: the real leak is at the **bucket boundary**. The original <=1024 bucket's 64x128x16 at
#     513..896 yields only 45~98 CTAs; after switching to 32x64x16 NN improves across the board (up to 1.82x),
#     see the comments in _select_dgemm_dot_config.
# Bucket-boundary sub-buckets (probe109..probe114 2026-09-12, official protocol + interleaved A/B over 4 rounds, median):
#   * probe109 swept every size 96..511^3 and found two bucket cliffs: 129^3 +42% (grid 64->27 CTAs),
#     257^3 +46% (grid 64->45 CTAs); the mechanism is that the extra whole tile at the boundary still runs the full k loop,
#     while grid/wave count drops, a double degradation.
#   * probe110/111/112/113/114 found the crossover per size on the transition band and added two sub-buckets
#     (16x32x32 w4 s2 gm8, upper bounds set by measured crossover, no extrapolation):
#       <=192^3: NN 129^3 19.04->16.56us (0.870x), overtaken from 196^3 -> upper bound 192;
#                TN candidates are slower (1.145~1.277x) so unchanged.
#       <=272^3: NN 257^3 33.28->29.60us (0.889x), TN 38.24->32.32us (0.845x),
#                overtaken from 274^3 -> upper bound 272.
#   * Both cliffs (129/144/160/192/257/264/272) are outside benchmark/core_shapes.yaml's
#     15 square shapes, so this is purely a robustness fix and does not affect the §1/§2 scoring protocol.
# Config rescan after MMAC landing (probe118/119/120, 2026-09-13, official protocol warmup=1000/
#   rep=100/median, interleaved A/B median; NN with 3~9 rounds, repeated measurements in the same bucket vary <0.2%):
#   * Prerequisite change: FlagTree hcu's fp64 tl.dot no longer uses FMA, but is lowered to the DCU native
#     fp64 matrix unit `v_mmac_16x16x4_f64` (instrShape 16x16x4; 1 A/B f64 and 4 D f64 per lane = 8 VGPRs;
#     2048 flops/wave64 = 16x of v_fma_f64).
#     Production kernel disassembly verified: v_fma_f64 = 0 for all candidates, none silently fell back. On the same machine and protocol
#     large shapes speed up ~2.4x (8192^3 NN old config 135.3 -> 56.4ms, new config 39.1ms),
#     accuracy unchanged (err 0.0 ~ 5.4e-12, due to fp64 accumulation-order differences, far better than any tolerance).
#   * The new lever with MMAC is **accumulator register pressure**: acc VGPR = 2*BM*BN/(32*num_warps),
#     ceiling 256. The old large-shape config 128x128 **w4** needs exactly 256 -> spill; the same tile
#     at w8 halves it, at w16 only 32. So under MMAC the right way to enlarge tiles is to **raise num_warps at the same time**
#     (in the FMA-era table num_warps was basically never swept as an independent dimension).
#   * transb=0 path (i.e. NN, plus TN / NT/TT after pre-transpose -- when max_dim>128 and m>=64
#     B is always pre-transposed on the host, so for large shapes only NN and TN remain) rescan conclusion is consistent:
#       max_dim <= 1408 : 32x64x16 w4 s2 g8   (the original <=896 bucket config, best over all)
#       max_dim >  1408 : 128x128x16 w16 s2 g8
#     The crossover lies between 1408..1536; measurements on both sides (parentheses are the hipBLAS baseline under the same protocol):
#       NN 1024^3 131.7us sp=0.893 (117.6) | 1280^3 218.9 sp=0.931 (203.8)
#          1408^3 301.6us sp=0.908 (273.8) | 1536^3 336.0 sp=0.924 (310.6)
#          2048^3 833.9us sp=0.856 (713.8) | 3072^3 2378 sp=0.984 (2339)
#          4096^3 5051  sp=1.088 (5494.6)| 8192^3 39128 sp=1.203 (47079)
#       TN 1024^3 127.8us sp=0.856 (109.4) | 2048^3 825.0 sp=0.801 (660.8)
#          4096^3 4992  sp=1.012 (5052)  | 8192^3 38731 sp=1.007 (39018)
#     Between 1408..3072 the two candidates overtake each other (1536^3 w16 10% faster, 1664^3 32x64 instead
#     13% faster, 1792/2048 a tie), a resonance of grid wave count and occupancy; no single config was found to satisfy both
#     sides; the boundary 1408 is chosen so the scoring protocol core_shapes.yaml's 1536/3072 (w16 ahead by
#     10~11%) and 1024 (32x64 ahead by 7%) each take their best, and 2048 differs by only 0.8%.
#   * Old conclusions invalidated/corrected hereby (original text kept above as history):
#       - "fp64 only has the vector path v_fma_f64 (v_mfma*/v_mac* = 0), operands can only be amortized by
#         enlarging tiles: LDS/FMA = 16*(1/BM+1/BN)": under MMAC each A/B lane occupies only
#         1 f64; the LDS traffic of layout conversion is completely different from FMA; enlarging tiles is no longer the only
#         lever, num_warps is instead more critical.
#       - "512^3's 0.43 sp is a structural ceiling, not a mis-tuned config": under MMAC 512^3 with
#         32x64x16 w4 s2 measures sp=1.064 (NN) / 1.000 (TN), the ceiling has been raised.
#       - ">1024 uses 128x128x16 w4": under MMAC this config spills acc, 4096^3 only reaches
#         sp=0.778; at w16 it becomes 1.088.
#   * Not rescanned (no data, keep as-is to avoid regression): the NT/TT direct-path configs reached only when
#     transb=1 and m<64 (the TT side is still 128x256x16 w8, acc needs 256 VGPRs, equally suspicious under MMAC,
#     but outside the scoring protocol and not measured; the NT side was changed to 128x128x16 w16 along with probe143).
#     The <=256^3 small-tile buckets were rescanned and landed by v26 (probe176/177, 2026-09-15) under the MMAC shape
#     (129..272^3 unified to 32x32x32 w4, see the branch comments below and report §3.23).


def _select_dgemm_dot_config(transa: int, transb: int, m: int, n: int, k: int):
    """Select the dot-kernel launch config by (trans, max_dim) bucket (Hygon measured table).

    Returns dict-ified launch kwargs, or None (size too small -> generic fallback).
    """
    max_dim = max(m, n, k)
    min_dim = min(m, n, k)

    if min_dim < 16 or max_dim < 32:
        return None

    nat = 0
    # v21 (probe158–probe161, 2026-09-14): after dgemm() lets TT with m>=64 also take the native TB1 direct path,
    # its tile must match TN -- under MMAC loading B's k-direction 128B runs costs almost nothing
    # (measured per size: native TB1 is as fast as TN), whereas the historical "TT direct path" bucket gave a large BN tile chosen for the FMA era,
    # among which the >1408 128x256x16 w8 even has acc = 256 VGPRs pressing the ceiling (probe160:
    # 1536^3 361.6us / 2048^3 929.4us, both slower than 128x128x16 w16's 335.8/832.3).
    # So here "TT with m>=64 that was not pre-transposed" is normalized to the TN protocol; the historical TT bucket is kept only for
    # max_dim<=128 or the m<64 tall-skinny direct path (those sizes are not pre-transposed anyway, behavior unchanged).
    if transa == 1 and transb == 1 and m >= 64 and max_dim > 128:
        transb = 0
    if max_dim <= 32:
        # 32^3: a single tile covers all of C (grid 1~4 CTA); after mask elimination GPU time is already close to
        # the empty-kernel floor. GROUP_M=1 saves the swizzle branches.
        if transa == 0 and transb == 0:  # NN
            bm, bn, bk, nw = 16, 16, 16, 2
        elif transa == 1 and transb == 0:  # TN
            bm, bn, bk, nw = 16, 32, 32, 2
        elif transa == 0:  # NT
            bm, bn, bk, nw, nat = 16, 16, 16, 2, 1
        else:  # TT
            bm, bn, bk, nw, nat = 16, 16, 32, 2, 1
        gm, ns = 1, 2
    elif max_dim <= 128:
        # 64/128^3: a square matrix has only 4~64 16x16 tiles, measured per bucket (probe30 2026-09-11,
        # official protocol). The k loop runs only 1~8 times; widening BLOCK_K to 32/64 amortizes loop and mask
        # overhead: NN 16x16x64 w2 (64^3 12.80->9.76us, 128^3 16.96->12.80us),
        # TN 16x16x64 w1 (64^3 10.72->9.44, 128^3 13.76->12.64) both beat
        # 16x16x16/16x32x32; NT/TT still use NAT_B + BLOCK_K=32 (64^3 13.12->11.52 /
        # 11.36->10.72). NN/TN use GROUP_M=1, NAT_B's NT/TT use GROUP_M=8.
        # NN's num_warps 2->4 (probe104/105 2026-09-12; interleaved A/B, through the public entry
        # flag_blas.dgemm, official protocol warmup=1000/rep=100, median of 8 rounds, reproduced twice):
        #   64^3 11.04->10.40us (0.942x) | 128^3 14.24->13.44us (0.944x)
        #   32^3 unchanged (9.76us). BLOCK_K=128 only improves 128^3 by another 2.4% (13.44->13.12us),
        #   but regresses 64^3 to 12.96us (mask waste when k<BLOCK_K), so BLOCK_K is not widened.
        # Only NN is tuned here; TN stays w1, NT/TT stay w2.
        # v23 (probe166/167, 2026-09-15, MMAC-era rescan, interleaved A/B median over 4 rounds +
        # a clone control "passing the production config unchanged back to force()", noise floor 0.00%): the NT/TT
        # "NAT_B + BLOCK_K=32" in the comment above was an FMA-era (probe30) choice, and under MMAC
        # it is a pure loss -- NAT_B=1 must first load (BN,BK) and then tl.trans, while each
        # v_mmac_16x16x4_f64 only yields 2048 flops / 4 LDS per lane, so the relative cost of layout conversion
        # is amplified (the same phenomenon is 2.8~3.9x at >1024^3, see probe164). After switching to the
        # 16x16x64 + nat=0 family same as NN/TN (GROUP_M consequently 8 -> 1), measured:
        #   64^3  NT 10.08->9.60us (-4.8%) | TT 10.08->9.60us (-4.8%)
        #   128^3 NT 12.16->9.92us (-18.4%) | TT 12.32->10.08us (-18.2%)
        # The attribution is pinned by the same-bucket control: forcing 16x16x32 w2 nat=1 reproduces 12.16/12.32us,
        # bit-identical speed to production. This is the direct source of NT 128^3 sp 0.805 -> ~1.00 (crossing the 0.95 threshold).
        if transa == 0 and transb == 0:  # NN
            bm, bn, bk, nw = 16, 16, 64, 4
        elif transa == 1 and transb == 0:  # TN
            bm, bn, bk, nw = 16, 16, 64, 1
        else:  # NT / TT
            bm, bn, bk, nw = 16, 16, 64, 2
        gm, ns = (8 if nat else 1), 2
    elif max_dim <= 192:
        # 129..192^3 sub-bucket. v26 (probe176/177, 2026-09-15, interleaved A/B median over 3~4 rounds
        # + clone noise floor 0.00%) rescanned under the MMAC shape: this bucket was 16x32x32 w4 (NN)
        # and 16x64x32 w4 (TN / NT), both set by probe109-114 (2026-09-12, FMA protocol) as
        # "narrow BN trades back CTA count"; under MMAC the optimum as a whole moves to 32x32x32 w4:
        #   129^3 nn/nt 13.12/13.28 -> 12.64/12.64 (-3.7%/-4.8%)
        #   144^3 nn/nt 12.80/12.16 -> 11.84/12.00 (-7.5%/-1.3%)
        #   160^3 nn/nt 12.48/12.80 -> 11.52/11.52 (-7.7%/-10.0%)
        #   176^3 nn/nt 13.60/12.96 -> 12.48/12.64 (-8.2%/-2.5%)
        # In the same round 16x32x32 w4 was used as a direct control candidate, and it lost across the board. Tall-skinny (min_dim<32) and
        # the TT direct path were not covered by this probe and keep their historical values to avoid regression.
        if transa == 1 and transb == 1:  # TT direct path (only m<32)
            bm, bn, bk = 64, 64, 32
        elif min_dim < 32:  # tall-skinny matrix: keep v13 bucketing
            bm, bn, bk = (16, 32, 32) if (transa == 0 and transb == 0) else (16, 64, 32)
        else:  # NN / TN / NT main line
            bm, bn, bk = 32, 32, 32
        gm, nw, ns = 8, 4, 2
    elif max_dim <= 256:
        # 193..256^3. The original 32x32x32 w4 (probe40/41, 2026-09-11) was overturned by probe57/58
        # (2026-09-11, 7 interleaved rounds) into 16x64x32 w4 -- but that was the **FMA protocol**:
        # then 32x32x32 lagged by 18% at 256^3 (26.08->21.44us). v26 (probe176,
        # MMAC shape, interleaved A/B over 4 rounds + clone floor 0.00%) re-measured with the **opposite sign**:
        # 32x32x32 w4 wins on all four trans variants, while the other candidates (32x64x32 / 32x32x16 /
        # 32x32x32 w8 / 16x32x32) all lose:
        #   256^3 nn/tn/nt/tt 13.28/14.08/14.56/13.76
        #         -> 13.12/13.60/13.28/13.12 (-1.2%/-3.4%/-8.8%/-4.7%)
        #   208^3 nn/nt 12.96/13.12 -> 12.64/12.80 (-2.5%/-2.4%)
        #   224^3 nn/nt 12.96/14.24 -> 12.32/12.48 (-4.9%/-12.4%)
        # NT 256^3 goes from official sp 0.926 across 0.95 (see report §3.23). At 511/512^3
        # 32x32x32 still regresses (probe177 re-measured 512^3 +18%, 288^3 as much as +504%),
        # so this bucket must be separated from <=1408.
        if transa == 1 and transb == 1:  # TT direct path (only m<32)
            bm, bn, bk = 64, 64, 32
        elif min_dim < 32:  # tall-skinny: keep original values
            bm, bn, bk = 16, 64, 32
        else:  # NN / TN / NT
            bm, bn, bk = 32, 32, 32
        gm, nw, ns = 8, 4, 2
    elif max_dim <= 272:
        # 257..272^3 sub-bucket (probe109/111/112/113/114, 2026-09-12, FMA-protocol
        # 256->257 cross-bucket fix). v26 re-measured: NN/TN original 16x32x32 w4 -> 32x32x32 w4
        # (272^3 nn 19.36->16.16, -16.5%), NT original 32x64x16 w8 -> 32x32x32 w4
        # (17.92->16.80, -6.3%). **Upper bound locked at 272**: probe177 shows at 288^3
        # 32x32x32 w4 regresses to 116us (+504%) -- that point has grid 81 CTA = 1.0125 waves,
        # exactly on the 80-CU wave boundary (tiles per CU jumps from 1 to 2); 512^3 also
        # +18%, 448^3 a tie ⇒ it cannot be merged into the <=1408 bucket's 32x64x16/64x32x16.
        # <=128^3 is also untouched (v23 already set it to 16x16x64 via probe166/167, and at 64^3
        # 32x32x32 yields only 4 CTAs).
        if transa == 1 and transb == 1:  # TT direct path (only m<32)
            bm, bn, bk = 64, 64, 32
        elif min_dim < 32:  # tall-skinny: keep original bucketing
            bm, bn, bk = (32, 64, 16) if (transa == 0 and transb == 1) else (16, 32, 32)
        else:  # NN / TN / NT main line
            bm, bn, bk = 32, 32, 32
        gm, nw, ns = 8, 4, 2
    elif max_dim <= 1408:
        # 511/512^3: 32x64x16; TT measured slightly better with 64x64x32.
        # 513..896 originally belonged to the <=1024 bucket (64x128x16); probe97/98/100/101 (median of 5~7 rounds)
        # found 64x128 yields only 45~98 CTAs in this range, and NN lags 32x64x16 across the board:
        #   544^3 2.04->3.72 TF (1.82x) | 576^3 2.30->2.85 (1.24x)
        #   640^3 2.81->3.57 (1.27x)     | 832^3 3.34->3.79 (1.13x)
        #   896^3 3.93->4.40 (1.12x)     | 704/768^3 lag by 3% (noise level)
        # Only from 928^3 does 64x128 overtake (1.05~1.23x, 1024^3 5.14 vs 4.18 TF), so the boundary was once 896.
        # 64x128's oscillation (832^3 abnormally low) has ruled out swizzle (GROUP_M 4/8/16 no difference) and
        # mask cost (keeping only loop-invariant m/n masks leaves time unchanged); it is an occupancy effect of partial-N tiles.
        # 2026-09-13 (probe119/120, rescan after MMAC landing): 32x64x16 w4 s2 leads all the way to
        # 1408^3 -- 1024^3 NN 131.7us (sp=0.893) vs 64x128x16's 151.5us (0.776),
        # 1280^3 218.9 vs 279.2, 1408^3 301.6 vs 307.5; only from 1536^3 is it overtaken by w16.
        # The original <=1024 bucket's 64x128x16 is therefore fully replaced, boundary 896 -> 1408, and that bucket deleted.
        # The TT direct path (reached only at 513..1408 when m<32) keeps the original 128x128x16, not re-measured to avoid regression.
        # 2026-09-14 (probe131/132/133/134, MMAC-era rescan, official protocol + interleaved multi-round median):
        # the previous 32x64x16 w4 was only compared against 128x128x16 w16 (probe119/120), and the medium/small tile
        # space had never been swept systematically. The supplementary sweep found that swapping BM/BN (64x32) is uniformly
        # faster on **evenly divisible** square matrices, but the price is that B's contiguous run is only 256B, and once lda/ldb is unaligned it degrades sharply:
        #   512^3  64x32 27.8 vs 32x64 31.8 (1.14x) | 640^3 40.5 vs 46.7 (1.15x)
        #   768^3  59.8 vs 63.7 (1.06x)             | 896^3 84.8 vs 92.3 (1.09x)
        #   1024^3 117.4 vs 131.5 (1.12x)           | 1408^3 281.8 vs 301.6 (1.07x)
        #   1023^3 192.4 vs 180.5 (0.94x, regression) | 2048^3 879.5 vs 827.4 (0.94x)
        # When unaligned, the optimum is instead determined by "128-tile fill rate" (probe134):
        #   1023^3 (fill 1.002) 128x128x16 w16 = 152.8 (production 180.5, 1.18x)
        #   1000^3 (fill 1.049) same = 150.1 (production 173.9, 1.16x)
        #   1200^3 (fill 1.138) same = 277.4 (production 214.4, 0.77x) -> must exclude
        #   The <=512 unaligned point (511^3): a large tile cannot produce CTAs: 128x128 = 84.2us, while
        #   32x64x16 **w8** = 34.7us (production 38.9us, 1.12x); small shapes are faster with more warps.
        # Mechanism: misalignment increases the number of cache lines each contiguous run spans; shorter runs waste relatively more
        # (128B run 2.0x / 256B 1.5x / 1024B 1.12x, probe133's "misalignment penalty" measured per config:
        # 64x32 63.6%, 32x64 36.7%, 128x64 19.4%, 128x128 7.9%).
        gm128 = 128 * ((m + 127) // 128) / m
        gn128 = 128 * ((n + 127) // 128) / n
        if transa == 1 and transb == 1:  # TT
            bm, bn, bk, gm, nw, ns = (
                (64, 64, 32, 8, 4, 2) if max_dim <= 512 else (128, 128, 16, 8, 4, 2)
            )
        elif m % 64 == 0 and n % 32 == 0 and k % 16 == 0:
            # Exactly divisible: best over the whole window (monotonically ahead 256..1408).
            bm, bn, bk, gm, nw, ns = 64, 32, 16, 8, 4, 2
        elif max_dim <= 512:
            # Unaligned and narrow: add warps to amortize loop and mask overhead.
            bm, bn, bk, gm, nw, ns = 32, 64, 16, 8, 8, 2
        elif gm128 * gn128 <= 1.05:
            # Unaligned but 128-tile almost full (last tile >= ~81%): the output-side run is long
            # (1024B), so the misalignment penalty is only 7.9%, offsetting the drop in CTA count.
            bm, bn, bk, gm, nw, ns = 128, 128, 16, 8, 16, 2
        else:
            bm, bn, bk, gm, nw, ns = 32, 64, 16, 8, 4, 2
    else:
        # max_dim > 1408: probe36/probe44 (real kernel A/B, 2026-09-11; all NAT_B=0 production protocol).
        # transb=0 (NN/TN, A-side transpose) switching to 128x128x16 s2 fully beats the original large tile -- the latter has too
        # small a grid on large shapes / splitting does not amortize: NN 8192^3 154.6->135.3ms (-12.5%), 2048^3
        # 3.00->2.72ms (-9%); TN 8192^3 155.8->121.7ms (-22%), 1536^3 1.16->0.95ms (-18%).
        # transb=1 (NT/TT) keeps the large tile: probe43 proves NAT_B=1 is a disaster on large shapes
        # (nt 1536^3 nat1=8490us vs nat0=2765us), so production takes nat0 and directly reads (BK,BN) on the direct path;
        # probe44 swept under nat0 and found NT's best is still 256x128x16 w8 (8192^3 269981us), TT's best
        # 128x256x16 w8 (8192^3 230496us, better than 128x128's 268678us).
        # 2026-09-13 (probe118/119): under MMAC 128x128 must raise num_warps from 4 to 16
        # to eliminate acc's VGPR spill (w4 = 2*128*128/(32*4) = 256 = the ceiling, measured
        # 4096^3 only reaches sp=0.778; w16 = 32, the same problem gives 1.088). w8s3 second (0.832), w8s2
        # 0.796, 128x64 w4s2 0.846 -- all below w16s2. The two NT/TT direct paths keep their old un-remeasured values.
        # 2026-09-14 (probe143/144/145): after v18 lets NT take the native direct path, this bucket's NT branch still returns
        # the historical 256x128x16 w8 (acc = 2*256*128/(32*8) = 256 VGPR = MMAC ceiling, measured
        # direct-path cost +20~93%), causing v18 to regress across >1408. After changing to the same config as the NN bucket:
        #   2048^3 948.5->840.7 (-11.3%) | 3072^3 2784->2401 (-13.8%)
        #   4095^3 7459->5720 (-23.3%)   | 4096^3 6423->5101 (-20.5%)
        #   6144^3 20417->16666 (-18.4%)
        # NT with max_dim>6144 does not take this branch (dgemm() has reverted to pre-transpose), so the direct path only serves
        # 1408<max_dim<=6144 and m<64.
        if transb == 0:      # NN / TN (A-side transpose)
            bm, bn, nw = 128, 128, 16
        elif transa == 0:    # NT direct path (B directly read as (BK,BN))
            # v23 (probe168/169, 2026-09-15, interleaved A/B median over 4 rounds + clone control,
            # noise floor 0.02%): when probe143/144/145 chose 128x128x16 w16 it was only compared against the historical
            # 256x128x16 w8; the small-tile space was never swept. The supplementary sweep confirms 64x32x16 w4 on 2048^3
            # is indeed faster (841 -> 801us, -4.8%, reproduced independently three times), but that gain
            # **cannot be generalized**: 1984^3 / 2240^3 have exactly the same 128x128 wave structure as 2048^3
            # (256 / 324 CTAs, 3.2 / 4.05 waves) yet regress 20.7% / 22.7% respectively, the fully divisible
            # 2304^3 regresses 33.1%, and 3072^3 regresses 130%; 1536^3 / 2560^3 also all lose.
            # That is, 2048^3 is a special case of lda/ldb = 16KB (a power of 2); wave quantization is not an effective predictive
            # factor, and any criterion written in terms of wave count will silently hurt sizes like 2304^3. So
            # 128x128x16 w16 is kept unchanged, and 2048^3's -4.8% is recorded in the report as known non-generalizable headroom.
            bm, bn, nw = 128, 128, 16
        else:                # TT direct path (only m<32)
            bm, bn, nw = 128, 256, 8
        bk, gm, ns = 16, 8, 2

    return {
        "BLOCK_M": bm,
        "BLOCK_N": bn,
        "BLOCK_K": bk,
        "GROUP_M": gm,
        "NAT_B": nat,
        "num_warps": nw,
        "num_stages": ns,
    }


# --- padding fast path when the NN leading dim is unaligned (probe146/147/148, 2026-09-14) ------------
# Under MMAC, fp64 tile loading is entirely determined by LDS/vectorization; 4095^3 (lda=ldb=ldc=4095) is
# 38% slower than 4096^3: probe137's controlled decomposition = ld misalignment ~17% + mask 6.7%, amplified by superposition (masks
# are only expensive when runs are unaligned). And probe139/140 already proved it **cannot be avoided inside the kernel**: a runtime
# branch splitting edge/interior tiles regresses aligned sizes by 17~26%, and tl.minimum clamping regresses aligned sizes by
# 33~81% (both break pipelining/vectorization analysis).
# The only effective approach is real host-side packing: copy A/B/C into a ceil-128 aligned border, rerun at the aligned size
# (then NO_MASK holds), and copy C's retained region back. probe146/147/148 used the real entry + including-copy
# timing (result max|diff| = 0, bit-exact):
#     2047->2048 1108.9->1018.2 (-8.2%) | 2303->2304 1544.3->1374.6 (-11.0%)
#     2561->2688 2055.5->1869.0 (-9.1%) | 3071->3072 3245.8->2768.1 (-14.7%)
#     3583->3584 4706.7->3955.6 (-16.0%)| 4095->4096 6970.6->5728.0 (-17.8%)
#       (4095^3 NN sp 0.792 -> 0.965)
#     1537->1664 -0.3% / 1793->1920 +1.4% (copy cost > gain), so the threshold is max_dim>=2048.
# Enabled only for NN: after NT takes the native direct path, 4095^3 is already 5720us (padding instead +1%); TN at that point
# already meets the target (sp 9.6); neither gains. Copy cost ~ s^2 while gemm ~ s^3, so the threshold can be relaxed with size.
_PAD_NN_MIN_DIM = 2048
_PAD_NN_MAX_GROWTH = 1.25     # upper bound on extra flops from the border (exclude cases like 2047->2176)
_PAD_NN_MAX_BYTES = 1 << 30   # total byte limit of the three border buffers
_FP64_BYTES = 8

# K_TAIL (k tail separation) is enabled only in the max_dim <= 1408 bucket. That bucket has a short k loop and is kernel-latency-bound;
# after dropping the k mask, A's contiguous-axis addressing can be vectorized, a positive gain (1023^3 kernel 153.8 -> 151.4us).
# The >1408 bucket's 128x128x16 s2 is a long-loop, throughput-bound config; the extra tail block disrupts s2's pipeline layout
# (4095^3 kernel 5767 -> 6867us, +19%; switching to s1/s3 turns positive, a scheduling sensitivity),
# so that bucket keeps the full k-masked loop. Data: this round's kernel A/B + probe163.
_K_TAIL_MAX_DIM = 1408


def _ceil_to(x: int, mult: int) -> int:
    return -(-x // mult) * mult


def _should_pad_nn(transa: int, transb: int, m: int, n: int, k: int) -> bool:
    """Whether to take the NN fast path "pack into a ceil-128 border and compute"."""
    if transa != 0 or transb != 0:
        return False
    if max(m, n, k) < _PAD_NN_MIN_DIM:
        return False
    if max(m, n) % 128 == 0 and k % 16 == 0:
        return False  # dimensions already aligned, no gain available
    pm, pn, pk = _ceil_to(m, 128), _ceil_to(n, 128), _ceil_to(k, 16)
    if pm * pn * pk > _PAD_NN_MAX_GROWTH * m * n * k:
        return False
    if (pm * pk + pk * pn + pm * pn) * _FP64_BYTES > _PAD_NN_MAX_BYTES:
        return False
    return True


def _dgemm_nn_padded(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc) -> None:
    """NN with unaligned ld: pack A/B/C -> aligned border -> rerun -> copy C back.

    Garbage outside the border does not pollute the retained region (A's extra columns multiply B's zero-padded rows; A's extra rows only
    affect the discarded extra rows of C), so only B's [k:pk) rows need zeroing; the whole border need not be cleared.
    """
    pm, pn, pk = _ceil_to(m, 128), _ceil_to(n, 128), _ceil_to(k, 16)
    ap = torch.empty((pm, pk), dtype=A.dtype, device=A.device)
    bp = torch.empty((pk, pn), dtype=B.dtype, device=B.device)
    cp = torch.empty((pm, pn), dtype=C.dtype, device=C.device)
    ap[:m, :k].copy_(A.as_strided((m, k), (lda, 1)))
    bp[:k, :n].copy_(B.as_strided((k, n), (ldb, 1)))
    bp[k:pk, :] = 0.0
    if beta != 0.0:
        cp[:m, :n].copy_(C.as_strided((m, n), (ldc, 1)))
    dgemm(0, 0, pm, pn, pk, alpha, ap, pk, bp, pn, beta, cp, pn)
    C.as_strided((m, n), (ldc, 1)).copy_(cp[:m, :n])


# --- feasibility check and driver for v25 tail-wave split-K (probe174/175, §3.22) ----------------
# Cost-model constants are taken from the report: saturated peak 28.0 TFLOPS (§3.15/§3.16), measured bandwidth 975 GB/s.
# The criterion is a **conservative gate** of "cost far smaller than gain", not a performance predictor -- enabling it only decides to cut one more slice
# on the tail wave; the main-axis path and existing buckets are unaffected.
_STREAMK_ENABLE = True
_STREAMK_SM_FALLBACK = 80
_STREAMK_PEAK_FLOPS = 28.0e12
_STREAMK_BW_BPS = 975.0e9
_streamk_sm_cache = None


def _streamk_sm() -> int:
    global _streamk_sm_cache
    if _streamk_sm_cache is None:
        try:
            _streamk_sm_cache = int(
                torch.cuda.get_device_properties(torch.cuda.current_device())
                .multi_processor_count
            )
        except Exception:  # pragma: no cover - should not be reached when there is no CUDA
            _streamk_sm_cache = _STREAMK_SM_FALLBACK
    return _streamk_sm_cache


def _streamk_plan(m: int, n: int, k: int, cfg) -> tuple:
    """Return (nsplit, bulk, tail) or None. Enabled only when the tail wave can be sliced and the gain far exceeds the cost."""
    if not _STREAMK_ENABLE or cfg is None or cfg.get("NAT_B") != 0:
        return None
    bm, bn, bk = cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["BLOCK_K"]
    if m % bm or n % bn or k % bk:
        return None                     # this path only serves the unmasked case
    sm = _streamk_sm()
    num_tiles = (m // bm) * (n // bn)
    nwaves = -(-num_tiles // sm)            # ceil(num_tiles / sm)
    if (num_tiles / sm) / nwaves >= 0.95:
        return None                     # wave quantization loss <5%: outside the approved protocol (8192^3 etc.)
    bulk = (num_tiles // sm) * sm
    tail = num_tiles - bulk
    if bulk == 0 or tail == 0:
        return None
    smax = sm // tail
    if smax < 2:
        return None                     # tail wave too large: splitting into ≥2 parts still cannot fill one wave, no gain
    ki = k // bk
    nsplit = next((d for d in range(smax, 1, -1) if ki % d == 0), None)
    if nsplit is None:
        return None
    tile_flops = 2.0 * bm * bn * k
    t_tile = tile_flops / _STREAMK_PEAK_FLOPS * sm      # wall clock for one CTA to run one tile
    gain = (1.0 - 1.0 / nsplit) * t_tile                # tail wave shrinks from 1 tile-time to 1/nsplit
    extra = 2.0 * nsplit * tail * bm * bn * 8 / _STREAMK_BW_BPS   # write + read of partial products
    if gain < 5.0 * extra:
        return None
    return nsplit, bulk, tail


def _dgemm_streamk(plan, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                   beta, C, ldc, cfg, beta_is_zero) -> None:
    """One launch for the main axis + one sliced launch for the tail wave + reduction write-back. 3 kernel launches in total."""
    nsplit, bulk, tail = plan
    bm, bn, bk = cfg["BLOCK_M"], cfg["BLOCK_N"], cfg["BLOCK_K"]
    gm, nw, ns = cfg["GROUP_M"], cfg["num_warps"], cfg["num_stages"]
    ki_per_slice = (k // bk) // nsplit
    part = torch.empty((nsplit * tail, bm, bn), dtype=torch.float64, device=A.device)
    _dgemm_dot_kernel[(bulk,)](
        A, B, C, alpha, beta, m, n, k, lda, ldb, ldc,
        transa, transb, beta_is_zero,
        NO_MASK=True, K_TAIL=False,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm, NAT_B=0,
        num_warps=nw, num_stages=ns,
    )
    _dgemm_splitk_kernel[(nsplit * tail,)](
        A, B, part, m, n, lda, ldb, bulk, tail, ki_per_slice,
        TRANS_A=transa, TRANS_B=transb,
        BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP_M=gm,
        num_warps=nw, num_stages=ns,
    )
    _dgemm_splitk_reduce_kernel[(tail,)](
        part, C, alpha, beta, m, n, ldc, bulk, nsplit,
        BETA_IS_ZERO=beta_is_zero,
        BLOCK_M=bm, BLOCK_N=bn, GROUP_M=gm, num_warps=16, num_stages=1,
    )


def dgemm(    transa: int,
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
    _validate_dgemm_args(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else float(beta)

    if m == 0 or n == 0 or k == 0 or alpha == 0.0:
        if beta == 0.0:
            C.zero_()
        elif beta != 1.0:
            C.mul_(beta)
        return

    # With transb=1, B's transposed load makes the dot kernel 2x slower (the right operand needs an smem layout conversion;
    # NAT_B/tl.trans can only mitigate <=256^3). First pre-transpose B into a contiguous (k,n) layout, then take
    # the transb=0 path (NT->NN, TT->TN). probe51 (real kernel, 2026-09-11): the large-shape gain
    # is the same order as medium/small shapes -- nt 1536/2048/4096 1959->1080 / 5204->2780 /
    # 36211->17464us (1.8-2.1x), tt the same magnitude.
    # The cost is one 16*n*k byte transpose; measured BW ~975GB/s, NN efficiency ~8 TFLOPS, so
    # transposed_gemm/transpose ~= m/64, break-even at m ~= 64: medium shapes (129..1024)
    # transpose <=8MB and measured m>=32 already amortizes; large shapes (>1024) transpose up to 512MB, take the more conservative
    # m >= 64.
    beta_is_zero = beta == 0.0

    with torch_device_fn.device(A.device):
        max_dim = max(m, n, k)
        if _should_pad_nn(transa, transb, m, n, k):
            # NN with unaligned ld, large shapes: pack into an aligned border and rerun (see probe146-148 above)
            _dgemm_nn_padded(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
            return
        # v18 (probe136/probe141, 2026-09-14): with transb=1, B pre-transpose is no longer "better by default".
        # Under MMAC each fp64 dot lane occupies only 1 f64, and layout-conversion cost is completely different from the FMA era:
        # native transb=1 (NAT_B=0, directly reading as (BK,BN)) on the NN bucket config costs only
        # +0.3~1.1% (128x128x16 w16) / +4~12% (small tiles), while pre-transpose pays an extra full-matrix
        # transpose (256^3 including torch.empty jitter +25.8us, 1024^3 +13us, 4096^3 +270us).
        # probe141 measured NT end-to-end per size (official protocol, median of 2 interleaved rounds):
        #   129^3 27.2->13.2 | 256^3 28.8->14.4 | 384^3 32.6->21.8 | 511^3 40.2->33.6
        #   512^3 33.0->30.4 | 768^3 67.4->63.2 | 1023^3 167.5->152.6
        #   1024^3 131.4->127.1 | 1536^3 366.4->338.3 | 2048^3 888.8->841.8
        #   3072^3 2511->2402 | 4096^3 5323->5105
        #   4095^3 7242->5718 (native TB1 is even 18% faster than TB0: when ld is unaligned, B's n-direction
        #          1024B runs can only be loaded scalar+predicated, while switching to k-direction 128B runs allows vectorization)
        # The only counterexample is 1408^3 (309.4->340.2, +10%): that point falls in the 64x32x16 w4 bucket and ld is unaligned,
        # so the native TB1 penalty reaches +20.7%. So NT keeps pre-transpose only in the (1024,1408] window (where the NN
        # bucket is 64x32x16 w4) and takes the native direct path for all other sizes.
        # v19 (probe143/144/145, 2026-09-14, official harness isomorphic tensors + real entry):
        #   1408 < max_dim <= 6144 direct path + 128x128x16 w16 (probe143/144):
        #     1536^3 364.2->338.9 | 2048^3 948.5->840.7 | 3072^3 2784->2400
        #     4095^3 7459->5720   | 4096^3 6423->5101   | 6144^3 20417->16666
        #     5120^3 -2.9%        (probe145, relative to "pre-transpose + NN")
        #   But when max_dim > 6144 the direct path fails and pre-transpose must be restored (probe145):
        #     7168^3 direct path 128x128x16 w16 = +0.65% (a tie) | 8192^3 = +45.1%!
        #     8192^3's best direct path 128x256x16 w16 is still +6.9%, so the upper bound is 6144
        #     (8192 is a power of 2; ldb=8192's 64KB row stride causes cache set conflicts).
        do_trans_b = transb == 1 and max_dim > 128
        if do_trans_b and transa == 0:      # NT: native direct path preferred (except two ranges)
            do_trans_b = m >= 64 and ((1024 < max_dim <= 1408) or max_dim > 6144)
        elif do_trans_b:                    # TT: same policy as NT since v21 (when m>=64)
            # v21 (probe158–probe161, 2026-09-14): TT originally always pre-transposed B (since v2),
            # but under MMAC that host-side transpose is a pure loss -- the transpose kernel itself is worth only ~1us (at the
            # do_bench floor), and the cost is entirely the fixed overhead of "one more host launch", which cannot be amortized on small shapes:
            # probe159 localized it with a stand-in method (real transpose 34.0us / torch.empty only 14.24us /
            # even skipping allocation 14.24us); probe160/161 measured per size that "native TB1
            # (NAT_B=0) + TN bucket tile" is uniformly faster (official protocol, median of 3 rounds):
            #   129^3 52.5->13.3 (3.95x) | 160^3 54.9->10.6 (5.20x)
            #   192^3 55.3->12.0 (4.61x) | 256^3 52.6->14.1 (3.81x)
            #   272^3 54.5->19.7 (2.77x) | 288^3 58.0->18.7 (3.10x)
            #   384^3 58.8->21.4 (2.74x) | 511^3 54.1->35.4 (1.53x)
            #   512^3 58.1->31.2 (1.86x) | 896^3 148.2->90.6 (1.64x)
            #   1023^3 187.7->153.3 (1.22x) | 1024^3 148.5->129.8 (1.14x)
            #   1536^3 362.7->335.8 (1.08x) | 2048^3 879.7->832.5 (1.06x)
            #   3072^3 2479->2378 (1.04x) | 4096^3 5264->5046 (1.04x)
            #   6144^3 16894->16498 (1.02x) | 8192^3 39967->39104 (1.02x)
            # Two exceptions keep pre-transpose: (a) the (1024,1408] window -- NT's direct path there is +20.7%
            # (probe141's 1408^3), TT measured a tie (0.99x); (b) >1408 and unaligned --
            # 4095^3 direct path instead 6035->6975 (+15.6%), because TB1's n-direction row stride is not a multiple of
            # 128B, shifting all k-direction runs; 4096^3 aligned is 1.04x.
            # NAT_B=1 is uniformly 2~4x worse, so the direct path uses nat=0's (BK,BN) direct load.
            if m < 64:                      # tall-skinny matrix: keep historical criterion, not re-measured
                do_trans_b = m >= (32 if max_dim <= 1024 else 64)
            elif max_dim > 1408:
                do_trans_b = not (m % 128 == 0 and n % 128 == 0 and k % 16 == 0)
            else:
                do_trans_b = 1024 < max_dim <= 1408
        if do_trans_b:
            B = _transpose_b(B, ldb, n, k)
            ldb = n
            transb = 0

        # TN: with transa=1, A is stored column-major as (k,m); the dot left operand's contiguous axis falls on dim 0,
        # ~7% slower than NN with large tiles (1023^3 474us vs 441us). In the range where the transpose cost
        # can be amortized (513..1024), pre-transpose A to row-major (m,k) and take the transa=0 path:
        # TN 1023^3 474->454us (sp 0.911->0.952), 1024^3 450->435 (0.938->0.970).
        # For <=512 the TN kernel is already faster than NN and the transpose cannot amortize, so it is not enabled.
        if transa == 1 and transb == 0 and 512 < max_dim <= 1024:
            A, lda = _transpose_a(A, lda, m, k)
            transa = 0

        launch_kwargs = _select_dgemm_dot_config(transa, transb, m, n, k)
        if launch_kwargs is not None:
            streamk = _streamk_plan(m, n, k, launch_kwargs)
            if streamk is not None:
                _dgemm_streamk(streamk, transa, transb, m, n, k, alpha, A, lda,
                               B, ldb, beta, C, ldc, launch_kwargs, beta_is_zero)
                return
            block_m = launch_kwargs["BLOCK_M"]
            block_n = launch_kwargs["BLOCK_N"]
            block_k = launch_kwargs["BLOCK_K"]
            grid = (triton.cdiv(m, block_m) * triton.cdiv(n, block_n),)
            _dgemm_dot_kernel[grid](
                A,
                B,
                C,
                alpha,
                beta,
                m,
                n,
                k,
                lda,
                ldb,
                ldc,
                transa,
                transb,
                beta_is_zero,
                NO_MASK=(m % block_m == 0 and n % block_n == 0 and k % block_k == 0),
                K_TAIL=(k % block_k != 0 and max_dim <= _K_TAIL_MAX_DIM),
                **launch_kwargs,
            )
        else:
            # Tiny-size fallback: generic scalar outer-product kernel (always correct, performance not on the critical path)
            from flag_blas.ops.level3.dgemm import _dgemm_kernel

            grid = lambda meta: (
                triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
            )
            _dgemm_kernel[grid](
                A,
                B,
                C,
                alpha,
                beta,
                m,
                n,
                k,
                lda,
                ldb,
                ldc,
                transa,
                transb,
                beta_is_zero,
            )


@triton.jit
def _transpose_f64_kernel(
    b_ptr,
    bt_ptr,
    n,
    k,
    ldb,
    ldbt,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """bt[j, i] = b[i, j]: (n x k, leading dim ldb) -> (k x n, leading dim ldbt)."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (offs_n[:, None] < n) & (offs_k[None, :] < k)
    x = tl.load(b_ptr + offs_n[:, None] * ldb + offs_k[None, :], mask=mask, other=0.0)
    tl.store(bt_ptr + offs_k[None, :] * ldbt + offs_n[:, None], x, mask=mask)


def _transpose_b(b_nt: torch.Tensor, ldb: int, n: int, k: int) -> torch.Tensor:
    """Pre-transpose transb=1's B into a contiguous (k, n) layout (for the transb=0 path)."""
    bt = torch.empty((k, n), dtype=b_nt.dtype, device=b_nt.device)
    block_n, block_k = 32, 32
    grid = (triton.cdiv(n, block_n), triton.cdiv(k, block_k))
    _transpose_f64_kernel[grid](
        b_nt, bt, n, k, ldb, n, block_n, block_k, num_warps=4, num_stages=2
    )
    return bt


def _transpose_a(a_nt: torch.Tensor, lda: int, m: int, k: int):
    """Pre-transpose transa=1's A (k x m, leading dim lda) into row-major (m, k_pad).

    Aligning the leading dim to 16 is intentional: when k is odd (e.g. 1023), not aligning makes the dot kernel's
    vectorized addressing ineffective; measured 1023^3 812us -> 454us. Returns (tensor, leading dim).
    """
    k_pad = (k + 15) // 16 * 16
    at = torch.empty((m, k_pad), dtype=a_nt.dtype, device=a_nt.device)
    block_n, block_m = 32, 32
    grid = (triton.cdiv(k, block_n), triton.cdiv(m, block_m))
    _transpose_f64_kernel[grid](
        a_nt, at, k, m, lda, k_pad, block_n, block_m, num_warps=4, num_stages=2
    )
    return at, k_pad
