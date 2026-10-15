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

import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.utils import libentry, libtuner

__all__ = ["grouped_bfgemm_kernel", "group_bfgemm"]


def _prune_group_bfgemm_configs(configs, named_args, **kwargs):
    shape = (named_args["K"], named_args["group_size"], named_args["N"])
    pinned_specs = {
        # The short-K direct-store shape is consistently best with WM2.  Leaving
        # it to the runtime tuner makes this threshold shape alternate between
        # WM2 and slower configurations across fresh cache databases.
        (96, 128, 2048): (128, 512, 32, 2, 1, 16, 2),
        # Shape 18 needs the otherwise globally excluded WM4/P2 configuration.
        (1536, 16, 4096): (256, 256, 64, 4, 2, 16, 2),
        # The swizzled N=7168 shape regresses with the cached WM16 selection.
        (2048, 17, 7168): (256, 256, 64, 8, 2, 16, 2),
        # Fallback for layouts that cannot use the balanced shape-27 path.
        (2048, 128, 1024): (256, 256, 64, 8, 2, 16, 2),
    }
    pinned = []
    other_configs = []
    for config in configs:
        meta = config.kwargs
        spec = (
            meta["BLOCK_M"],
            meta["BLOCK_N"],
            meta["BLOCK_K"],
            meta["NUM_WM"],
            meta["PIPE_STAGES"],
            config.num_warps,
            config.num_stages,
        )
        if spec == pinned_specs.get(shape):
            pinned.append(config)
        if spec != pinned_specs[(1536, 16, 4096)]:
            other_configs.append(config)
    if shape in pinned_specs:
        return pinned
    return other_configs


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_bfgemm"),
    key=["M", "N", "K", "group_size"],
    prune_configs_by={"early_config_prune": _prune_group_bfgemm_configs},
)
@triton.jit
def grouped_bfgemm_kernel(
    group_A,
    group_B,
    group_list,
    group_out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    NUM_WM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PIPE_STAGES: tl.constexpr = 1,
):
    tile_n = tl.program_id(0)
    window = tl.program_id(1)
    expert = tl.program_id(2)
    end = tl.load(group_list + expert).to(tl.int32)
    start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
    num_tiles = tl.cdiv(end - start, BLOCK_M)
    if expert == group_size - 1:
        num_tiles = (end - start) // 256 * (256 // BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    b_base = group_B + expert.to(tl.int64) * stride_be
    for w in tl.range(window, num_tiles, NUM_WM, num_stages=1):
        offs = start + w * BLOCK_M
        offs_m = offs + tl.arange(0, BLOCK_M)
        a_base = group_A + offs.to(tl.int64) * stride_am
        a_ptrs = (
            a_base
            + tl.arange(0, BLOCK_M)[:, None] * stride_am
            + offs_k[None, :] * stride_ak
        )
        b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if N % BLOCK_N == 0 and K % BLOCK_K == 0:
            for k in tl.range(
                0, K // BLOCK_K, num_stages=PIPE_STAGES, loop_unroll_factor=1
            ):
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                a = tl.load(a_ptrs)
                accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        else:
            for k in tl.range(
                0, tl.cdiv(K, BLOCK_K), num_stages=PIPE_STAGES, loop_unroll_factor=1
            ):
                b = tl.load(
                    b_ptrs,
                    mask=(offs_n[None, :] < N) & (offs_k[:, None] + k * BLOCK_K < K),
                    other=0,
                    eviction_policy="evict_last",
                )
                a = tl.load(a_ptrs, mask=offs_k[None, :] + k * BLOCK_K < K, other=0)
                accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        output = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
        output = tl.permute(output, (0, 2, 1))
        output0, output1 = tl.split(output)
        offs_n_half = tile_n * BLOCK_N + tl.arange(0, BLOCK_N // 2)
        out_rows = offs_m.to(tl.int64)
        out_ptrs = (
            group_out + out_rows[:, None] * stride_om + offs_n_half[None, :] * stride_on
        )
        out_mask = (out_rows[:, None] < end.to(tl.int64)) & (offs_n_half[None, :] < N)
        tl.store(out_ptrs, output0.to(group_out.dtype.element_ty), mask=out_mask)
        tl.store(
            out_ptrs + BLOCK_N // 2 * stride_on,
            output1.to(group_out.dtype.element_ty),
            mask=out_mask & (offs_n_half[None, :] + BLOCK_N // 2 < N),
        )


@triton.jit
def grouped_bfgemm_schedule_kernel(
    group_list,
    schedule,
    group_size: tl.constexpr,
    MAX_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    expert = tl.program_id(0)
    start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
    end = tl.load(group_list + expert).to(tl.int32)
    schedule_start = 0
    for previous in tl.range(0, expert, loop_unroll_factor=1):
        previous_start = tl.load(
            group_list + previous - 1, mask=previous > 0, other=0
        ).to(tl.int32)
        previous_end = tl.load(group_list + previous).to(tl.int32)
        schedule_start += tl.cdiv(previous_end - previous_start, BLOCK_M)
    num_tiles = tl.cdiv(end - start, BLOCK_M)
    if expert == group_size - 1:
        num_tiles = (end - start) // BLOCK_M
    for local_tile in tl.range(0, num_tiles, loop_unroll_factor=1):
        slot = schedule_start + local_tile
        row = start + local_tile * BLOCK_M
        tl.store(schedule + slot, row * group_size + expert)
    if expert == group_size - 1:
        for slot in tl.range(
            schedule_start + num_tiles, MAX_TILES, loop_unroll_factor=1
        ):
            tl.store(schedule + slot, -1)


@libentry()
@triton.jit
def grouped_bfgemm_balanced_kernel(
    group_A,
    group_B,
    group_list,
    group_out,
    schedule,
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    MAX_TILES: tl.constexpr,
    NUM_WM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PIPE_STAGES: tl.constexpr,
):
    tile_n = tl.program_id(0)
    window = tl.program_id(1)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    for task in tl.range(window, MAX_TILES, NUM_WM, loop_unroll_factor=1):
        scheduled_tile = tl.load(schedule + task)
        if scheduled_tile >= 0:
            expert = scheduled_tile % group_size
            offs = scheduled_tile // group_size
            end = tl.load(group_list + expert).to(tl.int32)
            offs_m = offs + tl.arange(0, BLOCK_M)
            a_base = group_A + offs * stride_am
            a_ptrs = (
                a_base
                + tl.arange(0, BLOCK_M)[:, None] * stride_am
                + offs_k[None, :] * stride_ak
            )
            b_ptrs = (
                group_B
                + expert * stride_be
                + offs_k[:, None] * stride_bk
                + offs_n[None, :] * stride_bn
            )
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for _ in tl.range(
                0, K // BLOCK_K, num_stages=PIPE_STAGES, loop_unroll_factor=1
            ):
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                a = tl.load(a_ptrs)
                accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
            output = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
            output = tl.permute(output, (0, 2, 1))
            output0, output1 = tl.split(output)
            offs_n_half = tile_n * BLOCK_N + tl.arange(0, BLOCK_N // 2)
            out_rows = offs_m
            out_ptrs = (
                group_out
                + out_rows[:, None] * stride_om
                + offs_n_half[None, :] * stride_on
            )
            out_mask = out_rows[:, None] < end
            tl.store(out_ptrs, output0.to(group_out.dtype.element_ty), mask=out_mask)
            tl.store(
                out_ptrs + BLOCK_N // 2 * stride_on,
                output1.to(group_out.dtype.element_ty),
                mask=out_mask,
            )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_bfgemm"),
    key=["M", "N", "K", "group_size"],
    prune_configs_by={"early_config_prune": _prune_group_bfgemm_configs},
)
@triton.jit
def grouped_bfgemm_direct_store_kernel(
    group_A,
    group_B,
    group_list,
    group_out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    NUM_WM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PIPE_STAGES: tl.constexpr = 1,
):
    tile_n = tl.program_id(0)
    window = tl.program_id(1)
    expert = tl.program_id(2)
    end = tl.load(group_list + expert).to(tl.int32)
    start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
    num_tiles = tl.cdiv(end - start, BLOCK_M)
    if expert == group_size - 1:
        num_tiles = (end - start) // 256 * (256 // BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    b_base = group_B + expert.to(tl.int64) * stride_be
    for w in tl.range(window, num_tiles, NUM_WM, num_stages=1):
        offs = start + w * BLOCK_M
        offs_m = offs + tl.arange(0, BLOCK_M)
        a_base = group_A + offs.to(tl.int64) * stride_am
        a_ptrs = (
            a_base
            + tl.arange(0, BLOCK_M)[:, None] * stride_am
            + offs_k[None, :] * stride_ak
        )
        b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if N % BLOCK_N == 0 and K % BLOCK_K == 0:
            for k in tl.range(
                0, K // BLOCK_K, num_stages=PIPE_STAGES, loop_unroll_factor=1
            ):
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        else:
            for k in tl.range(
                0, tl.cdiv(K, BLOCK_K), num_stages=PIPE_STAGES, loop_unroll_factor=1
            ):
                a = tl.load(a_ptrs, mask=offs_k[None, :] + k * BLOCK_K < K, other=0)
                b = tl.load(
                    b_ptrs,
                    mask=(offs_n[None, :] < N) & (offs_k[:, None] + k * BLOCK_K < K),
                    other=0,
                    eviction_policy="evict_last",
                )
                accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        out_base = group_out + offs.to(tl.int64) * stride_om
        out_ptrs = (
            out_base
            + tl.arange(0, BLOCK_M)[:, None] * stride_om
            + offs_n[None, :] * stride_on
        )
        if N % BLOCK_N == 0:
            tl.store(
                out_ptrs,
                accumulator.to(group_out.dtype.element_ty),
                mask=offs_m[:, None] < end,
            )
        else:
            tl.store(
                out_ptrs,
                accumulator.to(group_out.dtype.element_ty),
                mask=(offs_m[:, None] < end) & (offs_n[None, :] < N),
            )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_bfgemm"),
    key=["M", "N", "K", "group_size"],
    prune_configs_by={"early_config_prune": _prune_group_bfgemm_configs},
)
@triton.jit
def grouped_bfgemm_swz_kernel(
    group_A,
    group_B,
    group_list,
    group_out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    NUM_WM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_N: tl.constexpr = 4,
    PIPE_STAGES: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    nt = tl.cdiv(N, BLOCK_N)
    per_expert = nt * NUM_WM
    expert = pid // per_expert
    pe = pid % per_expert
    sb = pe // (GROUP_N * NUM_WM)
    r = pe % (GROUP_N * NUM_WM)
    tile_n = sb * GROUP_N + r % GROUP_N
    window = r // GROUP_N
    if tile_n < nt:
        end = tl.load(group_list + expert).to(tl.int32)
        start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
        num_tiles = tl.cdiv(end - start, BLOCK_M)
        if expert == group_size - 1:
            num_tiles = (end - start) // 256 * (256 // BLOCK_M)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        b_base = group_B + expert.to(tl.int64) * stride_be
        for w in tl.range(window, num_tiles, NUM_WM, num_stages=1):
            offs = start + w * BLOCK_M
            offs_m = offs + tl.arange(0, BLOCK_M)
            a_base = group_A + offs.to(tl.int64) * stride_am
            a_ptrs = (
                a_base
                + tl.arange(0, BLOCK_M)[:, None] * stride_am
                + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            if N % BLOCK_N == 0 and K % BLOCK_K == 0:
                for k in tl.range(
                    0, K // BLOCK_K, num_stages=PIPE_STAGES, loop_unroll_factor=1
                ):
                    b = tl.load(b_ptrs, eviction_policy="evict_last")
                    a = tl.load(a_ptrs)
                    accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                    a_ptrs += BLOCK_K * stride_ak
                    b_ptrs += BLOCK_K * stride_bk
            else:
                for k in tl.range(
                    0, tl.cdiv(K, BLOCK_K), num_stages=PIPE_STAGES, loop_unroll_factor=1
                ):
                    b = tl.load(
                        b_ptrs,
                        mask=(offs_n[None, :] < N)
                        & (offs_k[:, None] + k * BLOCK_K < K),
                        other=0,
                        eviction_policy="evict_last",
                    )
                    a = tl.load(a_ptrs, mask=offs_k[None, :] + k * BLOCK_K < K, other=0)
                    accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                    a_ptrs += BLOCK_K * stride_ak
                    b_ptrs += BLOCK_K * stride_bk
            output = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
            output = tl.permute(output, (0, 2, 1))
            output0, output1 = tl.split(output)
            offs_n_half = tile_n * BLOCK_N + tl.arange(0, BLOCK_N // 2)
            out_rows = offs_m.to(tl.int64)
            out_ptrs = (
                group_out
                + out_rows[:, None] * stride_om
                + offs_n_half[None, :] * stride_on
            )
            out_mask = (out_rows[:, None] < end.to(tl.int64)) & (
                offs_n_half[None, :] < N
            )
            tl.store(out_ptrs, output0.to(group_out.dtype.element_ty), mask=out_mask)
            tl.store(
                out_ptrs + BLOCK_N // 2 * stride_on,
                output1.to(group_out.dtype.element_ty),
                mask=out_mask & (offs_n_half[None, :] + BLOCK_N // 2 < N),
            )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_bfgemm_tail"),
    key=["M", "N", "K", "group_size"],
)
@triton.jit
def grouped_bfgemm_tail_kernel(
    group_A,
    group_B,
    group_list,
    group_out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_n = tl.program_id(0)
    tile_m = tl.program_id(1)
    expert = group_size - 1
    start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
    end = tl.load(group_list + expert).to(tl.int32)
    offs = start + (end - start) // 256 * 256 + tile_m * BLOCK_M
    if offs < end:
        offs_m = offs + tl.arange(0, BLOCK_M)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = (
            group_A
            + offs_m[:, None].to(tl.int64) * stride_am
            + offs_k[None, :] * stride_ak
        )
        b_ptrs = (
            group_B
            + expert.to(tl.int64) * stride_be
            + offs_k[:, None] * stride_bk
            + offs_n[None, :] * stride_bn
        )
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if N % BLOCK_N == 0 and K % BLOCK_K == 0:
            for k in tl.range(0, K // BLOCK_K, num_stages=1, loop_unroll_factor=1):
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                a = tl.load(a_ptrs, mask=offs_m[:, None] < end, other=0)
                accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        else:
            for k in tl.range(
                0, tl.cdiv(K, BLOCK_K), num_stages=1, loop_unroll_factor=1
            ):
                b = tl.load(
                    b_ptrs,
                    mask=(offs_n[None, :] < N) & (offs_k[:, None] + k * BLOCK_K < K),
                    other=0,
                    eviction_policy="evict_last",
                )
                a = tl.load(
                    a_ptrs,
                    mask=(offs_m[:, None] < end) & (offs_k[None, :] + k * BLOCK_K < K),
                    other=0,
                )
                accumulator = tl.dot(a, b, acc=accumulator, out_dtype=tl.float32)
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        out_rows = offs_m.to(tl.int64)
        out_ptrs = (
            group_out + out_rows[:, None] * stride_om + offs_n[None, :] * stride_on
        )
        tl.store(
            out_ptrs,
            accumulator.to(group_out.dtype.element_ty),
            mask=(out_rows[:, None] < end.to(tl.int64)) & (offs_n[None, :] < N),
        )


def group_bfgemm(group_A, group_B, group_list, group_out):
    assert group_A.ndim == 2 and group_B.ndim == 3 and group_list.ndim == 1
    M, K = group_A.shape
    group_size, BK, N = group_B.shape
    assert BK == K
    assert group_list.numel() == group_size
    assert group_out.shape == (M, N)
    if group_size == 0 or M == 0 or N == 0:
        return group_out
    args = (
        group_A,
        group_B,
        group_list,
        group_out,
        M,
        N,
        K,
        group_size,
        *group_A.stride(),
        *group_B.stride(),
        *group_out.stride(),
    )
    if (
        K == 2048
        and (
            (group_size == 128 and N in (192, 768, 1024, 1536))
            or (group_size == 512 and N == 256)
        )
        and M * group_size < 2**31
        and (M - 1) * group_A.stride(0) + (K - 1) * group_A.stride(1) < 2**31
        and (group_size - 1) * group_B.stride(0)
        + (K - 1) * group_B.stride(1)
        + (N - 1) * group_B.stride(2)
        < 2**31
        and (M - 1) * group_out.stride(0) + (N - 1) * group_out.stride(1) < 2**31
    ):
        block_m = 128 if N == 192 else 256
        block_n = 64 if N == 192 else 256
        num_wm = 256 if N == 768 else (1024 if N == 1024 else 2048)
        max_tiles = triton.cdiv(M, block_m) + group_size - 1
        schedule = group_list.new_empty((max_tiles,))
        grouped_bfgemm_schedule_kernel[(group_size,)](
            group_list,
            schedule,
            group_size,
            max_tiles,
            block_m,
        )
        grouped_bfgemm_balanced_kernel[(triton.cdiv(N, block_n), num_wm, 1)](
            group_A,
            group_B,
            group_list,
            group_out,
            schedule,
            N,
            K,
            group_size,
            *group_A.stride(),
            *group_B.stride(),
            *group_out.stride(),
            max_tiles,
            num_wm,
            block_m,
            block_n,
            64,
            2,
            num_warps=8 if N == 192 else 16,
            num_stages=2,
        )
    else:
        kernel = grouped_bfgemm_kernel
        grid = lambda meta: (
            triton.cdiv(N, meta["BLOCK_N"]),
            meta["NUM_WM"],
            group_size,
        )
        if K <= 768 and N >= 2048:
            kernel = grouped_bfgemm_direct_store_kernel
        elif N >= 7168:
            kernel = grouped_bfgemm_swz_kernel
            grid = lambda meta: (
                triton.cdiv(N, meta["BLOCK_N"]) * meta["NUM_WM"] * group_size,
            )
        kernel[grid](*args)
    grouped_bfgemm_tail_kernel[
        lambda meta: (
            triton.cdiv(N, meta["BLOCK_N"]),
            triton.cdiv(256, meta["BLOCK_M"]),
        )
    ](*args)
    return group_out


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_tf32gemm"),
    key=["M", "N", "K", "group_size"],
)
@triton.jit
def grouped_tf32gemm_kernel(
    group_A,
    group_B,
    group_list,
    group_out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    NUM_WM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PIPE_STAGES: tl.constexpr = 1,
):
    tile_n = tl.program_id(0)
    window = tl.program_id(1)
    expert = tl.program_id(2)
    end = tl.load(group_list + expert).to(tl.int32)
    start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
    num_tiles = tl.cdiv(end - start, BLOCK_M)
    if expert == group_size - 1:
        num_tiles = (end - start) // 256 * (256 // BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    b_base = group_B + expert.to(tl.int64) * stride_be
    for w in tl.range(window, num_tiles, NUM_WM, num_stages=1):
        offs = start + w * BLOCK_M
        offs_m = offs + tl.arange(0, BLOCK_M)
        a_base = group_A + offs.to(tl.int64) * stride_am
        a_ptrs = (
            a_base
            + tl.arange(0, BLOCK_M)[:, None] * stride_am
            + offs_k[None, :] * stride_ak
        )
        b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if N % BLOCK_N == 0 and K % BLOCK_K == 0:
            for k in tl.range(
                0, K // BLOCK_K, num_stages=PIPE_STAGES, loop_unroll_factor=1
            ):
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                a = tl.load(a_ptrs)
                accumulator = tl.dot(
                    a, b, acc=accumulator, input_precision="tf32", out_dtype=tl.float32
                )
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        else:
            for k in tl.range(
                0, tl.cdiv(K, BLOCK_K), num_stages=PIPE_STAGES, loop_unroll_factor=1
            ):
                b = tl.load(
                    b_ptrs,
                    mask=(offs_n[None, :] < N) & (offs_k[:, None] + k * BLOCK_K < K),
                    other=0,
                    eviction_policy="evict_last",
                )
                a = tl.load(a_ptrs, mask=offs_k[None, :] + k * BLOCK_K < K, other=0)
                accumulator = tl.dot(
                    a, b, acc=accumulator, input_precision="tf32", out_dtype=tl.float32
                )
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        output = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
        output = tl.permute(output, (0, 2, 1))
        output0, output1 = tl.split(output)
        offs_n_half = tile_n * BLOCK_N + tl.arange(0, BLOCK_N // 2)
        out_rows = offs_m.to(tl.int64)
        out_ptrs = (
            group_out + out_rows[:, None] * stride_om + offs_n_half[None, :] * stride_on
        )
        out_mask = (out_rows[:, None] < end.to(tl.int64)) & (offs_n_half[None, :] < N)
        tl.store(out_ptrs, output0.to(group_out.dtype.element_ty), mask=out_mask)
        tl.store(
            out_ptrs + BLOCK_N // 2 * stride_on,
            output1.to(group_out.dtype.element_ty),
            mask=out_mask & (offs_n_half[None, :] + BLOCK_N // 2 < N),
        )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_tf32gemm"),
    key=["M", "N", "K", "group_size"],
)
@triton.jit
def _grouped_tf32gemm_direct_store_kernel(
    group_A,
    group_B,
    group_list,
    group_out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    NUM_WM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PIPE_STAGES: tl.constexpr = 1,
):
    tile_n = tl.program_id(0)
    window = tl.program_id(1)
    expert = tl.program_id(2)
    end = tl.load(group_list + expert).to(tl.int32)
    start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
    num_tiles = tl.cdiv(end - start, BLOCK_M)
    if expert == group_size - 1:
        num_tiles = (end - start) // 256 * (256 // BLOCK_M)
    offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    b_base = group_B + expert.to(tl.int64) * stride_be
    for w in tl.range(window, num_tiles, NUM_WM, num_stages=1):
        offs = start + w * BLOCK_M
        offs_m = offs + tl.arange(0, BLOCK_M)
        a_base = group_A + offs.to(tl.int64) * stride_am
        a_ptrs = (
            a_base
            + tl.arange(0, BLOCK_M)[:, None] * stride_am
            + offs_k[None, :] * stride_ak
        )
        b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if N % BLOCK_N == 0 and K % BLOCK_K == 0:
            for k in tl.range(
                0, K // BLOCK_K, num_stages=PIPE_STAGES, loop_unroll_factor=1
            ):
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                accumulator = tl.dot(
                    a, b, acc=accumulator, input_precision="tf32", out_dtype=tl.float32
                )
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        else:
            for k in tl.range(
                0, tl.cdiv(K, BLOCK_K), num_stages=PIPE_STAGES, loop_unroll_factor=1
            ):
                a = tl.load(a_ptrs, mask=offs_k[None, :] + k * BLOCK_K < K, other=0)
                b = tl.load(
                    b_ptrs,
                    mask=(offs_n[None, :] < N) & (offs_k[:, None] + k * BLOCK_K < K),
                    other=0,
                    eviction_policy="evict_last",
                )
                accumulator = tl.dot(
                    a, b, acc=accumulator, input_precision="tf32", out_dtype=tl.float32
                )
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        out_base = group_out + offs.to(tl.int64) * stride_om
        out_ptrs = (
            out_base
            + tl.arange(0, BLOCK_M)[:, None] * stride_om
            + offs_n[None, :] * stride_on
        )
        if N % BLOCK_N == 0:
            tl.store(
                out_ptrs,
                accumulator.to(group_out.dtype.element_ty),
                mask=offs_m[:, None] < end,
            )
        else:
            tl.store(
                out_ptrs,
                accumulator.to(group_out.dtype.element_ty),
                mask=(offs_m[:, None] < end) & (offs_n[None, :] < N),
            )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_tf32gemm"),
    key=["M", "N", "K", "group_size"],
)
@triton.jit
def _grouped_tf32gemm_swz_kernel(
    group_A,
    group_B,
    group_list,
    group_out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    NUM_WM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_N: tl.constexpr = 4,
    PIPE_STAGES: tl.constexpr = 1,
):
    pid = tl.program_id(0)
    nt = tl.cdiv(N, BLOCK_N)
    per_expert = nt * NUM_WM
    expert = pid // per_expert
    pe = pid % per_expert
    sb = pe // (GROUP_N * NUM_WM)
    r = pe % (GROUP_N * NUM_WM)
    tile_n = sb * GROUP_N + r % GROUP_N
    window = r // GROUP_N
    if tile_n < nt:
        end = tl.load(group_list + expert).to(tl.int32)
        start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
        num_tiles = tl.cdiv(end - start, BLOCK_M)
        if expert == group_size - 1:
            num_tiles = (end - start) // 256 * (256 // BLOCK_M)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        b_base = group_B + expert.to(tl.int64) * stride_be
        for w in tl.range(window, num_tiles, NUM_WM, num_stages=1):
            offs = start + w * BLOCK_M
            offs_m = offs + tl.arange(0, BLOCK_M)
            a_base = group_A + offs.to(tl.int64) * stride_am
            a_ptrs = (
                a_base
                + tl.arange(0, BLOCK_M)[:, None] * stride_am
                + offs_k[None, :] * stride_ak
            )
            b_ptrs = b_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            if N % BLOCK_N == 0 and K % BLOCK_K == 0:
                for k in tl.range(
                    0, K // BLOCK_K, num_stages=PIPE_STAGES, loop_unroll_factor=1
                ):
                    b = tl.load(b_ptrs, eviction_policy="evict_last")
                    a = tl.load(a_ptrs)
                    accumulator = tl.dot(
                        a,
                        b,
                        acc=accumulator,
                        input_precision="tf32",
                        out_dtype=tl.float32,
                    )
                    a_ptrs += BLOCK_K * stride_ak
                    b_ptrs += BLOCK_K * stride_bk
            else:
                for k in tl.range(
                    0, tl.cdiv(K, BLOCK_K), num_stages=PIPE_STAGES, loop_unroll_factor=1
                ):
                    b = tl.load(
                        b_ptrs,
                        mask=(offs_n[None, :] < N)
                        & (offs_k[:, None] + k * BLOCK_K < K),
                        other=0,
                        eviction_policy="evict_last",
                    )
                    a = tl.load(a_ptrs, mask=offs_k[None, :] + k * BLOCK_K < K, other=0)
                    accumulator = tl.dot(
                        a,
                        b,
                        acc=accumulator,
                        input_precision="tf32",
                        out_dtype=tl.float32,
                    )
                    a_ptrs += BLOCK_K * stride_ak
                    b_ptrs += BLOCK_K * stride_bk
            output = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
            output = tl.permute(output, (0, 2, 1))
            output0, output1 = tl.split(output)
            offs_n_half = tile_n * BLOCK_N + tl.arange(0, BLOCK_N // 2)
            out_rows = offs_m.to(tl.int64)
            out_ptrs = (
                group_out
                + out_rows[:, None] * stride_om
                + offs_n_half[None, :] * stride_on
            )
            out_mask = (out_rows[:, None] < end.to(tl.int64)) & (
                offs_n_half[None, :] < N
            )
            tl.store(out_ptrs, output0.to(group_out.dtype.element_ty), mask=out_mask)
            tl.store(
                out_ptrs + BLOCK_N // 2 * stride_on,
                output1.to(group_out.dtype.element_ty),
                mask=out_mask & (offs_n_half[None, :] + BLOCK_N // 2 < N),
            )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("group_tf32gemm_tail"),
    key=["M", "N", "K", "group_size"],
)
@triton.jit
def _grouped_tf32gemm_tail_kernel(
    group_A,
    group_B,
    group_list,
    group_out,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    group_size: tl.constexpr,
    stride_am: tl.constexpr,
    stride_ak: tl.constexpr,
    stride_be: tl.constexpr,
    stride_bk: tl.constexpr,
    stride_bn: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_n = tl.program_id(0)
    tile_m = tl.program_id(1)
    expert = group_size - 1
    start = tl.load(group_list + expert - 1, mask=expert > 0, other=0).to(tl.int32)
    end = tl.load(group_list + expert).to(tl.int32)
    offs = start + (end - start) // 256 * 256 + tile_m * BLOCK_M
    if offs < end:
        offs_m = offs + tl.arange(0, BLOCK_M)
        offs_n = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = (
            group_A
            + offs_m[:, None].to(tl.int64) * stride_am
            + offs_k[None, :] * stride_ak
        )
        b_ptrs = (
            group_B
            + expert.to(tl.int64) * stride_be
            + offs_k[:, None] * stride_bk
            + offs_n[None, :] * stride_bn
        )
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        if N % BLOCK_N == 0 and K % BLOCK_K == 0:
            for k in tl.range(0, K // BLOCK_K, num_stages=1, loop_unroll_factor=1):
                b = tl.load(b_ptrs, eviction_policy="evict_last")
                a = tl.load(a_ptrs, mask=offs_m[:, None] < end, other=0)
                accumulator = tl.dot(
                    a, b, acc=accumulator, input_precision="tf32", out_dtype=tl.float32
                )
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        else:
            for k in tl.range(
                0, tl.cdiv(K, BLOCK_K), num_stages=1, loop_unroll_factor=1
            ):
                b = tl.load(
                    b_ptrs,
                    mask=(offs_n[None, :] < N) & (offs_k[:, None] + k * BLOCK_K < K),
                    other=0,
                    eviction_policy="evict_last",
                )
                a = tl.load(
                    a_ptrs,
                    mask=(offs_m[:, None] < end) & (offs_k[None, :] + k * BLOCK_K < K),
                    other=0,
                )
                accumulator = tl.dot(
                    a, b, acc=accumulator, input_precision="tf32", out_dtype=tl.float32
                )
                a_ptrs += BLOCK_K * stride_ak
                b_ptrs += BLOCK_K * stride_bk
        out_rows = offs_m.to(tl.int64)
        out_ptrs = (
            group_out + out_rows[:, None] * stride_om + offs_n[None, :] * stride_on
        )
        tl.store(
            out_ptrs,
            accumulator.to(group_out.dtype.element_ty),
            mask=(out_rows[:, None] < end.to(tl.int64)) & (offs_n[None, :] < N),
        )


def group_hgemm(group_A, group_B, group_list, group_out):
    assert group_A.ndim == 2 and group_B.ndim == 3 and group_list.ndim == 1
    M, K = group_A.shape
    group_size, BK, N = group_B.shape
    assert BK == K
    assert group_list.numel() == group_size
    assert group_out.shape == (M, N)
    if group_size == 0 or M == 0 or N == 0:
        return group_out
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), meta["NUM_WM"], group_size)
    if K <= 768 and N >= 2048:
        grouped_bfgemm_direct_store_kernel[grid](
            group_A,
            group_B,
            group_list,
            group_out,
            M,
            N,
            K,
            group_size,
            *group_A.stride(),
            *group_B.stride(),
            *group_out.stride(),
        )
    elif N >= 7168:
        swz_grid = lambda meta: (
            triton.cdiv(N, meta["BLOCK_N"]) * meta["NUM_WM"] * group_size,
        )
        grouped_bfgemm_swz_kernel[swz_grid](
            group_A,
            group_B,
            group_list,
            group_out,
            M,
            N,
            K,
            group_size,
            *group_A.stride(),
            *group_B.stride(),
            *group_out.stride(),
        )
    else:
        grouped_bfgemm_kernel[grid](
            group_A,
            group_B,
            group_list,
            group_out,
            M,
            N,
            K,
            group_size,
            *group_A.stride(),
            *group_B.stride(),
            *group_out.stride(),
        )
    tail_grid = lambda meta: (
        triton.cdiv(N, meta["BLOCK_N"]),
        triton.cdiv(256, meta["BLOCK_M"]),
    )
    grouped_bfgemm_tail_kernel[tail_grid](
        group_A,
        group_B,
        group_list,
        group_out,
        M,
        N,
        K,
        group_size,
        *group_A.stride(),
        *group_B.stride(),
        *group_out.stride(),
    )
    return group_out


def group_tf32gemm(group_A, group_B, group_list, group_out):
    assert group_A.ndim == 2 and group_B.ndim == 3 and group_list.ndim == 1
    M, K = group_A.shape
    group_size, BK, N = group_B.shape
    assert BK == K
    assert group_list.numel() == group_size
    assert group_out.shape == (M, N)
    if group_size == 0 or M == 0 or N == 0:
        return group_out
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), meta["NUM_WM"], group_size)
    if K <= 768 and N >= 2048:
        _grouped_tf32gemm_direct_store_kernel[grid](
            group_A,
            group_B,
            group_list,
            group_out,
            M,
            N,
            K,
            group_size,
            *group_A.stride(),
            *group_B.stride(),
            *group_out.stride(),
        )
    elif N >= 7168:
        swz_grid = lambda meta: (
            triton.cdiv(N, meta["BLOCK_N"]) * meta["NUM_WM"] * group_size,
        )
        _grouped_tf32gemm_swz_kernel[swz_grid](
            group_A,
            group_B,
            group_list,
            group_out,
            M,
            N,
            K,
            group_size,
            *group_A.stride(),
            *group_B.stride(),
            *group_out.stride(),
        )
    else:
        grouped_tf32gemm_kernel[grid](
            group_A,
            group_B,
            group_list,
            group_out,
            M,
            N,
            K,
            group_size,
            *group_A.stride(),
            *group_B.stride(),
            *group_out.stride(),
        )
    tail_grid = lambda meta: (
        triton.cdiv(N, meta["BLOCK_N"]),
        triton.cdiv(256, meta["BLOCK_M"]),
    )
    _grouped_tf32gemm_tail_kernel[tail_grid](
        group_A,
        group_B,
        group_list,
        group_out,
        M,
        N,
        K,
        group_size,
        *group_A.stride(),
        *group_B.stride(),
        *group_out.stride(),
    )
    return group_out
