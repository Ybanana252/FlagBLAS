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

"""Ascend-specific complex64 TPMV kernel.

The algorithm matches the public implementation while masking packed-matrix
offsets before pointer arithmetic, which Atlas A3 requires for masked loads.
"""

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level2._constants import (
    CUBLAS_DIAG_UNIT,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
)
from flag_blas.ops.level2.tpmv import (
    _check_tpmv,
    _mode_key,
    _prune_tpmv_configs,
    _row_major_tpmv_args,
    _stpmv_split_k,
    stpmv_kernel,
    stpmv_reduce_kernel,
    stpmv_splitk_kernel,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.runtime.backend._ascend.ops.hpmv import _current_device, _launch
from flag_blas.utils import libentry, libtuner

_TPMV_KEY = ["n", "mode_key"]
_TPMV_RESTORE = ["x_ptr"]


def stpmv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    AP: torch.Tensor,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert AP.dtype == torch.float32 == x.dtype
    _check_tpmv(AP, x, uplo, trans, diag, n, incx, complex_ok=False)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    uplo, trans, conj = _row_major_tpmv_args(uplo, trans)
    trans_flag = 0 if trans == CUBLAS_OP_N else 1
    split_k = _stpmv_split_k(n)

    with torch_device_fn.device(AP.device):
        if n <= 192 or (trans_flag == 1 and n <= 384):
            _stpmv_small_path(uplo, trans_flag, unit, n, AP, x, incx)
            return
        if trans_flag == 0 and 512 <= n <= 16384:
            block = 64
            splits = triton.cdiv(n, block)
            partial = torch.empty((splits, n), dtype=torch.float32, device=AP.device)
            _launch(
                _stpmv_partial_tile, (splits * (splits + 1) // 2,),
                (AP, x, partial), (),
                (n, incx, uplo, unit, block),
            )
            _launch(
                _stpmv_finish_tile, (splits,), (partial, x), (),
                (n, incx, splits, min(32, triton.next_power_of_2(splits)),
                 block, uplo, unit),
            )
            return
        xin = x.as_strided((n,), (incx,)).clone()
        if split_k > 1:
            partial = torch.empty((split_k, n), dtype=torch.float32, device=AP.device)
            grid_main = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]), split_k)
            stpmv_splitk_kernel[grid_main](
                AP,
                xin,
                partial,
                n,
                _mode_key(uplo, trans_flag, unit),
                SPLIT_K=split_k,
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
            )
            block_n = 1024
            stpmv_reduce_kernel[(triton.cdiv(n, block_n),)](
                partial,
                xin,
                x,
                n,
                incx,
                SPLIT_K=split_k,
                UNIT=unit,
                BLOCK_N=block_n,
                num_warps=4,
            )
        else:
            grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
            stpmv_kernel[grid](
                AP,
                xin,
                x,
                n,
                incx,
                _mode_key(uplo, trans_flag, unit),
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
            )


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ctpmv"),
    key=_TPMV_KEY,
    restore_value=_TPMV_RESTORE,
    prune_configs_by={"early_config_prune": _prune_tpmv_configs},
)
@triton.jit
def ctpmv_kernel(
    ap_ptr,
    xin_ptr,
    x_ptr,
    n,
    INCX,
    mode_key,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    INDEX64: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE_M
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_SIZE_M), BLOCK_SIZE_M)
    row_mask = rows < n
    acc_r = tl.zeros((BLOCK_SIZE_M, BLOCK_K), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_SIZE_M, BLOCK_K), dtype=tl.float32)

    n64 = tl.full((), n, tl.int64 if INDEX64 else tl.int32)
    rows64 = rows.to(tl.int64 if INDEX64 else tl.int32)
    offs_k = tl.arange(0, BLOCK_K)

    if TRANS == 1:
        if UPLO == 1:
            row_base = rows64 * (rows64 + 1) // 2
        else:
            row_base = rows64 * n64 - rows64 * (rows64 + 1) // 2

    if UPLO == TRANS:
        for kb in tl.range(0, row_start, BLOCK_K):
            j = kb + offs_k
            j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
            j_mask = j < n
            j64 = j.to(tl.int64 if INDEX64 else tl.int32)
            if TRANS == 0:
                col_base = j64 * n64 - j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                off = row_base[:, None] + j64[None, :]
            mask = row_mask[:, None] & j_mask[None, :]
            safe_off = tl.where(mask, off, 0)
            a_off = safe_off * 2
            x_off = j * 2
            ar = tl.load(
                ap_ptr + a_off, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            ai = tl.load(
                ap_ptr + a_off + 1, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            xr = tl.load(
                xin_ptr + x_off, mask=j_mask, other=0.0, eviction_policy="evict_last"
            )
            xi = tl.load(
                xin_ptr + x_off + 1,
                mask=j_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            if CONJ:
                ai = -ai
            acc_r += ar * xr[None, :] - ai * xi[None, :]
            acc_i += ar * xi[None, :] + ai * xr[None, :]
        diag_lo = row_start
    else:
        diag_lo = (row_start // BLOCK_K) * BLOCK_K

    for kb in tl.range(diag_lo, row_start + BLOCK_SIZE_M, BLOCK_K):
        j = kb + offs_k
        j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
        j_mask = j < n
        j64 = j.to(tl.int64 if INDEX64 else tl.int32)
        if TRANS == 0:
            if UPLO == 1:
                tri = j[None, :] >= rows[:, None]
                col_base = j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                tri = j[None, :] <= rows[:, None]
                col_base = j64 * n64 - j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
        else:
            if UPLO == 1:
                tri = j[None, :] <= rows[:, None]
            else:
                tri = j[None, :] >= rows[:, None]
            off = row_base[:, None] + j64[None, :]
        if UNIT:
            tri = tri & (j[None, :] != rows[:, None])
        mask = row_mask[:, None] & j_mask[None, :] & tri
        safe_off = tl.where(mask, off, 0)
        a_off = safe_off * 2
        x_off = j * 2
        ar = tl.load(
            ap_ptr + a_off, mask=mask, other=0.0, eviction_policy="evict_first"
        )
        ai = tl.load(
            ap_ptr + a_off + 1, mask=mask, other=0.0, eviction_policy="evict_first"
        )
        xr = tl.load(
            xin_ptr + x_off, mask=j_mask, other=0.0, eviction_policy="evict_last"
        )
        xi = tl.load(
            xin_ptr + x_off + 1, mask=j_mask, other=0.0, eviction_policy="evict_last"
        )
        if CONJ:
            ai = -ai
        acc_r += ar * xr[None, :] - ai * xi[None, :]
        acc_i += ar * xi[None, :] + ai * xr[None, :]

    if UPLO != TRANS:
        for kb in tl.range(row_start + BLOCK_SIZE_M, n, BLOCK_K):
            j = kb + offs_k
            j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
            j_mask = j < n
            j64 = j.to(tl.int64 if INDEX64 else tl.int32)
            if TRANS == 0:
                col_base = j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                off = row_base[:, None] + j64[None, :]
            mask = row_mask[:, None] & j_mask[None, :]
            safe_off = tl.where(mask, off, 0)
            a_off = safe_off * 2
            x_off = j * 2
            ar = tl.load(
                ap_ptr + a_off, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            ai = tl.load(
                ap_ptr + a_off + 1, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            xr = tl.load(
                xin_ptr + x_off, mask=j_mask, other=0.0, eviction_policy="evict_last"
            )
            xi = tl.load(
                xin_ptr + x_off + 1,
                mask=j_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            if CONJ:
                ai = -ai
            acc_r += ar * xr[None, :] - ai * xi[None, :]
            acc_i += ar * xi[None, :] + ai * xr[None, :]

    acc_r = tl.sum(acc_r, axis=1)
    acc_i = tl.sum(acc_i, axis=1)
    if UNIT:
        acc_r += tl.load(xin_ptr + rows * 2, mask=row_mask, other=0.0)
        acc_i += tl.load(xin_ptr + rows * 2 + 1, mask=row_mask, other=0.0)

    x_off_out = rows * INCX * 2
    tl.store(x_ptr + x_off_out, acc_r, mask=row_mask)
    tl.store(x_ptr + x_off_out + 1, acc_i, mask=row_mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("ctpmv"),
    key=_TPMV_KEY,
    prune_configs_by={"early_config_prune": _prune_tpmv_configs},
)
@triton.jit
def ctpmv_splitk_kernel(
    ap_ptr,
    xin_ptr,
    partial_ptr,
    n,
    mode_key,
    SPLIT_K: tl.constexpr,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    INDEX64: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    row_start = pid_m * BLOCK_SIZE_M
    rows = row_start + tl.arange(0, BLOCK_SIZE_M)
    rows = tl.max_contiguous(tl.multiple_of(rows, BLOCK_SIZE_M), BLOCK_SIZE_M)
    row_mask = rows < n
    acc_r = tl.zeros((BLOCK_SIZE_M, BLOCK_K), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_SIZE_M, BLOCK_K), dtype=tl.float32)

    n64 = tl.full((), n, tl.int64 if INDEX64 else tl.int32)
    rows64 = rows.to(tl.int64 if INDEX64 else tl.int32)
    offs_k = tl.arange(0, BLOCK_K)

    if TRANS == 1:
        if UPLO == 1:
            row_base = rows64 * (rows64 + 1) // 2
        else:
            row_base = rows64 * n64 - rows64 * (rows64 + 1) // 2

    if UPLO != TRANS:
        active_lo = row_start
        active_hi = n
    else:
        active_lo = 0
        active_hi = row_start + BLOCK_SIZE_M
    active_len = active_hi - active_lo
    total_tiles = (active_len + BLOCK_K - 1) // BLOCK_K
    tiles_per_chunk = (total_tiles + SPLIT_K - 1) // SPLIT_K
    my_tile_lo = pid_k * tiles_per_chunk
    my_tile_hi = tl.minimum((pid_k + 1) * tiles_per_chunk, total_tiles)
    my_lo = active_lo + my_tile_lo * BLOCK_K
    my_hi = active_lo + my_tile_hi * BLOCK_K

    diag_lo_v = row_start
    diag_hi_v = row_start + BLOCK_SIZE_M

    if UPLO == TRANS:
        pre_hi = tl.minimum(my_hi, diag_lo_v)
        for kb in tl.range(my_lo, pre_hi, BLOCK_K):
            j = kb + offs_k
            j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
            j_mask = j < n
            j64 = j.to(tl.int64 if INDEX64 else tl.int32)
            if TRANS == 0:
                col_base = j64 * n64 - j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                off = row_base[:, None] + j64[None, :]
            mask = row_mask[:, None] & j_mask[None, :]
            safe_off = tl.where(mask, off, 0)
            a_off = safe_off * 2
            x_off = j * 2
            ar = tl.load(
                ap_ptr + a_off, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            ai = tl.load(
                ap_ptr + a_off + 1, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            xr = tl.load(
                xin_ptr + x_off, mask=j_mask, other=0.0, eviction_policy="evict_last"
            )
            xi = tl.load(
                xin_ptr + x_off + 1,
                mask=j_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            if CONJ:
                ai = -ai
            acc_r += ar * xr[None, :] - ai * xi[None, :]
            acc_i += ar * xi[None, :] + ai * xr[None, :]

    diag_kb_lo = tl.maximum(my_lo, diag_lo_v)
    diag_kb_hi = tl.minimum(my_hi, diag_hi_v)
    for kb in tl.range(diag_kb_lo, diag_kb_hi, BLOCK_K):
        j = kb + offs_k
        j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
        j_mask = j < n
        j64 = j.to(tl.int64 if INDEX64 else tl.int32)
        if TRANS == 0:
            if UPLO == 1:
                tri = j[None, :] >= rows[:, None]
                col_base = j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                tri = j[None, :] <= rows[:, None]
                col_base = j64 * n64 - j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
        else:
            if UPLO == 1:
                tri = j[None, :] <= rows[:, None]
            else:
                tri = j[None, :] >= rows[:, None]
            off = row_base[:, None] + j64[None, :]
        if UNIT:
            tri = tri & (j[None, :] != rows[:, None])
        mask = row_mask[:, None] & j_mask[None, :] & tri
        safe_off = tl.where(mask, off, 0)
        a_off = safe_off * 2
        x_off = j * 2
        ar = tl.load(
            ap_ptr + a_off, mask=mask, other=0.0, eviction_policy="evict_first"
        )
        ai = tl.load(
            ap_ptr + a_off + 1, mask=mask, other=0.0, eviction_policy="evict_first"
        )
        xr = tl.load(
            xin_ptr + x_off, mask=j_mask, other=0.0, eviction_policy="evict_last"
        )
        xi = tl.load(
            xin_ptr + x_off + 1, mask=j_mask, other=0.0, eviction_policy="evict_last"
        )
        if CONJ:
            ai = -ai
        acc_r += ar * xr[None, :] - ai * xi[None, :]
        acc_i += ar * xi[None, :] + ai * xr[None, :]

    if UPLO != TRANS:
        post_lo = tl.maximum(my_lo, diag_hi_v)
        for kb in tl.range(post_lo, my_hi, BLOCK_K):
            j = kb + offs_k
            j = tl.max_contiguous(tl.multiple_of(j, BLOCK_K), BLOCK_K)
            j_mask = j < n
            j64 = j.to(tl.int64 if INDEX64 else tl.int32)
            if TRANS == 0:
                col_base = j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                off = row_base[:, None] + j64[None, :]
            mask = row_mask[:, None] & j_mask[None, :]
            safe_off = tl.where(mask, off, 0)
            a_off = safe_off * 2
            x_off = j * 2
            ar = tl.load(
                ap_ptr + a_off, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            ai = tl.load(
                ap_ptr + a_off + 1, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            xr = tl.load(
                xin_ptr + x_off, mask=j_mask, other=0.0, eviction_policy="evict_last"
            )
            xi = tl.load(
                xin_ptr + x_off + 1,
                mask=j_mask,
                other=0.0,
                eviction_policy="evict_last",
            )
            if CONJ:
                ai = -ai
            acc_r += ar * xr[None, :] - ai * xi[None, :]
            acc_i += ar * xi[None, :] + ai * xr[None, :]

    acc_r = tl.sum(acc_r, axis=1)
    acc_i = tl.sum(acc_i, axis=1)
    out_off = pid_k * n * 2 + rows * 2
    tl.store(partial_ptr + out_off, acc_r, mask=row_mask)
    tl.store(partial_ptr + out_off + 1, acc_i, mask=row_mask)


@libentry()
@triton.jit
def ctpmv_reduce_kernel(
    partial_ptr,
    xin_ptr,
    x_ptr,
    n,
    INCX,
    SPLIT_K: tl.constexpr,
    UNIT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < n
    acc_r = tl.zeros((BLOCK_N,), dtype=tl.float32)
    acc_i = tl.zeros((BLOCK_N,), dtype=tl.float32)
    for k in tl.static_range(0, SPLIT_K):
        v_r = tl.load(partial_ptr + k * n * 2 + offs * 2, mask=mask, other=0.0)
        v_i = tl.load(partial_ptr + k * n * 2 + offs * 2 + 1, mask=mask, other=0.0)
        acc_r += v_r
        acc_i += v_i
    if UNIT:
        acc_r += tl.load(xin_ptr + offs * 2, mask=mask, other=0.0)
        acc_i += tl.load(xin_ptr + offs * 2 + 1, mask=mask, other=0.0)
    out_off = offs * INCX * 2
    tl.store(x_ptr + out_off, acc_r, mask=mask)
    tl.store(x_ptr + out_off + 1, acc_i, mask=mask)




@libentry()
@triton.jit
def _stpmv_small_direct(
    AP, XIN, OUT, N: tl.constexpr, IX: tl.constexpr,
    UPLO: tl.constexpr, TRANS: tl.constexpr, UNIT: tl.constexpr,
    BM: tl.constexpr, BK: tl.constexpr,
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.arange(0, BK)
    if TRANS == 0:
        if UPLO == 1:
            base = (col * (col + 1)) >> 1
            valid = col[None, :] >= row[:, None]
        else:
            base = (col * (2 * N - col - 1)) >> 1
            valid = col[None, :] <= row[:, None]
        off = base[None, :] + row[:, None]
    else:
        if UPLO == 1:
            base = (row * (row + 1)) >> 1
            valid = col[None, :] <= row[:, None]
        else:
            base = (row * (2 * N - row - 1)) >> 1
            valid = col[None, :] >= row[:, None]
        off = base[:, None] + col[None, :]
    if UNIT:
        valid &= row[:, None] != col[None, :]
    valid &= (row[:, None] < N) & (col[None, :] < N)
    off = tl.where(valid, off, 0)
    av = tl.load(AP + off, valid, 0.0)
    xv = tl.load(XIN + col, col < N, 0.0)
    outv = tl.sum(av * xv[None, :], 1)
    if UNIT:
        outv += tl.load(XIN + row, row < N, 0.0)
    tl.store(OUT + row.to(tl.int64) * IX, outv, row < N)


def _stpmv_small_path(uplo, trans, unit, n, AP, x, incx):
    xin = x.as_strided((n,), (incx,)).clone()
    bm = 32 if n <= 128 else 16
    _launch(
        _stpmv_small_direct, (triton.cdiv(n, bm),), (AP, xin, x), (),
        (n, incx, uplo, trans, unit, bm, triton.next_power_of_2(n)),
    )


# A packed tile contributes to one output strip. The finish kernel masks out
# the opposite triangle, so no temporary-buffer initialization is required.
@libentry()
@triton.jit
def _stpmv_partial_tile(
    AP, X, P, N: tl.constexpr, IX: tl.constexpr,
    UPLO: tl.constexpr, UNIT: tl.constexpr, B: tl.constexpr,
):
    tile = tl.program_id(0)
    major = ((tl.sqrt(tile.to(tl.float32) * 8.0 + 1.0) - 1.0) * 0.5).to(tl.int32)
    major = tl.where(((major * (major + 1)) >> 1) > tile, major - 1, major)
    major = tl.where((((major + 1) * (major + 2)) >> 1) <= tile, major + 1, major)
    minor = tile - ((major * (major + 1)) >> 1)
    if UPLO == 1:
        rb, cb = major, minor
    else:
        rb, cb = minor, major
    lane = tl.arange(0, B)
    rows, cols = rb * B + lane, cb * B + lane
    if UPLO == 1:
        base = (rows * (rows + 1)) >> 1
        valid = (rows[:, None] < N) & (cols[None, :] < N) & (cols[None, :] <= rows[:, None])
    else:
        base = (rows * (2 * N - rows - 1)) >> 1
        valid = (rows[:, None] < N) & (cols[None, :] < N) & (cols[None, :] >= rows[:, None])
    if UNIT:
        valid &= rows[:, None] != cols[None, :]
    av = tl.load(AP + base[:, None] + cols[None, :], valid, 0.0)
    xr = tl.load(X + rows.to(tl.int64) * IX, rows < N, 0.0)
    out = tl.sum(av * xr[:, None], 0)
    tl.store(P + rb * N + cols, out, cols < N)


@libentry()
@triton.jit
def _stpmv_finish_tile(
    P, X, N: tl.constexpr, IX: tl.constexpr, SPLITS: tl.constexpr,
    R: tl.constexpr, B: tl.constexpr, UPLO: tl.constexpr,
    UNIT: tl.constexpr,
):
    block = tl.program_id(0)
    rows = block * B + tl.arange(0, B)
    acc = tl.full((R, B), 0.0, tl.float32)
    for start in range(0, SPLITS, R):
        split = start + tl.arange(0, R)
        mask = (split[:, None] < SPLITS) & (rows[None, :] < N)
        if UPLO == 1:
            mask &= split[:, None] >= block
        else:
            mask &= split[:, None] <= block
        acc += tl.load(P + split[:, None] * N + rows[None, :], mask, 0.0)
    result = tl.sum(acc, 0)
    if UNIT:
        result += tl.load(X + rows.to(tl.int64) * IX, rows < N, 0.0)
    tl.store(X + rows.to(tl.int64) * IX, result, rows < N)


@libentry()
@triton.jit
def _ctpmv_small_direct(
    AP, XIN, OUT, N: tl.constexpr, IX: tl.constexpr,
    UPLO: tl.constexpr, TRANS: tl.constexpr, CONJ: tl.constexpr,
    UNIT: tl.constexpr, BM: tl.constexpr, BK: tl.constexpr,
):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.arange(0, BK)
    if TRANS == 0:
        if UPLO == 1:
            base = (col * (col + 1)) >> 1
            valid = col[None, :] >= row[:, None]
        else:
            base = (col * (2 * N - col - 1)) >> 1
            valid = col[None, :] <= row[:, None]
        off = base[None, :] + row[:, None]
    else:
        if UPLO == 1:
            base = (row * (row + 1)) >> 1
            valid = col[None, :] <= row[:, None]
        else:
            base = (row * (2 * N - row - 1)) >> 1
            valid = col[None, :] >= row[:, None]
        off = base[:, None] + col[None, :]
    if UNIT:
        valid &= row[:, None] != col[None, :]
    valid &= (row[:, None] < N) & (col[None, :] < N)
    off = tl.where(valid, off, 0) * 2
    ar = tl.load(AP + off, valid, 0.0)
    ai = tl.load(AP + off + 1, valid, 0.0)
    if CONJ:
        ai = -ai
    xr = tl.load(XIN + col * 2, col < N, 0.0)
    xi = tl.load(XIN + col * 2 + 1, col < N, 0.0)
    rr = tl.sum(ar * xr[None, :] - ai * xi[None, :], 1)
    ii = tl.sum(ar * xi[None, :] + ai * xr[None, :], 1)
    if UNIT:
        rr += tl.load(XIN + row * 2, row < N, 0.0)
        ii += tl.load(XIN + row * 2 + 1, row < N, 0.0)
    out = row.to(tl.int64) * (IX * 2)
    tl.store(OUT + out, rr, row < N)
    tl.store(OUT + out + 1, ii, row < N)


def _ctpmv_small_path(uplo, trans, conj, unit, n, AP, x, incx):
    xin = x.as_strided((n,), (incx,)).clone()
    bm = 16 if n <= 128 else 8
    _launch(
        _ctpmv_small_direct, (triton.cdiv(n, bm),),
        (torch.view_as_real(AP), torch.view_as_real(xin), torch.view_as_real(x)), (),
        (n, incx, uplo, trans, conj, unit, bm, triton.next_power_of_2(n)),
    )


@libentry()
@triton.jit
def _ctpmv_packed_tile(
    AP, X, P, N: tl.constexpr, IX: tl.constexpr,
    UPLO: tl.constexpr, TRANS: tl.constexpr, UNIT: tl.constexpr,
    CONJ: tl.constexpr, B: tl.constexpr,
):
    tile = tl.program_id(0)
    major = ((tl.sqrt(tile.to(tl.float32) * 8.0 + 1.0) - 1.0) * 0.5).to(tl.int32)
    major = tl.where(((major * (major + 1)) >> 1) > tile, major - 1, major)
    major = tl.where((((major + 1) * (major + 2)) >> 1) <= tile, major + 1, major)
    minor = tile - ((major * (major + 1)) >> 1)
    if UPLO == 0:
        rb, cb = major, minor
    else:
        rb, cb = minor, major

    lanes = tl.arange(0, B)
    floats = tl.arange(0, 2 * B)
    parity = floats & 1
    rows = rb * B + lanes
    cols = cb * B + (floats >> 1)
    r = rows if N <= 32767 else rows.to(tl.int64)
    if UPLO == 0:
        base = (r * (r + 1)) >> 1
        triangle = cb * B * 2 + floats[None, :] < (rows[:, None] + 1) * 2
    else:
        base = (r * (2 * N - r - 1)) >> 1
        triangle = cb * B * 2 + floats[None, :] >= rows[:, None] * 2
    offsets = (base[:, None] + cb * B) * 2 + floats[None, :]
    mask = (
        (rows[:, None] < N)
        & (cb * B * 2 + floats[None, :] < N * 2)
        & triangle
    )
    av = tl.load(AP + offsets, mask, 0.0)
    if UNIT:
        av = tl.where(rows[:, None] == cols[None, :], 0.0, av)

    sign = tl.where(parity == 0, 1.0, -1.0)
    if TRANS == 0:
        xoff = (
            cb * B * 2 + floats
            if IX == 1
            else cols.to(tl.int64) * (2 * IX) + parity
        )
        xv = tl.load(X + xoff, cb * B * 2 + floats < N * 2, 0.0)
        result_r = tl.sum(av * (xv * sign)[None, :], 1)
        result_i = tl.sum(av * tl.gather(xv, floats ^ 1, 0)[None, :], 1)
        result = tl.where(
            parity == 0,
            tl.gather(result_r, floats >> 1, 0),
            tl.gather(result_i, floats >> 1, 0),
        )
        out = (cb * N + rb * B) * 2 + floats
        out_mask = rb * B * 2 + floats < N * 2
    else:
        row_float = rb * B + (floats >> 1)
        xoff = (
            rb * B * 2 + floats
            if IX == 1
            else row_float.to(tl.int64) * (2 * IX) + parity
        )
        xv = tl.load(X + xoff, rb * B * 2 + floats < N * 2, 0.0)
        xr = tl.gather(xv, lanes * 2, 0)
        xi = tl.gather(xv, lanes * 2 + 1, 0)
        times_r = tl.sum(av * xr[:, None], 0)
        times_i = tl.sum(av * xi[:, None], 0)
        swapped_i = tl.gather(times_i, floats ^ 1, 0)
        if CONJ:
            result = times_r * sign + swapped_i
        else:
            cross_sign = tl.where(parity == 0, -1.0, 1.0)
            result = times_r + cross_sign * swapped_i
        out = (rb * N + cb * B) * 2 + floats
        out_mask = cb * B * 2 + floats < N * 2
    tl.store(P + out, result, out_mask)

@libentry()
@triton.jit
def _ctpmv_packed_finish(
    P, X, N: tl.constexpr, IX: tl.constexpr, SPLITS: tl.constexpr,
    SPLIT_LE_BLOCK: tl.constexpr, UNIT: tl.constexpr,
    REDUCE: tl.constexpr, B: tl.constexpr,
):
    floats = tl.arange(0, 2 * B)
    rows = tl.program_id(0) * B + (floats >> 1)
    acc2d = tl.full((REDUCE, 2 * B), 0.0, tl.float32)
    for start in range(0, SPLITS, REDUCE):
        split = start + tl.arange(0, REDUCE)
        mask = (
            (split[:, None] < SPLITS)
            & (tl.program_id(0) * B * 2 + floats[None, :] < N * 2)
        )
        if SPLIT_LE_BLOCK:
            mask &= split[:, None] <= tl.program_id(0)
        else:
            mask &= split[:, None] >= tl.program_id(0)
        offsets = (
            (split[:, None] * N + tl.program_id(0) * B) * 2
            + floats[None, :]
        )
        acc2d += tl.load(P + offsets, mask, 0.0)
    result = tl.sum(acc2d, 0)
    xoff = (
        tl.program_id(0) * B * 2 + floats
        if IX == 1
        else rows.to(tl.int64) * (2 * IX) + (floats & 1)
    )
    output_mask = tl.program_id(0) * B * 2 + floats < N * 2
    if UNIT:
        result += tl.load(X + xoff, output_mask, 0.0)
    tl.store(X + xoff, result, output_mask)


def _ctpmv_packed_path(uplo, trans, conj, unit, n, AP, x, incx):
    block = 64
    splits = triton.cdiv(n, block)
    partial = torch.empty((splits, n, 2), dtype=torch.float32, device=AP.device)
    _launch(
        _ctpmv_packed_tile,
        (min(65535, splits * (splits + 1) // 2),),
        (AP, x, partial), (),
        (n, incx, uplo, trans, unit, conj, block),
    )
    finish_block = block
    _launch(
        _ctpmv_packed_finish, (triton.cdiv(n, finish_block),),
        (partial, x), (),
        (n, incx, splits, uplo == trans, unit,
         min(32, triton.next_power_of_2(splits)), finish_block),
    )


@libentry()
@triton.jit
def _ctpmv_tiny_inplace(
    AP, X, N: tl.constexpr, IX: tl.constexpr,
    UPLO: tl.constexpr, TRANS: tl.constexpr, CONJ: tl.constexpr,
    UNIT: tl.constexpr, B: tl.constexpr,
):
    # A single program loads the entire input before its only output store.
    rows = tl.arange(0, B)
    floats = tl.arange(0, 2 * B)
    cols = floats >> 1
    r = cols[None, :] if TRANS else rows[:, None]
    c = rows[:, None] if TRANS else cols[None, :]
    if UPLO == 0:
        base = (r * (r + 1)) >> 1
        mask = c <= r
    else:
        base = (r * (2 * N - r - 1)) >> 1
        mask = c >= r
    mask &= (r < N) & (c < N)
    if UNIT:
        mask &= r != c
    if TRANS:
        offsets = (base + c) * 2 + (floats[None, :] & 1)
    else:
        offsets = base * 2 + floats[None, :]
    av = tl.load(AP + offsets, mask, 0.0)
    sign = tl.where((floats & 1) == 0, 1.0, -1.0)
    if CONJ:
        av *= sign[None, :]
    xoff = cols * (2 * IX) + (floats & 1)
    xv = tl.load(X + xoff, cols < N, 0.0)
    rr = tl.sum(av * (xv * sign)[None, :], 1)
    ii = tl.sum(av * tl.gather(xv, floats ^ 1, 0)[None, :], 1)
    result = tl.where(
        (floats & 1) == 0,
        tl.gather(rr, floats >> 1, 0),
        tl.gather(ii, floats >> 1, 0),
    )
    if UNIT:
        result += xv
    tl.store(X + xoff, result, cols < N)


def _ctpmv_small_dispatch(uplo, trans, conj, unit, n, AP, x, incx):
    """Small-only dispatch; larger sizes keep their existing launch policy."""
    if n <= 32:
        _launch(
            _ctpmv_tiny_inplace, (1,), (AP, x), (),
            (n, incx, uplo, trans, conj, unit, triton.next_power_of_2(n)),
        )
    else:
        # The first kernel reads x and writes partials; only the second
        # kernel updates x, so the small path needs no input-vector clone.
        _ctpmv_packed_path(uplo, trans, conj, unit, n, AP, x, incx)


_CTPMV_SPLIT_K_NO_TRANS = (
    (511, 1),
    (2500, 2),
    (7500, 4),
    (13000, 3),
)
_CTPMV_SPLIT_K_TRANS = (
    (511, 1),
    (1500, 4),
    (3500, 3),
    (7500, 4),
    (11000, 3),
    (13000, 2),
)


def _ctpmv_split_k(n: int, trans: int) -> int:
    table = _CTPMV_SPLIT_K_NO_TRANS if trans == 0 else _CTPMV_SPLIT_K_TRANS
    for n_max, split_k in table:
        if n <= n_max:
            return split_k
    return 1


@libentry()
@triton.jit
def _stpmv_two_rhs_tile(
    AP, X0, X1, P0, P1, N: tl.constexpr,
    UPLO: tl.constexpr, B: tl.constexpr,
):
    tile = tl.program_id(0)
    major = ((tl.sqrt(tile.to(tl.float32) * 8.0 + 1.0) - 1.0) * 0.5).to(tl.int32)
    major = tl.where(((major * (major + 1)) >> 1) > tile, major - 1, major)
    major = tl.where((((major + 1) * (major + 2)) >> 1) <= tile, major + 1, major)
    minor = tile - ((major * (major + 1)) >> 1)
    if UPLO == 1:
        rb, cb = major, minor
    else:
        rb, cb = minor, major
    lane = tl.arange(0, B)
    rows, cols = rb * B + lane, cb * B + lane
    if UPLO == 1:
        base = (rows * (rows + 1)) >> 1
        valid = (rows[:, None] < N) & (cols[None, :] < N) & (cols[None, :] <= rows[:, None])
    else:
        base = (rows * (2 * N - rows - 1)) >> 1
        valid = (rows[:, None] < N) & (cols[None, :] < N) & (cols[None, :] >= rows[:, None])
    av = tl.load(AP + base[:, None] + cols[None, :], valid, 0.0)
    x0 = tl.load(X0 + rows, rows < N, 0.0)
    x1 = tl.load(X1 + rows, rows < N, 0.0)
    out0 = tl.sum(av * x0[:, None], 0)
    tl.store(P0 + rb * N + cols, out0, cols < N)
    out1 = tl.sum(av * x1[:, None], 0)
    tl.store(P1 + rb * N + cols, out1, cols < N)


@libentry()
@triton.jit
def _stpmv_two_rhs_row_tile(
    AP, X0, X1, P0, P1, N: tl.constexpr,
    UPLO: tl.constexpr, B: tl.constexpr,
):
    tile = tl.program_id(0)
    major = ((tl.sqrt(tile.to(tl.float32) * 8.0 + 1.0) - 1.0) * 0.5).to(tl.int32)
    major = tl.where(((major * (major + 1)) >> 1) > tile, major - 1, major)
    major = tl.where((((major + 1) * (major + 2)) >> 1) <= tile, major + 1, major)
    minor = tile - ((major * (major + 1)) >> 1)
    if UPLO == 1:
        rb, cb = major, minor
    else:
        rb, cb = minor, major
    lane = tl.arange(0, B)
    rows, cols = rb * B + lane, cb * B + lane
    if UPLO == 1:
        base = (rows * (rows + 1)) >> 1
        valid = (rows[:, None] < N) & (cols[None, :] < N) & (cols[None, :] <= rows[:, None])
    else:
        base = (rows * (2 * N - rows - 1)) >> 1
        valid = (rows[:, None] < N) & (cols[None, :] < N) & (cols[None, :] >= rows[:, None])
    av = tl.load(AP + base[:, None] + cols[None, :], valid, 0.0)
    x0 = tl.load(X0 + cols, cols < N, 0.0)
    x1 = tl.load(X1 + cols, cols < N, 0.0)
    out0 = tl.sum(av * x0[None, :], 1)
    tl.store(P0 + cb * N + rows, out0, rows < N)
    out1 = tl.sum(av * x1[None, :], 1)
    tl.store(P1 + cb * N + rows, out1, rows < N)


@libentry()
@triton.jit
def _stpmv_finish_row_tile(
    P, X, N: tl.constexpr, SPLITS: tl.constexpr,
    R: tl.constexpr, B: tl.constexpr, UPLO: tl.constexpr,
):
    block = tl.program_id(0)
    rows = block * B + tl.arange(0, B)
    acc = tl.full((R, B), 0.0, tl.float32)
    for start in range(0, SPLITS, R):
        split = start + tl.arange(0, R)
        mask = (split[:, None] < SPLITS) & (rows[None, :] < N)
        if UPLO == 1:
            mask &= split[:, None] <= block
        else:
            mask &= split[:, None] >= block
        acc += tl.load(P + split[:, None] * N + rows[None, :], mask, 0.0)
    tl.store(X + rows, tl.sum(acc, 0), rows < N)


def _stpmv_two_rhs_row(uplo, n, AP, x0, x1):
    block = 64
    splits = triton.cdiv(n, block)
    partial = torch.empty((2, splits, n), dtype=torch.float32, device=AP.device)
    _launch(
        _stpmv_two_rhs_row_tile, (splits * (splits + 1) // 2,),
        (AP, x0, x1, partial[0], partial[1]), (),
        (n, uplo, block),
    )
    for part, vector in enumerate((x0, x1)):
        _launch(
            _stpmv_finish_row_tile, (splits,), (partial[part], vector), (),
            (n, splits, min(32, triton.next_power_of_2(splits)), block, uplo),
        )


def _stpmv_two_rhs(uplo, n, AP, x0, x1):
    block = 64
    splits = triton.cdiv(n, block)
    partial = torch.empty((2, splits, n), dtype=torch.float32, device=AP.device)
    _launch(
        _stpmv_two_rhs_tile, (splits * (splits + 1) // 2,),
        (AP, x0, x1, partial[0], partial[1]), (),
        (n, uplo, block),
    )
    for part, vector in enumerate((x0, x1)):
        _launch(
            _stpmv_finish_tile, (splits,), (partial[part], vector), (),
            (n, 1, splits, min(32, triton.next_power_of_2(splits)),
             block, uplo, 0),
        )


@libentry()
@triton.jit
def _ctpmv_unit_diagonal(AR, AI, N: tl.constexpr, UPLO: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0) * B + tl.arange(0, B)
    if UPLO == 1:
        off = ((row * (row + 1)) >> 1) + row
    else:
        off = row * N - ((row * (row + 1)) >> 1) + row
    tl.store(AR + off, 1.0, row < N)
    tl.store(AI + off, 0.0, row < N)


@libentry()
@triton.jit
def _ctpmv_combine(P, Q, R, S, X, N: tl.constexpr, IX: tl.constexpr,
                   CONJ: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0) * B + tl.arange(0, B)
    mask = row < N
    p = tl.load(P + row, mask, 0.0)
    q = tl.load(Q + row, mask, 0.0)
    r = tl.load(R + row, mask, 0.0)
    s = tl.load(S + row, mask, 0.0)
    if CONJ:
        real = p + q
        imag = r - s
    else:
        real = p - q
        imag = r + s
    out = row.to(tl.int64) * (2 * IX)
    tl.store(X + out, real, mask)
    tl.store(X + out + 1, imag, mask)


def ctpmv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    AP: torch.Tensor,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert AP.dtype == torch.complex64 == x.dtype
    _check_tpmv(AP, x, uplo, trans, diag, n, incx, complex_ok=True)
    if n == 0:
        return
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    real_uplo = uplo
    real_trans = trans
    uplo, trans, conj = _row_major_tpmv_args(uplo, trans)
    trans_flag = 0 if trans == CUBLAS_OP_N else 1
    split_k = _ctpmv_split_k(n, trans_flag)

    if AP.device.index != _current_device():
        with torch_device_fn.device(AP.device):
            return ctpmv(real_uplo, real_trans, diag, n, AP, x, incx)
    if n <= 256:
        _ctpmv_small_dispatch(
            real_uplo, int(real_trans != CUBLAS_OP_N), conj,
            unit, n, AP, x, incx,
        )
        return
    if 384 <= n <= 16384:
        _ctpmv_packed_path(
            real_uplo, int(real_trans != CUBLAS_OP_N), conj,
            unit, n, AP, x, incx,
        )
        return
    xin = x.as_strided((n,), (incx,)).clone()
    AP_real = torch.view_as_real(AP)
    xin_real = torch.view_as_real(xin)
    x_real = torch.view_as_real(x)
    if split_k > 1:
        partial = torch.empty(
            (split_k, n, 2), dtype=torch.float32, device=AP.device
        )
        grid_main = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]), split_k)
        ctpmv_splitk_kernel[grid_main](
            AP_real,
            xin_real,
            partial,
            n,
            _mode_key(uplo, trans_flag, unit) | (conj << 8),
            SPLIT_K=split_k,
            UPLO=uplo,
            TRANS=trans_flag,
            UNIT=unit,
            CONJ=conj,
            INDEX64=n > 32767,
        )
        BLOCK_N = 1024
        grid_red = (triton.cdiv(n, BLOCK_N),)
        ctpmv_reduce_kernel[grid_red](
            partial,
            xin_real,
            x_real,
            n,
            incx,
            SPLIT_K=split_k,
            UNIT=unit,
            BLOCK_N=BLOCK_N,
            num_warps=4,
        )
    else:
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE_M"]),)
        ctpmv_kernel[grid](
            AP_real,
            xin_real,
            x_real,
            n,
            incx,
            _mode_key(uplo, trans_flag, unit) | (conj << 8),
            UPLO=uplo,
            TRANS=trans_flag,
            UNIT=unit,
            CONJ=conj,
            INDEX64=n > 32767,
            )


def _ctpmv_via_real(uplo, trans, n, AP, x, incx, unit, conj):
    packed = torch.view_as_real(AP)
    ar = packed[:, 0].contiguous()
    ai = packed[:, 1].contiguous()
    if unit:
        physical_uplo, _, _ = _row_major_tpmv_args(uplo, trans)
        _launch(
            _ctpmv_unit_diagonal, (triton.cdiv(n, 256),), (ar, ai), (),
            (n, physical_uplo, 256),
        )

    source = x.as_strided((n,), (incx,))
    components = torch.view_as_real(source)
    xr = components[:, 0].contiguous()
    xi = components[:, 1].contiguous()
    p, q, r, s = xr.clone(), xi.clone(), xi.clone(), xr.clone()
    if trans == CUBLAS_OP_T:
        physical_uplo, _, _ = _row_major_tpmv_args(uplo, trans)
        _stpmv_two_rhs(physical_uplo, n, ar, p, r)
        _stpmv_two_rhs(physical_uplo, n, ai, q, s)
    else:
        physical_uplo, _, _ = _row_major_tpmv_args(uplo, trans)
        _stpmv_two_rhs_row(physical_uplo, n, ar, p, r)
        _stpmv_two_rhs_row(physical_uplo, n, ai, q, s)
    _launch(
        _ctpmv_combine, (triton.cdiv(n, 256),), (p, q, r, s, x), (),
        (n, incx, conj, 256),
    )
