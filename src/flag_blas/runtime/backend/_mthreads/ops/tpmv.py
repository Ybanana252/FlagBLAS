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

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.ops.level2.tpmv import _check_tpmv, _row_major_tpmv_args
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

_CONFIGS = {
    (c.kwargs["BM"], c.kwargs["BK"], c.num_warps, c.num_stages): dict(
        c.kwargs, num_warps=c.num_warps, num_stages=c.num_stages
    )
    for c in runtime.get_tuned_config("mthreads_tpmv")
}


@libentry()
@triton.jit
def mthreads_stpmv_small_kernel(
    AP,
    X,
    N,
    SX,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    B: tl.constexpr,
):
    r = tl.arange(0, B).to(tl.int64)
    j = tl.arange(0, B).to(tl.int64)
    n64 = tl.full((), N, tl.int64)
    xin = tl.load(X + r * SX, r < N, 0)
    if TRANS == 0:
        if UPLO == 1:
            tri = j[None, :] >= r[:, None]
            off = j[None, :] * (j[None, :] + 1) // 2 + r[:, None]
        else:
            tri = j[None, :] <= r[:, None]
            off = j[None, :] * n64 - j[None, :] * (j[None, :] + 1) // 2 + r[:, None]
    else:
        if UPLO == 1:
            tri = j[None, :] <= r[:, None]
            off = r[:, None] * (r[:, None] + 1) // 2 + j[None, :]
        else:
            tri = j[None, :] >= r[:, None]
            off = r[:, None] * n64 - r[:, None] * (r[:, None] + 1) // 2 + j[None, :]
    if UNIT:
        tri &= j[None, :] != r[:, None]
    mask = (r[:, None] < N) & (j[None, :] < N) & tri
    a = tl.load(AP + tl.where(mask, off, 0), mask, 0)
    out = tl.sum(a * xin[None, :], axis=1)
    if UNIT:
        out += xin
    tl.debug_barrier()
    tl.store(X + r * SX, out, r < N)


@libentry()
@triton.jit
def mthreads_stpmv_kernel(
    ap_ptr,
    xin_ptr,
    x_ptr,
    n,
    INCX,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BM
    rows = row_start + tl.arange(0, BM)
    rows = tl.max_contiguous(tl.multiple_of(rows, BM), BM)
    row_mask = rows < n
    acc = tl.zeros((BM,), dtype=tl.float32)

    n64 = tl.full((), n, tl.int64)
    rows64 = rows.to(tl.int64)
    offs_k = tl.arange(0, BK)

    if TRANS == 1:
        if UPLO == 1:
            row_base = rows64 * (rows64 + 1) // 2
        else:
            row_base = rows64 * n64 - rows64 * (rows64 + 1) // 2

    if UPLO == TRANS:
        for kb in tl.range(0, row_start, BK):
            j = kb + offs_k
            j = tl.max_contiguous(tl.multiple_of(j, BK), BK)
            j_mask = j < n
            j64 = j.to(tl.int64)
            if TRANS == 0:
                col_base = j64 * n64 - j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                off = row_base[:, None] + j64[None, :]
            mask = row_mask[:, None] & j_mask[None, :]
            a_vals = tl.load(
                ap_ptr + off, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            x_vals = tl.load(
                xin_ptr + j, mask=j_mask, other=0.0, eviction_policy="evict_last"
            )
            acc += tl.sum(a_vals * x_vals[None, :], axis=1)
        diag_lo = row_start
    else:
        diag_lo = (row_start // BK) * BK

    for kb in tl.range(diag_lo, row_start + BM, BK):
        j = kb + offs_k
        j = tl.max_contiguous(tl.multiple_of(j, BK), BK)
        j_mask = j < n
        j64 = j.to(tl.int64)
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
        a_vals = tl.load(
            ap_ptr + off, mask=mask, other=0.0, eviction_policy="evict_first"
        )
        x_vals = tl.load(
            xin_ptr + j, mask=j_mask, other=0.0, eviction_policy="evict_last"
        )
        acc += tl.sum(a_vals * x_vals[None, :], axis=1)

    if UPLO != TRANS:
        for kb in tl.range(row_start + BM, n, BK):
            j = kb + offs_k
            j = tl.max_contiguous(tl.multiple_of(j, BK), BK)
            j_mask = j < n
            j64 = j.to(tl.int64)
            if TRANS == 0:
                col_base = j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                off = row_base[:, None] + j64[None, :]
            mask = row_mask[:, None] & j_mask[None, :]
            a_vals = tl.load(
                ap_ptr + off, mask=mask, other=0.0, eviction_policy="evict_first"
            )
            x_vals = tl.load(
                xin_ptr + j, mask=j_mask, other=0.0, eviction_policy="evict_last"
            )
            acc += tl.sum(a_vals * x_vals[None, :], axis=1)

    if UNIT:
        acc += tl.load(xin_ptr + rows, mask=row_mask, other=0.0)

    tl.store(x_ptr + rows.to(tl.int64) * INCX, acc, mask=row_mask)


@libentry()
@triton.jit
def mthreads_ctpmv_kernel(
    ap_ptr,
    xin_ptr,
    x_ptr,
    n,
    INCX,
    UPLO: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    CONJ: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BM
    rows = row_start + tl.arange(0, BM)
    rows = tl.max_contiguous(tl.multiple_of(rows, BM), BM)
    row_mask = rows < n
    acc_r = tl.zeros((BM,), dtype=tl.float32)
    acc_i = tl.zeros((BM,), dtype=tl.float32)

    n64 = tl.full((), n, tl.int64)
    rows64 = rows.to(tl.int64)
    offs_k = tl.arange(0, BK)

    if TRANS == 1:
        if UPLO == 1:
            row_base = rows64 * (rows64 + 1) // 2
        else:
            row_base = rows64 * n64 - rows64 * (rows64 + 1) // 2

    if UPLO == TRANS:
        for kb in tl.range(0, row_start, BK):
            j = kb + offs_k
            j = tl.max_contiguous(tl.multiple_of(j, BK), BK)
            j_mask = j < n
            j64 = j.to(tl.int64)
            if TRANS == 0:
                col_base = j64 * n64 - j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                off = row_base[:, None] + j64[None, :]
            mask = row_mask[:, None] & j_mask[None, :]
            a_off = off * 2
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
            acc_r += tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
            acc_i += tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)
        diag_lo = row_start
    else:
        diag_lo = (row_start // BK) * BK

    for kb in tl.range(diag_lo, row_start + BM, BK):
        j = kb + offs_k
        j = tl.max_contiguous(tl.multiple_of(j, BK), BK)
        j_mask = j < n
        j64 = j.to(tl.int64)
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
        a_off = off * 2
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
        acc_r += tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
        acc_i += tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)

    if UPLO != TRANS:
        for kb in tl.range(row_start + BM, n, BK):
            j = kb + offs_k
            j = tl.max_contiguous(tl.multiple_of(j, BK), BK)
            j_mask = j < n
            j64 = j.to(tl.int64)
            if TRANS == 0:
                col_base = j64 * (j64 + 1) // 2
                off = col_base[None, :] + rows64[:, None]
            else:
                off = row_base[:, None] + j64[None, :]
            mask = row_mask[:, None] & j_mask[None, :]
            a_off = off * 2
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
            acc_r += tl.sum(ar * xr[None, :] - ai * xi[None, :], axis=1)
            acc_i += tl.sum(ar * xi[None, :] + ai * xr[None, :], axis=1)

    if UNIT:
        acc_r += tl.load(xin_ptr + rows * 2, mask=row_mask, other=0.0)
        acc_i += tl.load(xin_ptr + rows * 2 + 1, mask=row_mask, other=0.0)

    x_off_out = rows.to(tl.int64) * INCX * 2
    tl.store(x_ptr + x_off_out, acc_r, mask=row_mask)
    tl.store(x_ptr + x_off_out + 1, acc_i, mask=row_mask)


def _tpmv(uplo, trans, diag, n, AP, x, incx, is_complex):
    assert AP.dtype == x.dtype == (torch.complex64 if is_complex else torch.float32)
    _check_tpmv(AP, x, uplo, trans, diag, n, incx, complex_ok=is_complex)
    assert AP.device.type == "musa"
    if n == 0:
        return
    physical_uplo, physical_trans, conj = _row_major_tpmv_args(uplo, trans)
    with torch_device_fn.device(AP.device):
        if not is_complex and n <= 64:
            tile = 16 if n <= 16 else 32 if n <= 32 else 64
            config = _CONFIGS[(tile, tile, 16 if tile == 64 else 4, 1)]
            # One CTA reads the input before any thread overwrites x.
            mthreads_stpmv_small_kernel[(1,)](
                AP,
                x,
                n,
                incx,
                UPLO=physical_uplo,
                TRANS=physical_trans,
                UNIT=diag,
                B=config["BM"],
                num_warps=config["num_warps"],
                num_stages=config["num_stages"],
            )
            return
        config = _CONFIGS[(16, 16, 4, 1)]
        # Copy the logical vector only; stride padding is excluded.
        xin = x.as_strided((n,), (incx,)).clone()
        a, xi, y = (
            (torch.view_as_real(v) for v in (AP, xin, x))
            if is_complex
            else (AP, xin, x)
        )
        kernel = mthreads_ctpmv_kernel if is_complex else mthreads_stpmv_kernel
        kwargs = dict(UPLO=physical_uplo, TRANS=physical_trans, UNIT=diag, **config)
        if is_complex:
            kwargs["CONJ"] = conj
        kernel[(triton.cdiv(n, config["BM"]),)](a, xi, y, n, incx, **kwargs)


def stpmv(uplo, trans, diag, n, AP, x, incx):
    return _tpmv(uplo, trans, diag, n, AP, x, incx, False)


def ctpmv(uplo, trans, diag, n, AP, x, incx):
    return _tpmv(uplo, trans, diag, n, AP, x, incx, True)
