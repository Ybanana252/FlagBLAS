"""Ascend row-major packed triangular solves.

Small systems keep the packed matrix and RHS on chip. Larger systems invert
independent diagonal blocks, then use one device-side substitution loop instead
of launching a kernel for every pivot. Layout conversion, when needed, shares
the diagonal preparation launch and is rebuilt from AP on every call.
"""

import importlib

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

_common = importlib.import_module("flag_blas.ops.level2.tpsv")
_BLOCK_SIZE = 32
_KERNEL_CACHE = {}


def _launch(kernel, grid, args, key):
    # Cache compiled code only, never matrix data or tensor pointers. Continue
    # through Triton's normal launcher to preserve stream and profiler hooks.
    cache_key = (kernel, key)
    compiled = _KERNEL_CACHE.get(cache_key)
    if compiled is None:
        compiled, _ = kernel[grid](*args, num_stages=1)
        if len(_KERNEL_CACHE) >= 512:
            _KERNEL_CACHE.clear()
        _KERNEL_CACHE[cache_key] = compiled
    else:
        compiled[(grid + (1, 1))[:3]](*args)


@triton.jit
def _packed_offset(row, col, n, UPLO: tl.constexpr, TRANS: tl.constexpr):
    if TRANS:
        row, col = col, row
    if UPLO == 1:
        return row * n - row * (row + 1) // 2 + col
    return row * (row + 1) // 2 + col


@libentry()
@triton.jit
def _tpsv_copy_vector(
    SRC, DST, n, incx, COMPLEX: tl.constexpr, SCATTER: tl.constexpr,
    B: tl.constexpr,
):
    r = tl.program_id(0) * B + tl.arange(0, B)
    src = r if SCATTER else r * incx
    dst = r * incx if SCATTER else r
    if COMPLEX:
        pair = tl.arange(0, 2)
        value = tl.load(SRC + src[:, None] * 2 + pair[None, :],
                        r[:, None] < n, other=0.0)
        tl.store(DST + dst[:, None] * 2 + pair[None, :], value, r[:, None] < n)
    else:
        value = tl.load(SRC + src, r < n, other=0.0)
        tl.store(DST + dst, value, r < n)


@triton.jit
def _tpsv_pack_tile(
    AP, TILES, n, padded,
    UPLO: tl.constexpr, TRANS: tl.constexpr, FORWARD: tl.constexpr,
    COMPLEX: tl.constexpr, B: tl.constexpr, M: tl.constexpr, SAFE: tl.constexpr,
):
    col_block = tl.program_id(0)
    row_base = (col_block + 1) * B + tl.program_id(1) * M
    if row_base < n:
        # Complex transpose temporaries need smaller subtiles to fit in UB.
        S: tl.constexpr = 64 if COMPLEX else M
        rr = tl.arange(0, S)
        cc = tl.arange(0, B)
        for segment in range(M // S):
            local_rows = segment * S + rr
            logical_rows = row_base + (local_rows if FORWARD else M - 1 - local_rows)
            logical_cols = col_block * B + (cc if FORWARD else B - 1 - cc)
            rows = logical_rows if FORWARD else n - 1 - logical_rows
            cols = logical_cols if FORWARD else n - 1 - logical_cols
            if SAFE:
                rows = tl.minimum(tl.maximum(rows, 0), n - 1)
            if TRANS:
                offsets = _packed_offset(cols[:, None], rows[None, :], n, UPLO, False)
                mask = (logical_cols[:, None] < n) & (logical_rows[None, :] < n)
            else:
                offsets = _packed_offset(rows[:, None], cols[None, :], n, UPLO, False)
                mask = (logical_rows[:, None] < n) & (logical_cols[None, :] < n)
            out = (
                col_block * (padded + M) * B
                + (row_base + local_rows[:, None]) * B + cc[None, :]
            )
            if COMPLEX:
                pair = tl.arange(0, 2)
                values = tl.load(AP + offsets[:, :, None] * 2 + pair[None, None, :],
                                 mask[:, :, None], other=0.0)
                ar, ai = tl.split(values)
                if TRANS:
                    ar, ai = tl.trans(ar), tl.trans(ai)
                tl.store(TILES + out, ar)
                tl.store(TILES + (padded + M) * padded + out, ai)
            else:
                ar = tl.load(AP + offsets, mask, other=0.0)
                if TRANS:
                    ar = tl.trans(ar)
                tl.store(TILES + out, ar)


@triton.jit
def _tpsv_pack(
    AP, TILES, n, padded,
    UPLO: tl.constexpr, TRANS: tl.constexpr, FORWARD: tl.constexpr,
    COMPLEX: tl.constexpr, B: tl.constexpr, M: tl.constexpr,
):
    row_base = (tl.program_id(0) + 1) * B + tl.program_id(1) * M
    # A separate tail path keeps masked addresses legal without losing the
    # affine, contiguous loads of full tiles. Active column panels are full.
    if row_base + M <= n:
        _tpsv_pack_tile(AP, TILES, n, padded, UPLO, TRANS, FORWARD, COMPLEX, B, M, False)
    else:
        _tpsv_pack_tile(AP, TILES, n, padded, UPLO, TRANS, FORWARD, COMPLEX, B, M, True)


@libentry()
@triton.jit
def _tpsv_prepare(
    AP, X, INV, RHS, TILES, n, padded, incx,
    UPLO: tl.constexpr, TRANS: tl.constexpr, UNIT: tl.constexpr,
    FORWARD: tl.constexpr, COMPLEX: tl.constexpr, CONJ: tl.constexpr,
    B: tl.constexpr, M: tl.constexpr, TILED: tl.constexpr,
):
    if tl.program_id(1) == 0:
        block = tl.program_id(0)
        r = tl.arange(0, B)
        logical = block * B + r
        rows = logical if FORWARD else n - 1 - logical
        valid = logical < n
        load_logical = logical if FORWARD else block * B + B - 1 - r
        load_rows = load_logical if FORWARD else n - 1 - load_logical
        load_rows = tl.minimum(tl.maximum(load_rows, 0), n - 1)
        offsets = _packed_offset(load_rows[:, None], load_rows[None, :], n, UPLO, False)
        mask = (load_logical[:, None] < n) & (load_logical[None, :] < n)
        if UPLO == 0:
            mask &= r[:, None] >= r[None, :]
        else:
            mask &= r[:, None] <= r[None, :]
        if UNIT:
            mask &= r[:, None] != r[None, :]
        if COMPLEX:
            pair = tl.arange(0, 2)
            values = tl.load(AP + offsets[:, :, None] * 2 + pair[None, None, :],
                             mask[:, :, None], other=0.0)
            ar, ai = tl.split(values)
            if TRANS:
                ar, ai = tl.trans(ar), tl.trans(ai)
            if not FORWARD:
                reverse = B * B - 1 - tl.arange(0, B * B)
                ar = tl.reshape(tl.gather(tl.reshape(ar, (B * B,)), reverse, 0), (B, B))
                ai = tl.reshape(tl.gather(tl.reshape(ai, (B * B,)), reverse, 0), (B, B))
            if CONJ:
                ai = -ai
            safe_rows = tl.minimum(tl.maximum(rows, 0), n - 1)
            xv = tl.load(X + safe_rows[:, None] * incx * 2 + pair[None, :],
                         valid[:, None], other=0.0)
            xr, xi = tl.split(xv)
            tl.store(RHS + logical, xr)
            tl.store(RHS + padded + logical, xi)
        else:
            ar = tl.load(AP + offsets, mask, other=0.0)
            if TRANS:
                ar = tl.trans(ar)
            if not FORWARD:
                reverse = B * B - 1 - tl.arange(0, B * B)
                ar = tl.reshape(tl.gather(tl.reshape(ar, (B * B,)), reverse, 0), (B, B))
            safe_rows = tl.minimum(tl.maximum(rows, 0), n - 1)
            xr = tl.load(X + safe_rows * incx, valid, other=0.0)
            tl.store(RHS + logical, xr)
        eye = r[:, None] == r[None, :]
        if UNIT:
            dr = tl.full((B,), 1.0, tl.float32)
            if COMPLEX:
                di = tl.full((B,), 0.0, tl.float32)
        else:
            dr = tl.sum(tl.where(eye, ar, 0.0), 1)
            dr = tl.where(valid, dr, 1.0)
            if COMPLEX:
                di = tl.sum(tl.where(eye, ai, 0.0), 1)
        if COMPLEX:
            scale = 1.0 / (dr * dr + di * di)
            ir, ii = dr * scale, -di * scale
            rr = tl.where(eye, ir[:, None], 0.0)
            ri = tl.where(eye, ii[:, None], 0.0)
            normalized = ar * ir[:, None] - ai * ii[:, None]
            ai = ar * ii[:, None] + ai * ir[:, None]
            ar = normalized
        else:
            ir = 1.0 / dr
            rr = tl.where(eye, ir[:, None], 0.0)
            ar = ar * ir[:, None]
        ar = tl.where(r[:, None] > r[None, :], ar, 0.0)
        if COMPLEX:
            ai = tl.where(r[:, None] > r[None, :], ai, 0.0)
        # All diagonal blocks are independent. Forward substitution on the
        # identity computes each inverse using only B sequential vector steps.
        for k in range(tl.minimum(B, n - block * B)):
            cr = tl.gather(tl.reshape(ar, (B * B,)), r * B + k, 0)
            vr = tl.gather(tl.reshape(rr, (B * B,)), k * B + r, 0)
            if COMPLEX:
                ci = tl.gather(tl.reshape(ai, (B * B,)), r * B + k, 0)
                vi = tl.gather(tl.reshape(ri, (B * B,)), k * B + r, 0)
                rr = rr - cr[:, None] * vr[None, :] + ci[:, None] * vi[None, :]
                ri = ri - cr[:, None] * vi[None, :] - ci[:, None] * vr[None, :]
            else:
                rr = rr - cr[:, None] * vr[None, :]
        out = block * B * B * (2 if COMPLEX else 1) + r[:, None] * B + r[None, :]
        tl.store(INV + out, rr)
        if COMPLEX:
            tl.store(INV + out + B * B, ri)

    if TILED:
        _tpsv_pack(AP, TILES, n, padded, UPLO, TRANS, FORWARD, COMPLEX, B, M)


@libentry()
@triton.jit
def _tpsv_solve_update(
    AP, X, INV, RHS, n, padded, incx,
    UPLO: tl.constexpr, TRANS: tl.constexpr,
    FORWARD: tl.constexpr, COMPLEX: tl.constexpr, CONJ: tl.constexpr,
    B: tl.constexpr, M: tl.constexpr, TILED: tl.constexpr,
):
    r = tl.arange(0, B)
    for start in range(0, n, B):
        inv_off = (start // B) * B * B * (2 if COMPLEX else 1) + r[:, None] * B + r[None, :]
        ir = tl.load(INV + inv_off)
        br = tl.load(RHS + start + r)
        if COMPLEX:
            ii = tl.load(INV + inv_off + B * B)
            bi = tl.load(RHS + padded + start + r)
            xr = tl.sum(ir * br[None, :] - ii * bi[None, :], 1)
            xi = tl.sum(ir * bi[None, :] + ii * br[None, :], 1)
        else:
            xr = tl.sum(ir * br[None, :], 1)
        logical_cols = start + r
        cols = logical_cols if FORWARD else n - 1 - logical_cols
        if not FORWARD:
            # Reverse only vectors in UB. Reversing GM accesses or entire
            # matrix tiles is considerably more expensive on Ascend.
            xr = tl.gather(xr, B - 1 - r, 0)
            if COMPLEX:
                xi = tl.gather(xi, B - 1 - r, 0)
            store_cols = n - start - B + r
            store_mask = store_cols >= 0
            store_cols = tl.maximum(store_cols, 0)
        else:
            store_cols = cols
            store_mask = logical_cols < n
        if COMPLEX:
            pair = tl.arange(0, 2)
            tl.store(X + store_cols[:, None] * incx * 2 + pair[None, :],
                     tl.join(xr, xi), store_mask[:, None])
        else:
            tl.store(X + store_cols * incx, xr, store_mask)
        # A single program owns the dependency chain and all RHS updates.
        for row_base in range(start + B, n, M):
            logical_rows = row_base + tl.arange(0, M)
            if TILED:
                off = (start // B) * (padded + M) * B + logical_rows[:, None] * B + r[None, :]
                # Packing initializes complete tiles, including padding.
                mask = tl.full((M, B), True, tl.int1)
            else:
                rows = logical_rows if FORWARD else n - 1 - logical_rows
                off = _packed_offset(rows[:, None], cols[None, :], n, UPLO, TRANS)
                mask = (logical_rows[:, None] < n) & (logical_cols[None, :] < n)
            if COMPLEX:
                if TILED:
                    ar = tl.load(AP + off)
                    ai = tl.load(AP + (padded + M) * padded + off)
                else:
                    pair = tl.arange(0, 2)
                    av = tl.load(AP + off[:, :, None] * 2 + pair[None, None, :],
                                 mask[:, :, None], other=0.0)
                    ar, ai = tl.split(av)
                if CONJ:
                    ai = -ai
                yr = tl.load(RHS + logical_rows, logical_rows < n, other=0.0)
                yi = tl.load(RHS + padded + logical_rows, logical_rows < n, other=0.0)
                delta_r = tl.sum(ar * xr[None, :] - ai * xi[None, :], 1)
                delta_i = tl.sum(ar * xi[None, :] + ai * xr[None, :], 1)
                if TILED and not FORWARD:
                    reverse = M - 1 - tl.arange(0, M)
                    delta_r = tl.gather(delta_r, reverse, 0)
                    delta_i = tl.gather(delta_i, reverse, 0)
                yr -= delta_r
                yi -= delta_i
                tl.store(RHS + logical_rows, yr, logical_rows < n)
                tl.store(RHS + padded + logical_rows, yi, logical_rows < n)
            else:
                a = tl.load(AP + off, mask, other=0.0)
                y = tl.load(RHS + logical_rows, logical_rows < n, other=0.0)
                delta = tl.sum(a * xr[None, :], 1)
                if TILED and not FORWARD:
                    delta = tl.gather(delta, M - 1 - tl.arange(0, M), 0)
                y -= delta
                tl.store(RHS + logical_rows, y, logical_rows < n)


@libentry()
@triton.jit
def _tpsv_small(
    AP, X, n, incx, MODE: tl.constexpr, N: tl.constexpr, PACKED: tl.constexpr,
):
    UPLO: tl.constexpr = MODE & 1
    TRANS: tl.constexpr = ((MODE >> 1) & 3) != 0
    CONJ: tl.constexpr = ((MODE >> 1) & 3) == 2
    UNIT: tl.constexpr = (MODE >> 3) & 1
    COMPLEX: tl.constexpr = (MODE >> 4) & 1
    FORWARD: tl.constexpr = (UPLO == 0) != TRANS
    r = tl.arange(0, N)
    physical = r if FORWARD else n - 1 - r
    p = tl.arange(0, PACKED)
    values = tl.load(AP + p, p < n * (n + 1) // 2 * (2 if COMPLEX else 1), other=0.0)
    if COMPLEX:
        values_r, values_i = tl.split(tl.reshape(values, (PACKED // 2, 2)))
        pair = tl.arange(0, 2)
        xv = tl.load(X + physical[:, None] * incx * 2 + pair[None, :],
                     r[:, None] < n, other=0.0)
        xr, xi = tl.split(xv)
    else:
        xr = tl.load(X + physical * incx, r < n, other=0.0)
    if not UNIT:
        diagonal = _packed_offset(physical, physical, n, UPLO, TRANS)
        diagonal = tl.where(r < n, diagonal, 0)
        if COMPLEX:
            dr = tl.gather(values_r, diagonal, 0)
            di = tl.gather(values_i, diagonal, 0)
            if CONJ:
                di = -di
            scale = 1.0 / (dr * dr + di * di)
            invr, invi = dr * scale, -di * scale
        else:
            invr = 1.0 / tl.gather(values, diagonal, 0)
    for k in range(n):
        idx = tl.full((1,), k, tl.int32)
        vr = tl.gather(xr, idx, 0)
        if COMPLEX:
            vi = tl.gather(xi, idx, 0)
        column = k if FORWARD else n - 1 - k
        off = _packed_offset(physical, column, n, UPLO, TRANS)
        off = tl.where((r >= k) & (r < n), off, 0)
        if COMPLEX:
            ar = tl.gather(values_r, off, 0)
            ai = tl.gather(values_i, off, 0)
            if CONJ:
                ai = -ai
            if not UNIT:
                pr = tl.gather(invr, idx, 0)
                pi = tl.gather(invi, idx, 0)
                nr = vr * pr - vi * pi
                vi = vi * pr + vr * pi
                vr = nr
        else:
            ar = tl.gather(values, off, 0)
            if not UNIT:
                pr = tl.gather(invr, idx, 0)
                vr = vr * pr
        ar = tl.where(r > k, ar, 0.0)
        if COMPLEX:
            ai = tl.where(r > k, ai, 0.0)
            xr = xr - ar * vr + ai * vi
            xi = xi - ar * vi - ai * vr
            xi = tl.where(r == k, vi, xi)
        else:
            xr = xr - ar * vr
        xr = tl.where(r == k, vr, xr)
    if COMPLEX:
        tl.store(X + physical[:, None] * incx * 2 + pair[None, :],
                 tl.join(xr, xi), r[:, None] < n)
    else:
        tl.store(X + physical * incx, xr, r < n)


def _tpsv(uplo, trans, diag, n, AP, x, incx, complex_data):
    _common._check_common(uplo, trans, diag, n, AP, x, incx)
    if n == 0:
        return x
    forward = (uplo == 0) != (trans != 0)
    components = 2 if complex_data else 1
    key = (
        AP.device.index, uplo, trans, diag, n, incx, complex_data,
        AP.data_ptr() % 16, x.data_ptr() % 16, _BLOCK_SIZE,
    )
    with torch_device_fn.device(AP.device):
        if complex_data:
            if (
                type(AP) is not torch.Tensor or type(x) is not torch.Tensor
                or AP.is_conj() or x.is_conj() or AP.is_neg() or x.is_neg()
            ):
                ap, xv = torch.view_as_real(AP), torch.view_as_real(x)
            else:
                ap = triton.reinterpret(AP, tl.float32)
                xv = triton.reinterpret(x, tl.float32)
        else:
            ap, xv = AP, x
        if incx != 1:
            # Keep the persistent solve contiguous and bound the UB workspace
            # required by strided I/O. Copy only active entries; gaps stay intact.
            work = torch.empty(n * components, dtype=torch.float32, device=x.device)
            grid = (triton.cdiv(n, 256),)
            _launch(
                _tpsv_copy_vector, grid,
                (xv, work, n, incx, complex_data, False, 256), key + (False,),
            )
            work_x = torch.view_as_complex(work.reshape(n, 2)) if complex_data else work
            _tpsv(uplo, trans, diag, n, AP, work_x, 1, complex_data)
            _launch(
                _tpsv_copy_vector, grid,
                (work, xv, n, incx, complex_data, True, 256), key + (True,),
            )
            return x
        if n <= (127 if complex_data else 160):
            args = (
                ap, xv, n, incx,
                uplo | (trans << 1) | (diag << 3) | (int(complex_data) << 4),
                triton.next_power_of_2(n),
                triton.next_power_of_2(n * (n + 1) // 2 * components),
            )
            _launch(_tpsv_small, (1,), args, key)
        else:
            block = _BLOCK_SIZE
            blocks = triton.cdiv(n, block)
            padded = blocks * block
            inverse = torch.empty(
                (blocks * block * block * components,),
                dtype=torch.float32, device=AP.device,
            )
            rhs = torch.empty(
                (padded * components,), dtype=torch.float32, device=AP.device,
            )
            tiled = uplo != 0 or trans != 0
            update_rows = (
                (64 if n < 512 else 256) if tiled and complex_data else 128
            )
            tiles = ap
            if tiled:
                tiles = torch.empty(
                    ((padded + update_rows) * padded * components,),
                    dtype=torch.float32, device=AP.device,
                )
            prepare_args = (
                ap, xv, inverse, rhs, tiles, n, padded, incx,
                uplo, trans != 0, diag == 1, forward,
                complex_data, trans == 2, block, update_rows, tiled,
            )
            prepare_grid = (blocks, triton.cdiv(n, update_rows) if tiled else 1)
            _launch(_tpsv_prepare, prepare_grid, prepare_args, key)
            solve_args = (
                tiles, xv, inverse, rhs, n, padded, incx,
                uplo, trans != 0, forward, complex_data, trans == 2,
                block, update_rows, tiled,
            )
            _launch(_tpsv_solve_update, (1,), solve_args, key)
    return x


def stpsv(uplo, trans, diag, n, AP, x, incx):
    assert AP.dtype == torch.float32 == x.dtype
    return _tpsv(uplo, trans, diag, n, AP, x, incx, False)


@libentry()
@triton.jit
def _ctpsv_pack_only(AP, DENSE, n, padded, C: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * C + tl.arange(0, C)
    pair = tl.arange(0, 2)
    safe_row = tl.minimum(row, n - 1)
    offset = safe_row * n - safe_row * (safe_row + 1) // 2 + col
    # Avoid all-masked DMA loads for empty rows/chunks on Ascend.
    if (row < n) & (tl.program_id(1) * C + C > row) & (tl.program_id(1) * C < n):
        value = tl.load(AP + offset[:, None] * 2 + pair[None, :],
                        ((col >= row) & (col < n))[:, None], other=0.0)
    else:
        value = tl.full((C, 2), 0.0, tl.float32)
    ar, ai = tl.split(value)
    out = row * padded + col
    tl.store(DENSE + out, ar)
    tl.store(DENSE + padded * padded + out, ai)


_ctpsv_diagonal = _tpsv_prepare.jit_function
_ctpsv_unpack = _ctpsv_pack_only.jit_function


@libentry()
@triton.jit
def _ctpsv_dense_prepare(
    AP, X, INV, RHS, DENSE, n, padded, CONJ: tl.constexpr,
    B: tl.constexpr, M: tl.constexpr, C: tl.constexpr,
):
    if (tl.program_id(0) * B < n) & (tl.program_id(1) == 0):
        _ctpsv_diagonal(AP, X, INV, RHS, DENSE, n, padded, 1,
                        1, True, False, True, True, CONJ, B, M, False)
    _ctpsv_unpack(AP, DENSE, n, padded, C)


@libentry()
@triton.jit
def _ctpsv_dense_solve(
    A, X, INV, RHS, n, padded, TRANS: tl.constexpr, CONJ: tl.constexpr,
    B: tl.constexpr, M: tl.constexpr,
):
    r = tl.arange(0, B)
    s = tl.arange(0, M)
    pair = tl.arange(0, 2)
    for start in range(0, n, B):
        inv_off = (start // B) * B * B * 2 + r[:, None] * B + r[None, :]
        ir = tl.load(INV + inv_off)
        ii = tl.load(INV + inv_off + B * B)
        br = tl.load(RHS + start + r)
        bi = tl.load(RHS + padded + start + r)
        xr = tl.sum(ir * br[None, :] - ii * bi[None, :], 1)
        xi = tl.sum(ir * bi[None, :] + ii * br[None, :], 1)
        if TRANS:
            col = start + r
            cmask = col < n
        else:
            xr = tl.gather(xr, B - 1 - r, 0)
            xi = tl.gather(xi, B - 1 - r, 0)
            col = n - start - B + r
            cmask = col >= 0
        safe_col = tl.maximum(col, 0)
        tl.store(X + safe_col[:, None] * 2 + pair[None, :],
                 tl.join(xr, xi), cmask[:, None])
        for base in range(start + B, n, M):
            logical = base + s
            if TRANS:
                off = safe_col[:, None] * padded + logical[None, :]
                mask = cmask[:, None] & (logical[None, :] < n)
                ar = tl.load(A + off, mask, other=0.0)
                ai = tl.load(A + padded * padded + off, mask, other=0.0)
                if CONJ:
                    ai = -ai
                dr = tl.sum(ar * xr[:, None] - ai * xi[:, None], 0)
                di = tl.sum(ar * xi[:, None] + ai * xr[:, None], 0)
            else:
                row = n - base - M + s
                off = tl.maximum(row[:, None], 0) * padded + safe_col[None, :]
                mask = (row[:, None] >= 0) & cmask[None, :]
                ar = tl.load(A + off, mask, other=0.0)
                ai = tl.load(A + padded * padded + off, mask, other=0.0)
                dr = tl.sum(ar * xr[None, :] - ai * xi[None, :], 1)
                di = tl.sum(ar * xi[None, :] + ai * xr[None, :], 1)
                dr = tl.gather(dr, M - 1 - s, 0)
                di = tl.gather(di, M - 1 - s, 0)
            yr = tl.load(RHS + logical, logical < n, other=0.0)
            yi = tl.load(RHS + padded + logical, logical < n, other=0.0)
            tl.store(RHS + logical, yr - dr, logical < n)
            tl.store(RHS + padded + logical, yi - di, logical < n)


# Scoped tuning: other dimensions and flags retain the original dispatcher.
_CTPSV_SPLIT_TRANSPOSE_SIZES = frozenset(
    (128, 129, 192, 255, 256, 257, 512, 513, 768, 1023, 1024)
)


@libentry()
@triton.jit
def _ctpsv_upper_pack(AP, TILES, n, padded, B: tl.constexpr, M: tl.constexpr):
    panel = tl.program_id(0)
    base = (panel + 1) * B + tl.program_id(1) * M
    if base < n:
        s = tl.arange(0, M)
        c = tl.arange(0, B * 2)
        row = n - base - M + s
        safe_row = tl.maximum(row, 0)
        col = n - panel * B - B
        off = (safe_row * n - safe_row * (safe_row + 1) // 2 + col) * 2
        value = tl.load(AP + off[:, None] + c[None, :],
                        row[:, None] >= 0, other=0.0)
        ar, ai = tl.split(tl.reshape(value, (M, B, 2)))
        out = panel * (padded + M) * B + (base + s[:, None]) * B + tl.arange(0, B)[None, :]
        tl.store(TILES + out, ar)
        tl.store(TILES + (padded + M) * padded + out, ai)


def _ctpsv_split_prepare(trans, n, AP, x):
    """Use layout-specific preparation for the selected upper systems only."""
    _common._check_common(1, trans, 0, n, AP, x, 1)
    block = 16 if n in (128, 129) else 32
    update_rows = 64 if trans == 0 or n < 512 else 256
    blocks = triton.cdiv(n, block)
    padded = triton.next_power_of_2(n)
    key = (
        "ctpsv_split", AP.device.index, trans, n,
        AP.data_ptr() % 16, x.data_ptr() % 16, block, update_rows,
    )
    with torch_device_fn.device(AP.device):
        ap = triton.reinterpret(AP, tl.float32)
        xv = triton.reinterpret(x, tl.float32)
        inverse = torch.empty(blocks * block * block * 2, dtype=torch.float32, device=AP.device)
        rhs = torch.empty(padded * 2, dtype=torch.float32, device=AP.device)
        if trans == 0:
            tiles = torch.empty((padded + update_rows) * padded * 2, dtype=torch.float32, device=AP.device)
            _launch(
                _tpsv_prepare, (blocks, 1),
                (ap, xv, inverse, rhs, tiles, n, padded, 1,
                 1, False, False, False, True, False, block, update_rows, False), key,
            )
            _launch(
                _ctpsv_upper_pack, (blocks, triton.cdiv(n, update_rows)),
                (ap, tiles, n, padded, block, update_rows), key,
            )
            _launch(
                _tpsv_solve_update, (1,),
                (tiles, xv, inverse, rhs, n, padded, 1,
                 1, False, False, True, False, block, update_rows, True), key,
            )
            return x
        tiles = torch.empty(padded * padded * 2, dtype=torch.float32, device=AP.device)
        _launch(
            _ctpsv_dense_prepare,
            (padded, triton.cdiv(padded, 512)),
            (ap, xv, inverse, rhs, tiles, n, padded, trans == 2,
             block, update_rows, min(padded, 512)), key,
        )
        _launch(
            _ctpsv_dense_solve, (1,),
            (tiles, xv, inverse, rhs, n, padded,
             trans != 0, trans == 2, block, update_rows), key,
        )
    return x


def ctpsv(uplo, trans, diag, n, AP, x, incx):
    assert AP.dtype == torch.complex64 == x.dtype
    split = (
        uplo == 1 and diag == 0 and incx == 1
        and (
            (trans in (1, 2) and n in _CTPSV_SPLIT_TRANSPOSE_SIZES)
            or (trans == 2 and n == 1025)
            or (trans == 0 and n == 4096)
        )
    )
    if (
        split and type(AP) is torch.Tensor and type(x) is torch.Tensor
        and not (AP.is_conj() or x.is_conj() or AP.is_neg() or x.is_neg())
    ):
        return _ctpsv_split_prepare(trans, n, AP, x)
    return _tpsv(uplo, trans, diag, n, AP, x, incx, True)
