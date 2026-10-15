# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Row-major Ascend SGBMV with contiguous band loads and fused output updates."""

import importlib
from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas import runtime
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner

ScalarType = Union[float, int, torch.Tensor]
_common = importlib.import_module("flag_blas.ops.level2.gbmv")
try:
    from triton.backends.ascend import utils as _ascend_driver_utils
    from triton.backends.ascend.driver import NPULauncher as _NPU_LAUNCHER
except ImportError:
    _NPU_LAUNCHER = None

try:
    from torch_npu._C import _npu_getCurrentRawStreamNoWait as _current_raw_stream
except ImportError:

    def _current_raw_stream(device):
        return triton.runtime.driver.active.get_current_stream(device)


try:
    from torch_npu._C import _npu_getDevice as _current_device
except ImportError:
    _current_device = torch_device_fn.current_device


# Cache compiled metadata only; input tensors, addresses and streams stay fresh.
_LAUNCH_CACHE = {}
_HOOK_CHAIN_TYPE = getattr(triton.knobs, "HookChain", None)


class _DevicePointer(int):
    """Validated NPU address with fresh tensor metadata for the profiler."""

    def size(self):
        return self.tensor.size()


def _entry(kernel, config_name):
    configs = runtime.get_tuned_config(config_name)
    if len(configs) != 1:
        raise ValueError(f"{config_name} needs one verified tile per structural family")
    return libentry()(kernel), configs[0]


@triton.jit
def _store(
    y,
    value,
    start,
    OUT: tl.constexpr,
    IY: tl.constexpr,
    ZERO: tl.constexpr,
    beta: tl.float32,
    BM: tl.constexpr,
):
    if IY == 1:
        rows = start.to(tl.int64) + tl.arange(0, BM)
        if not ZERO:
            value += beta * tl.load(y + rows, rows < OUT, other=0)
        tl.store(y + rows, value, rows < OUT)
    else:
        # Materializing a vector into a strided output is unsafe on Ascend.
        for j in range(0, BM):
            lane = tl.gather(value, tl.full((1,), j, tl.int32), axis=0)
            row = start.to(tl.int64) + j + tl.arange(0, 1)
            if not ZERO:
                lane += beta * tl.load(y + row * IY, row < OUT, other=0)
            tl.store(y + row * IY, lane, row < OUT)


@triton.jit
def _diagonal(
    a,
    x,
    y,
    alpha: tl.float32,
    beta: tl.float32,
    M: tl.constexpr,
    N: tl.constexpr,
    LDA: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    OUT: tl.constexpr,
    ZERO: tl.constexpr,
    BM: tl.constexpr,
):
    start = tl.program_id(0) * BM
    rows = start + tl.arange(0, BM)
    valid = (rows < M) & (rows < N)
    av = tl.load(a + rows * LDA, valid, other=0)
    xv = tl.load(x + rows * IX, valid, other=0)
    _store(y, alpha * av * xv, start, OUT, IY, ZERO, beta, BM)


@triton.jit
def _short(
    a,
    x,
    y,
    alpha: tl.float32,
    beta: tl.float32,
    M: tl.constexpr,
    N: tl.constexpr,
    KL: tl.constexpr,
    KU: tl.constexpr,
    LDA: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    ZERO: tl.constexpr,
    TRANS: tl.constexpr,
    BA: tl.constexpr,
    BX: tl.constexpr,
    PRUNE: tl.constexpr,
    BM: tl.constexpr,
):
    start = tl.program_id(0) * BM
    rr = tl.arange(0, BM)
    outs = start + rr
    if TRANS:
        abase = tl.maximum(start - KU, 0)
        xbase = abase
        xlen = M
        outlen = N
        limit = M + KU
    else:
        abase = start
        xbase = tl.maximum(start - KL, 0)
        xlen = N
        outlen = M
        limit = N + KL
    acc = tl.full((BM,), 0, tl.float32)
    if not PRUNE or start < limit:
        aa = abase * LDA + tl.arange(0, BA)
        abuf = tl.load(a + aa, aa < M * LDA, other=0)
        xx = xbase * IX + tl.arange(0, BX)
        xbuf = tl.load(x + xx, xx < 1 + (xlen - 1) * IX, other=0)
        for b in tl.static_range(0, KL + KU + 1):
            if TRANS:
                rows = KL + outs - b
                valid = (outs < N) & (rows >= 0) & (rows < M)
                ai = (rows - abase) * LDA + b
                xi = (rows - xbase) * IX
            else:
                cols = outs - KL + b
                valid = (outs < M) & (cols >= 0) & (cols < N)
                ai = rr * LDA + b
                xi = (cols - xbase) * IX
            av = tl.gather(abuf, tl.where(valid, ai, 0), axis=0)
            xv = tl.gather(xbuf, tl.where(valid, xi, 0), axis=0)
            acc += tl.where(valid, av * xv, 0)
    _store(y, alpha * acc, start, outlen, IY, ZERO, beta, BM)


@triton.jit
def _flat(
    a,
    x,
    y,
    alpha: tl.float32,
    beta: tl.float32,
    M: tl.constexpr,
    N: tl.constexpr,
    KL: tl.constexpr,
    KU: tl.constexpr,
    LDA: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    ZERO: tl.constexpr,
    TRANS: tl.constexpr,
    BB: tl.constexpr,
    BA: tl.constexpr,
    BX: tl.constexpr,
    BM: tl.constexpr,
):
    start = tl.program_id(0) * BM
    rr = tl.arange(0, BM)
    outs = start + rr
    b = tl.arange(0, BB)
    if TRANS:
        abase = tl.maximum(start - KU, 0)
        xbase = abase
        xi = KL + outs[:, None] - b[None, :]
        valid = (outs[:, None] < N) & (b[None, :] < KL + KU + 1) & (xi >= 0) & (xi < M)
        indices = (xi - abase) * LDA + b[None, :]
        xlen = M
        outlen = N
    else:
        abase = start
        xbase = tl.maximum(start - KL, 0)
        xi = outs[:, None] - KL + b[None, :]
        valid = (outs[:, None] < M) & (b[None, :] < KL + KU + 1) & (xi >= 0) & (xi < N)
        indices = rr[:, None] * LDA + b[None, :]
        xlen = N
        outlen = M
    aa = abase * LDA + tl.arange(0, BA)
    abuf = tl.load(a + aa, aa < M * LDA, other=0)
    av = tl.reshape(
        tl.gather(abuf, tl.reshape(tl.where(valid, indices, 0), (BM * BB,)), axis=0),
        (BM, BB),
    )
    xx = xbase * IX + tl.arange(0, BX)
    xbuf = tl.load(x + xx, xx < 1 + (xlen - 1) * IX, other=0)
    xv = tl.reshape(
        tl.gather(
            xbuf, tl.reshape(tl.where(valid, (xi - xbase) * IX, 0), (BM * BB,)), axis=0
        ),
        (BM, BB),
    )
    value = alpha * tl.sum(tl.where(valid, av * xv, 0), 1)
    _store(y, value, start, outlen, IY, ZERO, beta, BM)


@triton.jit
def _t_flat_tail(
    a,
    x,
    y,
    alpha: tl.float32,
    beta: tl.float32,
    M: tl.constexpr,
    N: tl.constexpr,
    KL: tl.constexpr,
    KU: tl.constexpr,
    LDA: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    ZERO: tl.constexpr,
    BB: tl.constexpr,
    BA: tl.constexpr,
    BX: tl.constexpr,
    ACTIVE: tl.constexpr,
    BM: tl.constexpr,
    ZB: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid < ACTIVE:
        _flat(
            a,
            x,
            y,
            alpha,
            beta,
            M,
            N,
            KL,
            KU,
            LDA,
            IX,
            IY,
            ZERO,
            True,
            BB,
            BA,
            BX,
            BM,
        )
    else:
        # Outputs beyond M + KU have no matrix contribution; fuse their update.
        start = ACTIVE * BM + (pid - ACTIVE) * ZB
        _store(y, alpha * tl.full((ZB,), 0, tl.float32), start, N, IY, ZERO, beta, ZB)


@triton.jit
def _compact_value(
    a,
    x,
    M: tl.constexpr,
    N: tl.constexpr,
    KL: tl.constexpr,
    KU: tl.constexpr,
    LDA: tl.constexpr,
    IX: tl.constexpr,
    BM: tl.constexpr,
    BB: tl.constexpr,
    BX: tl.constexpr,
    TAIL: tl.constexpr,
):
    start = tl.program_id(0) * BM
    rows = start + tl.arange(0, BM)
    band = tl.arange(0, BB)
    av = tl.load(
        a + rows[:, None] * LDA + band[None, :],
        (rows[:, None] < M) & (band[None, :] < KL + KU + 1),
        other=0,
    )
    base = tl.maximum(start - KL, 0)
    xx = base + tl.arange(0, BX)
    xbuf = tl.load(x + xx * IX, xx < N, other=0)
    cols = rows[:, None] - KL + band[None, :]
    valid = (
        (rows[:, None] < M) & (band[None, :] < KL + KU + 1) & (cols >= 0) & (cols < N)
    )
    indices = tl.reshape(tl.where(valid, cols - base, 0), (BM * BB,))
    xv = tl.reshape(tl.gather(xbuf, indices, axis=0), (BM, BB))
    value = tl.sum(tl.where(valid, av * xv, 0), 1)
    if TAIL:
        tail_col = rows - KL + BB
        tail_ok = (rows < M) & (tail_col >= 0) & (tail_col < N)
        last_a = tl.load(a + rows * LDA + BB, rows < M, other=0)
        last_x = tl.load(x + tl.where(tail_ok, tail_col, 0) * IX, tail_ok, other=0)
        value += tl.where(tail_ok, last_a * last_x, 0)
    return value


@triton.jit
def _n(
    a,
    x,
    y,
    alpha: tl.float32,
    beta: tl.float32,
    M: tl.constexpr,
    N: tl.constexpr,
    KL: tl.constexpr,
    KU: tl.constexpr,
    LDA: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    ZERO: tl.constexpr,
    BB: tl.constexpr,
    BX: tl.constexpr,
    TAIL: tl.constexpr,
    ACTIVE: tl.constexpr,
    PRUNE: tl.constexpr,
    BM: tl.constexpr,
    ZB: tl.constexpr,
):
    pid = tl.program_id(0)
    if PRUNE and pid >= ACTIVE:
        start = ACTIVE * BM + (pid - ACTIVE) * ZB
        _store(y, tl.full((ZB,), 0, tl.float32), start, M, IY, ZERO, beta, ZB)
    else:
        value = alpha * _compact_value(a, x, M, N, KL, KU, LDA, IX, BM, BB, BX, TAIL)
        _store(y, value, pid * BM, M, IY, ZERO, beta, BM)


@triton.jit
def _t(
    a,
    x,
    y,
    alpha: tl.float32,
    beta: tl.float32,
    M: tl.constexpr,
    N: tl.constexpr,
    KL: tl.constexpr,
    KU: tl.constexpr,
    LDA: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    ZERO: tl.constexpr,
    BB: tl.constexpr,
    BX: tl.constexpr,
    PRUNE: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    start = tl.program_id(0) * BM
    cols = start + tl.arange(0, BM)
    rr = tl.arange(0, BK)
    band = tl.arange(0, BB)
    acc = tl.zeros((BK, BM), tl.float32)
    if not PRUNE or start < M + KU:
        for off in range(0, BM + KL + KU, BK):
            rows = start - KU + off + rr
            rows_ok = (rows >= 0) & (rows < M)
            ab = tl.load(
                a + rows[:, None] * LDA + band[None, :],
                rows_ok[:, None] & (band[None, :] < KL + KU + 1),
                other=0,
            )
            b = KL + cols[None, :] - rows[:, None]
            valid = (
                rows_ok[:, None] & (cols[None, :] < N) & (b >= 0) & (b < KL + KU + 1)
            )
            av = tl.gather(ab, tl.where(valid, b, 0), axis=1)
            base = tl.maximum(start - KU + off, 0)
            physical = base * IX + tl.arange(0, BX)
            xbuf = tl.load(x + physical, physical < 1 + (M - 1) * IX, other=0)
            idx = tl.where(rows_ok, (rows - base) * IX, 0)
            xv = tl.gather(xbuf, idx, axis=0)
            acc += tl.where(valid, av * xv[:, None], 0)
    _store(y, alpha * tl.sum(acc, 0), start, N, IY, ZERO, beta, BM)


@triton.jit
def _t_compact(
    a,
    x,
    y,
    alpha: tl.float32,
    beta: tl.float32,
    M: tl.constexpr,
    N: tl.constexpr,
    KL: tl.constexpr,
    KU: tl.constexpr,
    LDA: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    ZERO: tl.constexpr,
    BB: tl.constexpr,
    BX: tl.constexpr,
    PRUNE: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
):
    start = tl.program_id(0) * BM
    cols = start + tl.arange(0, BM)
    rr = tl.arange(0, BK)
    acc = tl.zeros((BK, BM), tl.float32)
    if not PRUNE or start < M + KU:
        begin = tl.maximum(start - KU, 0)
        end = tl.minimum(start + BM + KL, M)
        for off in range(begin, end, BK):
            rows = off + rr
            rows_ok = rows < M
            # Row-major A(i,j) is at i*(LDA-1)+KL+j. Read just the
            # current output-column rectangle instead of the whole band.
            indices = rows[:, None] * (LDA - 1) + KL + cols[None, :]
            av = tl.load(
                a + indices,
                rows_ok[:, None] & (indices < M * LDA),
                other=0,
            )
            b = KL + cols[None, :] - rows[:, None]
            valid = (
                rows_ok[:, None] & (cols[None, :] < N) & (b >= 0) & (b < KL + KU + 1)
            )
            physical = off * IX + tl.arange(0, BX)
            xbuf = tl.load(x + physical, physical < 1 + (M - 1) * IX, other=0)
            xv = tl.gather(xbuf, rr * IX, axis=0)
            acc += tl.where(valid, av * xv[:, None], 0)
    _store(y, alpha * tl.sum(acc, 0), start, N, IY, ZERO, beta, BM)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("sgbmv_scale_ascend"),
    key=["OUT", "IY", "ZERO"],
    restore_value=["y"],
)
@triton.jit
def _scale(
    y,
    beta: tl.float32,
    OUT: tl.constexpr,
    IY: tl.constexpr,
    ZERO: tl.constexpr,
    BM: tl.constexpr,
):
    start = tl.program_id(0).to(tl.int64) * BM
    _store(y, tl.full((BM,), 0, tl.float32), start, OUT, IY, ZERO, beta, BM)


@triton.jit
def _x_window(x, base, LENGTH: tl.constexpr, IX: tl.constexpr, SIZE: tl.constexpr):
    if IX <= 32:
        # Read a contiguous physical span before selecting logical elements.
        physical = base * IX + tl.arange(0, triton.next_power_of_2(SIZE * IX))
        buf = tl.load(x + physical, physical < 1 + (LENGTH - 1) * IX, other=0)
        return tl.gather(buf, tl.arange(0, SIZE) * IX, axis=0)
    else:
        # Bound on-chip storage even when the physical vector stride is huge.
        offsets = tl.arange(0, SIZE)
        buf = tl.full((SIZE,), 0, tl.float32)
        for j in range(0, SIZE):
            index = base + j + tl.arange(0, 1)
            value = tl.load(x + index * IX, index < LENGTH, other=0)
            buf = tl.where(offsets == j, value, buf)
        return buf


@triton.jit
def _bounded(
    a,
    x,
    y,
    alpha: tl.float32,
    beta: tl.float32,
    M: tl.constexpr,
    N: tl.constexpr,
    KL: tl.constexpr,
    KU: tl.constexpr,
    LDA: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    ZERO: tl.constexpr,
    TRANS: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BX: tl.constexpr,
):
    # Fixed-size tiles keep arbitrary band widths/padding out of UB sizing.
    start = tl.program_id(0).to(tl.int64) * BM
    outs = start + tl.arange(0, BM)
    kk = tl.arange(0, BK)
    acc = tl.full((BM,), 0, tl.float32)
    if TRANS:
        for off in range(0, BM + KL + KU, BK):
            rows = start - KU + off + kk
            band = KL + outs[None, :] - rows[:, None]
            valid = (
                (rows[:, None] >= 0)
                & (rows[:, None] < M)
                & (outs[None, :] < N)
                & (band >= 0)
                & (band < KL + KU + 1)
            )
            av = tl.load(a + rows[:, None] * LDA + band, valid, other=0)
            base = tl.maximum(start - KU + off, 0)
            xbuf = _x_window(x, base, M, IX, BK)
            xv = tl.gather(
                xbuf,
                tl.where((rows >= 0) & (rows < M), rows - base, 0).to(tl.int32),
                axis=0,
            )
            acc += tl.sum(tl.where(valid, av * xv[:, None], 0), 0)
        outlen = N
    else:
        for off in range(0, KL + KU + 1, BK):
            band = off + kk
            cols = outs[:, None] - KL + band[None, :]
            valid = (
                (outs[:, None] < M)
                & (band[None, :] < KL + KU + 1)
                & (cols >= 0)
                & (cols < N)
            )
            av = tl.load(a + outs[:, None] * LDA + band[None, :], valid, other=0)
            base = tl.maximum(start - KL + off, 0)
            xbuf = _x_window(x, base, N, IX, BX)
            idx = tl.reshape(tl.where(valid, cols - base, 0), (BM * BK,)).to(tl.int32)
            xv = tl.reshape(tl.gather(xbuf, idx, axis=0), (BM, BK))
            acc += tl.sum(tl.where(valid, av * xv, 0), 1)
        outlen = M
    _store(y, alpha * acc, start, outlen, IY, ZERO, beta, BM)


_bounded_entry = _entry(_bounded, "sgbmv_bounded_ascend")
_diagonal_entry = _entry(_diagonal, "sgbmv_diagonal_ascend")
_short_n_entry = _entry(_short, "sgbmv_short_n_ascend")
_short_t_entry = _entry(_short, "sgbmv_short_t_ascend")
_short_strided_entry = _entry(_short, "sgbmv_short_strided_ascend")
_flat_entry = _entry(_flat, "sgbmv_flat_ascend")
_t_flat_tail_entry = _entry(_t_flat_tail, "sgbmv_t_flat_tail_ascend")
_n_tail_entry = _entry(_n, "sgbmv_n_tail_ascend")
_n_regular_entry = _entry(_n, "sgbmv_n_regular_ascend")
_t_entry = _entry(_t, "sgbmv_t_ascend")
_t_compact_entry = _entry(_t_compact, "sgbmv_t_compact_ascend")
_t_strided_entry = _entry(_t, "sgbmv_t_strided_ascend")


def _ceil_div(x, y):
    return (x + y - 1) // y


def _next_power_of_2(x):
    # Host launch dimensions are positive; keep device constexpr math unchanged.
    return 1 << (int(x) - 1).bit_length()


def _select_kernel(trans, m, n, kl, ku, alpha, lda, ix, beta, iy):
    band = kl + ku + 1
    out = n if trans else m
    zero = beta == 0.0
    shared = (alpha, beta, m, n, kl, ku, lda, ix, iy, zero)
    prune = (n > m + ku) if trans else (m > n + kl)
    if (
        band > 513
        or lda > band + 64
        or ix > 16
        # Leave room for the padded windows used by the int32 fast paths.
        or max(m * lda, max(m, n) * ix) >= 2**30
    ):
        kernel, config = _bounded_entry
        bm, bk = config.kwargs["BM"], config.kwargs["BK"]
        tail = shared + (bool(trans), bm, bk, _next_power_of_2(bm + bk - 1))
        grid = _ceil_div(out, bm)
    elif band == 1:
        kernel, config = _diagonal_entry
        bm = config.kwargs["BM"]
        tail = (alpha, beta, m, n, lda, ix, iy, out, zero, bm)
        grid = _ceil_div(out, bm)
    elif band <= 11:
        kernel, config = (
            _short_strided_entry
            if iy != 1
            else _short_t_entry if trans else _short_n_entry
        )
        bm = config.kwargs["BM"]
        rows = bm + band - 1 if trans else bm
        tail = shared + (
            bool(trans),
            _next_power_of_2(rows * lda),
            _next_power_of_2((bm + band - 1) * ix),
            prune,
            bm,
        )
        grid = _ceil_div(out, bm)
    elif band <= 65 and trans and prune and ix == 1 and iy == 1 and lda == band:
        kernel, config = _t_flat_tail_entry
        bm, zb = config.kwargs["BM"], config.kwargs["ZB"]
        active = _ceil_div(min(n, m + ku), bm)
        grid = active + _ceil_div(max(0, n - active * bm), zb)
        tail = shared + (
            max(8, _next_power_of_2(band)),
            _next_power_of_2((bm + band - 1) * lda),
            _next_power_of_2(bm + band - 1),
            active,
            bm,
            zb,
        )
    elif band <= 65 and not (not trans and prune):
        kernel, config = _flat_entry
        bm = config.kwargs["BM"]
        rows = bm + band - 1 if trans else bm
        tail = shared + (
            bool(trans),
            max(8, _next_power_of_2(band)),
            _next_power_of_2(rows * lda),
            _next_power_of_2((bm + band - 1) * ix),
            bm,
        )
        grid = _ceil_div(out, bm)
    elif not trans:
        last = ((band - 1) & (band - 2)) == 0
        kernel, config = _n_tail_entry if last else _n_regular_entry
        bm, zb = config.kwargs["BM"], config.kwargs["ZB"]
        bb = band - 1 if last else _next_power_of_2(band)
        active = _ceil_div(min(m, n + kl), bm) if prune else _ceil_div(m, bm)
        grid = active + _ceil_div(max(0, m - active * bm), zb)
        tail = shared + (bb, _next_power_of_2(bb + bm - 1), last, active, prune, bm, zb)
    else:
        if iy != 1:
            kernel, config = _t_strided_entry
        elif m <= 1024 and n <= 1024:
            kernel, config = _t_compact_entry
        else:
            kernel, config = _t_entry
        bm, bk = config.kwargs["BM"], config.kwargs["BK"]
        tail = shared + (
            _next_power_of_2(band),
            _next_power_of_2(bk * ix),
            prune,
            bm,
            bk,
        )
        grid = _ceil_div(n, bm)
    return kernel, grid, tail, config


def sgbmv(
    trans: int,
    m: int,
    n: int,
    kl: int,
    ku: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    assert A.dtype == torch.float32 == x.dtype == y.dtype
    _common._check_common(
        A, x, y, trans, m, n, kl, ku, lda, incx, incy, complex_ok=False
    )
    assert A.device.type == "npu"
    if m == 0 or n == 0:
        return
    alpha = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta = float(beta.item() if isinstance(beta, torch.Tensor) else beta)
    device = _current_device()
    if A.device.index not in (None, device):
        with torch_device_fn.device(A.device):
            return sgbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    if alpha == 0.0:
        if beta != 1.0:
            out = n if trans else m
            _scale[lambda meta: (_ceil_div(out, meta["BM"]),)](
                y, beta, out, incy, beta == 0.0
            )
        return
    a_pointer, x_pointer, y_pointer = A.data_ptr(), x.data_ptr(), y.data_ptr()
    key = (
        device,
        trans,
        m,
        n,
        kl,
        ku,
        alpha,
        beta,
        lda,
        incx,
        incy,
        a_pointer % 16,
        x_pointer % 16,
        y_pointer % 16,
    )
    entry = _LAUNCH_CACHE.get(key)
    if entry is None:
        kernel, grid, tail, config = _select_kernel(
            trans, m, n, kl, ku, alpha, lda, incx, beta, incy
        )
        compiled, _ = kernel[(grid,)](
            A, x, y, *tail, num_warps=config.num_warps, num_stages=config.num_stages
        )
        if len(_LAUNCH_CACHE) >= 512:
            _LAUNCH_CACHE.clear()
        _LAUNCH_CACHE[key] = (compiled, compiled.run, grid, tail)
        return

    compiled, run, grid, tail = entry
    stream = _current_raw_stream(device)
    knobs = triton.knobs.runtime
    enter_hook, exit_hook = knobs.launch_enter_hook, knobs.launch_exit_hook
    has_enter = enter_hook is not None and (
        type(enter_hook) is not _HOOK_CHAIN_TYPE or bool(enter_hook.calls)
    )
    has_exit = exit_hook is not None and (
        type(exit_hook) is not _HOOK_CHAIN_TYPE or bool(exit_hook.calls)
    )
    direct_launch = (
        type(run) is _NPU_LAUNCHER
        and not run.compile_only
        and not run.enable_msprof_register_tensor
        and not getattr(run.metadata, "debug_enabled", False)
    )
    ordinary_tensors = (
        type(A) is torch.Tensor
        and type(x) is torch.Tensor
        and type(y) is torch.Tensor
        and not A.is_conj()
        and not x.is_conj()
        and not y.is_conj()
        and not A.is_neg()
        and not x.is_neg()
        and not y.is_neg()
    )
    if not direct_launch or has_enter or has_exit or not ordinary_tensors:
        # Preserve runtime hooks, debug/profiling modes and other launchers.
        compiled[(grid, 1, 1)](A, x, y, *tail, stream=stream)
        return
    a_argument = _DevicePointer(a_pointer)
    x_argument = _DevicePointer(x_pointer)
    y_argument = _DevicePointer(y_pointer)
    a_argument.tensor = A
    x_argument.tensor = x
    y_argument.tensor = y
    registered = run.launch(
        grid,
        1,
        1,
        stream,
        compiled.function,
        compiled.packed_metadata,
        None,
        None,
        None,
        a_argument,
        x_argument,
        y_argument,
        *tail,
    )
    _ascend_driver_utils.TRITON_PROFILER_REGISTERED = registered == 1


@triton.jit
def _cstore_unit(
    y,
    sr,
    si,
    start,
    ar,
    ai,
    br,
    bi,
    OUT: tl.constexpr,
    ZERO: tl.constexpr,
    BM: tl.constexpr,
):
    rr = ar * sr - ai * si
    ri = ar * si + ai * sr
    offsets = start * 2 + tl.arange(0, BM * 2)
    if not ZERO:
        yp = tl.load(y + offsets, offsets < OUT * 2, other=0)
        yr, yi = tl.split(tl.reshape(yp, (BM, 2)))
        rr += br * yr - bi * yi
        ri += br * yi + bi * yr
    result = tl.reshape(tl.join(rr, ri), (BM * 2,))
    tl.store(y + offsets, result, offsets < OUT * 2)


@triton.jit
def _cshort(
    a,
    x,
    y,
    ar: tl.float32,
    ai: tl.float32,
    br: tl.float32,
    bi: tl.float32,
    M: tl.constexpr,
    N: tl.constexpr,
    KL: tl.constexpr,
    KU: tl.constexpr,
    LDA: tl.constexpr,
    TRANS: tl.constexpr,
    CONJ: tl.constexpr,
    ZERO: tl.constexpr,
    BM: tl.constexpr,
    BA: tl.constexpr,
    BX: tl.constexpr,
):
    start = tl.program_id(0) * BM
    if M * LDA * 2 >= 1073741824 or N * LDA * 2 >= 1073741824:
        start = tl.program_id(0).to(tl.int64) * BM
    lane = tl.arange(0, BM)
    out = start + lane
    if TRANS:
        abase = tl.maximum(start - KU, 0)
        xbase = abase
        inlen, outlen = M, N
    else:
        abase = start
        xbase = tl.maximum(start - KL, 0)
        inlen, outlen = N, M
    apos = abase * LDA * 2 + tl.arange(0, BA * 2)
    abuf = tl.load(a + apos, apos < M * LDA * 2, other=0)
    xoff = xbase * 2 + tl.arange(0, BX * 2)
    xbuf = tl.load(x + xoff, xoff < inlen * 2, other=0)
    sr = tl.full((BM,), 0, tl.float32)
    si = tl.full((BM,), 0, tl.float32)
    for b in tl.static_range(0, KL + KU + 1):
        if TRANS:
            xi = out + KL - b
            ap = (xi - abase) * LDA + b
        else:
            xi = out - KL + b
            ap = lane * LDA + b
        valid = (out < outlen) & (xi >= 0) & (xi < inlen)
        ap = tl.where(valid, ap * 2, 0)
        xp = tl.where(valid, (xi - xbase) * 2, 0)
        avr = tl.gather(abuf, ap, 0)
        avi = tl.gather(abuf, ap + 1, 0)
        xr = tl.gather(xbuf, xp, 0)
        ximag = tl.gather(xbuf, xp + 1, 0)
        if CONJ:
            avi = -avi
        sr += tl.where(valid, avr * xr - avi * ximag, 0)
        si += tl.where(valid, avr * ximag + avi * xr, 0)
    _cstore_unit(y, sr, si, start, ar, ai, br, bi, outlen, ZERO, BM)


@triton.jit
def _cstore(
    y,
    rr,
    ri,
    start,
    br,
    bi,
    OUT: tl.constexpr,
    IY: tl.constexpr,
    ZERO: tl.constexpr,
    YC: tl.constexpr,
    YN: tl.constexpr,
    BM: tl.constexpr,
):
    if IY == 1:
        off = start * 2 + tl.arange(0, BM * 2)
        if not ZERO:
            yr, yi = tl.split(
                tl.reshape(tl.load(y + off, off < OUT * 2, other=0), (BM, 2))
            )
            if YC:
                yi = -yi
            if YN:
                yr, yi = -yr, -yi
            rr += br * yr - bi * yi
            ri += br * yi + bi * yr
        if YC:
            ri = -ri
        if YN:
            rr, ri = -rr, -ri
        tl.store(y + off, tl.reshape(tl.join(rr, ri), (BM * 2,)), off < OUT * 2)
    else:
        # Keep each physical store contiguous; never write the stride gaps.
        for lane in range(BM):
            index = tl.full((1,), lane, tl.int32)
            vr, vi = tl.gather(rr, index, 0), tl.gather(ri, index, 0)
            row = start + lane
            off = row * IY * 2 + tl.arange(0, 2)
            if not ZERO:
                yr, yi = tl.split(
                    tl.reshape(tl.load(y + off, row < OUT, other=0), (1, 2))
                )
                if YC:
                    yi = -yi
                if YN:
                    yr, yi = -yr, -yi
                vr += br * yr - bi * yi
                vi += br * yi + bi * yr
            if YC:
                vi = -vi
            if YN:
                vr, vi = -vr, -vi
            tl.store(y + off, tl.reshape(tl.join(vr, vi), (2,)), row < OUT)


@triton.jit
def _cx_window(x, base, LENGTH: tl.constexpr, IX: tl.constexpr, SIZE: tl.constexpr):
    base = tl.minimum(base, LENGTH)
    if IX <= 16:
        physical = base * IX * 2 + tl.arange(0, triton.next_power_of_2(SIZE * IX * 2))
        buf = tl.load(x + physical, physical < (1 + (LENGTH - 1) * IX) * 2, other=0)
        ix = tl.arange(0, SIZE) * IX * 2
        return tl.gather(buf, ix, 0), tl.gather(buf, ix + 1, 0)
    else:
        lane = tl.arange(0, SIZE)
        xr = tl.full((SIZE,), 0, tl.float32)
        xi = tl.full((SIZE,), 0, tl.float32)
        for j in range(SIZE):
            row = base + j
            pair = tl.load(x + row * IX * 2 + tl.arange(0, 2), row < LENGTH, other=0)
            vr, vi = tl.split(tl.reshape(pair, (1, 2)))
            xr = tl.where(lane == j, vr, xr)
            xi = tl.where(lane == j, vi, xi)
        return xr, xi


@triton.jit
def _cgeneral(
    a,
    x,
    y,
    ar: tl.float32,
    ai: tl.float32,
    br: tl.float32,
    bi: tl.float32,
    M: tl.constexpr,
    N: tl.constexpr,
    KL: tl.constexpr,
    KU: tl.constexpr,
    LDA: tl.constexpr,
    IX: tl.constexpr,
    IY: tl.constexpr,
    TRANS: tl.constexpr,
    CONJ: tl.constexpr,
    ZERO: tl.constexpr,
    AC: tl.constexpr,
    AN: tl.constexpr,
    XC: tl.constexpr,
    XN: tl.constexpr,
    YC: tl.constexpr,
    YN: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BX: tl.constexpr,
    ACTIVE: tl.constexpr,
    ZB: tl.constexpr,
):
    pid = tl.program_id(0)
    # Reserve headroom for masked, rounded-up windows near the int32 boundary.
    if (
        M * LDA * 2 >= 1073741824
        or N * IX * 2 >= 1073741824
        or M * IX * 2 >= 1073741824
        or N * IY * 2 >= 1073741824
        or M * IY * 2 >= 1073741824
    ):
        pid = pid.to(tl.int64)
    outlen: tl.constexpr = N if TRANS else M
    if pid >= ACTIVE:
        start = ACTIVE * BM + (pid - ACTIVE) * ZB
        zeros = tl.full((ZB,), 0, tl.float32)
        _cstore(y, ar * zeros, ai * zeros, start, br, bi, outlen, IY, ZERO, YC, YN, ZB)
    else:
        start = pid * BM
        outs = start + tl.arange(0, BM)
        kk = tl.arange(0, BK)
        if TRANS:
            acc_r = tl.zeros((BK, BM), tl.float32)
            acc_i = tl.zeros((BK, BM), tl.float32)
            begin = tl.maximum(start - KU, 0)
            end = tl.minimum(start + BM + KL, M)
            for off in range(begin, end, BK):
                rows = off + kk
                band = KL + outs[None, :] - rows[:, None]
                valid = (
                    (rows[:, None] < M)
                    & (outs[None, :] < N)
                    & (band >= 0)
                    & (band < KL + KU + 1)
                )
                if BK * LDA <= 8192:
                    physical = off * LDA * 2 + tl.arange(
                        0, triton.next_power_of_2(BK * LDA * 2)
                    )
                    abuf = tl.load(a + physical, physical < M * LDA * 2, other=0)
                    indices = tl.reshape(
                        tl.where(valid, (kk[:, None] * LDA + band) * 2, 0).to(tl.int32),
                        (BK * BM,),
                    )
                    avr = tl.reshape(tl.gather(abuf, indices, 0), (BK, BM))
                    avi = tl.reshape(tl.gather(abuf, indices + 1, 0), (BK, BM))
                else:
                    cp = start * 2 + tl.arange(0, BM * 2)
                    addresses = rows[:, None] * ((LDA - 1) * 2) + KL * 2 + cp[None, :]
                    packed = tl.load(
                        a + addresses,
                        (rows[:, None] < M) & (addresses < M * LDA * 2),
                        other=0,
                    )
                    flat = tl.reshape(packed, (BK * BM * 2,))
                    pair = tl.arange(0, BK * BM) * 2
                    avr = tl.reshape(tl.gather(flat, pair, 0), (BK, BM))
                    avi = tl.reshape(tl.gather(flat, pair + 1, 0), (BK, BM))
                xr, xi = _cx_window(x, off, M, IX, BK)
                if AC != CONJ:
                    avi = -avi
                if AN:
                    avr, avi = -avr, -avi
                if XC:
                    xi = -xi
                if XN:
                    xr, xi = -xr, -xi
                acc_r += tl.where(valid, avr * xr[:, None] - avi * xi[:, None], 0)
                acc_i += tl.where(valid, avr * xi[:, None] + avi * xr[:, None], 0)
            sr, si = tl.sum(acc_r, 0), tl.sum(acc_i, 0)
        else:
            acc_r = tl.zeros((BM, BK), tl.float32)
            acc_i = tl.zeros((BM, BK), tl.float32)
            if BM * LDA <= 8192:
                physical = start * LDA * 2 + tl.arange(
                    0, triton.next_power_of_2(BM * LDA * 2)
                )
                abuf = tl.load(a + physical, physical < M * LDA * 2, other=0)
            for off in range(0, KL + KU + 1, BK):
                band = off + kk
                if BM * LDA <= 8192:
                    valid_a = (outs[:, None] < M) & (band[None, :] < KL + KU + 1)
                    indices = tl.reshape(
                        tl.where(
                            valid_a,
                            (tl.arange(0, BM)[:, None] * LDA + band[None, :]) * 2,
                            0,
                        ),
                        (BM * BK,),
                    )
                    avr = tl.reshape(tl.gather(abuf, indices, 0), (BM, BK))
                    avi = tl.reshape(tl.gather(abuf, indices + 1, 0), (BM, BK))
                else:
                    bp = off * 2 + tl.arange(0, BK * 2)
                    packed = tl.load(
                        a + outs[:, None] * LDA * 2 + bp[None, :],
                        (outs[:, None] < M) & (bp[None, :] < (KL + KU + 1) * 2),
                        other=0,
                    )
                    flat = tl.reshape(packed, (BM * BK * 2,))
                    pair = tl.arange(0, BM * BK) * 2
                    avr = tl.reshape(tl.gather(flat, pair, 0), (BM, BK))
                    avi = tl.reshape(tl.gather(flat, pair + 1, 0), (BM, BK))
                cols = outs[:, None] - KL + band[None, :]
                valid = (
                    (outs[:, None] < M)
                    & (band[None, :] < KL + KU + 1)
                    & (cols >= 0)
                    & (cols < N)
                )
                base = tl.maximum(start - KL + off, 0)
                xbr, xbi = _cx_window(x, base, N, IX, BX)
                index = tl.reshape(
                    tl.where(valid, cols - base, 0).to(tl.int32), (BM * BK,)
                )
                xr = tl.reshape(tl.gather(xbr, index, 0), (BM, BK))
                xi = tl.reshape(tl.gather(xbi, index, 0), (BM, BK))
                if AC:
                    avi = -avi
                if AN:
                    avr, avi = -avr, -avi
                if XC:
                    xi = -xi
                if XN:
                    xr, xi = -xr, -xi
                acc_r += tl.where(valid, avr * xr - avi * xi, 0)
                acc_i += tl.where(valid, avr * xi + avi * xr, 0)
            sr, si = tl.sum(acc_r, 1), tl.sum(acc_i, 1)
        _cstore(
            y,
            ar * sr - ai * si,
            ar * si + ai * sr,
            start,
            br,
            bi,
            outlen,
            IY,
            ZERO,
            YC,
            YN,
            BM,
        )


@triton.jit
def _cscale(
    y,
    br: tl.float32,
    bi: tl.float32,
    OUT: tl.constexpr,
    IY: tl.constexpr,
    ZERO: tl.constexpr,
    YC: tl.constexpr,
    YN: tl.constexpr,
    BM: tl.constexpr,
):
    start = tl.program_id(0) * BM
    if OUT * IY * 2 >= 1073741824:
        start = tl.program_id(0).to(tl.int64) * BM
    zeros = tl.full((BM,), 0, tl.float32)
    _cstore(y, zeros, zeros, start, br, bi, OUT, IY, ZERO, YC, YN, BM)


@triton.jit
def _cflat(
    a,
    x,
    y,
    ar: tl.float32,
    ai: tl.float32,
    br: tl.float32,
    bi: tl.float32,
    M: tl.constexpr,
    N: tl.constexpr,
    KL: tl.constexpr,
    KU: tl.constexpr,
    LDA: tl.constexpr,
    TRANS: tl.constexpr,
    CONJ: tl.constexpr,
    ZERO: tl.constexpr,
    BM: tl.constexpr,
    BA: tl.constexpr,
    BX: tl.constexpr,
    BB: tl.constexpr,
):
    start = tl.program_id(0) * BM
    if M * LDA * 2 >= 1073741824 or N * LDA * 2 >= 1073741824:
        start = tl.program_id(0).to(tl.int64) * BM
    lane = tl.arange(0, BM)
    out = start + lane
    b = tl.arange(0, BB)
    if TRANS:
        abase = tl.maximum(start - KU, 0)
        xbase = abase
        inlen, outlen = M, N
        xi = out[:, None] + KL - b[None, :]
        ap = (xi - abase) * LDA + b[None, :]
    else:
        abase = start
        xbase = tl.maximum(start - KL, 0)
        inlen, outlen = N, M
        xi = out[:, None] - KL + b[None, :]
        ap = lane[:, None] * LDA + b[None, :]
    valid = (
        (out[:, None] < outlen) & (b[None, :] < KL + KU + 1) & (xi >= 0) & (xi < inlen)
    )
    apos = abase * LDA * 2 + tl.arange(0, BA * 2)
    abuf = tl.load(a + apos, apos < M * LDA * 2, other=0)
    xoff = xbase * 2 + tl.arange(0, BX * 2)
    xbuf = tl.load(x + xoff, xoff < inlen * 2, other=0)
    ap = tl.reshape(tl.where(valid, ap * 2, 0), (BM * BB,))
    xp = tl.reshape(tl.where(valid, (xi - xbase) * 2, 0), (BM * BB,))
    avr = tl.reshape(tl.gather(abuf, ap, 0), (BM, BB))
    avi = tl.reshape(tl.gather(abuf, ap + 1, 0), (BM, BB))
    xr = tl.reshape(tl.gather(xbuf, xp, 0), (BM, BB))
    ximag = tl.reshape(tl.gather(xbuf, xp + 1, 0), (BM, BB))
    if CONJ:
        avi = -avi
    sr = tl.sum(tl.where(valid, avr * xr - avi * ximag, 0), 1)
    si = tl.sum(tl.where(valid, avr * ximag + avi * xr, 0), 1)
    _cstore_unit(y, sr, si, start, ar, ai, br, bi, outlen, ZERO, BM)


_cshort_entry = _entry(_cshort, "cgbmv_short_ascend")
_cflat_entry = _entry(_cflat, "cgbmv_flat_ascend")
_cn_entry = _entry(_cgeneral, "cgbmv_n_ascend")
_ct_entry = _entry(_cgeneral, "cgbmv_t_ascend")
_ct_strided_entry = _entry(_cgeneral, "cgbmv_t_strided_ascend")
_cbounded_entry = _entry(_cgeneral, "cgbmv_bounded_ascend")
_cscale_entry = _entry(_cscale, "cgbmv_scale_ascend")
_C_LAUNCH_CACHE = {}


class _CDevicePointer(int):
    """Fresh physical float-pair metadata for the NPU profiler."""

    def size(self):
        return (*self.tensor.size(), 2)


def _select_ckernel(trans, m, n, kl, ku, ar, ai, lda, ix, br, bi, iy, flags):
    band = kl + ku + 1
    out, inner = (n, m) if trans else (m, n)
    zero = br == 0 and bi == 0
    if ix == iy == 1 and not any(flags) and band <= 65:
        kernel, config = _cshort_entry if band <= 11 else _cflat_entry
        bm = config.kwargs["BM"]
        rows = bm + band - 1 if trans else bm
        ba = triton.next_power_of_2(rows * lda)
        if ba <= 8192:
            tail = (
                ar,
                ai,
                br,
                bi,
                m,
                n,
                kl,
                ku,
                lda,
                bool(trans),
                trans == 2,
                zero,
                bm,
                ba,
                triton.next_power_of_2(bm + band - 1),
            )
            if band > 11:
                tail += (max(8, triton.next_power_of_2(band)),)
            return kernel, triton.cdiv(out, bm), tail, config
    if ix > 16:
        kernel, config = _cbounded_entry
    elif not trans:
        kernel, config = _cn_entry
    elif iy != 1:
        kernel, config = _ct_strided_entry
    else:
        kernel, config = _ct_entry
    bm, bk = config.kwargs["BM"], config.kwargs["BK"]
    active = triton.cdiv(min(out, inner + (ku if trans else kl)), bm)
    zb = 256 if iy == 1 else 8
    grid = active + triton.cdiv(max(0, out - active * bm), zb)
    tail = (
        ar,
        ai,
        br,
        bi,
        m,
        n,
        kl,
        ku,
        lda,
        ix,
        iy,
        bool(trans),
        trans == 2,
        zero,
        *flags,
        bm,
        bk,
        triton.next_power_of_2(bm + bk - 1),
        active,
        zb,
    )
    return kernel, grid, tail, config


def cgbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy):
    assert A.dtype == x.dtype == y.dtype == torch.complex64
    _common._check_common(
        A, x, y, trans, m, n, kl, ku, lda, incx, incy, complex_ok=True
    )
    assert A.device.type == "npu"
    if m == 0 or n == 0:
        return
    ar, ai, br, bi = _common._complex_scalars(alpha, beta)
    device = _current_device()
    if A.device.index not in (None, device):
        with torch_device_fn.device(A.device):
            return cgbmv(trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy)
    flags = (A.is_conj(), A.is_neg(), x.is_conj(), x.is_neg(), y.is_conj(), y.is_neg())
    if ar == 0 and ai == 0:
        if br != 1 or bi != 0:
            kernel, config = _cscale_entry
            bm = config.kwargs["BM"] if incy == 1 else _cbounded_entry[1].kwargs["BM"]
            out = n if trans else m
            kernel[(triton.cdiv(out, bm),)](
                triton.reinterpret(y, tl.float32),
                br,
                bi,
                out,
                incy,
                br == 0 and bi == 0,
                *flags[-2:],
                bm,
                num_warps=config.num_warps,
                num_stages=config.num_stages
            )
        return
    ap, xp, yp = A.data_ptr(), x.data_ptr(), y.data_ptr()
    key = (
        device,
        trans,
        m,
        n,
        kl,
        ku,
        ar,
        ai,
        br,
        bi,
        lda,
        incx,
        incy,
        ap % 16,
        xp % 16,
        yp % 16,
        flags,
    )
    entry = _C_LAUNCH_CACHE.get(key)
    if entry is None:
        kernel, grid, tail, config = _select_ckernel(
            trans, m, n, kl, ku, ar, ai, lda, incx, br, bi, incy, flags
        )
        compiled, _ = kernel[(grid,)](
            triton.reinterpret(A, tl.float32),
            triton.reinterpret(x, tl.float32),
            triton.reinterpret(y, tl.float32),
            *tail,
            num_warps=config.num_warps,
            num_stages=config.num_stages
        )
        if len(_C_LAUNCH_CACHE) >= 512:
            _C_LAUNCH_CACHE.clear()
        _C_LAUNCH_CACHE[key] = (compiled, compiled.run, grid, tail)
        return
    compiled, run, grid, tail = entry
    stream = _current_raw_stream(device)
    knobs = triton.knobs.runtime
    enter, exit = knobs.launch_enter_hook, knobs.launch_exit_hook
    has_enter = enter is not None and (
        type(enter) is not _HOOK_CHAIN_TYPE or bool(enter.calls)
    )
    has_exit = exit is not None and (
        type(exit) is not _HOOK_CHAIN_TYPE or bool(exit.calls)
    )
    direct = (
        type(run) is _NPU_LAUNCHER
        and not run.compile_only
        and not run.enable_msprof_register_tensor
        and not getattr(run.metadata, "debug_enabled", False)
    )
    ordinary = (
        type(A) is torch.Tensor and type(x) is torch.Tensor and type(y) is torch.Tensor
    )
    if not direct or has_enter or has_exit or not ordinary:
        compiled[(grid, 1, 1)](
            triton.reinterpret(A, tl.float32),
            triton.reinterpret(x, tl.float32),
            triton.reinterpret(y, tl.float32),
            *tail,
            stream=stream
        )
        return
    aa, xx, yy = _CDevicePointer(ap), _CDevicePointer(xp), _CDevicePointer(yp)
    aa.tensor, xx.tensor, yy.tensor = A, x, y
    registered = run.launch(
        grid,
        1,
        1,
        stream,
        compiled.function,
        compiled.packed_metadata,
        None,
        None,
        None,
        aa,
        xx,
        yy,
        *tail
    )
    _ascend_driver_utils.TRITON_PROFILER_REGISTERED = registered == 1
