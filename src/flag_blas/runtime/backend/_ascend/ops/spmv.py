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

"""Packed-row streaming SPMV for Ascend, with deterministic partial sums."""

import torch
import triton
import triton.language as tl

from flag_blas.ops.level2.spmv import _check_common, _strided_y
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

# This version/hook-guarded launcher also accepts real tensors. It caches
# compiled code only and uses the current call's pointers and stream.
from .hpmv import _current_device, _launch


@libentry()
@triton.jit
def _fused(
    AP, X, Y, ALPHA: tl.float32, BETA: tl.float32,
    N: tl.constexpr, IX: tl.constexpr, IY: tl.constexpr,
    UPLO: tl.constexpr, B: tl.constexpr, BETA_ZERO: tl.constexpr,
):
    rb = tl.program_id(0)
    lane = tl.arange(0, B)
    acc = tl.full((B,), 0.0, tl.float32)
    for cb in range(triton.cdiv(N, B)):
        direct = rb >= cb if UPLO == 0 else rb <= cb
        sr, sc = tl.where(direct, rb, cb), tl.where(direct, cb, rb)
        rows, cols = sr * B + lane, sc * B + lane
        if UPLO == 0:
            base = (rows * (rows + 1)) >> 1
            triangle = cols[None, :] <= rows[:, None]
        else:
            base = (rows * (2 * N - rows - 1)) >> 1
            triangle = cols[None, :] >= rows[:, None]
        a = tl.load(
            AP + base[:, None] + sc * B + lane[None, :],
            (rows[:, None] < N) & (cols[None, :] < N) & triangle, 0.0,
        )
        x = tl.load(X + (cb * B + lane).to(tl.int64) * IX, cb * B + lane < N, 0.0)
        if direct:
            acc += tl.sum(a * x[None, :], 1)
        if (not direct) or rb == cb:
            a = tl.where(rows[:, None] == cols[None, :], 0.0, a)
            acc += tl.sum(a * x[:, None], 0)
    out_rows = rb * B + lane
    out = ALPHA * acc
    if not BETA_ZERO:
        out += BETA * tl.load(Y + out_rows.to(tl.int64) * IY, out_rows < N, 0.0)
    tl.store(Y + out_rows.to(tl.int64) * IY, out, out_rows < N)


@libentry()
@triton.jit
def _partial(
    AP, X, P, N: tl.constexpr, IX: tl.constexpr,
    UPLO: tl.constexpr, B: tl.constexpr,
):
    # A stored tile contributes to both output strips. Each contribution has
    # one unique slot in P, so no atomics or buffer initialization are needed.
    splits: tl.constexpr = triton.cdiv(N, B)
    tiles: tl.constexpr = splits * (splits + 1) // 2
    for tile in range(tl.program_id(0), tiles, tl.num_programs(0)):
        major = ((tl.sqrt(tile.to(tl.float32) * 8.0 + 1.0) - 1.0) * 0.5).to(tl.int32)
        major = tl.where(((major * (major + 1)) >> 1) > tile, major - 1, major)
        major = tl.where((((major + 1) * (major + 2)) >> 1) <= tile, major + 1, major)
        minor = tile - ((major * (major + 1)) >> 1)
        rb, cb = (major, minor) if UPLO == 0 else (minor, major)
        lane = tl.arange(0, B)
        rows, cols = rb * B + lane, cb * B + lane
        r = rows if N <= 32767 else rows.to(tl.int64)
        if UPLO == 0:
            base = (r * (r + 1)) >> 1
            triangle = cols[None, :] <= rows[:, None]
        else:
            base = (r * (2 * N - r - 1)) >> 1
            triangle = cols[None, :] >= rows[:, None]
        a = tl.load(
            AP + base[:, None] + cb * B + lane[None, :],
            (rows[:, None] < N) & (cols[None, :] < N) & triangle, 0.0,
        )
        xc = tl.load(X + cols.to(tl.int64) * IX, cols < N, 0.0)
        xr = tl.load(X + rows.to(tl.int64) * IX, rows < N, 0.0)
        direct = tl.sum(a * xc[None, :], 1)
        offdiag = tl.where(rows[:, None] == cols[None, :], 0.0, a)
        mirror = tl.sum(offdiag * xr[:, None], 0)
        if rb == cb:
            direct += mirror
        else:
            tl.store(P + rb * N + cb * B + lane, mirror, cols < N)
        tl.store(P + cb * N + rb * B + lane, direct, rows < N)


@libentry()
@triton.jit
def _finish(
    P, Y, ALPHA: tl.float32, BETA: tl.float32,
    N: tl.constexpr, IY: tl.constexpr, SPLITS: tl.constexpr,
    R: tl.constexpr, B: tl.constexpr, BETA_ZERO: tl.constexpr,
):
    lane = tl.arange(0, B)
    rows = tl.program_id(0) * B + lane
    acc = tl.full((R, B), 0.0, tl.float32)
    for start in range(0, SPLITS, R):
        split = start + tl.arange(0, R)
        acc += tl.load(
            P + split[:, None] * N + tl.program_id(0) * B + lane[None, :],
            (split[:, None] < SPLITS) & (rows[None, :] < N), 0.0,
        )
    out = ALPHA * tl.sum(acc, 0)
    if not BETA_ZERO:
        out += BETA * tl.load(Y + rows.to(tl.int64) * IY, rows < N, 0.0)
    tl.store(Y + rows.to(tl.int64) * IY, out, rows < N)


def sspmv(uplo, n, alpha, AP, x, incx, beta, y, incy):
    assert AP.dtype == torch.float32 == x.dtype == y.dtype
    _check_common(AP, x, y, uplo, n, incx, incy)
    if n == 0:
        return
    alpha = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta = float(beta.item() if isinstance(beta, torch.Tensor) else beta)
    if alpha == 0.0:
        y_view = _strided_y(y, n, incy)
        if beta == 0.0:
            y_view.zero_()
        elif beta != 1.0:
            y_view.mul_(beta)
        return
    if AP.device.index != _current_device():
        with torch_device_fn.device(AP.device):
            return sspmv(uplo, n, alpha, AP, x, incx, beta, y, incy)
    if n <= 1024:
        _launch(
            _fused, (triton.cdiv(n, 32),), (AP, x, y), (alpha, beta),
            (n, incx, incy, uplo, 32, beta == 0.0),
        )
        return
    block = 64
    splits = triton.cdiv(n, block)
    partial = torch.empty((splits, n), device=AP.device, dtype=torch.float32)
    _launch(
        _partial, (min(65535, splits * (splits + 1) // 2),),
        (AP, x, partial), (), (n, incx, uplo, block),
    )
    _launch(
        _finish, (triton.cdiv(n, 64),), (partial, y), (alpha, beta),
        (n, incy, splits, min(32, triton.next_power_of_2(splits)), 64, beta == 0.0),
    )
