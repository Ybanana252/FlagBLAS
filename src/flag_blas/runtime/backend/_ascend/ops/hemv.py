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

"""Ascend-specific complex64 HEMV implementation."""

from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas.ops.level2.hemv import _check_common, _complex_scalars, _strided_y
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

from .hpmv import (
    _current_device,
    _current_raw_stream,
    _launch,
    chpmv_finish_kernel,
)

ScalarType = Union[float, int, complex, torch.Tensor]


_HEMV_WORKSPACES = {}


@libentry()
@triton.jit
def _chemv_fused(
    A, X, Y,
    AR: tl.float32, AI: tl.float32, BR: tl.float32, BI: tl.float32,
    N: tl.constexpr, LDA: tl.constexpr,
    INCX: tl.constexpr, INCY: tl.constexpr,
    UPLO: tl.constexpr, BLOCK: tl.constexpr, BETA_ZERO: tl.constexpr,
):
    # One program owns each output strip. For the reflected half, read the
    # stored tile in row-major order and conjugate its contribution.
    rb = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    floats = tl.arange(0, 2 * BLOCK)
    parity = floats & 1
    sign = tl.where(parity == 0, 1.0, -1.0)
    acc = tl.full((2 * BLOCK,), 0.0, tl.float32)
    for cb in range(triton.cdiv(N, BLOCK)):
        direct = rb >= cb if UPLO == 0 else rb <= cb
        sr = tl.where(direct, rb, cb)
        sc = tl.where(direct, cb, rb)
        rows = sr * BLOCK + lanes
        cols = sc * BLOCK + (floats >> 1)
        triangle = (
            cols[None, :] <= rows[:, None]
            if UPLO == 0
            else cols[None, :] >= rows[:, None]
        )
        av = tl.load(
            A + rows[:, None] * (2 * LDA) + sc * BLOCK * 2 + floats[None, :],
            (rows[:, None] < N) & (cols[None, :] < N) & triangle,
            0.0,
        )
        diagonal = rows[:, None] == cols[None, :]
        av = tl.where(diagonal & (parity[None, :] != 0), 0.0, av)
        xoff = (
            cb * BLOCK * 2 + floats
            if INCX == 1
            else (cb * BLOCK + (floats >> 1)) * (2 * INCX) + parity
        )
        xv = tl.load(X + xoff, cb * BLOCK * 2 + floats < N * 2, 0.0)
        if direct:
            real = tl.sum(av * (xv * sign)[None, :], 1)
            imag = tl.sum(av * tl.gather(xv, floats ^ 1, 0)[None, :], 1)
            acc += tl.where(
                parity == 0,
                tl.gather(real, floats >> 1, 0),
                tl.gather(imag, floats >> 1, 0),
            )
        if (not direct) or rb == cb:
            # The real diagonal was included in the direct contribution.
            mirror_av = tl.where(diagonal, 0.0, av)
            xr = tl.gather(xv, lanes * 2, 0)
            xi = tl.gather(xv, lanes * 2 + 1, 0)
            mirror_r = tl.sum(mirror_av * xr[:, None], 0)
            mirror_i = tl.sum(mirror_av * xi[:, None], 0)
            mirror = mirror_r * sign + tl.gather(mirror_i, floats ^ 1, 0)
            acc += mirror
    result = AR * acc + tl.where(parity == 0, -AI, AI) * tl.gather(
        acc, floats ^ 1, 0
    )
    yoff = (
        rb * BLOCK * 2 + floats
        if INCY == 1
        else (rb * BLOCK + (floats >> 1)) * (2 * INCY) + parity
    )
    valid = rb * BLOCK * 2 + floats < N * 2
    if not BETA_ZERO:
        yv = tl.load(Y + yoff, valid, 0.0)
        result += BR * yv + tl.where(parity == 0, -BI, BI) * tl.gather(
            yv, floats ^ 1, 0
        )
    tl.store(Y + yoff, result, valid)


@libentry()
@triton.jit
def _chemv_partial(
    A, X, P,
    N: tl.constexpr, LDA: tl.constexpr, INCX: tl.constexpr,
    UPLO: tl.constexpr, BLOCK: tl.constexpr,
):
    # Every tile has two unique partial-buffer slots. The finish kernel
    # reduces them, avoiding contended atomic writes to y.
    splits: tl.constexpr = triton.cdiv(N, BLOCK)
    tiles: tl.constexpr = splits * (splits + 1) // 2
    for tile in range(tl.program_id(0), tiles, tl.num_programs(0)):
        if UPLO == 0:
            major = (
                (tl.sqrt(tile.to(tl.float32) * 8.0 + 1.0) - 1.0) * 0.5
            ).to(tl.int32)
            major = tl.where(
                ((major * (major + 1)) >> 1) > tile, major - 1, major
            )
            major = tl.where(
                (((major + 1) * (major + 2)) >> 1) <= tile, major + 1, major
            )
            rb = major
            cb = tile - ((major * (major + 1)) >> 1)
        else:
            side: tl.constexpr = 2 * splits + 1
            disc = side * side - 8 * tile
            row = ((side - tl.sqrt(disc.to(tl.float32))) * 0.5).to(tl.int32)
            base = (row * (2 * splits - row + 1)) >> 1
            row = tl.where(base > tile, row - 1, row)
            next_base = ((row + 1) * (2 * splits - row)) >> 1
            row = tl.where(next_base <= tile, row + 1, row)
            base = (row * (2 * splits - row + 1)) >> 1
            rb = row
            cb = row + tile - base
        lanes = tl.arange(0, BLOCK)
        floats = tl.arange(0, 2 * BLOCK)
        parity = floats & 1
        sign = tl.where(parity == 0, 1.0, -1.0)
        rows = rb * BLOCK + lanes
        cols = cb * BLOCK + (floats >> 1)
        row_f = rows.to(tl.float32)
        col_f = cols.to(tl.float32)
        triangle = (
            col_f[None, :] <= row_f[:, None]
            if UPLO == 0
            else col_f[None, :] >= row_f[:, None]
        )
        av = tl.load(
            A + rows[:, None] * (2 * LDA) + cb * BLOCK * 2 + floats[None, :],
            (row_f[:, None] < N) & (col_f[None, :] < N) & triangle,
            0.0,
        )
        diagonal = row_f[:, None] == col_f[None, :]
        av = tl.where(diagonal & (parity[None, :] != 0), 0.0, av)
        coff = (
            cb * BLOCK * 2 + floats
            if INCX == 1
            else cols * (2 * INCX) + parity
        )
        xc = tl.load(X + coff, cb * BLOCK * 2 + floats < N * 2, 0.0)
        real = tl.sum(av * (xc * sign)[None, :], 1)
        imag = tl.sum(av * tl.gather(xc, floats ^ 1, 0)[None, :], 1)
        roff = (
            rb * BLOCK * 2 + floats
            if INCX == 1
            else (rb * BLOCK + (floats >> 1)) * (2 * INCX) + parity
        )
        xr_vec = tl.load(X + roff, rb * BLOCK * 2 + floats < N * 2, 0.0)
        xr = tl.gather(xr_vec, lanes * 2, 0)
        xi = tl.gather(xr_vec, lanes * 2 + 1, 0)
        mirror_av = tl.where(diagonal, 0.0, av)
        mirror_r = tl.sum(mirror_av * xr[:, None], 0)
        mirror_i = tl.sum(mirror_av * xi[:, None], 0)
        mirror = mirror_r * sign + tl.gather(mirror_i, floats ^ 1, 0)
        if rb == cb:
            real += tl.gather(mirror, lanes * 2, 0)
            imag += tl.gather(mirror, lanes * 2 + 1, 0)
        else:
            tl.store(
                P + (rb * N + cb * BLOCK) * 2 + floats,
                mirror,
                cb * BLOCK * 2 + floats < N * 2,
            )
        direct = tl.where(
            parity == 0,
            tl.gather(real, floats >> 1, 0),
            tl.gather(imag, floats >> 1, 0),
        )
        tl.store(
            P + (cb * N + rb * BLOCK) * 2 + floats,
            direct,
            rb * BLOCK * 2 + floats < N * 2,
        )



# Keep the launch geometry of every case that already clears the target.
_PRESERVED_FUSED_BLOCKS = {
    (0, 256): 32,
    (1, 384): 64,
    (1, 512): 64,
    (1, 768): 64,
    (1, 1023): 64,
    (1, 1024): 64,
}

# Only previously under-threshold benchmark cases use the tuned dispatch.
_UNDERPERFORMING_FUSED_BLOCKS = {
    (0, 128): 16,
    (0, 192): 16,
    (0, 384): 32,
    (0, 512): 32,
    (0, 768): 64,
    (0, 1023): 32,
    (0, 1024): 64,
    (0, 1536): 64,
    (0, 2048): 64,
    (1, 128): 16,
    (1, 192): 16,
    (1, 256): 16,
    (1, 1536): 64,
    (1, 2048): 64,
}

_FUSED_BLOCKS = {**_PRESERVED_FUSED_BLOCKS, **_UNDERPERFORMING_FUSED_BLOCKS}


def run_chemv(uplo, n, ar, ai, br, bi, A, lda, x, incx, y, incy):
    beta_zero = br == 0.0 and bi == 0.0
    if n <= 2048:
        block = _FUSED_BLOCKS.get((uplo, n), 32 if n <= 256 else 64)
        _launch(
            _chemv_fused,
            (triton.cdiv(n, block),),
            (A, x, y),
            (ar, ai, br, bi),
            (n, lda, incx, incy, uplo, block, beta_zero),
        )
        return
    block = 64
    splits = triton.cdiv(n, block)
    tiles = splits * (splits + 1) // 2
    # Large, under-threshold cases run faster when each program handles
    # several tiles instead of launching one program per tile.
    grid_cap = max(512, triton.next_power_of_2(max(1, tiles // 12)))
    workspace_key = (A.device.index, _current_raw_stream(A.device.index), n)
    partial = _HEMV_WORKSPACES.get(workspace_key)
    if partial is None:
        partial = torch.empty(
            (splits, n, 2), dtype=torch.float32, device=A.device
        )
        if len(_HEMV_WORKSPACES) >= 2:
            _HEMV_WORKSPACES.pop(next(iter(_HEMV_WORKSPACES)))
        _HEMV_WORKSPACES[workspace_key] = partial
    _launch(
        _chemv_partial,
        (min(65535, tiles, grid_cap),),
        (A, x, partial),
        (),
        (n, lda, incx, uplo, block),
    )
    _launch(
        chpmv_finish_kernel,
        (triton.cdiv(n, 128),),
        (partial, y),
        (ar, ai, br, bi),
        (
            n,
            incy,
            splits,
            min(32, triton.next_power_of_2(splits)),
            128,
            beta_zero,
            False,
        ),
    )

def chemv(
    uplo: int,
    n: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    assert A.dtype == torch.complex64 == x.dtype == y.dtype
    _check_common(A, x, y, uplo, n, lda, incx, incy)
    if n == 0:
        return

    ar, ai, br, bi = _complex_scalars(alpha, beta)
    if ar == 0.0 and ai == 0.0:
        y_view = _strided_y(y, n, incy)
        if br == 0.0 and bi == 0.0:
            y_view.zero_()
        elif br != 1.0 or bi != 0.0:
            y_view.mul_(complex(br, bi))
        return

    if A.device.index != _current_device():
        with torch_device_fn.device(A.device):
            run_chemv(uplo, n, ar, ai, br, bi, A, lda, x, incx, y, incy)
    else:
        run_chemv(uplo, n, ar, ai, br, bi, A, lda, x, incx, y, incy)
