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

"""Ascend-specific complex64 HPMV kernel."""

from typing import Union

import torch
import triton
import triton.language as tl

from flag_blas.ops.level2.hpmv import (
    _check_common,
    _complex_scalars,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

ScalarType = Union[float, int, complex, torch.Tensor]
_LAUNCH_CACHE = {}
_FAST_LAUNCH_SUPPORTED = triton.__version__.split(".")[:2] == ["3", "5"]
_HOOK_CHAIN_TYPE = getattr(getattr(triton, "knobs", None), "HookChain", None)

try:
    from torch_npu._C import _npu_getCurrentRawStreamNoWait as _current_raw_stream
    from torch_npu._C import _npu_getDevice as _current_device
except ImportError:
    _current_device = torch_device_fn.current_device

    def _current_raw_stream(device):
        return triton.runtime.driver.active.get_current_stream(device)


try:
    from triton.backends.ascend.driver import NPULauncher as _NPU_LAUNCHER
except ImportError:
    _NPU_LAUNCHER = None


class _DevicePointer(int):
    """Retain each invocation's tensor until its launch is enqueued."""

    def size(self):
        shape = self.tensor.shape
        return (*shape, 2) if self.tensor.is_complex() else shape


def _real_tensors(tensors):
    return tuple(torch.view_as_real(t) if t.is_complex() else t for t in tensors)


def _launch(kernel, grid, tensors, scalars, constants):
    # Cache only compiled code, never inputs, outputs, pointers or streams.
    key = (
        kernel,
        tensors[0].device.index,
        constants,
        tuple(t.data_ptr() % 16 == 0 for t in tensors),
    )
    compiled = _LAUNCH_CACHE.get(key)
    if compiled is None:
        compiled, _ = kernel[grid](
            *_real_tensors(tensors), *scalars, *constants, num_warps=1, num_stages=1
        )
        if len(_LAUNCH_CACHE) >= 256:
            _LAUNCH_CACHE.clear()
        _LAUNCH_CACHE[key] = compiled
    else:
        launch_grid = (*grid, *(1 for _ in range(3 - len(grid))))
        run = compiled.run
        runtime_knobs = getattr(getattr(triton, "knobs", None), "runtime", None)
        hooks = (
            getattr(runtime_knobs, "launch_enter_hook", None),
            getattr(runtime_knobs, "launch_exit_hook", None),
        )
        has_hooks = any(
            hook is not None
            and (type(hook) is not _HOOK_CHAIN_TYPE or bool(hook.calls))
            for hook in hooks
        )
        # The runner ABI is checked for Triton-Ascend 3.5. Keep the normal
        # entry for other versions, hooks, profiling and special tensor views.
        if (
            not _FAST_LAUNCH_SUPPORTED
            or type(run) is not _NPU_LAUNCHER
            or runtime_knobs is None
            or has_hooks
            or getattr(run, "compile_only", False)
            or getattr(run, "enable_msprof_register_tensor", False)
            or getattr(compiled.metadata, "debug_enabled", False)
            or getattr(getattr(run, "metadata", None), "debug_enabled", False)
            or any(
                type(t) is not torch.Tensor or t.is_conj() or t.is_neg()
                for t in tensors
            )
        ):
            compiled[launch_grid](*_real_tensors(tensors), *scalars, *constants)
            return
        pointers = []
        for tensor in tensors:
            pointer = _DevicePointer(tensor.data_ptr())
            pointer.tensor = tensor
            pointers.append(pointer)
        run(
            *launch_grid,
            _current_raw_stream(tensors[0].device.index),
            compiled.function,
            compiled.packed_metadata,
            None,
            None,
            None,
            *pointers,
            *scalars,
            *constants,
        )


@libentry()
@triton.jit
def chpmv_small_kernel(
    AP,
    X,
    Y,
    AR: tl.float32,
    AI: tl.float32,
    BR: tl.float32,
    BI: tl.float32,
    N: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK: tl.constexpr,
    BETA_ZERO: tl.constexpr,
):
    row = tl.program_id(0)
    lanes = tl.arange(0, 2 * BLOCK)
    cols = lanes >> 1
    low, high = tl.minimum(row, cols), tl.maximum(row, cols)
    if UPLO == 0:
        offset = ((high * (high + 1)) >> 1) + low
        conjugate = cols > row
    else:
        offset = ((low * (2 * N - low - 1)) >> 1) + high
        conjugate = cols < row
    odd = (lanes & 1) != 0
    av = tl.load(AP + offset * 2 + (lanes & 1), cols < N, 0.0)
    av = tl.where(odd & conjugate, -av, av)
    av = tl.where(odd & (cols == row), 0.0, av)
    xoff = lanes if INCX == 1 else cols.to(tl.int64) * (2 * INCX) + (lanes & 1)
    xv = tl.load(X + xoff, lanes < 2 * N, 0.0)
    sr = tl.sum(av * xv * tl.where(odd, -1.0, 1.0), 0)
    si = tl.sum(av * tl.gather(xv, lanes ^ 1, 0), 0)
    rr, ri = AR * sr - AI * si, AR * si + AI * sr
    yo = row.to(tl.int64) * (2 * INCY)
    if not BETA_ZERO:
        yr, yi = tl.load(Y + yo), tl.load(Y + yo + 1)
        rr += BR * yr - BI * yi
        ri += BR * yi + BI * yr
    tl.store(Y + yo, rr)
    tl.store(Y + yo + 1, ri)


@libentry()
@triton.jit
def chpmv_fused_kernel(
    AP,
    X,
    Y,
    AR: tl.float32,
    AI: tl.float32,
    BR: tl.float32,
    BI: tl.float32,
    N: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK: tl.constexpr,
    BETA_ZERO: tl.constexpr,
):
    # Small matrices use one program per output strip, with the reduction
    # fused into the tile loop (no temporary allocation or second launch).
    rb = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    floats = tl.arange(0, 2 * BLOCK)
    parity = floats & 1
    sign = tl.where(parity == 0, 1.0, -1.0)
    acc = tl.full((2 * BLOCK,), 0.0, tl.float32)
    for cb in range(triton.cdiv(N, BLOCK)):
        direct = rb >= cb if UPLO == 0 else rb <= cb
        stored_r = tl.where(direct, rb, cb)
        stored_c = tl.where(direct, cb, rb)
        rows = stored_r * BLOCK + lanes
        cols = stored_c * BLOCK + (floats >> 1)
        if UPLO == 0:
            base = (rows * (rows + 1)) >> 1
            triangle = (
                stored_c * BLOCK * 2 + floats[None, :] < (rows[:, None] + 1) * 2
            )
        else:
            base = (rows * (2 * N - rows - 1)) >> 1
            triangle = (
                stored_c * BLOCK * 2 + floats[None, :] >= rows[:, None] * 2
            )
        av = tl.load(
            AP + (base[:, None] + stored_c * BLOCK) * 2 + floats[None, :],
            (rows[:, None] < N)
            & (stored_c * BLOCK * 2 + floats[None, :] < N * 2)
            & triangle,
            0.0,
        )
        diagonal = rows[:, None] == cols[None, :]
        av = tl.where(diagonal & (parity[None, :] != 0), 0.0, av)
        xoff = (
            cb * BLOCK * 2 + floats
            if INCX == 1
            else (cb * BLOCK + (floats >> 1)).to(tl.int64) * (2 * INCX) + parity
        )
        xv = tl.load(X + xoff, cb * BLOCK * 2 + floats < N * 2, 0.0)
        if direct:
            dr = tl.sum(av * (xv * sign)[None, :], 1)
            di = tl.sum(av * tl.gather(xv, floats ^ 1, 0)[None, :], 1)
            acc += tl.where(
                parity == 0,
                tl.gather(dr, floats >> 1, 0),
                tl.gather(di, floats >> 1, 0),
            )
        if (not direct) or rb == cb:
            av = tl.where(diagonal, 0.0, av)
            xr = tl.gather(xv, lanes * 2, 0)
            xi = tl.gather(xv, lanes * 2 + 1, 0)
            mr = tl.sum(av * xr[:, None], 0)
            mi = tl.sum(av * xi[:, None], 0)
            acc += mr * sign + tl.gather(mi, floats ^ 1, 0)
    result = AR * acc + tl.where(parity == 0, -AI, AI) * tl.gather(
        acc, floats ^ 1, 0
    )
    yoff = (
        rb * BLOCK * 2 + floats
        if INCY == 1
        else (rb * BLOCK + (floats >> 1)).to(tl.int64) * (2 * INCY) + parity
    )
    mask = rb * BLOCK * 2 + floats < N * 2
    if not BETA_ZERO:
        yv = tl.load(Y + yoff, mask, 0.0)
        result += BR * yv + tl.where(parity == 0, -BI, BI) * tl.gather(
            yv, floats ^ 1, 0
        )
    tl.store(Y + yoff, result, mask)


@libentry()
@triton.jit
def chpmv_partial_kernel(
    AP,
    X,
    P,
    N: tl.constexpr,
    INCX: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Read each stored tile once, in packed-row order. The same tile supplies
    # both A_ij*x_j and conj(A_ij)*x_i; distinct partial slots avoid atomics.
    # Bound the launch grid; larger matrices distribute tiles cyclically.
    splits: tl.constexpr = triton.cdiv(N, BLOCK)
    tiles: tl.constexpr = splits * (splits + 1) // 2
    for tile in range(tl.program_id(0), tiles, tl.num_programs(0)):
        major = ((tl.sqrt(tile.to(tl.float32) * 8.0 + 1.0) - 1.0) * 0.5).to(tl.int32)
        major = tl.where(((major * (major + 1)) >> 1) > tile, major - 1, major)
        major = tl.where((((major + 1) * (major + 2)) >> 1) <= tile, major + 1, major)
        minor = tile - ((major * (major + 1)) >> 1)
        rb, cb = (major, minor) if UPLO == 0 else (minor, major)
        lanes = tl.arange(0, BLOCK)
        floats = tl.arange(0, 2 * BLOCK)
        rows = rb * BLOCK + lanes
        cfloat = cb * BLOCK + (floats >> 1)
        r = rows if N <= 32767 else rows.to(tl.int64)
        if UPLO == 0:
            base = (r * (r + 1)) >> 1
            triangle = cb * BLOCK * 2 + floats[None, :] < (rows[:, None] + 1) * 2
        else:
            base = (r * (2 * N - r - 1)) >> 1
            triangle = cb * BLOCK * 2 + floats[None, :] >= rows[:, None] * 2
        offsets = (base[:, None] + cb * BLOCK) * 2 + floats[None, :]
        mask = (
            (rows[:, None] < N)
            & (cb * BLOCK * 2 + floats[None, :] < N * 2)
            & triangle
        )
        av = tl.load(AP + offsets, mask, 0.0)
        diagonal = rows[:, None] == cfloat[None, :]
        av = tl.where(diagonal & ((floats[None, :] & 1) != 0), 0.0, av)
        # Express consecutive real/imaginary lanes directly. Reconstructing
        # these as (lanes // 2) * 2 + lanes % 2 scalarizes Ascend memory ops.
        coff = (
            cb * BLOCK * 2 + floats
            if INCX == 1
            else cfloat.to(tl.int64) * (2 * INCX) + (floats & 1)
        )
        xc = tl.load(X + coff, cb * BLOCK * 2 + floats < N * 2, 0.0)
        sign = tl.where((floats & 1) == 0, 1.0, -1.0)
        swapped_x = tl.gather(xc, floats ^ 1, 0)
        direct_r = tl.sum(av * (xc * sign)[None, :], 1)
        direct_i = tl.sum(av * swapped_x[None, :], 1)
        rfloat = rb * BLOCK + (floats >> 1)
        roff = (
            rb * BLOCK * 2 + floats
            if INCX == 1
            else rfloat.to(tl.int64) * (2 * INCX) + (floats & 1)
        )
        xv = tl.load(X + roff, rb * BLOCK * 2 + floats < N * 2, 0.0)
        xr, xi = tl.gather(xv, lanes * 2, 0), tl.gather(xv, lanes * 2 + 1, 0)
        # The diagonal belongs only to the direct contribution.
        mirror_av = tl.where(diagonal, 0.0, av)
        mirror_xr = tl.sum(mirror_av * xr[:, None], 0)
        mirror_xi = tl.sum(mirror_av * xi[:, None], 0)
        mirror = mirror_xr * sign + tl.gather(mirror_xi, floats ^ 1, 0)
        if rb == cb:
            mirror_r = tl.gather(mirror, lanes * 2, 0)
            mirror_i = tl.gather(mirror, lanes * 2 + 1, 0)
            direct_r += mirror_r
            direct_i += mirror_i
        else:
            tl.store(
                P + (rb * N + cb * BLOCK) * 2 + floats,
                mirror,
                cb * BLOCK * 2 + floats < N * 2,
            )
        out = tl.where(
            (floats & 1) == 0,
            tl.gather(direct_r, floats >> 1, 0),
            tl.gather(direct_i, floats >> 1, 0),
        )
        tl.store(
            P + (cb * N + rb * BLOCK) * 2 + floats,
            out,
            rb * BLOCK * 2 + floats < N * 2,
        )


@libentry()
@triton.jit
def chpmv_finish_kernel(
    P,
    Y,
    AR: tl.float32,
    AI: tl.float32,
    BR: tl.float32,
    BI: tl.float32,
    N: tl.constexpr,
    INCY: tl.constexpr,
    SPLITS: tl.constexpr,
    REDUCE: tl.constexpr,
    BLOCK: tl.constexpr,
    BETA_ZERO: tl.constexpr,
    ALPHA_ZERO: tl.constexpr,
):
    lanes = tl.arange(0, 2 * BLOCK)
    rows = tl.program_id(0) * BLOCK + (lanes >> 1)
    if ALPHA_ZERO:
        result = tl.full((2 * BLOCK,), 0.0, tl.float32)
    else:
        acc2d = tl.full((REDUCE, 2 * BLOCK), 0.0, tl.float32)
        for start in range(0, SPLITS, REDUCE):
            splits = start + tl.arange(0, REDUCE)
            offsets = (
                (splits[:, None] * N + tl.program_id(0) * BLOCK) * 2
                + lanes[None, :]
            )
            values = tl.load(
                P + offsets,
                (splits[:, None] < SPLITS)
                & (tl.program_id(0) * BLOCK * 2 + lanes[None, :] < N * 2),
                0.0,
            )
            acc2d += values
        acc = tl.sum(acc2d, 0)
        swapped = tl.gather(acc, lanes ^ 1, 0)
        result = AR * acc + tl.where((lanes & 1) == 0, -AI, AI) * swapped
    yoff = (
        tl.program_id(0) * BLOCK * 2 + lanes
        if INCY == 1
        else rows.to(tl.int64) * (2 * INCY) + (lanes & 1)
    )
    if not BETA_ZERO:
        yv = tl.load(Y + yoff, tl.program_id(0) * BLOCK * 2 + lanes < N * 2, 0.0)
        swapped = tl.gather(yv, lanes ^ 1, 0)
        result += BR * yv + tl.where((lanes & 1) == 0, -BI, BI) * swapped
    tl.store(Y + yoff, result, tl.program_id(0) * BLOCK * 2 + lanes < N * 2)


def chpmv(
    uplo: int,
    n: int,
    alpha: ScalarType,
    AP: torch.Tensor,
    x: torch.Tensor,
    incx: int,
    beta: ScalarType,
    y: torch.Tensor,
    incy: int,
) -> None:
    assert AP.dtype == torch.complex64 == x.dtype == y.dtype
    _check_common(AP, x, y, uplo, n, incx, incy)
    if n == 0:
        return
    ar, ai, br, bi = _complex_scalars(alpha, beta)
    alpha_zero = ar == 0.0 and ai == 0.0
    if alpha_zero and br == 1.0 and bi == 0.0:
        return
    if AP.device.index != _current_device():
        with torch_device_fn.device(AP.device):
            return chpmv(
                uplo, n, complex(ar, ai), AP, x, incx, complex(br, bi), y, incy
            )
    if n <= 32 and not alpha_zero:
        _launch(
            chpmv_small_kernel,
            (n,),
            (AP, x, y),
            (ar, ai, br, bi),
            (n, incx, incy, uplo, triton.next_power_of_2(n), br == 0.0 and bi == 0.0),
        )
        return
    if n <= 512 and not alpha_zero:
        _launch(
            chpmv_fused_kernel,
            (triton.cdiv(n, 32),),
            (AP, x, y),
            (ar, ai, br, bi),
            (n, incx, incy, uplo, 32, br == 0.0 and bi == 0.0),
        )
        return
    block = 64
    splits = triton.cdiv(n, block)
    partial = (
        y
        if alpha_zero
        else torch.empty((splits, n, 2), dtype=torch.float32, device=AP.device)
    )
    if not alpha_zero:
        _launch(
            chpmv_partial_kernel,
            (min(65535, splits * (splits + 1) // 2),),
            (AP, x, partial),
            (),
            (n, incx, uplo, block),
        )
    _launch(
        chpmv_finish_kernel,
        (triton.cdiv(n, 32),),
        (partial, y),
        (ar, ai, br, bi),
        (
            n,
            incy,
            splits,
            min(32, triton.next_power_of_2(splits)),
            32,
            br == 0.0 and bi == 0.0,
            alpha_zero,
        ),
    )
