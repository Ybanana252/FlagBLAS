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

import logging
from functools import lru_cache
from importlib import import_module

import torch
import triton
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _sum_single(
    X,
    O,
    N: tl.constexpr,
    INCX: tl.constexpr,
    COMPLEX: tl.constexpr,
    SQUARE: tl.constexpr,
    PACKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, O.dtype.element_ty)
    for start in range(tl.cdiv(N, BLOCK)):
        idx = start * BLOCK + i
        if COMPLEX:
            if PACKED:
                pair = tl.load(
                    X + (idx * INCX * 2)[:, None] + tl.arange(0, 2)[None, :],
                    mask=(idx < N)[:, None],
                    other=0,
                )
                a, b = tl.split(pair)
            else:
                a = tl.load(X + idx * INCX * 2, mask=idx < N, other=0)
                b = tl.load(X + idx * INCX * 2 + 1, mask=idx < N, other=0)
            if SQUARE:
                value = a * a + b * b
            else:
                value = tl.abs(a) + tl.abs(b)
        else:
            a = tl.load(X + idx * INCX, mask=idx < N, other=0)
            value = a * a if SQUARE else tl.abs(a)
        acc += value
    total = tl.sum(acc, 0)
    if SQUARE:
        total = tl.sqrt(total)
    tl.store(O, total)


@libentry()
@triton.jit
def _sum_stage1(
    X,
    P,
    N: tl.constexpr,
    INCX: tl.constexpr,
    COMPLEX: tl.constexpr,
    SQUARE: tl.constexpr,
    PACKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    grid = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, P.dtype.element_ty)
    for tile in range(pid, tl.cdiv(N, BLOCK), grid):
        idx = tile * BLOCK + lane
        if COMPLEX:
            if PACKED:
                pair = tl.load(
                    X + (idx * INCX * 2)[:, None] + tl.arange(0, 2)[None, :],
                    mask=(idx < N)[:, None],
                    other=0,
                )
                a, b = tl.split(pair)
            else:
                a = tl.load(X + idx * INCX * 2, mask=idx < N, other=0)
                b = tl.load(X + idx * INCX * 2 + 1, mask=idx < N, other=0)
            if SQUARE:
                value = a * a + b * b
            else:
                value = tl.abs(a) + tl.abs(b)
        else:
            a = tl.load(X + idx * INCX, mask=idx < N, other=0)
            value = a * a if SQUARE else tl.abs(a)
        acc += value
    tl.store(P + pid, tl.sum(acc, 0))


@libentry()
@triton.jit
def _sum_finish(
    P, Output, COUNT: tl.constexpr, SQUARE: tl.constexpr, BLOCK: tl.constexpr
):
    i = tl.arange(0, BLOCK)
    value = tl.load(P + i, mask=i < COUNT, other=0)
    total = tl.sum(value, 0)
    if SQUARE:
        total = tl.sqrt(total)
    tl.store(Output, total)


@lru_cache(maxsize=None)
def _configured_kernel(name, warps):
    # Keep separate entry caches for distinct launch configurations.
    return libentry()(globals()[name].jit_function)


def _run_reduction(n, x, incx, result, square, config):
    block, warps, grid_cap, packed = config
    raw = torch.view_as_real(x) if x.is_complex() else x
    launch = dict(num_warps=warps, num_stages=1, enable_fp_fusion=False)
    with torch_device_fn.device(x.device):
        if grid_cap == 0:
            _configured_kernel("_sum_single", warps)[(1,)](
                raw, result, n, incx, x.is_complex(), square, packed, block, **launch
            )
        else:
            grid = min(triton.cdiv(n, block), grid_cap)
            partial = torch.empty(grid, dtype=result.dtype, device=x.device)
            _configured_kernel("_sum_stage1", warps)[(grid,)](
                raw, partial, n, incx, x.is_complex(), square, packed, block, **launch
            )
            _configured_kernel("_sum_finish", 4)[(1,)](
                partial,
                result,
                grid,
                square,
                triton.next_power_of_2(grid),
                num_warps=4,
                num_stages=1,
            )


def _nrm2_impl(n, x, incx, result, is_complex):
    if n <= 0 or incx <= 0:
        result.zero_()
        return
    if 8192 < n <= 131072:
        common = import_module("flag_blas.ops.level1.nrm2")
        common._nrm2_impl(n, x, incx, result, is_complex)
        return

    required_size = 1 + (n - 1) * incx
    assert x.numel() >= required_size, (
        f"x is too short: need at least {required_size} elements "
        f"for n={n}, incx={incx}, got {x.numel()}"
    )
    assert x.is_contiguous(), "x must be contiguous"
    assert (
        result.dim() == 1 and result.numel() == 1
    ), "result must be a 1-element tensor"
    assert x.device == result.device, "x and result must be on the same device"

    if n <= 8192:
        if x.dtype == torch.float32:
            config = (256, 1, 0, False) if n <= 2048 else (8192, 2, 0, False)
        elif x.dtype == torch.float64:
            config = (512, 2, 0, False) if n <= 2048 else (8192, 2, 0, False)
        elif x.dtype == torch.complex64:
            config = (256, 1, 0, True) if n <= 2048 else (8192, 1, 0, True)
        else:
            config = (1024, 1, 0, True) if n <= 2048 else (8192, 1, 0, True)
    elif x.dtype == torch.complex64:
        config = (1024, 4, 160, True)
    elif x.dtype in (torch.float64, torch.complex128):
        config = (512, 4, 160, False)
    else:
        config = (1024, 4, 160, False)
    _run_reduction(n, x, incx, result, True, config)


def snrm2(n: int, x: torch.Tensor, incx: int, result: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS SNRM2")
    assert x.dtype == torch.float32, "x must be float32"
    assert result.dtype == torch.float32, "result must be float32"
    _nrm2_impl(n, x, incx, result, is_complex=False)


def dnrm2(n: int, x: torch.Tensor, incx: int, result: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS DNRM2")
    assert x.dtype == torch.float64, "x must be float64"
    assert result.dtype == torch.float64, "result must be float64"
    _nrm2_impl(n, x, incx, result, is_complex=False)


def scnrm2(n: int, x: torch.Tensor, incx: int, result: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS SCNRM2")
    assert x.dtype == torch.complex64, "x must be complex64"
    assert result.dtype == torch.float32, "result for scnrm2 must be float32"
    _nrm2_impl(n, x, incx, result, is_complex=True)


def dznrm2(n: int, x: torch.Tensor, incx: int, result: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS DZNRM2")
    assert x.dtype == torch.complex128, "x must be complex128"
    assert result.dtype == torch.float64, "result for dznrm2 must be float64"
    _nrm2_impl(n, x, incx, result, is_complex=True)
