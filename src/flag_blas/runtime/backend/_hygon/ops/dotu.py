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
def _dot_complex(
    X,
    Y,
    O,
    N: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    CONJUGATE: tl.constexpr,
    PACKED: tl.constexpr,
    BLOCK: tl.constexpr,
    PARTIAL: tl.constexpr,
):
    pid = tl.program_id(0)
    grid = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    acc_real = tl.full((BLOCK,), 0, O.dtype.element_ty)
    acc_imag = tl.full((BLOCK,), 0, O.dtype.element_ty)
    for tile in range(pid, tl.cdiv(N, BLOCK), grid):
        idx = tile * BLOCK + lane
        if PACKED == 2:
            flat = tl.arange(0, BLOCK * 2)
            logical = tile * BLOCK + flat // 2
            xx = tl.load(X + logical * INCX * 2 + flat % 2, mask=logical < N, other=0)
            yy = tl.load(Y + logical * INCY * 2 + flat % 2, mask=logical < N, other=0)
            xr, xi = tl.split(tl.reshape(xx, (BLOCK, 2)))
            yr, yi = tl.split(tl.reshape(yy, (BLOCK, 2)))
        elif PACKED:
            pair_x = tl.load(
                X + (idx * INCX * 2)[:, None] + tl.arange(0, 2)[None, :],
                mask=(idx < N)[:, None],
                other=0,
            )
            pair_y = tl.load(
                Y + (idx * INCY * 2)[:, None] + tl.arange(0, 2)[None, :],
                mask=(idx < N)[:, None],
                other=0,
            )
            xr, xi = tl.split(pair_x)
            yr, yi = tl.split(pair_y)
        else:
            xr = tl.load(X + idx * INCX * 2, mask=idx < N, other=0)
            xi = tl.load(X + idx * INCX * 2 + 1, mask=idx < N, other=0)
            yr = tl.load(Y + idx * INCY * 2, mask=idx < N, other=0)
            yi = tl.load(Y + idx * INCY * 2 + 1, mask=idx < N, other=0)
        if CONJUGATE:
            acc_real += xr * yr + xi * yi
            acc_imag += xr * yi - xi * yr
        else:
            acc_real += xr * yr - xi * yi
            acc_imag += xr * yi + xi * yr
    tl.store(O + pid * 2, tl.sum(acc_real, 0))
    tl.store(O + pid * 2 + 1, tl.sum(acc_imag, 0))


@libentry()
@triton.jit
def _dot_small(
    X,
    Y,
    O,
    N: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    CONJUGATE: tl.constexpr,
    PACKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    idx = tl.arange(0, BLOCK)
    if PACKED:
        xx = tl.load(
            X + (idx * INCX * 2)[:, None] + tl.arange(0, 2)[None, :],
            mask=(idx < N)[:, None],
            other=0,
        )
        yy = tl.load(
            Y + (idx * INCY * 2)[:, None] + tl.arange(0, 2)[None, :],
            mask=(idx < N)[:, None],
            other=0,
        )
        xr, xi = tl.split(xx)
        yr, yi = tl.split(yy)
    else:
        xr = tl.load(X + idx * INCX * 2, mask=idx < N, other=0)
        xi = tl.load(X + idx * INCX * 2 + 1, mask=idx < N, other=0)
        yr = tl.load(Y + idx * INCY * 2, mask=idx < N, other=0)
        yi = tl.load(Y + idx * INCY * 2 + 1, mask=idx < N, other=0)
    if CONJUGATE:
        real = xr * yr + xi * yi
        imag = xr * yi - xi * yr
    else:
        real = xr * yr - xi * yi
        imag = xr * yi + xi * yr
    tl.store(O, tl.sum(real, 0))
    tl.store(O + 1, tl.sum(imag, 0))


@libentry()
@triton.jit
def _dot_finish(P, Output, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    a = tl.load(P + i * 2, mask=i < COUNT, other=0)
    b = tl.load(P + i * 2 + 1, mask=i < COUNT, other=0)
    tl.store(Output, tl.sum(a, 0))
    tl.store(Output + 1, tl.sum(b, 0))


@libentry()
@triton.jit
def _conjugate_result(Output):
    imaginary = tl.load(Output + 1)
    tl.store(Output + 1, -imaginary)


@lru_cache(maxsize=None)
def _dot_kernel(name, warps):
    return libentry()(globals()[name].jit_function)


def _complex_config(n, dtype, incx, incy):
    if dtype == torch.complex64:
        if 2048 < n <= 8192:
            return (8192, 8, 0, True)
        if incx == incy == 1 and n >= 67108864:
            return (1024, 2, 160, 2)
        if incx == incy == 2 and n >= 1048576:
            return (512, 2, 320, 2) if n < 8388608 else (4096, 8, 320, 2)
        if incx == incy == 1 and 2097152 <= n <= 4194304:
            return (512, 4, 160, True)
    elif n >= 33554432 and (incx == 3 or incy == 3):
        return (2048, 4, 1280 if incx == incy == 3 else 320, 2)
    return None


def _validate_dotc_inputs(n, x, incx, y, incy, result):
    if n <= 0:
        result.zero_()
        return

    assert incx > 0, "incx must be positive"
    assert incy > 0, "incy must be positive"
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert (
        x.device == y.device == result.device
    ), "all tensors must reside on the same device"
    assert (
        x.dtype == y.dtype == result.dtype
    ), "x, y and result must have identical complex dtypes"
    assert result.numel() == 1, "result must be a scalar tensor"

    required_x = 1 + (n - 1) * incx
    required_y = 1 + (n - 1) * incy
    assert x.numel() >= required_x, (
        f"x is too short: need at least {required_x} elements "
        f"for n={n}, incx={incx}, got {x.numel()}"
    )
    assert y.numel() >= required_y, (
        f"y is too short: need at least {required_y} elements "
        f"for n={n}, incy={incy}, got {y.numel()}"
    )


def _run_complex_dot(n, x, incx, y, incy, result, conjugate):
    config = _complex_config(n, x.dtype, incx, incy)
    common = import_module(
        "flag_blas.ops.level1.dotc" if conjugate else "flag_blas.ops.level1.dotu"
    )
    if (
        x.dtype == torch.complex128
        and 1048576 <= n <= 16777216
        and incx == 3
        and incy == 2
        and x.is_contiguous()
        and y.is_contiguous()
        and result.is_contiguous()
    ):
        if conjugate:
            _validate_dotc_inputs(n, x, incx, y, incy, result)
        else:
            common._validate_dotu_inputs(n, x, incx, y, incy, result)
        getattr(common, "zdotc" if conjugate else "zdotu")(n, y, incy, x, incx, result)
        if conjugate:
            with torch_device_fn.device(result.device):
                _conjugate_result[(1,)](
                    torch.view_as_real(result), num_warps=1, num_stages=1
                )
        return
    if config is None or not (
        x.is_contiguous() and y.is_contiguous() and result.is_contiguous()
    ):
        name = ("c" if x.dtype == torch.complex64 else "z") + (
            "dotc" if conjugate else "dotu"
        )
        getattr(common, name)(n, x, incx, y, incy, result)
        return
    if conjugate:
        _validate_dotc_inputs(n, x, incx, y, incy, result)
    else:
        common._validate_dotu_inputs(n, x, incx, y, incy, result)
    block, warps, grid_cap, packed = config
    raw_x, raw_y, raw_result = (
        torch.view_as_real(x),
        torch.view_as_real(y),
        torch.view_as_real(result),
    )
    launch = dict(num_warps=warps, num_stages=1, enable_fp_fusion=False)
    with torch_device_fn.device(x.device):
        if grid_cap == 0:
            _dot_kernel("_dot_small", warps)[(1,)](
                raw_x,
                raw_y,
                raw_result,
                n,
                incx,
                incy,
                conjugate,
                packed,
                block,
                **launch,
            )
        else:
            grid = min(triton.cdiv(n, block), grid_cap)
            partial = torch.empty(grid, dtype=result.dtype, device=x.device)
            _dot_kernel("_dot_complex", warps)[(grid,)](
                raw_x,
                raw_y,
                torch.view_as_real(partial),
                n,
                incx,
                incy,
                conjugate,
                packed,
                block,
                True,
                **launch,
            )
            _dot_kernel("_dot_finish", 4)[(1,)](
                torch.view_as_real(partial),
                raw_result,
                grid,
                triton.next_power_of_2(grid),
                num_warps=4,
                num_stages=1,
            )


def _dotu_impl(n, x, incx, y, incy, result):
    _run_complex_dot(n, x, incx, y, incy, result, False)


def cdotu(
    n: int, x: torch.Tensor, incx: int, y: torch.Tensor, incy: int, result: torch.Tensor
) -> None:
    logger.debug("FLAG_BLAS CDOTU")
    assert x.dtype == torch.complex64, "x must be complex64"
    _dotu_impl(n, x, incx, y, incy, result)


def zdotu(
    n: int, x: torch.Tensor, incx: int, y: torch.Tensor, incy: int, result: torch.Tensor
) -> None:
    logger.debug("FLAG_BLAS ZDOTU")
    assert x.dtype == torch.complex128, "x must be complex128"
    _dotu_impl(n, x, incx, y, incy, result)
