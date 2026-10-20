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

from .nrm2 import _configured_kernel as _sum_kernel

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _dot_real(
    X,
    Y,
    O,
    N: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    grid = tl.num_programs(0)
    lane = tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, O.dtype.element_ty)
    for tile in range(pid, tl.cdiv(N, BLOCK), grid):
        idx = tile * BLOCK + lane
        x = tl.load(X + idx * INCX, mask=idx < N, other=0)
        y = tl.load(Y + idx * INCY, mask=idx < N, other=0)
        acc += x * y
    tl.store(O + pid, tl.sum(acc, 0))


@lru_cache(maxsize=1)
def _sdot_kernel():
    return libentry()(_dot_real.jit_function)


def _dot_impl(n, x, incx, y, incy, result):
    if not (
        8388608 <= n <= 16777216
        and incx == incy == 1
        and x.is_contiguous()
        and y.is_contiguous()
        and result.is_contiguous()
    ):
        common = import_module("flag_blas.ops.level1.dot")
        common._dot_impl(n, x, incx, y, incy, result)
        return
    assert x.dim() == 1, "x must be 1-dimensional"
    assert y.dim() == 1, "y must be 1-dimensional"
    assert x.device == y.device, "x and y must be on the same device"
    assert x.device == result.device, "result must be on the same device"
    assert result.dtype == x.dtype, "result dtype must match input dtype"
    assert result.numel() == 1, "result must be a scalar tensor"
    assert x.numel() >= n, f"x too short: need {n}, got {x.numel()}"
    assert y.numel() >= n, f"y too short: need {n}, got {y.numel()}"
    grid = min(triton.cdiv(n, 2048), 160)
    partial = torch.empty(grid, dtype=result.dtype, device=x.device)
    with torch_device_fn.device(x.device):
        _sdot_kernel()[(grid,)](
            x,
            y,
            partial,
            n,
            incx,
            incy,
            2048,
            num_warps=8,
            num_stages=1,
            enable_fp_fusion=False,
        )
        _sum_kernel("_sum_finish", 4)[(1,)](
            partial,
            result,
            grid,
            False,
            triton.next_power_of_2(grid),
            num_warps=4,
            num_stages=1,
        )


def sdot(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    result: torch.Tensor,
) -> None:
    logger.debug("FLAG_BLAS SDOT")
    assert x.dtype == torch.float32, "x must be float32"
    assert y.dtype == torch.float32, "y must be float32"
    assert result.dtype == torch.float32, "result must be float32"
    _dot_impl(n, x, incx, y, incy, result)
