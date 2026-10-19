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
from importlib import import_module

import torch

from .nrm2 import _run_reduction

logger = logging.getLogger(__name__)


def _asum_impl(n, x, incx, result, is_complex):
    if n <= 0 or incx <= 0:
        result.zero_()
        return
    common_name = {
        torch.float32: "sasum",
        torch.float64: "dasum",
        torch.complex64: "scasum",
        torch.complex128: "dzasum",
    }[x.dtype]
    use_single = n <= 8192 and result.numel() == 1 and x.device == result.device
    use_optimized = use_single or (
        1048576 <= n <= 4194304
        and result.numel() == 1
        and x.device == result.device
        and not (x.dtype == torch.float64 and incx == 1)
        and not (x.dtype == torch.complex128 and incx == 2)
    )
    if not use_optimized:
        common = import_module("flag_blas.ops.level1.asum")
        getattr(common, common_name)(n, x, incx, result)
        return
    required_size = 1 + (n - 1) * incx
    assert x.numel() >= required_size, (
        f"x is too short: need at least {required_size} elements "
        f"for n={n}, incx={incx}, got {x.numel()}"
    )
    assert x.is_contiguous(), "x must be contiguous"
    if use_single:
        config = (
            max(256, 1 << (n - 1).bit_length()),
            1 if n <= 2048 else 2,
            0,
            is_complex,
        )
    elif x.dtype == torch.float64:
        config = (1024, 8, 160, False)
    elif x.dtype == torch.complex64:
        config = (1024, 4, 160, True)
    elif x.dtype == torch.complex128:
        config = (512, 4, 160, True)
    else:
        config = (1024, 4, 160, False)
    _run_reduction(n, x, incx, result, False, config)


def sasum(n: int, x: torch.Tensor, incx: int, result: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS SASUM")
    assert x.dtype == torch.float32, "x must be float32"
    assert result.dtype == torch.float32, "result must be float32"
    _asum_impl(n, x, incx, result, is_complex=False)


def dasum(n: int, x: torch.Tensor, incx: int, result: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS DASUM")
    assert x.dtype == torch.float64, "x must be float64"
    assert result.dtype == torch.float64, "result must be float64"
    _asum_impl(n, x, incx, result, is_complex=False)


def scasum(n: int, x: torch.Tensor, incx: int, result: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS SCASUM")
    assert x.dtype == torch.complex64, "x must be complex64"
    assert result.dtype == torch.float32, "result for scasum must be float32"
    _asum_impl(n, x, incx, result, is_complex=True)


def dzasum(n: int, x: torch.Tensor, incx: int, result: torch.Tensor) -> None:
    logger.debug("FLAG_BLAS DZASUM")
    assert x.dtype == torch.complex128, "x must be complex128"
    assert result.dtype == torch.float64, "result for dzasum must be float64"
    _asum_impl(n, x, incx, result, is_complex=True)
