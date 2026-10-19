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

import torch

from .dotu import _run_complex_dot

logger = logging.getLogger(__name__)


def _dotc_impl(n, x, incx, y, incy, result):
    _run_complex_dot(n, x, incx, y, incy, result, True)


def cdotc(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    result: torch.Tensor,
) -> None:
    logger.debug("FLAG_BLAS CDOTC")
    assert x.dtype == torch.complex64, "x must be complex64"
    assert y.dtype == torch.complex64, "y must be complex64"
    assert result.dtype == torch.complex64, "result must be complex64"
    _dotc_impl(n, x, incx, y, incy, result)


def zdotc(
    n: int,
    x: torch.Tensor,
    incx: int,
    y: torch.Tensor,
    incy: int,
    result: torch.Tensor,
) -> None:
    logger.debug("FLAG_BLAS ZDOTC")
    assert x.dtype == torch.complex128, "x must be complex128"
    assert y.dtype == torch.complex128, "y must be complex128"
    assert result.dtype == torch.complex128, "result must be complex128"
    _dotc_impl(n, x, incx, y, incy, result)
