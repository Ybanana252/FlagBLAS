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

import torch

from flag_blas.ops.level2.tbsv import (
    CUBLAS_DIAG_UNIT,
    CUBLAS_OP_N,
    _check_tbsv,
    _complex_tbsv_kernel,
    _real_tbsv_kernel,
    _row_major_tbsv_args,
)
from flag_blas.runtime import torch_device_fn


def _real_tbsv(
    dtype: torch.dtype,
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert A.dtype == dtype == x.dtype
    _check_tbsv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=False)
    if n == 0:
        return

    uplo, trans, _ = _row_major_tbsv_args(uplo, trans)
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    trans_flag = 0 if trans == CUBLAS_OP_N else 1

    # Multi-program TBSV kernels use a spin-wait flag that can deadlock on T-Head.
    with torch_device_fn.device(A.device):
        _real_tbsv_kernel[(1,)](
            A,
            x,
            n,
            k,
            lda,
            incx,
            UPLO=uplo,
            TRANS=trans_flag,
            UNIT=unit,
        )


def _complex_tbsv(
    dtype: torch.dtype,
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    assert A.dtype == dtype == x.dtype
    _check_tbsv(A, x, uplo, trans, diag, n, k, lda, incx, complex_ok=True)
    if n == 0:
        return

    uplo, trans, conj = _row_major_tbsv_args(uplo, trans)
    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0

    with torch_device_fn.device(A.device):
        _complex_tbsv_kernel[(1,)](
            torch.view_as_real(A),
            torch.view_as_real(x),
            n,
            k,
            lda,
            incx,
            UPLO=uplo,
            TRANS=trans,
            UNIT=unit,
            CONJ=conj,
        )


def stbsv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    """Solve a single-precision triangular banded system in-place."""
    _real_tbsv(torch.float32, uplo, trans, diag, n, k, A, lda, x, incx)


def dtbsv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    """Solve a double-precision triangular banded system in-place."""
    _real_tbsv(torch.float64, uplo, trans, diag, n, k, A, lda, x, incx)


def ctbsv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    """Solve a complex single-precision triangular banded system in-place."""
    _complex_tbsv(torch.complex64, uplo, trans, diag, n, k, A, lda, x, incx)


def ztbsv(
    uplo: int,
    trans: int,
    diag: int,
    n: int,
    k: int,
    A: torch.Tensor,
    lda: int,
    x: torch.Tensor,
    incx: int,
) -> None:
    """Solve a complex double-precision triangular banded system in-place."""
    _complex_tbsv(torch.complex128, uplo, trans, diag, n, k, A, lda, x, incx)
