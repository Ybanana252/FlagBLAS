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
import triton

from flag_blas.ops.level2._constants import CUBLAS_DIAG_UNIT
from flag_blas.ops.level2.trsv import (
    _check_trsv,
    _forward,
    _panel_ranges,
    _row_major_dispatch,
    ctrsv_panel_kernel,
    ctrsv_update_kernel,
    dtrsv_panel_kernel,
    dtrsv_update_kernel,
    ztrsv_panel_kernel,
    ztrsv_update_kernel,
)
from flag_blas.runtime import torch_device_fn


def _real_trsv(dtype, uplo, trans, diag, n, A, lda, x, incx):
    assert A.dtype == dtype == x.dtype
    _check_trsv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=False)
    if n == 0:
        return

    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    uplo, trans_flag, _ = _row_major_dispatch(uplo, trans)
    forward = _forward(uplo, trans_flag)
    block_n = 16
    block_m = 128

    # Multi-program TRSV kernels spin on a global flag and can deadlock on
    # T-Head. Launch each dependency-ordered panel on the current stream.
    with torch_device_fn.device(A.device):
        for start, end in _panel_ranges(n, block_n, forward):
            dtrsv_panel_kernel[(1,)](
                A,
                x,
                start,
                end,
                lda,
                incx,
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
                FORWARD=forward,
                BLOCK_N=block_n,
            )
            if forward:
                row_base = end
                row_count = n - end
            else:
                row_base = 0
                row_count = start
            if row_count > 0:
                dtrsv_update_kernel[(triton.cdiv(row_count, block_m),)](
                    A,
                    x,
                    row_base,
                    row_count,
                    start,
                    end,
                    lda,
                    incx,
                    TRANS=trans_flag,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                )


def _complex_trsv(dtype, panel_kernel, update_kernel, block_n, block_m,
                  uplo, trans, diag, n, A, lda, x, incx):
    assert A.dtype == dtype == x.dtype
    _check_trsv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=True)
    if n == 0:
        return

    unit = 1 if diag == CUBLAS_DIAG_UNIT else 0
    uplo, trans_flag, conj = _row_major_dispatch(uplo, trans)
    forward = _forward(uplo, trans_flag)
    A_real = torch.view_as_real(A)
    x_real = torch.view_as_real(x)

    with torch_device_fn.device(A.device):
        for start, end in _panel_ranges(n, block_n, forward):
            panel_kernel[(1,)](
                A_real,
                x_real,
                start,
                end,
                lda,
                incx,
                UPLO=uplo,
                TRANS=trans_flag,
                UNIT=unit,
                FORWARD=forward,
                CONJ=conj,
                BLOCK_N=block_n,
            )
            if forward:
                row_base = end
                row_count = n - end
            else:
                row_base = 0
                row_count = start
            if row_count > 0:
                update_kernel[(triton.cdiv(row_count, block_m),)](
                    A_real,
                    x_real,
                    row_base,
                    row_count,
                    start,
                    end,
                    lda,
                    incx,
                    TRANS=trans_flag,
                    CONJ=conj,
                    BLOCK_M=block_m,
                    BLOCK_N=block_n,
                )


def strsv(uplo, trans, diag, n, A, lda, x, incx):
    """Solve a single-precision triangular system in-place."""
    _real_trsv(torch.float32, uplo, trans, diag, n, A, lda, x, incx)


def dtrsv(uplo, trans, diag, n, A, lda, x, incx):
    """Solve a double-precision triangular system in-place."""
    _real_trsv(torch.float64, uplo, trans, diag, n, A, lda, x, incx)


def ctrsv(uplo, trans, diag, n, A, lda, x, incx):
    """Solve a complex single-precision triangular system in-place."""
    _complex_trsv(
        torch.complex64,
        ctrsv_panel_kernel,
        ctrsv_update_kernel,
        16,
        128,
        uplo,
        trans,
        diag,
        n,
        A,
        lda,
        x,
        incx,
    )


def ztrsv(uplo, trans, diag, n, A, lda, x, incx):
    """Solve a complex double-precision triangular system in-place."""
    _complex_trsv(
        torch.complex128,
        ztrsv_panel_kernel,
        ztrsv_update_kernel,
        8,
        64,
        uplo,
        trans,
        diag,
        n,
        A,
        lda,
        x,
        incx,
    )
