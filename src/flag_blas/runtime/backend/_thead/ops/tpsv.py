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

from flag_blas.ops.level2.tpsv import (
    CUBLAS_OP_C,
    _check_common,
    _complex_tpsv_kernel,
    _real_tpsv_kernel,
    _row_major_tpsv_args,
)
from flag_blas.runtime import torch_device_fn


def _real_tpsv(dtype, uplo, trans, diag, n, AP, x, incx):
    assert trans != CUBLAS_OP_C
    _check_common(uplo, trans, diag, n, AP, x, incx)
    assert AP.dtype is dtype and x.dtype is dtype
    if n == 0:
        return x

    uplo, trans, _ = _row_major_tpsv_args(uplo, trans)
    with torch_device_fn.device(AP.device):
        _real_tpsv_kernel[(1,)](uplo, trans, diag, n, AP, x, incx)
    return x


def _complex_tpsv(dtype, uplo, trans, diag, n, AP, x, incx):
    _check_common(uplo, trans, diag, n, AP, x, incx)
    assert AP.dtype is dtype and x.dtype is dtype
    if n == 0:
        return x

    uplo, trans, conj = _row_major_tpsv_args(uplo, trans)
    with torch_device_fn.device(AP.device):
        _complex_tpsv_kernel[(1,)](
            uplo,
            trans,
            diag,
            n,
            torch.view_as_real(AP),
            torch.view_as_real(x),
            incx,
            CONJ=conj,
        )
    return x


def stpsv(uplo, trans, diag, n, AP, x, incx):
    return _real_tpsv(torch.float32, uplo, trans, diag, n, AP, x, incx)


def dtpsv(uplo, trans, diag, n, AP, x, incx):
    return _real_tpsv(torch.float64, uplo, trans, diag, n, AP, x, incx)


def ctpsv(uplo, trans, diag, n, AP, x, incx):
    return _complex_tpsv(torch.complex64, uplo, trans, diag, n, AP, x, incx)


def ztpsv(uplo, trans, diag, n, AP, x, incx):
    return _complex_tpsv(torch.complex128, uplo, trans, diag, n, AP, x, incx)
