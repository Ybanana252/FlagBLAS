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

import importlib

import torch
import triton

from flag_blas.runtime import torch_device_fn

_common = importlib.import_module("flag_blas.ops.level2.tpsv")

CUBLAS_DIAG_NON_UNIT = 0
CUBLAS_FILL_MODE_LOWER = 0
CUBLAS_OP_N = 0


def ctpsv(uplo, trans, diag, n, AP, x, incx):
    """Solve CTPSV with a smaller dependency chunk for medium MUSA cases.

    The common kernel groups four preceding panels for forward solves up to
    n=256.  On MUSA that creates a 32x128 complex tile even when most columns
    are masked.  Keeping one panel per chunk removes that register-heavy
    discontinuity while retaining the common implementation elsewhere.
    """

    _common._check_common(uplo, trans, diag, n, AP, x, incx)
    assert AP.dtype is torch.complex64 and x.dtype is torch.complex64
    if n == 0:
        return x

    if not (incx == 1 and diag == CUBLAS_DIAG_NON_UNIT and 65 <= n <= 256):
        return _common.ctpsv(uplo, trans, diag, n, AP, x, incx)

    physical_uplo, physical_trans, conj = _common._row_major_tpsv_args(uplo, trans)
    trans_flag = int(physical_trans != CUBLAS_OP_N)
    lower_eff = int(
        (physical_uplo == CUBLAS_FILL_MODE_LOWER) ^ (physical_trans != CUBLAS_OP_N)
    )
    block_n = 16
    panel_count = triton.cdiv(n, block_n)

    with torch_device_fn.device(AP.device):
        _common._complex_tpsv_blocked_kernel[(panel_count,)](
            torch.view_as_real(AP),
            torch.view_as_real(x),
            _common._tpsv_flags(AP.device),
            n,
            UPLO=physical_uplo,
            TRANS=trans_flag,
            UNIT=False,
            CONJ=conj,
            LOWER_EFF=lower_eff,
            FORWARD=bool(lower_eff),
            IS_DOUBLE=False,
            BLOCK_N=block_n,
            CHUNK=1,
            num_warps=4,
        )
    return x
