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

import copy
import importlib

import torch
import triton

from flag_blas.runtime import torch_device_fn


_common = importlib.import_module("flag_blas.ops.level2.hemv")

# Derive BLOCK_N inside autotune, after AABS has adjusted the tunable config.
# Supplying it at the call site lets AABS also add it to Config.kwargs, which
# makes the final launch receive BLOCK_N twice for non-power-of-two sizes.
chemv_small_kernel = triton.autotune(
    configs=copy.deepcopy(_common._CHEMV_SMALL_CONFIGS),
    key=list(_common._HEMV_KEY),
    restore_value=["y_ptr"],
)(
    triton.heuristics(
        {"BLOCK_N": lambda args: triton.next_power_of_2(args["n"])}
    )(_common.chemv_small_kernel.fn)
)


def chemv(uplo, n, alpha, A, lda, x, incx, beta, y, incy):
    if not (
        uplo == _common.CUBLAS_FILL_MODE_LOWER
        and 0 < n <= 192
        and incx == 1
        and incy == 1
    ):
        return _common.chemv(uplo, n, alpha, A, lda, x, incx, beta, y, incy)

    assert A.dtype == torch.complex64 == x.dtype == y.dtype
    _common._check_common(A, x, y, uplo, n, lda, incx, incy)
    ar, ai, br, bi = _common._complex_scalars(alpha, beta)
    if ar == 0.0 and ai == 0.0:
        return _common.chemv(uplo, n, alpha, A, lda, x, incx, beta, y, incy)

    with torch_device_fn.device(A.device):
        chemv_small_kernel[lambda meta: (triton.cdiv(n, meta["BLOCK_M"]),)](
            torch.view_as_real(A),
            torch.view_as_real(x),
            torch.view_as_real(y),
            ar,
            ai,
            br,
            bi,
            n,
            lda,
            incx,
            incy,
            UPLO=uplo,
            BETA_IS_ZERO=br == 0.0 and bi == 0.0,
        )
