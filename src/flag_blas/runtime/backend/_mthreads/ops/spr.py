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
import triton.language as tl

from flag_blas.ops.level2.spr import ScalarType, _check_spr_args
from flag_blas.ops.level2.spr import sspr as _generic_sspr
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry


@libentry()
@triton.jit
def sspr_small_mthreads_kernel(
    ap_ptr,
    x_ptr,
    alpha: tl.float32,
    N: tl.constexpr,
    INCX: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    # AP indices fit int32 because this kernel is restricted to N <= 512.
    # Widen vector offsets separately to preserve support for large INCX.
    xr = tl.load(x_ptr + row.to(tl.int64) * INCX)
    if UPLO == 0:
        offsets = row * (row + 1) // 2 + cols
        mask = (cols < N) & (cols <= row)
    else:
        offsets = row * (2 * N - row + 1) // 2 + cols - row
        mask = (cols < N) & (cols >= row)
    xc = tl.load(x_ptr + cols.to(tl.int64) * INCX, mask, other=0.0)
    ap = tl.load(ap_ptr + offsets, mask, other=0.0)
    tl.store(ap_ptr + offsets, ap + (alpha * xr) * xc, mask)


def sspr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    AP: torch.Tensor,
) -> None:
    if n > 512:
        return _generic_sspr(uplo, n, alpha, x, incx, AP)

    _check_spr_args(torch.float32, uplo, n, alpha, x, incx, AP)
    if n == 0:
        return
    alpha = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha == 0.0:
        return

    block_n = triton.next_power_of_2(n)
    # Use 4/8/16 warps for up to 128/256/512 columns, respectively.
    # Public UPLO already describes row-packed storage; do not flip it here.
    with torch_device_fn.device(AP.device):
        sspr_small_mthreads_kernel[(n,)](
            AP,
            x,
            alpha,
            N=n,
            INCX=incx,
            UPLO=uplo,
            BLOCK_N=block_n,
            num_warps=max(4, block_n // 32),
            num_stages=1,
        )
