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

from flag_blas import runtime
from flag_blas.ops.level2._constants import CUBLAS_FILL_MODE_LOWER
from flag_blas.ops.level2.spr import (
    ScalarType,
    _check_spr_args,
    _f64_to_i64,
    dspr as _generic_dspr,
    dspr_kernel as _generic_dspr_kernel,
    sspr as _generic_sspr,
    sspr_kernel as _generic_sspr_kernel,
)
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry, libtuner


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_spr_row"),
    key=["n", "INCX", "UPLO"],
    restore_value=["ap_ptr"],
)
@triton.jit
def sspr_thead_kernel(
    ap_ptr,
    x_ptr,
    alpha: tl.float32,
    n,
    INCX,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    program_count = tl.num_programs(0)
    lanes = tl.arange(0, BLOCK_SIZE)

    while row < n:
        n64 = tl.full((), n, tl.int64)
        row64 = row.to(tl.int64)

        if UPLO == 0:
            first_col = 0
            col_count = row + 1
            packed_start = row64 * (row64 + 1) // 2
        else:
            first_col = row
            col_count = n - row
            packed_start = row64 * (2 * n64 - row64 + 1) // 2

        x_row = tl.load(x_ptr + row * INCX)
        chunk = 0
        while chunk < col_count:
            cols = first_col + chunk + lanes
            mask = chunk + lanes < col_count
            x_cols = tl.load(x_ptr + cols * INCX, mask=mask, other=0.0)
            offsets = packed_start + chunk + lanes
            packed = tl.load(ap_ptr + offsets, mask=mask, other=0.0)
            tl.store(ap_ptr + offsets, packed + alpha * x_row * x_cols, mask=mask)
            chunk += BLOCK_SIZE
        row += program_count


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("thead_spr_row"),
    key=["n", "INCX", "UPLO"],
    restore_value=["ap_ptr"],
)
@triton.jit
def dspr_thead_kernel(
    ap_ptr,
    x_ptr,
    alpha_int: tl.int64,
    n,
    INCX,
    UPLO: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    program_count = tl.num_programs(0)
    lanes = tl.arange(0, BLOCK_SIZE)
    alpha = alpha_int.to(tl.float64, bitcast=True)

    while row < n:
        n64 = tl.full((), n, tl.int64)
        row64 = row.to(tl.int64)

        if UPLO == 0:
            first_col = 0
            col_count = row + 1
            packed_start = row64 * (row64 + 1) // 2
        else:
            first_col = row
            col_count = n - row
            packed_start = row64 * (2 * n64 - row64 + 1) // 2

        x_row = tl.load(x_ptr + row * INCX)
        chunk = 0
        while chunk < col_count:
            cols = first_col + chunk + lanes
            mask = chunk + lanes < col_count
            x_cols = tl.load(x_ptr + cols * INCX, mask=mask, other=0.0)
            offsets = packed_start + chunk + lanes
            packed = tl.load(ap_ptr + offsets, mask=mask, other=0.0)
            tl.store(ap_ptr + offsets, packed + alpha * x_row * x_cols, mask=mask)
            chunk += BLOCK_SIZE
        row += program_count


# Reuse the row kernels with measured configs for the medium-size, unit-stride
# region. Keep the original autotuner (and its cache) for all other sizes.
sspr_thead_configured_kernel = libentry()(sspr_thead_kernel.jit_function)
dspr_thead_configured_kernel = libentry()(dspr_thead_kernel.jit_function)
_SSPR_MEDIUM_CONFIG = runtime.get_tuned_config("thead_sspr_row_medium")[0]
_DSPR_MEDIUM_CONFIG = runtime.get_tuned_config("thead_dspr_row_medium")[0]
_SPR_WIDE_CONFIG = runtime.get_tuned_config("thead_spr_row_wide")[0]

# Small upper-packed matrices benefit from 2-D tiles instead of row loops.
# Only the T-Head dispatch/config changes; the shared kernels stay untouched.
sspr_thead_small_kernel = libentry()(_generic_sspr_kernel.jit_function)
dspr_thead_small_kernel = libentry()(_generic_dspr_kernel.jit_function)
_SPR_SMALL_CONFIG = runtime.get_tuned_config("thead_spr_tile_small")[0]


def sspr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    AP: torch.Tensor,
) -> None:
    if uplo == CUBLAS_FILL_MODE_LOWER:
        _generic_sspr(uplo, n, alpha, x, incx, AP)
        return
    _check_spr_args(torch.float32, uplo, n, alpha, x, incx, AP)
    if n == 0:
        return
    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha_value == 0.0:
        return

    with torch_device_fn.device(AP.device):
        if incx == 1 and 64 <= n <= 192:
            config = _SPR_SMALL_CONFIG
            grid = (triton.cdiv(n, config.kwargs["BLOCK_SIZE"]),) * 2
            sspr_thead_small_kernel[grid](
                AP, x, alpha_value, n, incx, UPLO=1 - uplo, **config.all_kwargs()
            )
        elif incx == 1 and 160 <= n <= 1024:
            config = _SSPR_MEDIUM_CONFIG if n <= 512 else _SPR_WIDE_CONFIG
            sspr_thead_configured_kernel[(n,)](
                AP, x, alpha_value, n, incx, UPLO=uplo, **config.all_kwargs()
            )
        else:
            sspr_thead_kernel[(min(n, 65535),)](
                AP, x, alpha_value, n, incx, UPLO=uplo
            )


def dspr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    AP: torch.Tensor,
) -> None:
    if uplo == CUBLAS_FILL_MODE_LOWER:
        _generic_dspr(uplo, n, alpha, x, incx, AP)
        return
    _check_spr_args(torch.float64, uplo, n, alpha, x, incx, AP)
    if n == 0:
        return
    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    if alpha_value == 0.0:
        return

    with torch_device_fn.device(AP.device):
        if incx == 1 and 64 <= n <= 224:
            config = _SPR_SMALL_CONFIG
            grid = (triton.cdiv(n, config.kwargs["BLOCK_SIZE"]),) * 2
            dspr_thead_small_kernel[grid](
                AP,
                x,
                _f64_to_i64(alpha_value),
                n,
                incx,
                UPLO=1 - uplo,
                **config.all_kwargs(),
            )
        elif incx == 1 and 191 <= n <= 1024:
            config = _DSPR_MEDIUM_CONFIG if n <= 256 else _SPR_WIDE_CONFIG
            dspr_thead_configured_kernel[(n,)](
                AP,
                x,
                _f64_to_i64(alpha_value),
                n,
                incx,
                UPLO=uplo,
                **config.all_kwargs(),
            )
        else:
            dspr_thead_kernel[(min(n, 65535),)](
                AP, x, _f64_to_i64(alpha_value), n, incx, UPLO=uplo
            )
