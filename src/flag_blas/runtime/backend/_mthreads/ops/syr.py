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

from flag_blas.ops.level2.syr import ScalarType, _check_syr_args
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry


@libentry()
@triton.jit
def _syr_rows_kernel(
    A,
    X,
    alpha_r,
    alpha_i,
    N: tl.constexpr,
    LDA: tl.constexpr,
    INCX: tl.constexpr,
    UPLO: tl.constexpr,
    COMPLEX: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    r0 = tl.program_id(0) * BM
    c0 = tl.program_id(1) * BN
    if UPLO == 0:
        if r0 + BM <= c0:
            return
    else:
        if c0 + BN <= r0:
            return
    r = r0 + tl.arange(0, BM)
    c = c0 + tl.arange(0, BN)
    mask = (r[:, None] < N) & (c[None, :] < N)
    if UPLO == 0:
        mask = mask & (r[:, None] >= c[None, :])
    else:
        mask = mask & (r[:, None] <= c[None, :])
    off = r[:, None].to(tl.int64) * LDA + c[None, :].to(tl.int64)
    if COMPLEX:
        xr = tl.load(X + r.to(tl.int64) * INCX * 2, r < N, other=0.0)
        xi = tl.load(X + r.to(tl.int64) * INCX * 2 + 1, r < N, other=0.0)
        yr = tl.load(X + c.to(tl.int64) * INCX * 2, c < N, other=0.0)
        yi = tl.load(X + c.to(tl.int64) * INCX * 2 + 1, c < N, other=0.0)
        pr = xr[:, None] * yr[None, :] - xi[:, None] * yi[None, :]
        pi = xr[:, None] * yi[None, :] + xi[:, None] * yr[None, :]
        ur = alpha_r * pr - alpha_i * pi
        ui = alpha_r * pi + alpha_i * pr
        oldr = tl.load(A + off * 2, mask, other=0.0)
        oldi = tl.load(A + off * 2 + 1, mask, other=0.0)
        tl.store(A + off * 2, oldr + ur, mask)
        tl.store(A + off * 2 + 1, oldi + ui, mask)
    else:
        xr = tl.load(X + r.to(tl.int64) * INCX, r < N, other=0.0)
        xc = tl.load(X + c.to(tl.int64) * INCX, c < N, other=0.0)
        old = tl.load(A + off, mask, other=0.0)
        tl.store(A + off, old + alpha_r * xr[:, None] * xc[None, :], mask)


@libentry()
@triton.jit
def _ssyr_tiles_kernel(
    A,
    x,
    alpha,
    n: tl.constexpr,
    lda: tl.constexpr,
    incx: tl.constexpr,
    uplo: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    major = ((tl.sqrt(8.0 * pid + 1.0) - 1.0) * 0.5).to(tl.int32)
    # Correct floating-point inverse rounding at triangular tile boundaries.
    base = major * (major + 1) // 2
    major = tl.where(base > pid, major - 1, major)
    next_base = (major + 1) * (major + 2) // 2
    major = tl.where(next_base <= pid, major + 1, major)
    minor = pid - major * (major + 1) // 2
    if uplo == 0:
        pid_m = major
        pid_n = minor
    else:
        pid_m = minor
        pid_n = major
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_bounds = (rows[:, None] < n) & (cols[None, :] < n)
    if uplo == 0:
        mask_tri = rows[:, None] >= cols[None, :]
    else:
        mask_tri = rows[:, None] <= cols[None, :]
    mask = mask_bounds & mask_tri
    xv_r = tl.load(x + rows * incx, mask=rows < n, other=0.0)
    xv_c = tl.load(x + cols * incx, mask=cols < n, other=0.0)
    offs = rows[:, None].to(tl.int64) + cols[None, :].to(tl.int64) * lda
    old = tl.load(A + offs, mask=mask, other=0.0)
    val = old + alpha * xv_r[:, None] * xv_c[None, :]
    tl.store(A + offs, val, mask=mask)


@libentry()
@triton.jit
def _csyr_tiles_kernel(
    A,
    x,
    alpha_r,
    alpha_i,
    n: tl.constexpr,
    lda: tl.constexpr,
    incx: tl.constexpr,
    uplo: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_bounds = (rows[:, None] < n) & (cols[None, :] < n)
    if uplo == 0:
        mask_tri = rows[:, None] >= cols[None, :]
    else:
        mask_tri = rows[:, None] <= cols[None, :]
    mask = mask_bounds & mask_tri

    xr = tl.load(x + (rows * incx) * 2, mask=rows < n, other=0.0)
    xi = tl.load(x + (rows * incx) * 2 + 1, mask=rows < n, other=0.0)
    yr = tl.load(x + (cols * incx) * 2, mask=cols < n, other=0.0)
    yi = tl.load(x + (cols * incx) * 2 + 1, mask=cols < n, other=0.0)

    prod_r = xr[:, None] * yr[None, :] - xi[:, None] * yi[None, :]
    prod_i = xr[:, None] * yi[None, :] + xi[:, None] * yr[None, :]
    alpha_real = alpha_r
    alpha_imag = alpha_i
    upd_r = alpha_real * prod_r - alpha_imag * prod_i
    upd_i = alpha_real * prod_i + alpha_imag * prod_r

    elem = rows[:, None].to(tl.int64) + cols[None, :].to(tl.int64) * lda
    old_r = tl.load(A + elem * 2, mask=mask, other=0.0)
    old_i = tl.load(A + elem * 2 + 1, mask=mask, other=0.0)
    tl.store(A + elem * 2, old_r + upd_r, mask=mask)
    tl.store(A + elem * 2 + 1, old_i + upd_i, mask=mask)


def ssyr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    A: torch.Tensor,
    lda: int,
):
    _check_syr_args(uplo, n, x, incx, A, lda, torch.float32)
    if n == 0:
        return A
    alpha_value = float(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    # Aligned upper triangles favor tiles once a row spans more than two
    # 128-element slices. Keep row slices for smaller or unaligned matrices.
    use_tiles = n >= 512 or (uplo == 1 and n > 256 and lda % 128 == 0)
    with torch_device_fn.device(A.device):
        if not use_tiles:
            # Logical row-major addressing; UPLO is unchanged.
            _syr_rows_kernel[(n, triton.cdiv(n, 128))](
                A,
                x,
                alpha_value,
                0.0,
                n,
                lda,
                incx,
                uplo,
                COMPLEX=False,
                BM=1,
                BN=128,
                num_warps=4,
                num_stages=1,
            )
        else:
            # Column-major coordinates over row-major storage flip the triangle.
            tiles = triton.cdiv(n, 16)
            _ssyr_tiles_kernel[(tiles * (tiles + 1) // 2,)](
                A,
                x,
                alpha_value,
                n,
                lda,
                incx,
                1 - uplo,
                BLOCK_M=16,
                BLOCK_N=16,
                num_warps=4,
                num_stages=1,
            )
    return A


def csyr(
    uplo: int,
    n: int,
    alpha: ScalarType,
    x: torch.Tensor,
    incx: int,
    A: torch.Tensor,
    lda: int,
):
    _check_syr_args(uplo, n, x, incx, A, lda, torch.complex64)
    if n == 0:
        return A
    alpha_value = complex(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    a_real = torch.view_as_real(A)
    x_real = torch.view_as_real(x)
    with torch_device_fn.device(A.device):
        # Row slices benefit small matrices and moderately sized unaligned pitches.
        if n < 512 or (n < 1536 and lda % 32 != 0):
            _syr_rows_kernel[(n, triton.cdiv(n, 128))](
                a_real,
                x_real,
                alpha_value.real,
                alpha_value.imag,
                n,
                lda,
                incx,
                uplo,
                COMPLEX=True,
                BM=1,
                BN=128,
                num_warps=4,
                num_stages=1,
            )
        else:
            tiles = triton.cdiv(n, 16)
            _csyr_tiles_kernel[(tiles, tiles)](
                a_real,
                x_real,
                alpha_value.real,
                alpha_value.imag,
                n,
                lda,
                incx,
                1 - uplo,
                BLOCK_M=16,
                BLOCK_N=16,
                num_warps=4,
                num_stages=1,
            )
    return A
