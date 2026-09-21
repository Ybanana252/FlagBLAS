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
import triton.language as tl

from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

_common = importlib.import_module("flag_blas.ops.level2.trsv")


@libentry()
@triton.jit
def _trsv_small_kernel(
    a_ptr,
    x_ptr,
    lda,
    incx,
    N: tl.constexpr,
    TRANS: tl.constexpr,
    UNIT: tl.constexpr,
    FORWARD: tl.constexpr,
    COMPLEX: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Solve small real or complex unit-diagonal systems in one program."""
    tl.static_assert(not COMPLEX or UNIT)
    rows = tl.arange(0, BLOCK_N)
    valid = rows < N
    if COMPLEX:
        rhs_r = tl.load(x_ptr + 2 * rows * incx, mask=valid, other=0.0)
        rhs_i = tl.load(x_ptr + 2 * rows * incx + 1, mask=valid, other=0.0)
    else:
        rhs_r = tl.load(x_ptr + rows * incx, mask=valid, other=0.0)
    if not UNIT:
        inverse_diag = 1.0 / tl.load(a_ptr + rows * lda + rows, mask=valid, other=1.0)

    for step in tl.static_range(0, N):
        row = step if FORWARD else N - 1 - step
        index = tl.full((1,), row, tl.int32)
        # A gather broadcasts the pivot without a masked vector reduction.
        value_r = tl.sum(tl.gather(rhs_r, index, axis=0), axis=0)
        if COMPLEX:
            value_i = tl.sum(tl.gather(rhs_i, index, axis=0), axis=0)
        if not UNIT:
            value_r *= tl.sum(tl.gather(inverse_diag, index, axis=0), axis=0)
        rhs_r = tl.where(rows == row, value_r, rhs_r)
        if COMPLEX:
            rhs_i = tl.where(rows == row, value_i, rhs_i)
        future = rows > row if FORWARD else rows < row
        if TRANS == 0:
            matrix_offset = rows * lda + row
        else:
            matrix_offset = row * lda + rows
        if COMPLEX:
            ar = tl.load(a_ptr + 2 * matrix_offset, mask=valid & future, other=0.0)
            ai = tl.load(a_ptr + 2 * matrix_offset + 1, mask=valid & future, other=0.0)
            if TRANS == 2:
                ai = -ai
            rhs_r = tl.where(future, rhs_r - (ar * value_r - ai * value_i), rhs_r)
            rhs_i = tl.where(future, rhs_i - (ar * value_i + ai * value_r), rhs_i)
        else:
            matrix = tl.load(a_ptr + matrix_offset, mask=valid & future, other=0.0)
            rhs_r = tl.where(future, rhs_r - matrix * value_r, rhs_r)
    if COMPLEX:
        tl.store(x_ptr + 2 * rows * incx, rhs_r, mask=valid)
        tl.store(x_ptr + 2 * rows * incx + 1, rhs_i, mask=valid)
    else:
        tl.store(x_ptr + rows * incx, rhs_r, mask=valid)


@libentry()
@triton.jit
def _ctrsv_small_nonunit_kernel(
    a_ptr,
    x_ptr,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compact lower, non-transposed solve with complex diagonal reciprocals."""
    rows = tl.arange(0, BLOCK_N)
    valid = rows < N
    rhs_r = tl.load(x_ptr + 2 * rows, mask=valid, other=0.0)
    rhs_i = tl.load(x_ptr + 2 * rows + 1, mask=valid, other=0.0)
    diag_offset = 2 * (rows * N + rows)
    dr = tl.load(a_ptr + diag_offset, mask=valid, other=1.0)
    di = tl.load(a_ptr + diag_offset + 1, mask=valid, other=0.0)
    denominator = dr * dr + di * di
    inverse_r = dr / denominator
    inverse_i = -di / denominator
    for row in tl.static_range(0, N):
        index = tl.full((1,), row, tl.int32)
        vr = tl.sum(tl.gather(rhs_r, index, axis=0), axis=0)
        vi = tl.sum(tl.gather(rhs_i, index, axis=0), axis=0)
        rr = tl.sum(tl.gather(inverse_r, index, axis=0), axis=0)
        ri = tl.sum(tl.gather(inverse_i, index, axis=0), axis=0)
        value_r = vr * rr - vi * ri
        value_i = vi * rr + vr * ri
        rhs_r = tl.where(rows == row, value_r, rhs_r)
        rhs_i = tl.where(rows == row, value_i, rhs_i)
        future = (rows > row) & valid
        offset = 2 * (rows * N + row)
        ar = tl.load(a_ptr + offset, mask=future, other=0.0)
        ai = tl.load(a_ptr + offset + 1, mask=future, other=0.0)
        rhs_r = tl.where(future, rhs_r - (ar * value_r - ai * value_i), rhs_r)
        rhs_i = tl.where(future, rhs_i - (ar * value_i + ai * value_r), rhs_i)
    tl.store(x_ptr + 2 * rows, rhs_r, mask=valid)
    tl.store(x_ptr + 2 * rows + 1, rhs_i, mask=valid)


def strsv(uplo, trans, diag, n, A, lda, x, incx):
    if n > 64:
        return _common.strsv(uplo, trans, diag, n, A, lda, x, incx)
    assert A.dtype == torch.float32 == x.dtype
    _common._check_trsv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=False)
    if n == 0:
        return
    forward = (uplo == 0) if trans == 0 else (uplo == 1)
    with torch_device_fn.device(A.device):
        _trsv_small_kernel[(1,)](
            A,
            x,
            lda,
            incx,
            N=n,
            TRANS=trans,
            UNIT=diag == 1,
            FORWARD=forward,
            COMPLEX=False,
            BLOCK_N=triton.next_power_of_2(n),
            num_warps=1,
            num_stages=1,
        )


def ctrsv(uplo, trans, diag, n, A, lda, x, incx):
    # A single predecessor panel per chunk avoids the large complex update
    # tiles and the scalar-expanded n=256/512 path in the common dispatch.
    specialized = (
        incx == 1
        and lda == n
        and uplo == 0
        and trans == 0
        and 64 < n <= (4096 if diag == 1 else 256)
    )
    small_unit = 0 < n <= 64 and diag == 1
    small_nonunit = (
        0 < n <= 64
        and diag == 0
        and uplo == 0
        and trans == 0
        and incx == 1
        and lda == n
    )
    if not (specialized or small_unit or small_nonunit):
        return _common.ctrsv(uplo, trans, diag, n, A, lda, x, incx)
    assert A.dtype == torch.complex64 == x.dtype
    _common._check_trsv(A, x, uplo, trans, diag, n, lda, incx, complex_ok=True)
    if small_nonunit:
        with torch_device_fn.device(A.device):
            _ctrsv_small_nonunit_kernel[(1,)](
                torch.view_as_real(A),
                torch.view_as_real(x),
                N=n,
                BLOCK_N=triton.next_power_of_2(n),
                num_warps=1,
                num_stages=1,
            )
        return
    if small_unit:
        forward = (uplo == 0) if trans == 0 else (uplo == 1)
        with torch_device_fn.device(A.device):
            _trsv_small_kernel[(1,)](
                torch.view_as_real(A),
                torch.view_as_real(x),
                lda,
                incx,
                N=n,
                TRANS=trans,
                UNIT=True,
                FORWARD=forward,
                COMPLEX=True,
                BLOCK_N=triton.next_power_of_2(n),
                num_warps=1,
                num_stages=1,
            )
        return
    internal_uplo, trans_flag, conj = _common._row_major_dispatch(uplo, trans)
    unit = int(diag == 1)
    mode_key = _common._mode_key(internal_uplo, trans_flag, unit) | (conj << 8)
    with torch_device_fn.device(A.device):
        _common.ctrsv_fwd_fused_kernel[(triton.cdiv(n, 32),)](
            torch.view_as_real(A),
            torch.view_as_real(x),
            _common._trsv_flags(A.device),
            n,
            lda,
            mode_key,
            TRANS=trans_flag,
            UNIT=unit,
            CONJ=conj,
            LOWER_EFF=1,
            BLOCK_N=32,
            CHUNK=1,
            ROWLOAD=1,
            INV_DOT=0,
            VEC64=int(n >= 1024),
            num_warps=4,
        )
