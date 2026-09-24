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

"""Hermitian band matrix-vector product for Ascend vector cores."""

import torch
import triton
import triton.language as tl

from flag_blas.ops.level2.hbmv import _check_common, _complex_scalars
from flag_blas.ops.level2.hbmv import chbmv as _common_chbmv
from flag_blas.runtime import torch_device_fn
from flag_blas.utils import libentry

from . import hpmv as _launcher
from .hpmv import _current_device, _launch

try:
    from triton.backends.ascend import utils as _ascend_utils
except ImportError:
    _ascend_utils = None


# Band-storage HBMV kernels and dispatch.
# Only the previously underperforming layouts are redirected. In particular,
# N=512/K=4/upper and N=1024/lower retain their existing dispatch.
SMALL_CASES = frozenset(
    [(256, k, uplo) for k in (0, 1, 4) for uplo in (0, 1)]
    + [(512, k, 0) for k in (0, 1, 4)]
    + [(512, k, 1) for k in (0, 1)]
    + [(1024, k, 1) for k in (1, 4, 16)]
)
_HBMV_LAUNCH_CACHE = {}
_CACHE = _HBMV_LAUNCH_CACHE  # Compatibility for existing guard tests.


@libentry()
@triton.jit(do_not_specialize=["N"])
def _chbmv_narrow(
    A,
    X,
    Y,
    AR: tl.float32,
    AI: tl.float32,
    BR: tl.float32,
    BI: tl.float32,
    N,
    K: tl.constexpr,
    LDA: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK: tl.constexpr,
    BUFFER: tl.constexpr,
    ALPHA_ZERO: tl.constexpr,
    BETA_ZERO: tl.constexpr,
):
    # Keep real/imaginary pairs together in vector loads and stores.
    lanes = tl.arange(0, 2 * BLOCK)
    yf = tl.program_id(0) * BLOCK * 2 + lanes
    rows = tl.program_id(0) * BLOCK + (lanes >> 1)
    odd = (lanes & 1) != 0
    acc = tl.full((2 * BLOCK,), 0.0, tl.float32)
    if not ALPHA_ZERO:
        # Load a contiguous slab, then gather band entries inside UB. A gather
        # directly from global memory is scalarized by the Ascend compiler.
        start = tl.program_id(0) * BLOCK - K
        offsets = start * (2 * LDA) + tl.arange(0, BUFFER)
        slab = tl.load(A + offsets, (offsets >= 0) & (offsets < N * LDA * 2), 0.0)
        for d in tl.static_range(-K, K + 1):
            cols = rows + d
            if UPLO == 1:
                stored_row = tl.minimum(rows, cols)
                band = d if d >= 0 else -d
                conjugate = d < 0
            else:
                stored_row = tl.maximum(rows, cols)
                band = K - (d if d >= 0 else -d)
                conjugate = d > 0
            # The mask must also use consecutive float lanes; a mask expressed
            # through (lanes >> 1) prevents vectorized Ascend memory accesses.
            xf = yf + 2 * d
            valid = (yf < N * 2) & (xf >= 0) & (xf < N * 2)
            av = tl.gather(
                slab, ((stored_row - start) * LDA + band) * 2 + (lanes & 1), 0
            )
            av = tl.where(odd & conjugate, -av, av)
            if d == 0:
                av = tl.where(odd, 0.0, av)
            # Unused boundary-band slots may contain NaNs: multiplying them
            # by a masked-out zero X would still contaminate valid outputs.
            av = tl.where(valid, av, 0.0)
            xoff = (
                (tl.program_id(0) * BLOCK + d) * 2 + lanes
                if INCX == 1
                else cols * (2 * INCX) + (lanes & 1)
            )
            xv = tl.load(X + xoff, valid, 0.0)
            xr = tl.gather(xv, lanes & ~1, 0)
            xi = tl.gather(xv, lanes | 1, 0)
            acc += (
                av * xr
                + tl.where(odd, 1.0, -1.0) * tl.gather(av, lanes ^ 1, 0) * xi
            )
        result = AR * acc + tl.where(odd, AI, -AI) * tl.gather(acc, lanes ^ 1, 0)
    else:
        result = acc
    yoff = (
        tl.program_id(0) * BLOCK * 2 + lanes
        if INCY == 1
        else rows * (2 * INCY) + (lanes & 1)
    )
    if not BETA_ZERO:
        yv = tl.load(Y + yoff, yf < N * 2, 0.0)
        result += BR * yv + tl.where(odd, BI, -BI) * tl.gather(yv, lanes ^ 1, 0)
    tl.store(Y + yoff, result, yf < N * 2)


@libentry()
@triton.jit(do_not_specialize=["N"])
def _chbmv_band(
    A,
    X,
    Y,
    AR: tl.float32,
    AI: tl.float32,
    BR: tl.float32,
    BI: tl.float32,
    N,
    K: tl.constexpr,
    LDA: tl.constexpr,
    INCX: tl.constexpr,
    INCY: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK: tl.constexpr,
    BETA_ZERO: tl.constexpr,
):
    rb = tl.program_id(0)
    lanes = tl.arange(0, BLOCK)
    floats = tl.arange(0, 2 * BLOCK)
    parity = floats & 1
    sign = tl.where(parity == 0, 1.0, -1.0)
    acc = tl.full((2 * BLOCK,), 0.0, tl.float32)
    reach: tl.constexpr = triton.cdiv(K, BLOCK)
    for cb in range(
        tl.maximum(0, rb - reach),
        tl.minimum(tl.cdiv(N, BLOCK), rb + reach + 1),
    ):
        direct = rb >= cb if UPLO == 0 else rb <= cb
        stored_r = tl.where(direct, rb, cb)
        stored_c = tl.where(direct, cb, rb)
        rows = stored_r * BLOCK + lanes
        cols = stored_c * BLOCK + (floats >> 1)
        cf = stored_c * BLOCK * 2 + floats
        if UPLO == 0:
            base = rows * (LDA - 1) + K
            valid = (cf[None, :] >= (rows[:, None] - K) * 2) & (
                cf[None, :] < (rows[:, None] + 1) * 2
            )
        else:
            base = rows * (LDA - 1)
            valid = (cf[None, :] >= rows[:, None] * 2) & (
                cf[None, :] < (rows[:, None] + K + 1) * 2
            )
        av = tl.load(
            A + base[:, None] * 2 + stored_c * BLOCK * 2 + floats[None, :],
            valid & (rows[:, None] < N) & (cf[None, :] < N * 2),
            0.0,
        )
        diagonal = rows[:, None] == cols[None, :]
        av = tl.where(diagonal & (parity[None, :] != 0), 0.0, av)
        xoff = (
            cb * BLOCK * 2 + floats
            if INCX == 1
            else (cb * BLOCK + (floats >> 1)) * (2 * INCX) + parity
        )
        xv = tl.load(X + xoff, cb * BLOCK * 2 + floats < N * 2, 0.0)
        if direct:
            dr = tl.sum(av * (xv * sign)[None, :], 1)
            di = tl.sum(av * tl.gather(xv, floats ^ 1, 0)[None, :], 1)
            acc += tl.where(
                parity == 0,
                tl.gather(dr, floats >> 1, 0),
                tl.gather(di, floats >> 1, 0),
            )
        if (not direct) or rb == cb:
            av = tl.where(diagonal, 0.0, av)
            xr = tl.gather(xv, lanes * 2, 0)
            xi = tl.gather(xv, lanes * 2 + 1, 0)
            mr = tl.sum(av * xr[:, None], 0)
            mi = tl.sum(av * xi[:, None], 0)
            acc += mr * sign + tl.gather(mi, floats ^ 1, 0)
    result = AR * acc + tl.where(parity == 0, -AI, AI) * tl.gather(acc, floats ^ 1, 0)
    yoff = (
        rb * BLOCK * 2 + floats
        if INCY == 1
        else (rb * BLOCK + (floats >> 1)) * (2 * INCY) + parity
    )
    mask = rb * BLOCK * 2 + floats < N * 2
    if not BETA_ZERO:
        yv = tl.load(Y + yoff, mask, 0.0)
        result += BR * yv + tl.where(parity == 0, -BI, BI) * tl.gather(
            yv, floats ^ 1, 0
        )
    tl.store(Y + yoff, result, mask)


@libentry()
@triton.jit
def _chbmv_small(
    A,
    X,
    Y,
    AR: tl.float32,
    AI: tl.float32,
    BR: tl.float32,
    BI: tl.float32,
    N: tl.constexpr,
    K: tl.constexpr,
    UPLO: tl.constexpr,
    BLOCK: tl.constexpr,
    BETA_ZERO: tl.constexpr,
):
    lanes = tl.arange(0, 2 * BLOCK)
    base = tl.program_id(0) * BLOCK
    yf = base * 2 + lanes
    odd = (lanes & 1) != 0
    if K == 0:
        # A real diagonal scales each complex X; no band gather is needed.
        av = tl.load(A + yf)
        xv = tl.load(X + yf)
        acc = tl.gather(av, lanes & ~1, 0) * xv
    else:
        rows = base + (lanes >> 1)
        start = tl.maximum(0, base - K)
        ABUF: tl.constexpr = triton.next_power_of_2((BLOCK + 2 * K) * (K + 1) * 2)
        XBUF: tl.constexpr = triton.next_power_of_2((BLOCK + 2 * K) * 2)
        af = start * ((K + 1) * 2) + tl.arange(0, ABUF)
        xf = start * 2 + tl.arange(0, XBUF)
        amat = tl.load(A + af, af < N * (K + 1) * 2, 0.0)
        xvec = tl.load(X + xf, xf < N * 2, 0.0)
        acc = tl.full((2 * BLOCK,), 0.0, tl.float32)
        for d in tl.static_range(-K, K + 1):
            cols = rows + d
            if UPLO == 1:
                stored = rows + (d if d < 0 else 0)
                band = d if d >= 0 else -d
                conjugate = d < 0
            else:
                stored = rows + (d if d > 0 else 0)
                band = K - (d if d >= 0 else -d)
                conjugate = d > 0
            valid = (cols >= 0) & (cols < N)
            ai = ((stored - start) * (K + 1) + band) * 2 + (lanes & 1)
            av = tl.gather(amat, tl.maximum(ai, 0), 0)
            av = tl.where(odd & conjugate, -av, av)
            if d == 0:
                av = tl.where(odd, 0.0, av)
            av = tl.where(valid, av, 0.0)
            xi = (cols - start) * 2 + (lanes & 1)
            xv = tl.gather(xvec, tl.maximum(xi, 0), 0)
            xv = tl.where(valid, xv, 0.0)
            xr = tl.gather(xv, lanes & ~1, 0)
            xi = tl.gather(xv, lanes | 1, 0)
            acc += av * xr + tl.where(odd, 1.0, -1.0) * tl.gather(
                av, lanes ^ 1, 0
            ) * xi
    result = AR * acc + tl.where(odd, AI, -AI) * tl.gather(acc, lanes ^ 1, 0)
    if not BETA_ZERO:
        yv = tl.load(Y + yf)
        result += BR * yv + tl.where(odd, BI, -BI) * tl.gather(yv, lanes ^ 1, 0)
    tl.store(Y + yf, result)


def launch_small(n, k, uplo, A, x, y, ar, ai, br, bi):
    """Return False without launching when the guarded fast path is unavailable.

    The caller has already checked shapes, strides, dtypes and current device.
    Cache compiled code and layout metadata, never tensors, invocation pointers,
    alpha/beta values or streams.
    """
    if (n, k, uplo) not in SMALL_CASES:
        return False
    if not _launcher._FAST_LAUNCH_SUPPORTED or _ascend_utils is None:
        return False
    if (
        type(A) is not torch.Tensor
        or type(x) is not torch.Tensor
        or type(y) is not torch.Tensor
    ):
        return False
    if (
        A.is_conj() or x.is_conj() or y.is_conj()
        or A.is_neg() or x.is_neg() or y.is_neg()
    ):
        return False
    knobs = triton.knobs.runtime
    enter, leave = knobs.launch_enter_hook, knobs.launch_exit_hook
    chain = _launcher._HOOK_CHAIN_TYPE
    if (enter is not None and (type(enter) is not chain or enter.calls)) or (
        leave is not None and (type(leave) is not chain or leave.calls)
    ):
        return False
    ap, xp, yp = A.data_ptr(), x.data_ptr(), y.data_ptr()
    if (ap | xp | yp) & 15:
        return False
    device = A.device.index
    beta_zero = br == 0.0 and bi == 0.0
    key = (device, n, k, uplo, beta_zero)
    entry = _HBMV_LAUNCH_CACHE.get(key)
    if entry is None:
        block = 32
        if k == 16:
            # The existing rectangular device kernel is already efficient for
            # K=16. Only specialize its host dispatch; do not replace the kernel.
            kernel = _chbmv_band
            constants = (n, k, k + 1, 1, 1, uplo, block, beta_zero)
        else:
            kernel = _chbmv_small
            constants = (n, k, uplo, block, beta_zero)
        grid = n // block
        compiled, _ = kernel[(grid,)](
            torch.view_as_real(A),
            torch.view_as_real(x),
            torch.view_as_real(y),
            ar, ai, br, bi, *constants,
            num_warps=1, num_stages=1,
        )
        if len(_HBMV_LAUNCH_CACHE) >= 256:
            _HBMV_LAUNCH_CACHE.clear()
        _HBMV_LAUNCH_CACHE[key] = (compiled, compiled.run, constants, grid)
        return True
    compiled, run, constants, grid = entry
    if (
        type(run) is not _launcher._NPU_LAUNCHER
        or compiled.run is not run
        or run.compile_only
        or run.enable_msprof_register_tensor
        or getattr(compiled.metadata, "debug_enabled", False)
        or getattr(run.metadata, "debug_enabled", False)
    ):
        return False
    # This is NPULauncher's checked 3.5 ABI, with its profiler status update.
    # A/x/y stay alive as this frame's arguments until enqueue completes. Read
    # every address and the current stream anew on every invocation.
    registered = run.launch(
        grid, 1, 1,
        _launcher._current_raw_stream(device),
        compiled.function,
        compiled.packed_metadata,
        None, None, None,
        ap, xp, yp, ar, ai, br, bi, *constants,
    )
    _ascend_utils.TRITON_PROFILER_REGISTERED = registered == 1
    return True


def chbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy):
    assert A.dtype == torch.complex64 == x.dtype == y.dtype
    _check_common(A, x, y, uplo, n, k, lda, incx, incy)
    if n == 0:
        return
    ar, ai, br, bi = _complex_scalars(alpha, beta)
    alpha_zero = ar == 0.0 and ai == 0.0
    if alpha_zero and br == 1.0 and bi == 0.0:
        return
    if k > 256 and not alpha_zero:
        return _common_chbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    if A.device.index != _current_device():
        with torch_device_fn.device(A.device):
            return chbmv(uplo, n, k, alpha, A, lda, x, incx, beta, y, incy)
    beta_zero = br == 0.0 and bi == 0.0
    if (
        n <= 1024
        and (n, k, uplo) in SMALL_CASES
        and not alpha_zero
        and lda == k + 1
        and incx == 1
        and incy == 1
        and launch_small(n, k, uplo, A, x, y, ar, ai, br, bi)
    ):
        return
    # Use the vectorized halo path for full, unit-stride strips and its tuned
    # bandwidths. Other bandwidths (notably K=3, whose unrolled variant triggers
    # ADDR_MISALIGN on Triton-Ascend 3.5) use the rectangular kernel.
    # Bound the contiguous slab in UB even when callers heavily pad LDA.
    narrow = (
        k in (0, 1, 4)
        and n % 128 == 0
        and incx == 1
        and incy == 1
        and (128 + 2 * k) * lda * 2 <= 8192
    )
    if alpha_zero or narrow:
        block = 128
        # Host-side Triton math helpers invoke their JIT wrapper. Plain integer
        # arithmetic avoids that overhead on launch-bound narrow-band cases.
        elements = 1 if alpha_zero else (block + 2 * k) * lda * 2
        buffer = 1 << (elements - 1).bit_length()
        _launch(
            _chbmv_narrow,
            ((n + block - 1) // block,),
            (A, x, y),
            (ar, ai, br, bi, n),
            (k, lda, incx, incy, uplo, block, buffer, alpha_zero, beta_zero),
        )
    else:
        block = 32
        _launch(
            _chbmv_band,
            ((n + block - 1) // block,),
            (A, x, y),
            (ar, ai, br, bi, n),
            (k, lda, incx, incy, uplo, block, beta_zero),
        )
