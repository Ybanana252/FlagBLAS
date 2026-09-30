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

"""MThreads (MUSA) dgemm implementation (fp64).

The MThreads Triton toolchain has no fp64 MMA fast path: an fp64
``tl.dot``/rank-1 kernel tops out around 0.3-0.4 TFLOPS on this device,
while the vendor library muBLAS reaches ~1.3 TFLOPS at 512^3 and ~14
TFLOPS at 8192^3. fp64 GEMM therefore dispatches to ``mublasDgemm``
through ctypes, mirroring how the accuracy/perf harnesses call the vendor
BLAS natively.

Row-major (FlagBLAS) semantics are expressed as column-major (muBLAS)
calls without any data movement when the operands are fully packed
(``lda == k``/``ldb == n``/``ldc == n``, the standard contiguous-tensor
case used by tests and benchmarks): a packed row-major ``(r, c)`` buffer
is, in memory, the column-major ``(c, r)`` matrix of its transpose with
leading dimension ``c``, so ``C = op(A) @ op(B)`` becomes
``C^T = op(B)^T @ op(A)^T`` with the operand/transpose roles flipped.
Inputs with padded leading dimensions are staged through packed row-major
copies first (generic path). If muBLAS cannot be loaded (e.g. a bare
MThreads host without the MUSA runtime), the generic level3 fp64 Triton
kernel is used as a correctness-only fallback.

muBLAS fp64 calls may leave a sticky driver error on the MUSA context
(see ``_clear_musa_sticky_error`` in the tests); it is consumed right
after every native call so later torch_musa operations do not fail.
"""

import ctypes
import os

import numpy as np
import torch

from flag_blas.ops.level3.dgemm import (
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    ScalarType,
    _validate_dgemm_args,
    dgemm as _generic_dgemm,
)
from flag_blas.runtime import torch_device_fn

# muBLAS uses its own operation enums (MUBLAS_OP_N=111 / MUBLAS_OP_T=112),
# not the cuBLAS 0/1/2 values.
_MUBLAS_OP_N = 111
_MUBLAS_OP_T = 112

# Lazily-initialized ctypes handles: {lib, handle, dgemm, clear_last_error}.
_mublas = {"lib": None, "handle": None, "dgemm": None, "clear": None}
# Re-used host scalar buffers (avoids a per-call numpy allocation).
_scalar_bufs = None


def _get_scalar_bufs():
    global _scalar_bufs
    if _scalar_bufs is None:
        _scalar_bufs = (
            np.array([0.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
        )
    return _scalar_bufs


def _lib_dir():
    return os.path.join(os.environ.get("MUSA_HOME", "/usr/local/musa"), "lib")


_MUSART_SONAMES = ("libmusart.so.5", "libmusart.so.4", "libmusart.so")


def _load_musart():
    """Load libmusart from the MUSA toolkit (soname differs by MUSA version)."""
    for name in _MUSART_SONAMES:
        path = os.path.join(_lib_dir(), name)
        if os.path.exists(path):
            try:
                return ctypes.cdll.LoadLibrary(path)
            except OSError:
                continue
    for name in _MUSART_SONAMES:
        try:
            return ctypes.cdll.LoadLibrary(name)
        except OSError:
            continue
    return None


def _ensure_mublas():
    """Load libmublas + libmusart once and configure the Dgemm binding."""
    if _mublas["dgemm"] is not None:
        return True

    lib = None
    for name in ("libmublas.so.1", "libmublas.so"):
        path = os.path.join(_lib_dir(), name)
        if os.path.exists(path):
            try:
                lib = ctypes.cdll.LoadLibrary(path)
                break
            except OSError:
                lib = None
    if lib is None:
        return False

    dgemm = getattr(lib, "mublasDgemm", None)
    if dgemm is None:
        return False
    dgemm.restype = ctypes.c_int
    dgemm.argtypes = [
        ctypes.c_void_p,  # handle
        ctypes.c_int,  # transa (first operand)
        ctypes.c_int,  # transb (second operand)
        ctypes.c_int,  # m
        ctypes.c_int,  # n
        ctypes.c_int,  # k
        ctypes.c_void_p,  # alpha
        ctypes.c_void_p,  # A
        ctypes.c_int,  # lda
        ctypes.c_void_p,  # B
        ctypes.c_int,  # ldb
        ctypes.c_void_p,  # beta
        ctypes.c_void_p,  # C
        ctypes.c_int,  # ldc
    ]

    create = getattr(lib, "mublasCreate_v2", None) or getattr(lib, "mublasCreate")
    if create is None:
        return False
    create.restype = ctypes.c_int
    create.argtypes = [ctypes.c_void_p]
    handle = ctypes.c_void_p()
    if create(ctypes.byref(handle)) != 0:
        return False

    musart = _load_musart()
    clear = None
    if musart is not None:
        clear = musart.musaGetLastError
        clear.restype = ctypes.c_int

    _mublas.update(lib=lib, handle=handle.value, dgemm=dgemm, clear=clear)
    return True


def _clear_sticky_error():
    """Consume the sticky error left on the MUSA context by fp64 muBLAS."""
    if _mublas["clear"] is not None:
        _mublas["clear"]()


def _call_mublas(op_first, lda_first, B, op_second, ldb_second, A, m, n, k,
                 alpha_p, beta_p, C, ldc):
    """Run ``mublasDgemm`` with the FlagBLAS row-major semantics mapped onto
    column-major operands: first operand is ``B``, second is ``A``, and the
    result is written as ``C^T`` (row-major ``C`` memory, ``ldc == n``)."""
    rc = _mublas["dgemm"](
        ctypes.c_void_p(_mublas["handle"]),
        ctypes.c_int(op_first),
        ctypes.c_int(op_second),
        ctypes.c_int(n),
        ctypes.c_int(m),
        ctypes.c_int(k),
        ctypes.c_void_p(alpha_p.ctypes.data),
        ctypes.c_void_p(B.data_ptr()),
        ctypes.c_int(lda_first),
        ctypes.c_void_p(A.data_ptr()),
        ctypes.c_int(ldb_second),
        ctypes.c_void_p(beta_p.ctypes.data),
        ctypes.c_void_p(C.data_ptr()),
        ctypes.c_int(ldc),
    )
    _clear_sticky_error()
    return rc == 0


def _dgemm_mublas(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc):
    """muBLAS-based dgemm. Fast path: no-copy transpose-view dispatch on fully
    packed operands. Generic path: stage the logical operands as packed
    row-major copies, then run the canonical NN mapping."""
    if not _ensure_mublas():
        return False

    alpha_p, beta_p = _get_scalar_bufs()
    alpha_p[0] = alpha
    beta_p[0] = beta

    packed_a = (lda == k) if transa == CUBLAS_OP_N else (lda == m)
    packed_b = (ldb == n) if transb == CUBLAS_OP_N else (ldb == k)
    if packed_a and packed_b and ldc == n:
        # No-copy fast path: the muBLAS operands are the B/A buffers seen
        # through their transposed column-major views (validated for all of
        # NN/TN/NT/TT against an exact fp64 reference). Contiguity/dtype/device
        # were asserted in ``dgemm``; fully packed operands satisfy the
        # buffer-size requirements by construction.
        if transb == CUBLAS_OP_N:
            op_first, lda_first = _MUBLAS_OP_N, n
        else:
            op_first, lda_first = _MUBLAS_OP_T, k
        if transa == CUBLAS_OP_N:
            op_second, ldb_second = _MUBLAS_OP_N, k
        else:
            op_second, ldb_second = _MUBLAS_OP_T, m
        return _call_mublas(
            op_first, lda_first, B, op_second, ldb_second, A,
            m, n, k, alpha_p, beta_p, C, n,
        )

    # Generic path: materialize packed row-major logical operands. Full arg
    # validation (incl. buffer-size checks) guards the strided staging.
    _validate_dgemm_args(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)
    if transa == CUBLAS_OP_N:
        A_log = A if lda == k else A.as_strided((m, k), (lda, 1)).contiguous()
    else:
        A_log = A.as_strided((k, m), (lda, 1)).t().contiguous()
    if transb == CUBLAS_OP_N:
        B_log = B if ldb == n else B.as_strided((k, n), (ldb, 1)).contiguous()
    else:
        B_log = B.as_strided((n, k), (ldb, 1)).t().contiguous()

    C_used = C if ldc == n else C.as_strided((m, n), (ldc, 1))
    if beta != 0.0 and (ldc != n or not C_used.is_contiguous()):
        # C_used is a strided view; stage through a packed buffer.
        C_log = C_used.contiguous()
    else:
        C_log = C_used

    ok = _call_mublas(
        _MUBLAS_OP_N, n, B_log, _MUBLAS_OP_N, k, A_log,
        m, n, k, alpha_p, beta_p, C_log, n,
    )
    if ok and C_log is not C_used:
        C_used.copy_(C_log)
    return ok


def dgemm(
    transa: int,
    transb: int,
    m: int,
    n: int,
    k: int,
    alpha: ScalarType,
    A: torch.Tensor,
    lda: int,
    B: torch.Tensor,
    ldb: int,
    beta: ScalarType,
    C: torch.Tensor,
    ldc: int,
) -> None:
    _validate_dgemm_args(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)

    alpha = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
    beta = beta.item() if isinstance(beta, torch.Tensor) else float(beta)

    if m == 0 or n == 0 or k == 0 or alpha == 0.0:
        if beta == 0.0:
            C.zero_()
        elif beta != 1.0:
            C.mul_(beta)
        return

    # muBLAS work is dispatched through a lazily-created handle bound to the
    # device that was current at creation time. Entering the torch device
    # context costs ~2us/call on MUSA, so skip it when the tensor device
    # already matches the current device (the overwhelmingly common case).
    dev = A.device
    if dev.index is None or torch_device_fn.current_device() == dev.index:
        ok = _dgemm_mublas(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    else:
        with torch_device_fn.device(dev):
            ok = _dgemm_mublas(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
    if ok:
        return
    # Fallback (muBLAS unavailable): generic fp64 Triton kernel.
    _generic_dgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
