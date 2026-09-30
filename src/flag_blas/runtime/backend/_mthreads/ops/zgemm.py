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

"""MThreads (MUSA) zgemm implementation (fp64 complex).

The MThreads Triton toolchain can only emulate complex GEMM as four fp64
``tl.dot`` passes (see the generic level3 kernel), which tops out far below
the vendor library. fp64-complex GEMM therefore dispatches to
``mublasZgemm`` through ctypes, mirroring how the accuracy/perf harnesses
call the vendor BLAS natively (and how the fp64 dgemm / fp32-complex cgemm
backends work).

Row-major (FlagBLAS) semantics are expressed as column-major (muBLAS)
calls without any data movement when the operands are fully packed
(``lda == k``/``ldb == n``/``ldc == n`` for the untransposed shapes, the
standard contiguous-tensor case used by tests and benchmarks): a packed
row-major ``(r, c)`` buffer is, in memory, the column-major ``(c, r)``
matrix of its transpose with leading dimension ``c``, so the row-major
``C = alpha * op(A) @ op(B) + beta * C`` becomes the column-major
``C^T = op(B)^T @ op(A)^T`` with the operand/transpose roles flipped.

For real GEMMs an ``op`` of T means "transpose". For complex GEMMs the op
is one of N / T / C (C = conjugate transpose), and the role flip keeps the
muBLAS op code identical to the FlagBLAS one (verified against an exact
reference for all nine N/T/C x N/T/C combinations): transposing the whole
product only ever turns an operand's op into its plain transpose, whose
conjugate part is carried by the C code. ``conj`` itself is therefore never
needed on the host -- no data copy is required beyond the generic staging
path for inputs with padded leading dimensions.

muBLAS complex calls may leave a sticky driver error on the MUSA context
(see ``_clear_musa_sticky_error`` in the tests); it is consumed right after
every native call so later torch_musa operations do not fail.
"""

import ctypes
import os

import numpy as np
import torch

from flag_blas.ops.level3.zgemm import (
    CUBLAS_OP_C,
    CUBLAS_OP_N,
    CUBLAS_OP_T,
    ScalarType,
    _validate_zgemm_args,
    zgemm as _generic_zgemm,
)
from flag_blas.runtime import torch_device_fn

# muBLAS uses its own operation enums (MUBLAS_OP_N=111 / MUBLAS_OP_T=112 /
# MUBLAS_OP_C=113), not the cuBLAS 0/1/2 values.
_MUBLAS_OP_N = 111
_MUBLAS_OP_T = 112
_MUBLAS_OP_C = 113

# Lazily-initialized ctypes handles: {lib, handle, zgemm, clear_last_error}.
_mublas = {"lib": None, "handle": None, "zgemm": None, "clear": None}
# Re-used host scalar buffers (avoids a per-call numpy allocation). muBLAS
# complex scalars are two fp64 (real, imag) -- the cuDoubleComplex layout.
_scalar_bufs = None


def _get_scalar_bufs():
    global _scalar_bufs
    if _scalar_bufs is None:
        _scalar_bufs = (
            np.zeros(2, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
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
    """Load libmublas + libmusart once and configure the Zgemm binding."""
    if _mublas["zgemm"] is not None:
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

    zgemm = getattr(lib, "mublasZgemm", None)
    if zgemm is None:
        return False
    zgemm.restype = ctypes.c_int
    zgemm.argtypes = [
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

    _mublas.update(lib=lib, handle=handle.value, zgemm=zgemm, clear=clear)
    return True


def _clear_sticky_error():
    """Consume the sticky error left on the MUSA context by muBLAS."""
    if _mublas["clear"] is not None:
        _mublas["clear"]()


def _call_mublas(op_first, lda_first, B, op_second, ldb_second, A, m, n, k,
                 alpha_p, beta_p, C, ldc):
    """Run ``mublasZgemm`` with the FlagBLAS row-major semantics mapped onto
    column-major operands: first operand is ``B``, second is ``A``, and the
    result is written as ``C^T`` (row-major ``C`` memory, ``ldc == n``)."""
    rc = _mublas["zgemm"](
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


# muBLAS kernel choice is asymmetric in which operand carries the transpose
# flag: its {N,T} and {T,N} microkernels have size-dependent speeds, and the
# no-copy row->column-major role-flip mapping pins the flags on fixed
# operands, so for a few core shapes the flip call lands on the vendor's slow
# kernel for that size (the same call mirrored -- or the NN path -- runs much
# faster). For those keys we stage a physical transpose and re-issue the call
# in the vendor-preferred orientation. Keys are exact packed-square shapes
# measured on MTT S5000 (2026-09-08):
#   (128^3, NT): the flip mapping issues ops (T,N) on crossed slots and lands
#   on the slow kernel (213us); re-issuing the native replica (ops (N,T))
#   matches the native reference's fast kernel (~186us).
_SQUARE_OVERRIDES = {
    (128, 128, 128, CUBLAS_OP_N, CUBLAS_OP_T): "native",
}


def _run_native_square(transa, transb, m, n, k, A, B, C, alpha_p, beta_p):
    """Square-only native-layout replica reached through ``_SQUARE_OVERRIDES``.

    For a transposed-op (single-transpose) call the no-copy flip mapping lands
    the transpose flag on a fixed operand where the vendor micro-kernel is slow
    for that size. Re-staging the packed row-major operands as their physical
    transposes (the layout the native reference path feeds to muBLAS) and
    re-issuing the call in the vendor-preferred orientation selects the fast
    kernel instead. ``m == n == k`` here, so every leading dimension collapses
    to ``m`` and the column-major output write into row-major ``C`` is
    identity.

    The slot assignment is the transpose trick that makes the result land
    untransposed in ``C``: the physically-transposed copy of the *second*
    FlagBLAS operand is fed to the muBLAS first (A-)slot and the transposed
    copy of the *first* operand to the second (B-)slot, with the op codes
    still read off ``transa``/``transb``. (This reproduces the exact buffer +
    op pairing the benchmark harness' native reference uses for these shapes,
    validated numerically against the flip mapping.)
    """
    if transa == CUBLAS_OP_N:
        op_a = _MUBLAS_OP_N
    elif transa == CUBLAS_OP_T:
        op_a = _MUBLAS_OP_T
    else:
        op_a = _MUBLAS_OP_C
    if transb == CUBLAS_OP_N:
        op_b = _MUBLAS_OP_N
    elif transb == CUBLAS_OP_T:
        op_b = _MUBLAS_OP_T
    else:
        op_b = _MUBLAS_OP_C
    # Physical transposes of the packed row-major operands, fed to the muBLAS
    # slots *crossed* (B^T in the A-slot, A^T in the B-slot): that pairing is
    # what makes the square column-major result land untransposed in C.
    A_n = A.t().contiguous()
    B_n = B.t().contiguous()
    rc = _mublas["zgemm"](
        ctypes.c_void_p(_mublas["handle"]),
        ctypes.c_int(op_a),
        ctypes.c_int(op_b),
        ctypes.c_int(m),
        ctypes.c_int(n),
        ctypes.c_int(k),
        ctypes.c_void_p(alpha_p.ctypes.data),
        ctypes.c_void_p(B_n.data_ptr()),
        ctypes.c_int(m),
        ctypes.c_void_p(A_n.data_ptr()),
        ctypes.c_int(m),
        ctypes.c_void_p(beta_p.ctypes.data),
        ctypes.c_void_p(C.data_ptr()),
        ctypes.c_int(m),
    )
    _clear_sticky_error()
    return rc == 0


def _run_mublas_packed(transa, transb, m, n, k, A, B, C, alpha_p, beta_p):
    """Run the no-copy muBLAS mapping on fully-packed row-major operands.

    ``A``/``B``/``C`` are packed in their *stored* orientation: ``A`` is
    ``(m, k)`` when ``transa`` is N and ``(k, m)`` otherwise; ``B`` is
    ``(k, n)`` when ``transb`` is N and ``(n, k)`` otherwise; ``C`` is
    ``(m, n)`` with ``ldc == n``. The op codes passed to muBLAS mirror the
    FlagBLAS ones: the leading dimension the caller supplied for the stored
    matrix is exactly the "number of rows" muBLAS needs for the flipped
    column-major reading of the *transposed* matrix.
    """
    if transa == CUBLAS_OP_N:
        op_second, ldb_second = _MUBLAS_OP_N, k
    elif transa == CUBLAS_OP_T:
        op_second, ldb_second = _MUBLAS_OP_T, m
    else:
        op_second, ldb_second = _MUBLAS_OP_C, m
    if transb == CUBLAS_OP_N:
        op_first, lda_first = _MUBLAS_OP_N, n
    elif transb == CUBLAS_OP_T:
        op_first, lda_first = _MUBLAS_OP_T, k
    else:
        op_first, lda_first = _MUBLAS_OP_C, k
    return _call_mublas(
        op_first, lda_first, B, op_second, ldb_second, A,
        m, n, k, alpha_p, beta_p, C, n,
    )


def _zgemm_mublas(transa, transb, m, n, k, alpha_c, beta_c, A, lda, B, ldb,
                  C, ldc):
    """muBLAS-based zgemm. Fast path: no-copy transpose-view dispatch on fully
    packed operands. Generic path: stage the stored operands as packed
    row-major copies (in their transposed orientation when the op says so),
    then run the canonical mapping."""
    if not _ensure_mublas():
        return False

    alpha_p, beta_p = _get_scalar_bufs()
    alpha_p[0], alpha_p[1] = alpha_c.real, alpha_c.imag
    beta_p[0], beta_p[1] = beta_c.real, beta_c.imag

    packed_a = (lda == k) if transa == CUBLAS_OP_N else (lda == m)
    packed_b = (ldb == n) if transb == CUBLAS_OP_N else (ldb == k)
    if packed_a and packed_b and ldc == n:
        # No-copy fast path: the muBLAS operands are the B/A buffers seen
        # through their transposed column-major views (validated for all nine
        # N/T/C x N/T/C combinations against an exact complex reference).
        # Contiguity/dtype/device were asserted in ``zgemm``; fully packed
        # operands satisfy the buffer-size requirements by construction.
        mode = _SQUARE_OVERRIDES.get((m, n, k, transa, transb))
        if mode == "nn_a":
            # The vendor runs the transposed-mirror orientation much faster
            # for this size, so transpose A once and run the (numerically
            # identical) all-NN mapping.
            return _run_mublas_packed(
                CUBLAS_OP_N, transb, m, n, k, A.t().contiguous(), B, C,
                alpha_p, beta_p,
            )
        if mode == "nn_b":
            # Mirror of ``nn_a``: the vendor prefers the all-NN mapping when
            # B is transposed once, so stage B instead.
            return _run_mublas_packed(
                transa, CUBLAS_OP_N, m, n, k, A, B.t().contiguous(), C,
                alpha_p, beta_p,
            )
        if mode == "native":
            return _run_native_square(
                transa, transb, m, n, k, A, B, C, alpha_p, beta_p
            )
        return _run_mublas_packed(
            transa, transb, m, n, k, A, B, C, alpha_p, beta_p
        )

    # Generic path: materialize packed copies of the stored operands (each in
    # its own natural orientation) and re-use the same no-copy mapping. The
    # conjugate part of a C op is applied by muBLAS itself, never on the host.
    _validate_zgemm_args(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)
    if transa == CUBLAS_OP_N:
        A_p = A if lda == k else A.as_strided((m, k), (lda, 1)).contiguous()
    else:
        A_p = A if lda == m else A.as_strided((k, m), (lda, 1)).contiguous()
    if transb == CUBLAS_OP_N:
        B_p = B if ldb == n else B.as_strided((k, n), (ldb, 1)).contiguous()
    else:
        B_p = B if ldb == k else B.as_strided((n, k), (ldb, 1)).contiguous()

    C_used = C if ldc == n else C.as_strided((m, n), (ldc, 1))
    if beta_c != 0.0 and (ldc != n or not C_used.is_contiguous()):
        # C_used is a strided view; stage through a packed buffer.
        C_log = C_used.contiguous()
    else:
        C_log = C_used

    ok = _run_mublas_packed(
        transa, transb, m, n, k, A_p, B_p, C_log, alpha_p, beta_p
    )
    if ok and C_log is not C_used:
        C_used.copy_(C_log)
    return ok


def zgemm(
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
    _validate_zgemm_args(transa, transb, m, n, k, A, lda, B, ldb, C, ldc)

    alpha_c = complex(alpha.item() if isinstance(alpha, torch.Tensor) else alpha)
    beta_c = complex(beta.item() if isinstance(beta, torch.Tensor) else beta)

    if m == 0 or n == 0 or k == 0 or alpha_c == 0.0:
        if beta_c == 0.0:
            C.zero_()
        elif beta_c != 1.0:
            C.mul_(beta_c)
        return

    # muBLAS work is dispatched through a lazily-created handle bound to the
    # device that was current at creation time. Entering the torch device
    # context costs ~2us/call on MUSA, so skip it when the tensor device
    # already matches the current device (the overwhelmingly common case).
    dev = A.device
    if dev.index is None or torch_device_fn.current_device() == dev.index:
        ok = _zgemm_mublas(
            transa, transb, m, n, k, alpha_c, beta_c, A, lda, B, ldb, C, ldc
        )
    else:
        with torch_device_fn.device(dev):
            ok = _zgemm_mublas(
                transa, transb, m, n, k, alpha_c, beta_c, A, lda, B, ldb, C, ldc
            )
    if ok:
        return
    # Fallback (muBLAS unavailable): generic complex128 Triton kernel.
    _generic_zgemm(transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
