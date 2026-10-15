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

"""Lazy muBLAS reference bindings for the shared vendor-reference tests."""

import atexit
import ctypes
import ctypes.util
import os
from threading import RLock

import torch


class MuComplex(ctypes.Structure):
    _fields_ = [("real", ctypes.c_float), ("imag", ctypes.c_float)]


class MuDoubleComplex(ctypes.Structure):
    _fields_ = [("real", ctypes.c_double), ("imag", ctypes.c_double)]


# The SDK exports the compute-type overload with C++ linkage.
_GEMM_EX_SYMBOL = (
    "_Z12mublasGemmExP15_mublasHandle_t17mublasOperation_tS1_iiiPKvS3_"
    "14musaDataType_tiS3_S4_iS3_PvS4_i19mublasComputeType_t16mublasGemmAlgo_t"
)
_LIBRARY = None
_HANDLES = {}
_LOCK = RLock()


def check_mublas_status(status, operation):
    if status != 0:
        raise RuntimeError(f"{operation} failed with muBLAS status {status}")


class _MuBLASLibrary:
    def __init__(self, raw):
        self._raw = raw

    def __getattr__(self, name):
        # Existing tests set ctypes signatures and pass muBLAS-compatible
        # enums themselves. Return the real function, without argument guessing.
        symbol = name
        if symbol.startswith("hipblas"):
            symbol = "mublas" + symbol[len("hipblas") :]
        symbol = symbol.removesuffix("_v2")
        if symbol == "mublasGemmEx":
            symbol = _GEMM_EX_SYMBOL
        return getattr(self._raw, symbol)


def get_mublas_library():
    global _LIBRARY
    with _LOCK:
        if _LIBRARY is not None:
            return _LIBRARY
        names = [ctypes.util.find_library("mublas"), "libmublas.so"]
        musa_home = os.environ.get("MUSA_HOME") or os.environ.get("MUSA_PATH")
        if musa_home:
            names.extend(
                os.path.join(musa_home, subdir, "libmublas.so")
                for subdir in ("lib", "lib64")
            )
        errors = []
        for name in dict.fromkeys(names):
            if not name:
                continue
            try:
                raw = ctypes.CDLL(name)
                break
            except OSError as exc:
                errors.append(f"{name}: {exc}")
        else:
            raise RuntimeError("Unable to load libmublas.so: " + "; ".join(errors))

        raw.mublasCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        raw.mublasDestroy.argtypes = [ctypes.c_void_p]
        raw.mublasSetStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        raw.mublasSetPointerMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        for name in (
            "mublasCreate",
            "mublasDestroy",
            "mublasSetStream",
            "mublasSetPointerMode",
        ):
            getattr(raw, name).restype = ctypes.c_int
        _LIBRARY = _MuBLASLibrary(raw)
        return _LIBRARY


def get_mublas_context(tensor):
    if tensor.device.type != "musa":
        raise ValueError("muBLAS reference requires a MUSA tensor")
    device_index = tensor.device.index
    if device_index is None:
        device_index = torch.musa.current_device()
    library = get_mublas_library()
    with _LOCK, torch.musa.device(device_index):
        handle = _HANDLES.get(device_index)
        if handle is None:
            handle = ctypes.c_void_p()
            check_mublas_status(library.mublasCreate(ctypes.byref(handle)), "mublasCreate")
            try:
                check_mublas_status(
                    library.mublasSetPointerMode(handle, 0), "mublasSetPointerMode"
                )
            except Exception:
                library.mublasDestroy(handle)
                raise
            _HANDLES[device_index] = handle
        stream = torch.musa.current_stream(device_index).musa_stream
        check_mublas_status(
            library.mublasSetStream(handle, ctypes.c_void_p(stream)), "mublasSetStream"
        )
        return library, handle


def _destroy_handles():
    if _LIBRARY is None:
        return
    for device_index, handle in tuple(_HANDLES.items()):
        try:
            with torch.musa.device(device_index):
                _LIBRARY.mublasDestroy(handle)
        except Exception:
            pass
    _HANDLES.clear()


atexit.register(_destroy_handles)
