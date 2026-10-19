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

import ctypes
import ctypes.util
from typing import Generator

import pytest
import torch

import flag_blas
from benchmark.attri_util import L1_STRIDE_SHAPES, L1_VECTOR_SHAPES
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from flag_blas.utils import shape_utils

IS_HYGON = flag_blas.vendor_name == "hygon"
if IS_HYGON:
    from tests.hipblas_reference import (
        HipComplex,
        HipDoubleComplex,
        check_hipblas_status,
        get_hipblas_context,
        get_hipblas_library,
    )
else:
    import cupy as cp


def load_cublas():
    lib_names = ["libcublas.so", "libcublas.so.12", "libcublas.so.11"]
    found_path = ctypes.util.find_library("cublas")
    if found_path:
        lib_names.insert(0, found_path)

    for name in lib_names:
        try:
            return ctypes.cdll.LoadLibrary(name)
        except OSError:
            continue
    raise RuntimeError("Cannot find libcublas.so in the system")


_cublas = None if IS_HYGON else load_cublas()


class cuComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]


class cuDoubleComplex(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]


def _rot_host_scalars(dtype, c, s):
    c = c.item() if isinstance(c, torch.Tensor) else c
    s = s.item() if isinstance(s, torch.Tensor) else s
    if dtype == torch.float32:
        return ctypes.c_float(c), ctypes.c_float(s)
    if dtype == torch.float64:
        return ctypes.c_double(c), ctypes.c_double(s)
    if dtype == torch.complex64:
        complex_type = HipComplex if IS_HYGON else cuComplex
        return ctypes.c_float(c), complex_type(s.real, s.imag)
    complex_type = HipDoubleComplex if IS_HYGON else cuDoubleComplex
    return ctypes.c_double(c), complex_type(s.real, s.imag)


def _resolve_cublas_rot(dtype):
    symbol = {
        torch.float32: "cublasSrot_v2",
        torch.float64: "cublasDrot_v2",
        torch.complex64: "cublasCrot_v2",
        torch.complex128: "cublasZrot_v2",
    }[dtype]
    function = getattr(_cublas, symbol)
    function.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    function.restype = ctypes.c_int
    return function


def blas_srot_wrapper(
    x,
    y,
    c,
    s,
    incx=1,
    incy=1,
    n=None,
    handle=None,
    c_func=None,
    c_host=None,
    s_host=None,
):
    flag_blas.srot(n, x, incx, y, incy, c, s)
    return x, y


def blas_drot_wrapper(
    x,
    y,
    c,
    s,
    incx=1,
    incy=1,
    n=None,
    handle=None,
    c_func=None,
    c_host=None,
    s_host=None,
):
    flag_blas.drot(n, x, incx, y, incy, c, s)
    return x, y


def blas_crot_wrapper(
    x,
    y,
    c,
    s,
    incx=1,
    incy=1,
    n=None,
    handle=None,
    c_func=None,
    c_host=None,
    s_host=None,
):
    flag_blas.crot(n, x, incx, y, incy, c, s)
    return x, y


def blas_zrot_wrapper(
    x,
    y,
    c,
    s,
    incx=1,
    incy=1,
    n=None,
    handle=None,
    c_func=None,
    c_host=None,
    s_host=None,
):
    flag_blas.zrot(n, x, incx, y, incy, c, s)
    return x, y


if IS_HYGON:

    def _resolve_hipblas_rot(dtype):
        symbol = {
            torch.float32: "hipblasSrot",
            torch.float64: "hipblasDrot",
            torch.complex64: "hipblasCrot_v2",
            torch.complex128: "hipblasZrot_v2",
        }[dtype]
        function = getattr(get_hipblas_library(), symbol)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        function.restype = ctypes.c_int
        return function

    def hipblas_rot_baseline(
        x,
        y,
        c,
        s,
        incx=1,
        incy=1,
        n=None,
        handle=None,
        c_func=None,
        c_host=None,
        s_host=None,
    ):
        if n <= 0:
            return x, y
        check_hipblas_status(
            c_func(
                handle,
                n,
                x.data_ptr(),
                incx,
                y.data_ptr(),
                incy,
                ctypes.byref(c_host),
                ctypes.byref(s_host),
            ),
            "hipBLAS ROT",
        )
        return x, y

    def _hygon_rot_wrapper(op):
        def wrapper(
            x,
            y,
            c,
            s,
            incx=1,
            incy=1,
            n=None,
            handle=None,
            c_func=None,
            c_host=None,
            s_host=None,
        ):
            return op(x, y, c, s, incx=incx, incy=incy, n=n)

        return wrapper

    cublas_rot_reference = hipblas_rot_baseline
    blas_srot_wrapper = _hygon_rot_wrapper(blas_srot_wrapper)
    blas_drot_wrapper = _hygon_rot_wrapper(blas_drot_wrapper)
    blas_crot_wrapper = _hygon_rot_wrapper(blas_crot_wrapper)
    blas_zrot_wrapper = _hygon_rot_wrapper(blas_zrot_wrapper)

else:

    def cublas_rot_reference(
        x,
        y,
        c,
        s,
        incx=1,
        incy=1,
        n=None,
        handle=None,
        c_func=None,
        c_host=None,
        s_host=None,
    ):
        if n <= 0:
            return x, y
        status = c_func(
            handle,
            n,
            x.data_ptr(),
            incx,
            y.data_ptr(),
            incy,
            ctypes.byref(c_host),
            ctypes.byref(s_host),
        )
        if status != 0:
            raise RuntimeError(f"cuBLAS rot execution failed, error code: {status}")
        return x, y


class RotBenchmark(Benchmark):
    def __init__(self, *args, c=0.8, s=0.6, **kwargs):
        super().__init__(*args, **kwargs)
        self.c = c
        self.s = s

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        return L1_VECTOR_SHAPES

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            c_func = _resolve_hipblas_rot(cur_dtype)
        else:
            handle = cp.cuda.device.get_cublas_handle()
            c_func = _resolve_cublas_rot(cur_dtype)
            status = _cublas.cublasSetPointerMode_v2(ctypes.c_void_p(handle), 0)
            if status != 0:
                raise RuntimeError(f"cuBLAS pointer mode setup failed: {status}")
        c_host, s_host = _rot_host_scalars(cur_dtype, self.c, self.s)
        for shape in self.shapes:
            inp1 = torch.randn(shape, dtype=cur_dtype, device=self.device)
            inp2 = torch.randn(shape, dtype=cur_dtype, device=self.device)
            n = inp1.numel()
            if n > 0:
                kwargs = {"c": self.c, "s": self.s, "n": n}
                if IS_HYGON:
                    _, handle = get_hipblas_context(inp1)
                kwargs.update(
                    handle=handle, c_func=c_func, c_host=c_host, s_host=s_host
                )
                yield inp1, inp2, kwargs

    def get_gbps(self, args, latency):
        inp1, inp2 = args[0], args[1]
        io_amount = 2 * shape_utils.size_in_bytes(inp1) + 2 * shape_utils.size_in_bytes(
            inp2
        )
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        x = args[0]
        incx = kwargs.get("incx", getattr(self, "incx", 1))
        n = kwargs.get("n", x.numel() // incx)
        # CROT/ZROT use real c and complex s: twenty real FLOPs per pair.
        return (20 if x.dtype.is_complex else 6) * n


class RotStrideBenchmark(Benchmark):
    def __init__(self, *args, incx=1, incy=1, c=0.8, s=0.6, **kwargs):
        super().__init__(*args, **kwargs)
        self.incx = incx
        self.incy = incy
        self.c = c
        self.s = s

    def set_more_metrics(self):
        return ["tflops", "gbps"]

    def set_more_shapes(self):
        return L1_STRIDE_SHAPES

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            c_func = _resolve_hipblas_rot(cur_dtype)
        else:
            handle = cp.cuda.device.get_cublas_handle()
            c_func = _resolve_cublas_rot(cur_dtype)
            status = _cublas.cublasSetPointerMode_v2(ctypes.c_void_p(handle), 0)
            if status != 0:
                raise RuntimeError(f"cuBLAS pointer mode setup failed: {status}")
        c_host, s_host = _rot_host_scalars(cur_dtype, self.c, self.s)
        for shape in self.shapes:
            n = shape[0]
            inp1 = torch.randn(n * self.incx, dtype=cur_dtype, device=self.device)
            inp2 = torch.randn(n * self.incy, dtype=cur_dtype, device=self.device)
            if n > 0:
                kwargs = {
                    "c": self.c,
                    "s": self.s,
                    "incx": self.incx,
                    "incy": self.incy,
                    "n": n,
                }
                if IS_HYGON:
                    _, handle = get_hipblas_context(inp1)
                kwargs.update(
                    handle=handle, c_func=c_func, c_host=c_host, s_host=s_host
                )
                yield inp1, inp2, kwargs

    def get_gbps(self, args, latency):
        inp1 = args[0]
        n = (inp1.numel() + self.incx - 1) // self.incx
        element_size = inp1.element_size()
        io_amount = 4 * n * element_size
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        x = args[0]
        incx = kwargs.get("incx", getattr(self, "incx", 1))
        n = kwargs.get("n", x.numel() // incx)
        # CROT/ZROT use real c and complex s: twenty real FLOPs per pair.
        return (20 if x.dtype.is_complex else 6) * n


if IS_HYGON:
    RotBenchmark.correctness_reference = "hipBLAS"
    RotStrideBenchmark.correctness_reference = "hipBLAS"


@pytest.mark.srot
def test_perf_srot():
    bench = RotBenchmark(
        op_name="srot",
        torch_op=cublas_rot_reference,
        blas_op=blas_srot_wrapper,
        dtypes=[torch.float32],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.drot
def test_perf_drot():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = RotBenchmark(
        op_name="drot",
        torch_op=cublas_rot_reference,
        blas_op=blas_drot_wrapper,
        dtypes=[torch.float64],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.crot
def test_perf_crot():
    bench = RotBenchmark(
        op_name="crot",
        torch_op=cublas_rot_reference,
        blas_op=blas_crot_wrapper,
        dtypes=[torch.complex64],
        c=0.8,
        s=0.36 + 0.48j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zrot
def test_perf_zrot():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = RotBenchmark(
        op_name="zrot",
        torch_op=cublas_rot_reference,
        blas_op=blas_zrot_wrapper,
        dtypes=[torch.complex128],
        c=0.8,
        s=0.36 + 0.48j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.srot
@pytest.mark.parametrize("incx,incy", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_perf_srot_stride(incx, incy):
    bench = RotStrideBenchmark(
        op_name=f"srot_stride_incx{incx}_incy{incy}",
        torch_op=cublas_rot_reference,
        blas_op=blas_srot_wrapper,
        dtypes=[torch.float32],
        incx=incx,
        incy=incy,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.drot
@pytest.mark.parametrize("incx,incy", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_perf_drot_stride(incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = RotStrideBenchmark(
        op_name=f"drot_stride_incx{incx}_incy{incy}",
        torch_op=cublas_rot_reference,
        blas_op=blas_drot_wrapper,
        dtypes=[torch.float64],
        incx=incx,
        incy=incy,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.crot
@pytest.mark.parametrize("incx,incy", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_perf_crot_stride(incx, incy):
    bench = RotStrideBenchmark(
        op_name=f"crot_stride_incx{incx}_incy{incy}",
        torch_op=cublas_rot_reference,
        blas_op=blas_crot_wrapper,
        dtypes=[torch.complex64],
        incx=incx,
        incy=incy,
        c=0.8,
        s=0.36 + 0.48j,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zrot
@pytest.mark.parametrize("incx,incy", [(2, 2), (2, 3), (3, 2), (3, 3)])
def test_perf_zrot_stride(incx, incy):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = RotStrideBenchmark(
        op_name=f"zrot_stride_incx{incx}_incy{incy}",
        torch_op=cublas_rot_reference,
        blas_op=blas_zrot_wrapper,
        dtypes=[torch.complex128],
        incx=incx,
        incy=incy,
        c=0.8,
        s=0.36 + 0.48j,
    )
    run_correctness_then_benchmark(bench)
