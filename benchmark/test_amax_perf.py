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

from typing import Generator

import pytest
import torch

import flag_blas
from benchmark.attri_util import L1_AMAX_STRIDE_SHAPES, L1_VECTOR_SHAPES
from benchmark.performance_utils import Benchmark, run_correctness_then_benchmark
from flag_blas.utils import shape_utils

IS_HYGON = flag_blas.vendor_name == "hygon"


if IS_HYGON:
    import ctypes

    from tests.accuracy_utils import get_l1_hipblas_context
    from tests.hipblas_reference import check_hipblas_status, get_hipblas_library
else:
    import cupy as cp
    from cupy_backends.cuda.libs import cublas


def blas_samax_wrapper(x, incx=1, n=None, handle=None, result=None):
    flag_blas.samax(n, x, incx, result)
    return result


def blas_damax_wrapper(x, incx=1, n=None, handle=None, result=None):
    flag_blas.damax(n, x, incx, result)
    return result


def blas_camax_wrapper(x, incx=1, n=None, handle=None, result=None):
    flag_blas.camax(n, x, incx, result)
    return result


def blas_zamax_wrapper(x, incx=1, n=None, handle=None, result=None):
    flag_blas.zamax(n, x, incx, result)
    return result


if IS_HYGON:

    def _resolve_hipblas_amax(dtype):
        symbol = {
            torch.float32: "hipblasIsamax",
            torch.float64: "hipblasIdamax",
            torch.complex64: "hipblasIcamax_v2",
            torch.complex128: "hipblasIzamax_v2",
        }[dtype]
        function = getattr(get_hipblas_library(), symbol)
        function.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        function.restype = ctypes.c_int
        return function

    def hipblas_amax_baseline(x, incx=1, n=None, handle=None, result=None, c_func=None):
        check_hipblas_status(
            c_func(handle, n, x.data_ptr(), incx, result.data_ptr()), "hipBLAS AMAX"
        )
        return result

    def _hygon_amax_wrapper(op):
        def wrapper(x, incx=1, n=None, handle=None, result=None, c_func=None):
            return op(x, incx=incx, n=n, handle=handle, result=result)

        return wrapper

    cublas_isamax = hipblas_amax_baseline
    blas_samax_wrapper = _hygon_amax_wrapper(blas_samax_wrapper)
    cublas_idamax = hipblas_amax_baseline
    blas_damax_wrapper = _hygon_amax_wrapper(blas_damax_wrapper)
    cublas_icamax = hipblas_amax_baseline
    blas_camax_wrapper = _hygon_amax_wrapper(blas_camax_wrapper)
    cublas_izamax = hipblas_amax_baseline
    blas_zamax_wrapper = _hygon_amax_wrapper(blas_zamax_wrapper)

else:

    def cublas_isamax(x, incx=1, n=None, handle=None, result=None):
        cublas.isamax(handle, n, x.data_ptr(), incx, result.data_ptr())
        return result

    def cublas_idamax(x, incx=1, n=None, handle=None, result=None):
        cublas.idamax(handle, n, x.data_ptr(), incx, result.data_ptr())
        return result

    def cublas_icamax(x, incx=1, n=None, handle=None, result=None):
        cublas.icamax(handle, n, x.data_ptr(), incx, result.data_ptr())
        return result

    def cublas_izamax(x, incx=1, n=None, handle=None, result=None):
        cublas.izamax(handle, n, x.data_ptr(), incx, result.data_ptr())
        return result


class AmaxBenchmark(Benchmark):
    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        return L1_VECTOR_SHAPES

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            c_func = _resolve_hipblas_amax(cur_dtype)
        else:
            handle = cp.cuda.device.get_cublas_handle()
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)

        for shape in self.shapes:
            inp = torch.randn(shape, dtype=cur_dtype, device=self.device)
            n = inp.numel()

            result = torch.zeros(1, dtype=torch.int32, device=self.device)

            if n > 0:
                if IS_HYGON:
                    _, handle = get_l1_hipblas_context(inp)
                kwargs = {"n": n, "handle": handle, "result": result}
                if IS_HYGON:
                    kwargs["c_func"] = c_func
                yield inp, kwargs

    def get_gbps(self, args, latency):
        inp = args[0]
        io_amount = shape_utils.size_in_bytes(inp) + 4
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        x = args[0]
        incx = kwargs.get("incx", getattr(self, "incx", 1))
        n = kwargs.get("n", x.numel() // incx)
        # BLAS complex magnitude is abs(real) + abs(imag); comparisons are not FLOPs.
        return n if x.dtype.is_complex else 0


class AmaxStrideBenchmark(Benchmark):
    def __init__(self, *args, incx=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.incx = incx

    def set_more_metrics(self):
        return ["gbps"]

    def set_more_shapes(self):
        return L1_AMAX_STRIDE_SHAPES

    def get_input_iter(self, cur_dtype) -> Generator:
        if IS_HYGON:
            c_func = _resolve_hipblas_amax(cur_dtype)
        else:
            handle = cp.cuda.device.get_cublas_handle()
            cublas.setPointerMode(handle, cublas.CUBLAS_POINTER_MODE_DEVICE)

        for shape in self.shapes:
            n = shape[0]
            inp = torch.randn(n * self.incx, dtype=cur_dtype, device=self.device)

            result = torch.zeros(1, dtype=torch.int32, device=self.device)

            if n > 0:
                if IS_HYGON:
                    _, handle = get_l1_hipblas_context(inp)
                kwargs = {
                    "incx": self.incx,
                    "n": n,
                    "handle": handle,
                    "result": result,
                }
                if IS_HYGON:
                    kwargs["c_func"] = c_func
                yield inp, kwargs

    def get_gbps(self, args, latency):
        inp = args[0]
        n = (inp.numel() + self.incx - 1) // self.incx
        element_size = inp.element_size()
        io_amount = n * element_size + 4
        return io_amount * 1e-9 / (latency * 1e-3)

    def get_tflops(self, op, *args, **kwargs):
        x = args[0]
        incx = kwargs.get("incx", getattr(self, "incx", 1))
        n = kwargs.get("n", x.numel() // incx)
        # BLAS complex magnitude is abs(real) + abs(imag); comparisons are not FLOPs.
        return n if x.dtype.is_complex else 0


if IS_HYGON:
    AmaxBenchmark.correctness_reference = "hipBLAS"
    AmaxStrideBenchmark.correctness_reference = "hipBLAS"


@pytest.mark.samax
def test_perf_samax():
    bench = AmaxBenchmark(
        op_name="samax",
        torch_op=cublas_isamax,
        blas_op=blas_samax_wrapper,
        dtypes=[torch.float32],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.damax
def test_perf_damax():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AmaxBenchmark(
        op_name="damax",
        torch_op=cublas_idamax,
        blas_op=blas_damax_wrapper,
        dtypes=[torch.float64],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.camax
def test_perf_camax():
    bench = AmaxBenchmark(
        op_name="camax",
        torch_op=cublas_icamax,
        blas_op=blas_camax_wrapper,
        dtypes=[torch.complex64],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zamax
def test_perf_zamax():
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AmaxBenchmark(
        op_name="zamax",
        torch_op=cublas_izamax,
        blas_op=blas_zamax_wrapper,
        dtypes=[torch.complex128],
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.samax
@pytest.mark.parametrize("incx", [2, 5])
def test_perf_samax_stride(incx):
    bench = AmaxStrideBenchmark(
        op_name=f"samax_stride_incx{incx}",
        torch_op=cublas_isamax,
        blas_op=blas_samax_wrapper,
        dtypes=[torch.float32],
        incx=incx,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.damax
@pytest.mark.parametrize("incx", [2, 5])
def test_perf_damax_stride(incx):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AmaxStrideBenchmark(
        op_name=f"damax_stride_incx{incx}",
        torch_op=cublas_idamax,
        blas_op=blas_damax_wrapper,
        dtypes=[torch.float64],
        incx=incx,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.camax
@pytest.mark.parametrize("incx", [2, 5])
def test_perf_camax_stride(incx):
    bench = AmaxStrideBenchmark(
        op_name=f"camax_stride_incx{incx}",
        torch_op=cublas_icamax,
        blas_op=blas_camax_wrapper,
        dtypes=[torch.complex64],
        incx=incx,
    )
    run_correctness_then_benchmark(bench)


@pytest.mark.zamax
@pytest.mark.parametrize("incx", [2, 5])
def test_perf_zamax_stride(incx):
    if not flag_blas.runtime.device.support_fp64:
        pytest.skip("Device does not support float64")
    bench = AmaxStrideBenchmark(
        op_name=f"zamax_stride_incx{incx}",
        torch_op=cublas_izamax,
        blas_op=blas_zamax_wrapper,
        dtypes=[torch.complex128],
        incx=incx,
    )
    run_correctness_then_benchmark(bench)
