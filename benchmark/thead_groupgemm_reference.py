import json
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OP_NAME = "group_tf32gemm"
DEFAULT_BASELINE = "benchmark/baselines/h100.json"
DEFAULT_HARDWARE = "benchmark/baselines/hardware.json"

OP_RESOURCES = {
    "group_tf32gemm": "tensor_tf32",
    "group_bfgemm": "tensor_bf16",
    "group_hgemm": "tensor_fp16",
}


class THeadGroupGemmReference:
    def __init__(self, op_name=OP_NAME, reference_file=None, hardware_file=None):
        self.op_name = op_name
        self.mode = os.environ.get("FLAGBLAS_GROUPGEMM_REFERENCE", "h100")
        self.bottleneck = "tensor"
        self.threshold = float(os.environ.get("FLAGBLAS_GROUPGEMM_THRESHOLD", "0.9"))
        if self.mode not in ("none", "h100", "legacy"):
            raise ValueError(
                "FLAGBLAS_GROUPGEMM_REFERENCE must be none, h100 or legacy"
            )
        if not math.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("FLAGBLAS_GROUPGEMM_THRESHOLD must be finite and positive")
        if self.mode == "none":
            return

        self.path = (
            Path(reference_file)
            if reference_file
            else (
                PROJECT_ROOT
                / os.environ.get("FLAGBLAS_GROUPGEMM_REFERENCE_FILE", DEFAULT_BASELINE)
            )
        )
        self.baseline_id = self.path.name.rsplit(".", 1)[0]
        self.cases = {}
        rows = json.loads(self.path.read_text(encoding="utf-8"))
        if not rows:
            raise ValueError("GroupGEMM baseline is empty")
        for row in rows:
            if row.get("op") != self.op_name:
                continue
            key = self.key(row)
            if key in self.cases:
                raise ValueError(f"Duplicate GroupGEMM baseline case: {key}")
            value = float(row["latency_ms"])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid GroupGEMM reference latency: {key}")
            self.cases[key] = row
        if not self.cases:
            raise ValueError(
                f"GroupGEMM baseline has no rows for op {self.op_name!r}: {self.path}"
            )

        hardware_path = (
            Path(hardware_file) if hardware_file else PROJECT_ROOT / DEFAULT_HARDWARE
        )
        hardware = json.loads(hardware_path.read_text(encoding="utf-8"))["thead"]
        resource = hardware[OP_RESOURCES[self.op_name]]
        self.factor = (
            resource["h100_theoretical_dense"] / resource["thead_measured_gemm"]
        )
        self.normalization_basis = "measured_gemm"
        self.calibration = resource
        override = os.environ.get("FLAGBLAS_GROUPGEMM_CONVERSION_FACTOR")
        self.overridden = override is not None
        if override is not None:
            self.factor = float(override)
        if not math.isfinite(self.factor) or self.factor <= 0:
            raise ValueError("GroupGEMM conversion factor must be finite and positive")

    @staticmethod
    def key(row):
        return (
            row["op"],
            row["dtype"],
            int(row["K"]),
            int(row["E"]),
            int(row["N"]),
        )

    def lookup(self, op, dtype, k, e, n):
        if self.mode == "none":
            return None
        key = (op, str(dtype), int(k), int(e), int(n))
        row = self.cases.get(key)
        if row is None:
            raise ValueError(
                f"Missing H100 GroupGEMM baseline case: {key} "
                f"(baseline={self.baseline_id})"
            )
        return row

    def describe(self):
        if self.mode == "none":
            print("[conversion] reference=none; equivalent metrics not evaluated")
            return
        reference_api = "saved H100 grouped-GEMM timings"
        provenance = self.path.with_name(
            self.path.name.replace(".json", ".provenance.json")
        )
        if provenance.is_file():
            try:
                meta = json.loads(provenance.read_text(encoding="utf-8"))
                ref = meta.get("reference", {})
                if ref.get("library") and ref.get("api"):
                    reference_api = (
                        f"{ref['library']} {ref['api']} (saved H100 timings)"
                    )
            except (ValueError, OSError):
                pass
        print(
            f"[conversion] baseline={self.baseline_id}; file={self.path}; "
            f"reference={reference_api}"
        )
        resource_name = OP_RESOURCES[self.op_name]
        resource = self.calibration
        print(
            f"[conversion] {resource_name}: "
            f"K={resource['h100_theoretical_dense']:g} / "
            f"{resource['thead_measured_gemm']:.12g} {resource['unit']} = "
            f"{self.factor:.12g}; basis={self.normalization_basis}; "
            f"override={self.overridden}"
        )
        print(
            f"[conversion] selected={self.bottleneck} (compute-bound, fixed); "
            f"PASS iff equivalent_score > {self.threshold:g}; "
            "equivalent_score = K * T_H100_saved / T_PPU_measured"
        )


class THeadGroupGemmTf32Reference(THeadGroupGemmReference):
    def __init__(self, reference_file=None, hardware_file=None):
        super().__init__(
            op_name="group_tf32gemm",
            reference_file=reference_file,
            hardware_file=hardware_file,
        )


def _run_thead_groupgemm(bench, op_name, results_tag):
    import gc

    import flag_blas
    from benchmark.attri_util import BenchmarkMetrics, BenchmarkResult
    from benchmark.conftest import Config, emit_record_logger

    if flag_blas.vendor_name != "thead":
        raise RuntimeError("PPU GroupGEMM equivalent path used on another platform")
    if Config.query:
        return bench.run()
    bench.init_user_config()
    reference = THeadGroupGemmReference(op_name=op_name)
    resource_name = OP_RESOURCES[op_name]
    reference.describe()
    for dtype in bench.to_bench_dtypes:
        metrics, records = [], []
        for inputs in bench.get_input_iter(dtype):
            args, kwargs = bench.unpack_to_args_kwargs(inputs)
            group_A, group_B, group_list = args[0], args[1], args[2]
            k, e, n = kwargs["K"], kwargs["group_size"], kwargs["N"]
            total_m = kwargs["M"]
            saved = reference.lookup(op_name, dtype, k, e, n)
            metric = BenchmarkMetrics()
            metric.shape_detail = bench.record_shapes(*args, **kwargs)
            metric.latency = float(bench.get_latency(bench.gems_op, *args, **kwargs))
            if not math.isfinite(metric.latency) or metric.latency <= 0:
                raise ValueError("PPU GroupGEMM latency must be finite and positive")
            metric.latency_base = saved["latency_ms"]
            metric.gbps_base = saved["gbps"]
            flops = 2 * total_m * n * k
            metric.tflops = flops / metric.latency / 1e9
            io_bytes = (
                group_A.numel() * group_A.element_size()
                + group_B.numel() * group_B.element_size()
                + group_list.numel() * group_list.element_size()
                + total_m * n * group_A.element_size()
            )
            metric.gbps = io_bytes / metric.latency / 1e6
            raw_ratio = metric.latency_base / metric.latency
            equivalent_score = reference.factor * raw_ratio
            metric.speedup = equivalent_score
            records.append(
                dict(
                    op=op_name,
                    dtype=str(dtype),
                    K=k,
                    E=e,
                    N=n,
                    M=total_m,
                    h100_ms=metric.latency_base,
                    ppu_ms=metric.latency,
                    raw_ratio=raw_ratio,
                    conversion_factor=reference.factor,
                    equivalent_score=equivalent_score,
                    performance_status=(
                        "PASS"
                        if equivalent_score > reference.threshold
                        else "UNDERPERFORM"
                    ),
                    resource=resource_name,
                    normalization_basis=reference.normalization_basis,
                    threshold=reference.threshold,
                    flops=flops,
                    useful_tflops=metric.tflops,
                    baseline_id=reference.baseline_id,
                )
            )
            metrics.append(metric)
            gc.collect()
        result = BenchmarkResult(
            level=Config.bench_level.value,
            op_name=bench.op_name,
            dtype=str(dtype),
            mode=Config.mode.value,
            result=metrics,
        )
        print(
            "Metrics: groupgemm-equivalent-v1; thead_l2=true; "
            "reference=h100_saved; "
            "Torch=H100 saved grouped-GEMM, Gems=PPU FlagBLAS; "
            f"Gems Speedup=equivalent score ({resource_name} normalized)"
        )
        print(result, end="")
        print(f"[{results_tag}] " + json.dumps(records, allow_nan=False))
        emit_record_logger(result.to_json())
    gc.collect()
    flag_blas.runtime.torch_device_fn.empty_cache()


def run_thead_group_tf32gemm(bench):
    return _run_thead_groupgemm(bench, "group_tf32gemm", "thead-groupgemm-tf32-results")


def run_thead_group_bfgemm(bench):
    return _run_thead_groupgemm(bench, "group_bfgemm", "thead-groupgemm-bf16-results")


def run_thead_group_hgemm(bench):
    return _run_thead_groupgemm(bench, "group_hgemm", "thead-groupgemm-fp16-results")
