# Shared experiment utilities
from common.model_benchmark import (  # noqa: F401
    BenchmarkResult,
    benchmark_model,
    benchmark_detr_model,
    compute_gflops,
    compute_params,
    measure_fps,
    format_benchmark_report,
    benchmark_to_dict,
    log_benchmark,
)
