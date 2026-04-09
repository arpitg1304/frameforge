"""Benchmark suite for comparing video decoding backends."""

from frameforge.benchmark.configs import BenchmarkConfig
from frameforge.benchmark.runner import BenchmarkResult, run_benchmark

__all__ = ["BenchmarkConfig", "BenchmarkResult", "run_benchmark"]
