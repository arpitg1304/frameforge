"""Benchmark runner: measures throughput, latency, and memory per backend."""

from __future__ import annotations

import json
import os
import platform
import random
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger

from frameforge.benchmark.configs import BenchmarkConfig
from frameforge.reader import VideoReader


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    backend: str
    video_path: str
    num_trials: int
    clip_length_frames: int
    seek_mode: str
    decode_throughput_fps: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    memory_peak_mb: float
    gpu_utilization_pct: float | None = None
    system_info: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, data: str | dict) -> BenchmarkResult:
        if isinstance(data, str):
            data = json.loads(data)
        return cls(**data)


def _get_system_info() -> dict:
    """Collect system information for the benchmark report."""
    info: dict = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return info


def _get_gpu_utilization() -> float | None:
    """Get current GPU utilization via pynvml, if available."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        pynvml.nvmlShutdown()
        return float(util.gpu)
    except Exception:
        return None


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Execute a benchmark for a single backend configuration.

    Decodes clips of *clip_length_frames* for *num_trials* iterations,
    measuring per-trial latency, overall throughput, and peak memory.
    """
    logger.info(
        "Benchmarking backend={} video={} trials={} clip_len={} seek={}",
        config.backend,
        config.video_path,
        config.num_trials,
        config.clip_length_frames,
        config.seek_mode,
    )

    reader = VideoReader(
        config.video_path,
        backend=config.backend,
        device=config.device,
    )

    total_frames_in_video = len(reader)
    clip_len = min(config.clip_length_frames, total_frames_in_video)
    max_start = total_frames_in_video - clip_len

    # Build start indices
    if config.seek_mode == "sequential":
        starts = [
            (i * clip_len) % (max_start + 1)
            for i in range(config.warmup_trials + config.num_trials)
        ]
    else:
        starts = [
            random.randint(0, max_start)
            for _ in range(config.warmup_trials + config.num_trials)
        ]

    # Warmup
    for i in range(config.warmup_trials):
        _ = reader[starts[i] : starts[i] + clip_len]

    # Timed trials
    tracemalloc.start()
    latencies: list[float] = []

    for i in range(config.warmup_trials, config.warmup_trials + config.num_trials):
        t0 = time.perf_counter()
        _ = reader[starts[i] : starts[i] + clip_len]
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    latencies_arr = np.array(latencies)
    total_frames_decoded = config.num_trials * clip_len
    total_time_sec = latencies_arr.sum() / 1000

    gpu_util = _get_gpu_utilization() if config.device != "cpu" else None

    reader.close()

    return BenchmarkResult(
        backend=config.backend,
        video_path=str(config.video_path),
        num_trials=config.num_trials,
        clip_length_frames=clip_len,
        seek_mode=config.seek_mode,
        decode_throughput_fps=total_frames_decoded / total_time_sec if total_time_sec > 0 else 0,
        latency_p50_ms=float(np.percentile(latencies_arr, 50)),
        latency_p95_ms=float(np.percentile(latencies_arr, 95)),
        latency_p99_ms=float(np.percentile(latencies_arr, 99)),
        memory_peak_mb=peak_memory / (1024 * 1024),
        gpu_utilization_pct=gpu_util,
        system_info=_get_system_info(),
    )
