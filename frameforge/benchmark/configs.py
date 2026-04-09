"""Benchmark configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run.

    Attributes:
        backend: Backend name (e.g. "pyav", "decord", "torchcodec", "opencv").
        video_path: Path to the video file to benchmark.
        num_trials: Number of decoding trials to average over.
        clip_length_frames: Number of frames per clip to decode.
        seek_mode: One of "sequential" (read from start) or "random" (random seeks).
        warmup_trials: Number of warmup trials (not counted in results).
        device: Device for decoding ("cpu" or "cuda:0").
    """

    backend: str
    video_path: str | Path
    num_trials: int = 100
    clip_length_frames: int = 16
    seek_mode: str = "sequential"  # "sequential" | "random"
    warmup_trials: int = 5
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.video_path = Path(self.video_path)
        if self.seek_mode not in ("sequential", "random"):
            raise ValueError(f"seek_mode must be 'sequential' or 'random', got '{self.seek_mode}'")


# Pre-built configs for common scenarios
DEFAULT_CONFIGS: dict[str, list[str]] = {
    "quick": ["pyav"],
    "full": ["pyav", "torchcodec", "decord", "opencv"],
}
