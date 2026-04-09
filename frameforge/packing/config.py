"""Configuration and result dataclasses for data packing."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class PackConfig:
    """Configuration for packing episode clips into shard videos.

    Args:
        episode_paths: List of source video file paths.
        output_dir: Directory to write shards and manifest.
        clip_length: Number of frames per clip.
        clip_stride: Step between clip start frames within an episode.
            Defaults to clip_length (non-overlapping). Set smaller for overlap.
        clips_per_episode: Max clips to extract per episode. None = all possible.
        clips_per_shard: Target clips per shard file.
        max_shard_size_mb: Optional cap on shard file size (overrides clips_per_shard).
        cameras: List of camera names for multi-camera datasets.
            If provided, episode_paths should be a dict mapping camera → list of paths.
        codec: Video codec for shards. "h264", "mjpeg", or "ffv1".
        crf: Constant rate factor (quality). Only used for h264.
        resolution: Optional (width, height) to downscale at pack time.
        fps: Override FPS. None = keep source FPS.
        seed: Random seed for clip shuffling. Same seed = reproducible shards.
        backend: Backend to use for reading source episodes.
        num_workers: Parallel workers for packing.
        skip_short_episodes: Skip episodes shorter than clip_length.
        metadata: Optional user metadata dict stored in manifest.
    """

    episode_paths: list[str | Path] | dict[str, list[str | Path]]
    output_dir: str | Path
    clip_length: int = 16
    clip_stride: int | None = None  # defaults to clip_length
    clips_per_episode: int | None = None
    clips_per_shard: int = 1000
    max_shard_size_mb: float | None = None
    cameras: list[str] | None = None
    codec: str = "h264"
    crf: int = 18
    resolution: tuple[int, int] | None = None
    fps: float | None = None
    seed: int = 42
    backend: str = "pyav"
    num_workers: int = 1
    skip_short_episodes: bool = True
    metadata: dict | None = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        if self.clip_stride is None:
            self.clip_stride = self.clip_length
        if self.codec not in ("h264", "mjpeg", "ffv1"):
            raise ValueError(f"Unsupported codec: {self.codec}. Use 'h264', 'mjpeg', or 'ffv1'.")

        # Normalize episode_paths
        if isinstance(self.episode_paths, dict):
            if self.cameras is None:
                self.cameras = list(self.episode_paths.keys())
        else:
            self.episode_paths = [Path(p) for p in self.episode_paths]


@dataclass
class PackResult:
    """Result of a packing operation."""

    output_dir: Path
    num_shards: int
    total_clips: int
    total_frames: int
    total_size_bytes: int
    manifest_path: Path
    elapsed_sec: float

    def summary(self) -> str:
        size_mb = self.total_size_bytes / (1024 * 1024)
        return (
            f"Packed {self.total_clips} clips ({self.total_frames} frames) "
            f"into {self.num_shards} shards ({size_mb:.1f} MB) "
            f"in {self.elapsed_sec:.1f}s"
        )
