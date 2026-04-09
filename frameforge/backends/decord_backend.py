"""Decord backend — fast CPU/GPU video decoding optimized for ML workloads.

Known seeking limitations:
    Decord uses its own seeking implementation that handles B-frames correctly
    for most containers. However, seeking in variable-frame-rate (VFR) videos
    may not be frame-accurate. For constant-frame-rate H.264/HEVC content,
    seeking is reliable. GPU decoding is supported via NVDEC but requires
    building decord with CUDA support.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from loguru import logger

from frameforge.backends.base import AbstractBackend, BackendNotAvailable

try:
    import decord as _decord
    from decord import VideoReader as DecordVideoReader
    from decord import cpu, gpu
except ImportError as exc:
    raise BackendNotAvailable(
        "decord is not installed. Install it with: pip install frameforge[decord]"
    ) from exc


class DecordBackend(AbstractBackend):
    """Video decoding backend using decord."""

    def __init__(self, device: str = "cpu") -> None:
        self._reader: DecordVideoReader | None = None
        self._path: str | None = None
        if device == "cpu":
            self._ctx = cpu(0)
        else:
            dev_id = 0
            if ":" in device:
                dev_id = int(device.split(":")[1])
            self._ctx = gpu(dev_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, path: str | Path) -> None:
        path = str(path)
        _decord.bridge.set_bridge("torch")
        self._reader = DecordVideoReader(path, ctx=self._ctx)
        self._path = path
        logger.debug("decord opened {} ({} frames)", path, self.num_frames)

    def close(self) -> None:
        self._reader = None

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def read_frame(self, idx: int) -> np.ndarray:
        self._ensure_open()
        assert self._reader is not None

        if idx < 0 or idx >= self.num_frames:
            raise IndexError(f"Frame index {idx} out of range [0, {self.num_frames})")

        frame = self._reader[idx]
        # decord with torch bridge returns (H, W, C) tensor
        if isinstance(frame, torch.Tensor):
            return frame.numpy().astype(np.uint8)
        return np.asarray(frame, dtype=np.uint8)

    def read_clip(self, start: int, end: int) -> torch.Tensor:
        self._ensure_open()
        assert self._reader is not None

        if start < 0 or end > self.num_frames or start >= end:
            raise IndexError(
                f"Clip range [{start}, {end}) invalid for video with {self.num_frames} frames"
            )

        indices = list(range(start, end))
        frames = self._reader.get_batch(indices)
        # get_batch returns (T, H, W, C) tensor with torch bridge
        if isinstance(frames, torch.Tensor):
            return frames.to(dtype=torch.uint8)
        return torch.from_numpy(np.asarray(frames, dtype=np.uint8))

    def seek(self, timestamp_sec: float) -> int:
        self._ensure_open()
        assert self._reader is not None
        frame_idx = int(timestamp_sec * self.fps)
        return max(0, min(frame_idx, self.num_frames - 1))

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        self._ensure_open()
        assert self._reader is not None
        return float(self._reader.get_avg_fps())

    @property
    def num_frames(self) -> int:
        self._ensure_open()
        assert self._reader is not None
        return len(self._reader)

    @property
    def duration_sec(self) -> float:
        return self.num_frames / self.fps

    @property
    def width(self) -> int:
        self._ensure_open()
        assert self._reader is not None
        return self._reader[0].shape[1]

    @property
    def height(self) -> int:
        self._ensure_open()
        assert self._reader is not None
        return self._reader[0].shape[0]

    @property
    def codec(self) -> str:
        # decord doesn't expose codec info directly
        return "unknown"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._reader is None:
            raise RuntimeError("No video is open. Call open() first.")
