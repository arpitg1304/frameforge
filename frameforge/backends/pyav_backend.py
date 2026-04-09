"""PyAV (FFmpeg) backend — the most portable default backend.

Known seeking limitations:
    PyAV uses FFmpeg's av_seek_frame which seeks to the nearest keyframe.
    For videos with long GOP structures, the actual frame returned after a
    seek may be several frames before the requested position. This backend
    decodes forward from the keyframe to reach the exact requested frame,
    which is accurate but can be slow for random access in long-GOP files.
    B-frame reordering is handled correctly by PyAV's decoder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from loguru import logger

from frameforge.backends.base import AbstractBackend, BackendNotAvailable

try:
    import av
except ImportError as exc:
    raise BackendNotAvailable(
        "PyAV is not installed. Install it with: pip install frameforge[pyav]"
    ) from exc


class PyAVBackend(AbstractBackend):
    """Video decoding backend using PyAV (FFmpeg bindings)."""

    def __init__(self) -> None:
        self._container: av.container.InputContainer | None = None
        self._stream: av.video.stream.VideoStream | None = None
        self._path: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, path: str | Path) -> None:
        path = str(path)
        self._container = av.open(path)
        self._stream = self._container.streams.video[0]
        # Enable multi-threaded decoding
        self._stream.thread_type = "AUTO"
        self._path = path
        logger.debug("PyAV opened {} ({} frames)", path, self.num_frames)

    def close(self) -> None:
        if self._container is not None:
            self._container.close()
            self._container = None
            self._stream = None

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def read_frame(self, idx: int) -> np.ndarray:
        self._ensure_open()
        assert self._container is not None and self._stream is not None

        if idx < 0 or idx >= self.num_frames:
            raise IndexError(f"Frame index {idx} out of range [0, {self.num_frames})")

        # Seek to nearest keyframe, then decode forward
        target_pts = int(idx * self._stream.time_base.denominator
                         / (self._stream.time_base.numerator * self.fps))
        self._container.seek(target_pts, stream=self._stream)

        for i, frame in enumerate(self._container.decode(video=0)):
            frame_idx = int(frame.pts * float(self._stream.time_base) * self.fps)
            if frame_idx >= idx:
                return frame.to_ndarray(format="rgb24")

        raise RuntimeError(f"Could not decode frame {idx}")

    def read_clip(self, start: int, end: int) -> torch.Tensor:
        self._ensure_open()
        assert self._container is not None and self._stream is not None

        if start < 0 or end > self.num_frames or start >= end:
            raise IndexError(
                f"Clip range [{start}, {end}) invalid for video with {self.num_frames} frames"
            )

        # Seek to nearest keyframe before start
        target_pts = int(start * self._stream.time_base.denominator
                         / (self._stream.time_base.numerator * self.fps))
        self._container.seek(target_pts, stream=self._stream)

        frames: list[np.ndarray] = []
        for frame in self._container.decode(video=0):
            frame_idx = int(frame.pts * float(self._stream.time_base) * self.fps)
            if frame_idx >= end:
                break
            if frame_idx >= start:
                frames.append(frame.to_ndarray(format="rgb24"))

        if len(frames) != (end - start):
            logger.warning(
                "Expected {} frames but decoded {} for clip [{}, {})",
                end - start,
                len(frames),
                start,
                end,
            )

        return torch.from_numpy(np.stack(frames))

    def seek(self, timestamp_sec: float) -> int:
        self._ensure_open()
        assert self._container is not None and self._stream is not None

        frame_idx = int(timestamp_sec * self.fps)
        frame_idx = max(0, min(frame_idx, self.num_frames - 1))
        return frame_idx

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        self._ensure_open()
        assert self._stream is not None
        return float(self._stream.average_rate)  # type: ignore[arg-type]

    @property
    def num_frames(self) -> int:
        self._ensure_open()
        assert self._stream is not None
        n = self._stream.frames
        if n == 0:
            # Some containers don't report frame count; estimate from duration
            n = int(self.duration_sec * self.fps)
        return n

    @property
    def duration_sec(self) -> float:
        self._ensure_open()
        assert self._stream is not None
        return float(self._stream.duration * self._stream.time_base)

    @property
    def width(self) -> int:
        self._ensure_open()
        assert self._stream is not None
        return self._stream.width

    @property
    def height(self) -> int:
        self._ensure_open()
        assert self._stream is not None
        return self._stream.height

    @property
    def codec(self) -> str:
        self._ensure_open()
        assert self._stream is not None
        return self._stream.codec_context.name

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._container is None:
            raise RuntimeError("No video is open. Call open() first.")
