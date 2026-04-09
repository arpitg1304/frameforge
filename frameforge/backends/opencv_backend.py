"""OpenCV (cv2) backend — widely available fallback.

Known seeking limitations:
    OpenCV's VideoCapture.set(CAP_PROP_POS_FRAMES, idx) is NOT frame-accurate
    for many codecs. It seeks to the nearest keyframe and may not decode
    forward to the exact requested frame. For H.264 with long GOPs, the
    returned frame can be off by several frames. B-frame reordering is
    handled by the underlying FFmpeg/GStreamer decoder, but seek precision
    is the weakest of all backends. Best used as a fallback when no other
    backend is available.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from loguru import logger

from frameforge.backends.base import AbstractBackend, BackendNotAvailable

try:
    import cv2
except ImportError as exc:
    raise BackendNotAvailable(
        "OpenCV is not installed. Install it with: pip install frameforge[opencv]"
    ) from exc


class OpenCVBackend(AbstractBackend):
    """Video decoding backend using OpenCV's VideoCapture."""

    def __init__(self) -> None:
        self._cap: cv2.VideoCapture | None = None
        self._path: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, path: str | Path) -> None:
        path = str(path)
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {path}")
        self._path = path
        logger.debug("OpenCV opened {} ({} frames)", path, self.num_frames)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def read_frame(self, idx: int) -> np.ndarray:
        self._ensure_open()
        assert self._cap is not None

        if idx < 0 or idx >= self.num_frames:
            raise IndexError(f"Frame index {idx} out of range [0, {self.num_frames})")

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {idx}")
        # OpenCV returns BGR — convert to RGB
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def read_clip(self, start: int, end: int) -> torch.Tensor:
        self._ensure_open()
        assert self._cap is not None

        if start < 0 or end > self.num_frames or start >= end:
            raise IndexError(
                f"Clip range [{start}, {end}) invalid for video with {self.num_frames} frames"
            )

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames: list[np.ndarray] = []
        for _ in range(end - start):
            ret, frame = self._cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

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
        assert self._cap is not None
        frame_idx = int(timestamp_sec * self.fps)
        return max(0, min(frame_idx, self.num_frames - 1))

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        self._ensure_open()
        assert self._cap is not None
        return self._cap.get(cv2.CAP_PROP_FPS)

    @property
    def num_frames(self) -> int:
        self._ensure_open()
        assert self._cap is not None
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def duration_sec(self) -> float:
        return self.num_frames / self.fps if self.fps > 0 else 0.0

    @property
    def width(self) -> int:
        self._ensure_open()
        assert self._cap is not None
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        self._ensure_open()
        assert self._cap is not None
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def codec(self) -> str:
        self._ensure_open()
        assert self._cap is not None
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        return "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._cap is None:
            raise RuntimeError("No video is open. Call open() first.")
