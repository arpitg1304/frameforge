"""Unified VideoWriter — encode frames/clips back to video files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from loguru import logger

try:
    import av
except ImportError:
    av = None  # type: ignore[assignment]


class VideoWriter:
    """Write frames to a video file using PyAV (FFmpeg).

    Usage::

        with VideoWriter("output.mp4", fps=30, width=640, height=480) as w:
            for frame in frames:
                w.write_frame(frame)

    Args:
        path: Output file path.
        fps: Frame rate.
        width: Frame width.
        height: Frame height.
        codec: Video codec name (default "h264").
        pix_fmt: Pixel format (default "yuv420p").
    """

    def __init__(
        self,
        path: str | Path,
        fps: float = 30.0,
        width: int = 640,
        height: int = 480,
        codec: str = "h264",
        pix_fmt: str = "yuv420p",
    ) -> None:
        if av is None:
            raise ImportError("PyAV is required for VideoWriter. Install: pip install frameforge[pyav]")

        self._path = str(path)
        self._fps = fps
        self._width = width
        self._height = height
        self._codec = codec
        self._pix_fmt = pix_fmt
        self._container: av.container.OutputContainer | None = None
        self._stream: av.video.stream.VideoStream | None = None
        self._frame_count = 0

    def open(self) -> None:
        self._container = av.open(self._path, mode="w")
        self._stream = self._container.add_stream(self._codec, rate=self._fps)
        self._stream.width = self._width
        self._stream.height = self._height
        self._stream.pix_fmt = self._pix_fmt
        self._frame_count = 0
        logger.debug("VideoWriter opened {}", self._path)

    def write_frame(self, frame: np.ndarray | torch.Tensor) -> None:
        """Write a single RGB frame of shape (H, W, C) with dtype uint8."""
        assert self._container is not None and self._stream is not None

        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        frame = np.ascontiguousarray(frame, dtype=np.uint8)

        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in self._stream.encode(av_frame):
            self._container.mux(packet)
        self._frame_count += 1

    def write_clip(self, clip: torch.Tensor | np.ndarray) -> None:
        """Write a clip of shape (T, H, W, C)."""
        if isinstance(clip, torch.Tensor):
            clip = clip.cpu().numpy()
        for i in range(clip.shape[0]):
            self.write_frame(clip[i])

    def close(self) -> None:
        if self._container is not None:
            assert self._stream is not None
            for packet in self._stream.encode():
                self._container.mux(packet)
            self._container.close()
            self._container = None
            self._stream = None
            logger.debug("VideoWriter closed {} ({} frames)", self._path, self._frame_count)

    def __enter__(self) -> VideoWriter:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.close()
