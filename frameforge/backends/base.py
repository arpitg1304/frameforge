"""Abstract base class for all video decoding backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import numpy as np
import torch


class BackendNotAvailable(ImportError):
    """Raised when a backend's underlying library is not installed."""


class AbstractBackend(ABC):
    """Unified interface that every video backend must implement.

    Subclasses handle library-specific decoding while exposing a consistent
    API for reading frames, clips, and metadata.
    """

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def open(self, path: str | Path) -> None:
        """Open a video file for reading."""

    @abstractmethod
    def close(self) -> None:
        """Release all resources associated with the opened video."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.close()

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    @abstractmethod
    def read_frame(self, idx: int) -> np.ndarray:
        """Read a single frame by index.

        Returns:
            np.ndarray of shape (H, W, C) with dtype uint8, in RGB order.
        """

    @abstractmethod
    def read_clip(self, start: int, end: int) -> torch.Tensor:
        """Read a contiguous clip of frames [start, end).

        Returns:
            torch.Tensor of shape (T, H, W, C) with dtype uint8.
        """

    @abstractmethod
    def seek(self, timestamp_sec: float) -> int:
        """Seek to the nearest frame at *timestamp_sec*.

        Returns:
            The frame index that the backend seeked to.
        """

    # ------------------------------------------------------------------
    # Metadata properties
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def fps(self) -> float:
        """Frames per second of the video stream."""

    @property
    @abstractmethod
    def num_frames(self) -> int:
        """Total number of frames in the video."""

    @property
    @abstractmethod
    def duration_sec(self) -> float:
        """Duration of the video in seconds."""

    @property
    @abstractmethod
    def width(self) -> int:
        """Frame width in pixels."""

    @property
    @abstractmethod
    def height(self) -> int:
        """Frame height in pixels."""

    @property
    @abstractmethod
    def codec(self) -> str:
        """Codec name (e.g. 'h264', 'hevc')."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> dict:
        """Return a summary dict of video metadata."""
        return {
            "fps": self.fps,
            "num_frames": self.num_frames,
            "duration_sec": self.duration_sec,
            "width": self.width,
            "height": self.height,
            "codec": self.codec,
        }
