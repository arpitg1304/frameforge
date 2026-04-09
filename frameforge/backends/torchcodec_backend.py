"""torchcodec backend — GPU-accelerated video decoding via PyTorch.

Known seeking limitations:
    torchcodec uses pts-based seeking internally. Frame-accurate random access
    is well-supported for keyframe-aligned seeks. For B-frame content, the
    library handles reordering transparently. GPU decoding (NVDEC) requires
    CUDA and may not support all codecs — H.264 and HEVC are generally safe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from loguru import logger

from frameforge.backends.base import AbstractBackend, BackendNotAvailable

try:
    import torchcodec  # noqa: F401
    from torchcodec.decoders import VideoDecoder
except ImportError as exc:
    raise BackendNotAvailable(
        "torchcodec is not installed. Install it with: pip install frameforge[torchcodec]"
    ) from exc


class TorchCodecBackend(AbstractBackend):
    """Video decoding backend using torchcodec."""

    def __init__(self, device: str = "cpu") -> None:
        self._decoder: VideoDecoder | None = None
        self._device = device
        self._path: str | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, path: str | Path) -> None:
        path = str(path)
        self._decoder = VideoDecoder(path, device=self._device)
        self._path = path
        logger.debug("torchcodec opened {} ({} frames)", path, self.num_frames)

    def close(self) -> None:
        self._decoder = None

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def read_frame(self, idx: int) -> np.ndarray:
        self._ensure_open()
        assert self._decoder is not None

        if idx < 0 or idx >= self.num_frames:
            raise IndexError(f"Frame index {idx} out of range [0, {self.num_frames})")

        frame = self._decoder[idx]
        # torchcodec returns (C, H, W) — convert to (H, W, C) uint8
        if frame.ndim == 3 and frame.shape[0] in (1, 3):
            frame = frame.permute(1, 2, 0)
        return frame.cpu().numpy().astype(np.uint8)

    def read_clip(self, start: int, end: int) -> torch.Tensor:
        self._ensure_open()
        assert self._decoder is not None

        if start < 0 or end > self.num_frames or start >= end:
            raise IndexError(
                f"Clip range [{start}, {end}) invalid for video with {self.num_frames} frames"
            )

        frames = self._decoder[start:end]
        # torchcodec batch returns (T, C, H, W) — convert to (T, H, W, C)
        if frames.ndim == 4 and frames.shape[1] in (1, 3):
            frames = frames.permute(0, 2, 3, 1)
        return frames.to(dtype=torch.uint8)

    def seek(self, timestamp_sec: float) -> int:
        self._ensure_open()
        assert self._decoder is not None
        frame_idx = int(timestamp_sec * self.fps)
        return max(0, min(frame_idx, self.num_frames - 1))

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def fps(self) -> float:
        self._ensure_open()
        assert self._decoder is not None
        meta = self._decoder.metadata
        return meta.average_fps  # type: ignore[return-value]

    @property
    def num_frames(self) -> int:
        self._ensure_open()
        assert self._decoder is not None
        meta = self._decoder.metadata
        return meta.num_frames  # type: ignore[return-value]

    @property
    def duration_sec(self) -> float:
        self._ensure_open()
        assert self._decoder is not None
        meta = self._decoder.metadata
        return meta.duration_seconds  # type: ignore[return-value]

    @property
    def width(self) -> int:
        self._ensure_open()
        assert self._decoder is not None
        meta = self._decoder.metadata
        return meta.width  # type: ignore[return-value]

    @property
    def height(self) -> int:
        self._ensure_open()
        assert self._decoder is not None
        meta = self._decoder.metadata
        return meta.height  # type: ignore[return-value]

    @property
    def codec(self) -> str:
        self._ensure_open()
        assert self._decoder is not None
        meta = self._decoder.metadata
        return meta.codec  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._decoder is None:
            raise RuntimeError("No video is open. Call open() first.")
