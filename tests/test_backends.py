"""Tests for individual backends — currently PyAV (always available in dev)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from frameforge.backends.pyav_backend import PyAVBackend

NUM_FRAMES = 60
WIDTH = 320
HEIGHT = 240
FPS = 30


class TestPyAVBackend:
    def test_open_and_metadata(self, synthetic_video: Path) -> None:
        backend = PyAVBackend()
        backend.open(synthetic_video)

        assert backend.fps == pytest.approx(FPS, abs=1)
        assert backend.num_frames == NUM_FRAMES
        assert backend.width == WIDTH
        assert backend.height == HEIGHT
        assert backend.codec == "h264"
        assert backend.duration_sec == pytest.approx(NUM_FRAMES / FPS, abs=0.1)

        backend.close()

    def test_read_frame(self, synthetic_video: Path) -> None:
        with PyAVBackend() as backend:
            backend.open(synthetic_video)
            frame = backend.read_frame(0)

            assert isinstance(frame, np.ndarray)
            assert frame.shape == (HEIGHT, WIDTH, 3)
            assert frame.dtype == np.uint8

    def test_read_frame_out_of_range(self, synthetic_video: Path) -> None:
        with PyAVBackend() as backend:
            backend.open(synthetic_video)
            with pytest.raises(IndexError):
                backend.read_frame(9999)

    def test_read_clip(self, synthetic_video: Path) -> None:
        with PyAVBackend() as backend:
            backend.open(synthetic_video)
            clip = backend.read_clip(0, 10)

            assert isinstance(clip, torch.Tensor)
            assert clip.shape == (10, HEIGHT, WIDTH, 3)
            assert clip.dtype == torch.uint8

    def test_read_clip_invalid_range(self, synthetic_video: Path) -> None:
        with PyAVBackend() as backend:
            backend.open(synthetic_video)
            with pytest.raises(IndexError):
                backend.read_clip(50, 70)

    def test_seek(self, synthetic_video: Path) -> None:
        with PyAVBackend() as backend:
            backend.open(synthetic_video)
            idx = backend.seek(0.5)
            assert 0 <= idx < NUM_FRAMES

    def test_context_manager(self, synthetic_video: Path) -> None:
        with PyAVBackend() as backend:
            backend.open(synthetic_video)
            assert backend.num_frames == NUM_FRAMES
        # After exit, internal state should be cleaned up
        assert backend._container is None

    def test_metadata_dict(self, synthetic_video: Path) -> None:
        with PyAVBackend() as backend:
            backend.open(synthetic_video)
            meta = backend.metadata
            assert "fps" in meta
            assert "num_frames" in meta
            assert "width" in meta
            assert "height" in meta
            assert "codec" in meta
