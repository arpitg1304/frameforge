"""Tests for the unified VideoReader."""

from pathlib import Path

import numpy as np
import pytest
import torch

from frameforge.reader import VideoReader

NUM_FRAMES = 60
WIDTH = 320
HEIGHT = 240


class TestVideoReader:
    def test_auto_backend(self, synthetic_video: Path) -> None:
        reader = VideoReader(synthetic_video)
        assert len(reader) == NUM_FRAMES
        reader.close()

    def test_explicit_pyav(self, synthetic_video: Path) -> None:
        reader = VideoReader(synthetic_video, backend="pyav")
        assert len(reader) == NUM_FRAMES
        reader.close()

    def test_single_frame_indexing(self, synthetic_video: Path) -> None:
        with VideoReader(synthetic_video, backend="pyav") as reader:
            frame = reader[0]
            assert isinstance(frame, torch.Tensor)
            assert frame.shape == (HEIGHT, WIDTH, 3)

    def test_negative_indexing(self, synthetic_video: Path) -> None:
        with VideoReader(synthetic_video, backend="pyav") as reader:
            frame = reader[-1]
            assert frame.shape == (HEIGHT, WIDTH, 3)

    def test_slice_indexing(self, synthetic_video: Path) -> None:
        with VideoReader(synthetic_video, backend="pyav") as reader:
            clip = reader[10:20]
            assert isinstance(clip, torch.Tensor)
            assert clip.shape == (10, HEIGHT, WIDTH, 3)

    def test_fancy_indexing(self, synthetic_video: Path) -> None:
        with VideoReader(synthetic_video, backend="pyav") as reader:
            frames = reader[[0, 5, 10]]
            assert isinstance(frames, torch.Tensor)
            assert frames.shape == (3, HEIGHT, WIDTH, 3)

    def test_numpy_output(self, synthetic_video: Path) -> None:
        with VideoReader(synthetic_video, backend="pyav", output="numpy") as reader:
            frame = reader[0]
            assert isinstance(frame, np.ndarray)

    def test_metadata(self, synthetic_video: Path) -> None:
        reader = VideoReader(synthetic_video, backend="pyav")
        meta = reader.metadata
        assert meta["width"] == WIDTH
        assert meta["height"] == HEIGHT
        assert meta["codec"] == "h264"
        reader.close()

    def test_properties(self, synthetic_video: Path) -> None:
        with VideoReader(synthetic_video, backend="pyav") as reader:
            assert reader.fps == pytest.approx(30.0, abs=1)
            assert reader.num_frames == NUM_FRAMES
            assert reader.duration_sec == pytest.approx(2.0, abs=0.1)

    def test_context_manager(self, synthetic_video: Path) -> None:
        with VideoReader(synthetic_video) as reader:
            assert len(reader) == NUM_FRAMES
        # Backend should be closed
        assert reader._backend is None

    def test_pickling(self, synthetic_video: Path) -> None:
        import pickle

        reader = VideoReader(synthetic_video, backend="pyav")
        _ = reader[0]  # force backend init

        data = pickle.dumps(reader)
        reader2 = pickle.loads(data)
        # Backend should be None after unpickling (lazy init)
        assert reader2._backend is None
        # But should work fine
        frame = reader2[0]
        assert frame.shape == (HEIGHT, WIDTH, 3)
        reader.close()
        reader2.close()

    def test_invalid_backend(self, synthetic_video: Path) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            VideoReader(synthetic_video, backend="nonexistent")[0]
