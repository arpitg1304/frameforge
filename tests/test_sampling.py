"""Tests for temporal clip samplers."""

from pathlib import Path

import pytest
import torch

from frameforge.reader import VideoReader
from frameforge.sampling.temporal import (
    DenseSampler,
    EpisodeSampler,
    RandomSampler,
    UniformSampler,
)

NUM_FRAMES = 60
HEIGHT = 240
WIDTH = 320


class TestUniformSampler:
    def test_basic(self, synthetic_video: Path) -> None:
        sampler = UniformSampler(num_frames=8)
        reader = VideoReader(synthetic_video, backend="pyav")
        clip = sampler.sample(reader)
        assert clip.shape == (8, HEIGHT, WIDTH, 3)
        reader.close()

    def test_indices_evenly_spaced(self) -> None:
        sampler = UniformSampler(num_frames=5)
        indices = sampler.get_indices(100)
        assert len(indices) == 5
        assert indices[0] == 0
        assert indices[-1] == 99

    def test_more_frames_than_video(self) -> None:
        sampler = UniformSampler(num_frames=100)
        indices = sampler.get_indices(10)
        assert len(indices) == 10  # capped to video length

    def test_invalid_num_frames(self) -> None:
        with pytest.raises(ValueError):
            UniformSampler(num_frames=0)


class TestRandomSampler:
    def test_basic(self, synthetic_video: Path) -> None:
        sampler = RandomSampler(num_frames=4, clip_duration_sec=0.5)
        reader = VideoReader(synthetic_video, backend="pyav")
        clip = sampler.sample(reader)
        assert clip.shape[0] == 4
        assert clip.shape[1:] == (HEIGHT, WIDTH, 3)
        reader.close()

    def test_indices_within_clip_window(self) -> None:
        sampler = RandomSampler(num_frames=4, clip_duration_sec=0.5)
        indices = sampler.get_indices(100, fps=30.0)
        assert len(indices) == 4
        # All indices should be within a 15-frame window
        assert max(indices) - min(indices) < 15

    def test_sorted_indices(self) -> None:
        sampler = RandomSampler(num_frames=8, clip_duration_sec=1.0)
        indices = sampler.get_indices(100, fps=30.0)
        assert indices == sorted(indices)


class TestDenseSampler:
    def test_stride_1(self, synthetic_video: Path) -> None:
        sampler = DenseSampler(stride=1)
        indices = sampler.get_indices(NUM_FRAMES)
        assert len(indices) == NUM_FRAMES

    def test_stride_5(self) -> None:
        sampler = DenseSampler(stride=5)
        indices = sampler.get_indices(100)
        assert indices == list(range(0, 100, 5))

    def test_invalid_stride(self) -> None:
        with pytest.raises(ValueError):
            DenseSampler(stride=0)


class TestEpisodeSampler:
    def test_basic(self, synthetic_video: Path) -> None:
        sampler = EpisodeSampler(episode_boundaries=[0, 30], num_frames=8)
        reader = VideoReader(synthetic_video, backend="pyav")
        clip = sampler.sample(reader)
        assert clip.shape[0] == 8
        reader.close()

    def test_indices_within_episode(self) -> None:
        sampler = EpisodeSampler(episode_boundaries=[0, 50, 100], num_frames=8)
        # Run multiple times to check boundaries
        for _ in range(20):
            indices = sampler.get_indices(150)
            assert len(indices) == 8
            # Check that all indices fall within a single episode
            if indices[0] < 50:
                assert all(i < 50 for i in indices)
            elif indices[0] < 100:
                assert all(50 <= i < 100 for i in indices)
            else:
                assert all(100 <= i < 150 for i in indices)

    def test_empty_boundaries(self) -> None:
        with pytest.raises(ValueError):
            EpisodeSampler(episode_boundaries=[], num_frames=8)
