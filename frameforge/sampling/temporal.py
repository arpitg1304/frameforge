"""Temporal clip sampling strategies common in video understanding.

Each sampler implements ``sample(reader) -> torch.Tensor`` returning a
(T, H, W, C) uint8 tensor.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

import torch

from frameforge.reader import VideoReader


class BaseSampler(ABC):
    """Base class for all temporal samplers."""

    @abstractmethod
    def sample(self, reader: VideoReader) -> torch.Tensor:
        """Sample frames from *reader*, returning (T, H, W, C) uint8 tensor."""

    @abstractmethod
    def get_indices(self, num_frames: int) -> list[int]:
        """Return frame indices that would be sampled from a video of length *num_frames*."""


class UniformSampler(BaseSampler):
    """Evenly spaced frames across the full video.

    Selects *num_frames* indices uniformly distributed from frame 0 to the
    last frame, inclusive of endpoints when possible.
    """

    def __init__(self, num_frames: int) -> None:
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        self.num_frames = num_frames

    def get_indices(self, total_frames: int) -> list[int]:
        if total_frames <= self.num_frames:
            return list(range(total_frames))
        # Linspace-style uniform indices
        step = (total_frames - 1) / (self.num_frames - 1) if self.num_frames > 1 else 0
        return [int(round(i * step)) for i in range(self.num_frames)]

    def sample(self, reader: VideoReader) -> torch.Tensor:
        indices = self.get_indices(len(reader))
        return reader[indices]


class RandomSampler(BaseSampler):
    """Random clip window with randomly sampled frames within it.

    1. Pick a random window of *clip_duration_sec* seconds.
    2. Randomly sample *num_frames* frames from that window.
    """

    def __init__(self, num_frames: int, clip_duration_sec: float) -> None:
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        if clip_duration_sec <= 0:
            raise ValueError("clip_duration_sec must be > 0")
        self.num_frames = num_frames
        self.clip_duration_sec = clip_duration_sec

    def get_indices(self, total_frames: int, fps: float = 30.0) -> list[int]:  # type: ignore[override]
        clip_frames = int(self.clip_duration_sec * fps)
        clip_frames = min(clip_frames, total_frames)

        max_start = total_frames - clip_frames
        start = random.randint(0, max(0, max_start))
        end = start + clip_frames

        n = min(self.num_frames, clip_frames)
        indices = sorted(random.sample(range(start, end), n))
        return indices

    def sample(self, reader: VideoReader) -> torch.Tensor:
        indices = self.get_indices(len(reader), reader.fps)
        return reader[indices]


class DenseSampler(BaseSampler):
    """Every Nth frame (dense sampling with stride).

    Reads every *stride*-th frame starting from frame 0.
    """

    def __init__(self, stride: int = 1) -> None:
        if stride < 1:
            raise ValueError("stride must be >= 1")
        self.stride = stride

    def get_indices(self, total_frames: int) -> list[int]:
        return list(range(0, total_frames, self.stride))

    def sample(self, reader: VideoReader) -> torch.Tensor:
        indices = self.get_indices(len(reader))
        return reader[indices]


class EpisodeSampler(BaseSampler):
    """Robotics-specific: sample within episode boundaries.

    Given a list of episode boundary frame indices, randomly picks an
    episode and uniformly samples *num_frames* from within it.

    Args:
        episode_boundaries: Sorted list of frame indices where episodes
            start. The last episode extends to the end of the video.
        num_frames: Number of frames to sample per episode.
    """

    def __init__(
        self, episode_boundaries: list[int], num_frames: int = 16
    ) -> None:
        if not episode_boundaries:
            raise ValueError("episode_boundaries must be non-empty")
        if num_frames < 1:
            raise ValueError("num_frames must be >= 1")
        self.episode_boundaries = sorted(episode_boundaries)
        self.num_frames = num_frames

    def get_indices(self, total_frames: int) -> list[int]:
        # Pick a random episode
        ep_idx = random.randint(0, len(self.episode_boundaries) - 1)
        start = self.episode_boundaries[ep_idx]
        end = (
            self.episode_boundaries[ep_idx + 1]
            if ep_idx + 1 < len(self.episode_boundaries)
            else total_frames
        )

        length = end - start
        if length <= self.num_frames:
            return list(range(start, end))

        # Uniform sampling within the episode
        step = (length - 1) / (self.num_frames - 1) if self.num_frames > 1 else 0
        return [start + int(round(i * step)) for i in range(self.num_frames)]

    def sample(self, reader: VideoReader) -> torch.Tensor:
        indices = self.get_indices(len(reader))
        return reader[indices]
