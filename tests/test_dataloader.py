"""Tests for DataLoader wrappers."""

from pathlib import Path

import pytest
import torch

from frameforge.dataloader.collate import video_collate, video_collate_with_mask
from frameforge.dataloader.dataset import VideoClipDataset, VideoStreamDataset
from frameforge.sampling.temporal import UniformSampler

HEIGHT = 240
WIDTH = 320


class TestVideoClipDataset:
    def test_getitem(self, synthetic_video: Path) -> None:
        sampler = UniformSampler(num_frames=8)
        ds = VideoClipDataset([synthetic_video], sampler=sampler, backend="pyav")
        assert len(ds) == 1
        clip = ds[0]
        assert isinstance(clip, torch.Tensor)
        assert clip.shape == (8, HEIGHT, WIDTH, 3)

    def test_with_transform(self, synthetic_video: Path) -> None:
        sampler = UniformSampler(num_frames=4)

        def normalize(x: torch.Tensor) -> torch.Tensor:
            return x.float() / 255.0

        ds = VideoClipDataset(
            [synthetic_video], sampler=sampler, backend="pyav", transform=normalize
        )
        clip = ds[0]
        assert clip.dtype == torch.float32
        assert clip.max() <= 1.0

    def test_multiple_videos(self, synthetic_video_pair: tuple[Path, Path]) -> None:
        cam0, cam1 = synthetic_video_pair
        sampler = UniformSampler(num_frames=4)
        ds = VideoClipDataset([cam0, cam1], sampler=sampler, backend="pyav")
        assert len(ds) == 2
        clip0 = ds[0]
        clip1 = ds[1]
        assert clip0.shape == clip1.shape


class TestVideoStreamDataset:
    def test_iteration(self, synthetic_video: Path) -> None:
        sampler = UniformSampler(num_frames=8)
        ds = VideoStreamDataset(
            synthetic_video, sampler=sampler, backend="pyav", clip_length=16
        )
        clips = list(ds)
        assert len(clips) > 0
        assert all(c.shape == (16, HEIGHT, WIDTH, 3) for c in clips)


class TestCollate:
    def test_video_collate_same_length(self) -> None:
        batch = [torch.zeros(8, 64, 64, 3, dtype=torch.uint8) for _ in range(4)]
        result = video_collate(batch)
        assert result.shape == (4, 8, 64, 64, 3)

    def test_video_collate_variable_length(self) -> None:
        batch = [
            torch.zeros(8, 64, 64, 3, dtype=torch.uint8),
            torch.zeros(12, 64, 64, 3, dtype=torch.uint8),
        ]
        result = video_collate(batch)
        assert result.shape == (2, 12, 64, 64, 3)

    def test_collate_with_mask(self) -> None:
        batch = [
            torch.ones(5, 32, 32, 3, dtype=torch.uint8),
            torch.ones(10, 32, 32, 3, dtype=torch.uint8),
        ]
        padded, mask = video_collate_with_mask(batch)
        assert padded.shape == (2, 10, 32, 32, 3)
        assert mask.shape == (2, 10)
        assert mask[0, :5].all()
        assert not mask[0, 5:].any()
        assert mask[1].all()
