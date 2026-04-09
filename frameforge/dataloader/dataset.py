"""PyTorch Dataset and IterableDataset wrappers for video clips.

Worker safety: VideoReader is initialized inside __getitem__ / __iter__,
not in __init__, so file handles are never shared across forked workers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from frameforge.reader import VideoReader
from frameforge.sampling.temporal import BaseSampler


class VideoClipDataset(Dataset):
    """Map-style dataset that returns sampled clips from a list of videos.

    Each ``__getitem__`` call opens a VideoReader, samples a clip via the
    provided sampler, optionally applies a transform, and closes the reader.
    This is worker-safe by design.

    Args:
        video_paths: List of paths to video files.
        sampler: A temporal sampler instance (e.g. UniformSampler).
        backend: Backend name or "auto".
        transform: Optional callable applied to the (T, H, W, C) tensor.
        device: Device for decoding.
    """

    def __init__(
        self,
        video_paths: list[str | Path],
        sampler: BaseSampler,
        backend: str = "auto",
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        device: str = "cpu",
    ) -> None:
        self.video_paths = [Path(p) for p in video_paths]
        self.sampler = sampler
        self.backend = backend
        self.transform = transform
        self.device = device

    def __len__(self) -> int:
        return len(self.video_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        with VideoReader(self.video_paths[idx], backend=self.backend, device=self.device) as reader:
            clip = self.sampler.sample(reader)

        if self.transform is not None:
            clip = self.transform(clip)
        return clip


class VideoStreamDataset(IterableDataset):
    """Iterable dataset that yields sequential clips from a single video.

    Handles worker sharding via ``get_worker_info()`` — each worker
    processes a different portion of the video.

    Args:
        video_path: Path to the video file.
        sampler: A temporal sampler (typically DenseSampler or UniformSampler).
        backend: Backend name or "auto".
        clip_length: Number of frames per yielded clip.
        device: Device for decoding.
    """

    def __init__(
        self,
        video_path: str | Path,
        sampler: BaseSampler,
        backend: str = "auto",
        clip_length: int = 16,
        device: str = "cpu",
    ) -> None:
        self.video_path = Path(video_path)
        self.sampler = sampler
        self.backend = backend
        self.clip_length = clip_length
        self.device = device

    def __iter__(self):  # noqa: ANN204
        reader = VideoReader(self.video_path, backend=self.backend, device=self.device)
        total = len(reader)

        # Worker sharding
        worker_info = get_worker_info()
        if worker_info is not None:
            per_worker = total // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else total
        else:
            start = 0
            end = total

        # Yield non-overlapping clips
        for clip_start in range(start, end - self.clip_length + 1, self.clip_length):
            clip = reader[clip_start : clip_start + self.clip_length]
            yield clip

        reader.close()
