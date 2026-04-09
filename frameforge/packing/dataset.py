"""Shard datasets for training: map-style and iterable-style readers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from frameforge.reader import VideoReader


def _load_manifest(shard_dir: str | Path) -> dict:
    """Load and validate manifest.json from a shard directory."""
    shard_dir = Path(shard_dir)
    manifest_path = shard_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json found in {shard_dir}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version", 0) < 2:
        raise ValueError(f"Unsupported manifest version: {manifest.get('version')}")
    return manifest


class ShardDataset(Dataset):
    """Map-style dataset over packed shard videos.

    Supports random access to any clip by index. Since shards use GOP=1,
    seeking to any clip is efficient (every frame is a keyframe).

    Good for: evaluation, debugging, small-scale training, any workflow
    that needs random access by clip index.

    Args:
        shard_dir: Path to the directory containing shards and manifest.json.
        camera: Which camera to read for multi-camera shards. None for single-cam.
        transform: Optional callable applied to each (T, H, W, C) clip tensor.
        backend: Video decode backend.
    """

    def __init__(
        self,
        shard_dir: str | Path,
        camera: str | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        backend: str = "pyav",
    ) -> None:
        self._shard_dir = Path(shard_dir)
        self._camera = camera
        self._transform = transform
        self._backend = backend

        self._manifest = _load_manifest(shard_dir)
        self._config = self._manifest["pack_config"]
        self._clip_length = self._config["clip_length"]

        # Build clip → shard mapping
        self._clips = self._manifest["clips"]

        # Build shard_id → file path mapping for the requested camera
        self._shard_files: dict[int, Path] = {}
        for shard_info in self._manifest["shards"]:
            if shard_info.get("camera") == camera or (camera is None and shard_info.get("camera") is None):
                self._shard_files[shard_info["shard_id"]] = self._shard_dir / shard_info["file"]

        if not self._shard_files:
            available = set(s.get("camera") for s in self._manifest["shards"])
            raise ValueError(
                f"No shards found for camera={camera!r}. Available cameras: {available}"
            )

        # Lazy-opened readers (one per shard, opened on demand)
        self._readers: dict[int, VideoReader] = {}

    def __len__(self) -> int:
        return len(self._clips)

    def __getitem__(self, idx: int) -> torch.Tensor | tuple[torch.Tensor, dict]:
        clip_meta = self._clips[idx]
        shard_id = clip_meta["shard_id"]
        clip_index = clip_meta["clip_index"]

        # Open reader lazily
        if shard_id not in self._readers:
            path = self._shard_files[shard_id]
            self._readers[shard_id] = VideoReader(path, backend=self._backend)

        reader = self._readers[shard_id]

        # Clips are packed sequentially: clip N starts at frame N * clip_length
        frame_start = clip_index * self._clip_length
        frame_end = frame_start + self._clip_length
        clip = reader[frame_start:frame_end]

        # Ensure exact clip length (PyAV B-frame edge cases)
        if clip.shape[0] > self._clip_length:
            clip = clip[:self._clip_length]

        if self._transform is not None:
            clip = self._transform(clip)

        return clip

    def get_clip_metadata(self, idx: int) -> dict:
        """Get the manifest metadata for clip at index *idx*."""
        return self._clips[idx]

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __del__(self) -> None:
        self.close()

    @property
    def manifest(self) -> dict:
        return self._manifest

    @property
    def num_clips(self) -> int:
        return len(self._clips)

    @property
    def num_shards(self) -> int:
        return len(self._shard_files)


class ShardStreamDataset(IterableDataset):
    """Iterable dataset that reads shard videos sequentially for max throughput.

    Each worker reads a different subset of shards. Shard order is shuffled
    per epoch. Within each shard, clips are read sequentially (no seeking).

    Good for: large-scale training where throughput matters most.

    Args:
        shard_dir: Path to shard directory with manifest.json.
        camera: Which camera for multi-camera shards.
        transform: Optional callable applied to each (T, H, W, C) clip tensor.
        backend: Video decode backend.
        prefetch: Enable decode prefetching for even faster sequential reads.
        shuffle_shards: Shuffle shard order each iteration. Default True.
        seed: Base seed for shard shuffling (combined with epoch/worker).
    """

    def __init__(
        self,
        shard_dir: str | Path,
        camera: str | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        backend: str = "pyav",
        prefetch: bool = True,
        shuffle_shards: bool = True,
        seed: int = 42,
    ) -> None:
        self._shard_dir = Path(shard_dir)
        self._camera = camera
        self._transform = transform
        self._backend = backend
        self._prefetch = prefetch
        self._shuffle = shuffle_shards
        self._seed = seed
        self._epoch = 0

        self._manifest = _load_manifest(shard_dir)
        self._config = self._manifest["pack_config"]
        self._clip_length = self._config["clip_length"]
        self._clips = self._manifest["clips"]

        # Shard files for the requested camera
        self._shard_list: list[tuple[int, Path, int]] = []  # (shard_id, path, num_clips)
        for shard_info in self._manifest["shards"]:
            if shard_info.get("camera") == camera or (camera is None and shard_info.get("camera") is None):
                self._shard_list.append((
                    shard_info["shard_id"],
                    self._shard_dir / shard_info["file"],
                    shard_info["num_clips"],
                ))

        if not self._shard_list:
            available = set(s.get("camera") for s in self._manifest["shards"])
            raise ValueError(
                f"No shards found for camera={camera!r}. Available cameras: {available}"
            )

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for deterministic shard shuffling."""
        self._epoch = epoch

    def __iter__(self):  # noqa: ANN204
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # Shuffle shard order (deterministic per epoch + seed)
        shards = list(self._shard_list)
        if self._shuffle:
            rng = random.Random(self._seed + self._epoch)
            rng.shuffle(shards)

        # Distribute shards across workers
        my_shards = shards[worker_id::num_workers]

        # Build clip index for metadata lookup
        clips_by_shard: dict[int, list[dict]] = {}
        for clip in self._clips:
            sid = clip["shard_id"]
            if sid not in clips_by_shard:
                clips_by_shard[sid] = []
            clips_by_shard[sid].append(clip)

        for shard_id, shard_path, num_clips in my_shards:
            reader = VideoReader(
                shard_path,
                backend=self._backend,
                prefetch=self._prefetch,
                prefetch_ahead=self._clip_length * 2,
            )

            for ci in range(num_clips):
                frame_start = ci * self._clip_length
                frame_end = frame_start + self._clip_length
                clip = reader[frame_start:frame_end]

                # Ensure exact clip length (PyAV B-frame edge cases)
                if clip.shape[0] > self._clip_length:
                    clip = clip[:self._clip_length]
                elif clip.shape[0] < self._clip_length:
                    continue  # skip short clips

                if self._transform is not None:
                    clip = self._transform(clip)

                yield clip

            reader.close()

    @property
    def manifest(self) -> dict:
        return self._manifest

    @property
    def num_clips(self) -> int:
        return len(self._clips)

    @property
    def num_shards(self) -> int:
        return len(self._shard_list)
