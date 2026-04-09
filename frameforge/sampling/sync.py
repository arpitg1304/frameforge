"""Multi-camera synchronization primitives for robotics setups."""

from __future__ import annotations

import torch

from frameforge.reader import VideoReader


def sync_readers_by_timestamp(
    readers: dict[str, VideoReader],
    timestamps_sec: list[float],
) -> dict[str, torch.Tensor]:
    """Read synchronized frames from multiple cameras at given timestamps.

    Args:
        readers: Mapping of camera name → VideoReader.
        timestamps_sec: List of target timestamps in seconds.

    Returns:
        Mapping of camera name → (T, H, W, C) tensor where T = len(timestamps_sec).
    """
    result: dict[str, torch.Tensor] = {}
    for cam_name, reader in readers.items():
        backend = reader._ensure_backend()
        indices = [backend.seek(ts) for ts in timestamps_sec]
        result[cam_name] = reader[indices]
    return result
