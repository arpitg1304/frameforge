"""Custom collate functions for video tensor batches."""

from __future__ import annotations

import torch


def video_collate(batch: list[torch.Tensor]) -> torch.Tensor:
    """Collate variable-length video clips by padding to the longest clip.

    Args:
        batch: List of (T_i, H, W, C) tensors with potentially different T_i.

    Returns:
        (B, T_max, H, W, C) tensor zero-padded along the time dimension.
    """
    max_t = max(clip.shape[0] for clip in batch)
    h, w, c = batch[0].shape[1], batch[0].shape[2], batch[0].shape[3]

    padded = torch.zeros(len(batch), max_t, h, w, c, dtype=batch[0].dtype)
    for i, clip in enumerate(batch):
        padded[i, : clip.shape[0]] = clip

    return padded


def video_collate_with_mask(
    batch: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collate with a boolean mask indicating valid (non-padded) frames.

    Returns:
        Tuple of (B, T_max, H, W, C) padded tensor and (B, T_max) bool mask.
    """
    max_t = max(clip.shape[0] for clip in batch)
    h, w, c = batch[0].shape[1], batch[0].shape[2], batch[0].shape[3]

    padded = torch.zeros(len(batch), max_t, h, w, c, dtype=batch[0].dtype)
    mask = torch.zeros(len(batch), max_t, dtype=torch.bool)

    for i, clip in enumerate(batch):
        t = clip.shape[0]
        padded[i, :t] = clip
        mask[i, :t] = True

    return padded, mask
