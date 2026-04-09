"""Using frameforge with PyTorch DataLoader."""

import torch
from torch.utils.data import DataLoader

from frameforge.dataloader import VideoClipDataset
from frameforge.dataloader.collate import video_collate
from frameforge.sampling import UniformSampler

video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]
sampler = UniformSampler(num_frames=16)


def normalize(clip: torch.Tensor) -> torch.Tensor:
    return clip.float() / 255.0


dataset = VideoClipDataset(
    video_paths=video_paths,
    sampler=sampler,
    backend="pyav",
    transform=normalize,
)

loader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=2,
    collate_fn=video_collate,
)

for batch in loader:
    print(f"Batch shape: {batch.shape}")  # (B, T, H, W, C)
    # Feed to your model here
    break
