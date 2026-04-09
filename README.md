# frameforge

[![CI](https://github.com/arpitg1304/frameforge/actions/workflows/ci.yml/badge.svg)](https://github.com/arpitg1304/frameforge/actions/workflows/ci.yml)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arpitg1304/frameforge/blob/main/notebooks/getting_started_colab.ipynb)

Load video clips for robotics model training. Episode-aware, fork-safe, no frame extraction.

```python
from frameforge import VideoReader
from frameforge.sampling import EpisodeSampler
from frameforge.dataloader import VideoClipDataset

dataset = VideoClipDataset(
    video_paths=episode_files,
    sampler=EpisodeSampler(boundaries=[0, 3000, 6000], num_frames=16),
)
loader = DataLoader(dataset, batch_size=32, num_workers=8)  # just works
```

## The problem

You have 10,000 manipulation episode MP4s. You need a PyTorch DataLoader that:

1. **Samples 16-frame clips within episode boundaries** — not across two different tasks
2. **Works with `num_workers > 0`** — no silent corruption from forked file handles
3. **Doesn't require extracting every frame to JPEG** — your dataset is 2 TB as video, 18 TB as images

Every robotics lab writes 200 lines of custom code for this. frameforge is those 200 lines, tested and reusable.

## Install

```bash
pip install frameforge[pyav]
```

## What it does

### Episode-aware sampling

```python
from frameforge.sampling import EpisodeSampler

# Never samples across task boundaries
sampler = EpisodeSampler(episode_boundaries=[0, 3000, 6000], num_frames=16)
clip = sampler.sample(reader)  # all 16 frames from the same episode
```

Also: `UniformSampler` (eval), `RandomSampler` (augmentation), `DenseSampler` (every Nth frame).

### Fork-safe DataLoader

```python
from frameforge.dataloader import VideoClipDataset

dataset = VideoClipDataset(
    video_paths=["ep1.mp4", "ep2.mp4", "ep3.mp4"],
    sampler=EpisodeSampler(boundaries, num_frames=16),
    backend="pyav",
    transform=lambda x: x.float() / 255.0,
)

# This won't crash. Readers open inside __getitem__, not __init__.
loader = DataLoader(dataset, batch_size=8, num_workers=8)
```

### Multi-camera sync

```python
from frameforge.sampling.sync import sync_readers_by_timestamp

readers = {
    "wrist": VideoReader("wrist.mp4"),
    "overhead": VideoReader("overhead.mp4"),
}
synced = sync_readers_by_timestamp(readers, timestamps_sec=[0.0, 0.5, 1.0])
# synced["wrist"] → (3, H, W, C), synced["overhead"] → (3, H, W, C)
```

### Read from any backend

```python
reader = VideoReader("episode.mp4")                                    # auto-select
reader = VideoReader("episode.mp4", backend="pyav")                    # explicit
reader = VideoReader("episode.mp4", backend="torchcodec", device="cuda:0")  # GPU

frame = reader[0]              # single frame → (H, W, C)
clip = reader[100:116]         # slice → (16, H, W, C)
frames = reader[[0, 50, 100]]  # fancy index → (3, H, W, C)
```

| Backend | GPU | Seeking | Install |
|---------|-----|---------|---------|
| PyAV | No | Accurate | `pip install av` |
| torchcodec | NVDEC | Accurate | `pip install torchcodec` |
| decord | NVDEC | Good | `pip install decord` |
| OpenCV | No | Approximate | `pip install opencv-python` |

## Performance

```python
# Disk-cached keyframe index — faster seeks on large-GOP videos
reader = VideoReader("episode.mp4", cache_index=True)

# Background decode-ahead — prefetches sequential frames in a separate thread
reader = VideoReader("episode.mp4", prefetch=True)

# Both — best for DataLoader workloads
reader = VideoReader("episode.mp4", cache_index=True, prefetch=True)
```

### Pre-pack shards for maximum throughput

For large-scale training: extract clips offline into sequential shard videos. Eliminates file-open and seek overhead.

```python
from frameforge.packing import PackConfig, pack_shards, ShardStreamDataset

pack_shards(PackConfig(
    episode_paths=all_episodes,
    output_dir="shards/v1/",
    clip_length=16,
    seed=42,
))

dataset = ShardStreamDataset("shards/v1/", prefetch=True)
loader = DataLoader(dataset, batch_size=256, num_workers=4)
```

## Who this is for

- **Training manipulation policies** (ACT, Diffusion Policy, VLAs) from episode MP4s
- **Multi-camera robotics rigs** that need synchronized frame reads
- **Teams with existing video datasets** (not in LeRobot/RLDS format) who don't want to extract frames
- **Anyone who's debugged fork-safety crashes** in video DataLoaders

## Roadmap

- [ ] PyNvVideoCodec backend (direct NVDEC, zero-copy GPU decode)
- [ ] LeRobot dataset format integration
- [ ] WebDataset / tar-based sharding for distributed training
- [ ] Ray Data source for multi-node decode
