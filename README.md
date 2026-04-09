# frameforge

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/arpitg1304/frameforge/blob/main/notebooks/getting_started_colab.ipynb)

Train VLAs directly from compressed video. No frame extraction, no storage blowup, no fork-safety crashes.

```python
from frameforge import VideoReader

reader = VideoReader("episode_001.mp4")  # auto-selects best backend
clip = reader[100:116]                   # (16, H, W, C) uint8 tensor — that's it
```

## Why frameforge?

**The storage problem**: A 1000-hour manipulation dataset stored as extracted JPEG frames takes **~18 TB**. The same data as compressed H.264 video: **~2 TB**. That's a 10x difference, and it only grows with multi-camera rigs.

**The fragmentation problem**: You want to train from video, but PyAV, decord, torchcodec, and OpenCV all have different APIs, tensor formats, and seeking behavior. Switch backends? Rewrite your pipeline. Fork workers? Silent data corruption.

**The robotics problem**: Generic video loaders don't know about episode boundaries. Your manipulation policy sees 8 frames of "pick up cup" followed by 8 frames of "open drawer" in the same training clip. Garbage in, garbage out.

frameforge fixes all three:

| | |
|---|---|
| **10-30x less storage** | Train directly from compressed MP4, no frame extraction |
| **`reader[100:116]`** | Three lines from video file to training tensor |
| **Swap backends in one arg** | `backend="pyav"` → `backend="torchcodec"` — zero code changes |
| **No fork-safety bugs** | Lazy init + pickle-safe — workers just work |
| **Episode-aware sampling** | `EpisodeSampler` never crosses task boundaries |
| **Multi-camera sync** | Time-aligned reads across wrist / overhead / third-person |
| **89x prefetch speedup** | Background decode-ahead with LRU cache for sequential reads |
| **Built for VLAs** | Designed for RT-2, Octo, OpenVLA, pi0, LeRobot-style pipelines |

## Install

```bash
pip install frameforge                    # core (numpy + torch)
pip install frameforge[pyav]              # + PyAV backend (recommended)
pip install frameforge[torchcodec]        # + torchcodec (GPU decode)
pip install frameforge[decord]            # + decord
pip install frameforge[all]               # everything
pip install frameforge[dev]               # + pytest for development
```

## Quick start

```python
from frameforge import VideoReader

# Auto-selects the best available backend
reader = VideoReader("episode_001.mp4")

# Pythonic indexing
frame = reader[0]                # single frame → (H, W, C) tensor
clip = reader[100:116]           # slice → (16, H, W, C) tensor
selected = reader[[0, 50, 100]]  # fancy index → (3, H, W, C) tensor

print(reader.metadata)
# {'fps': 30.0, 'num_frames': 9000, 'duration_sec': 300.0,
#  'width': 640, 'height': 480, 'codec': 'h264'}
```

### Swap backends

```python
reader = VideoReader("episode_001.mp4", backend="decord")
reader = VideoReader("episode_001.mp4", backend="torchcodec", device="cuda:0")
```

### Clip sampling

```python
from frameforge.sampling import UniformSampler, EpisodeSampler

sampler = UniformSampler(num_frames=16)
clip = sampler.sample(reader)  # (16, H, W, C)

# Robotics: sample within episode boundaries
ep_sampler = EpisodeSampler(episode_boundaries=[0, 3000, 6000], num_frames=16)
clip = ep_sampler.sample(reader)
```

### DataLoader integration

```python
from torch.utils.data import DataLoader
from frameforge.dataloader import VideoClipDataset
from frameforge.sampling import UniformSampler

dataset = VideoClipDataset(
    video_paths=["ep1.mp4", "ep2.mp4", "ep3.mp4"],
    sampler=UniformSampler(16),
    backend="pyav",
)

loader = DataLoader(dataset, batch_size=8, num_workers=4)
```

### Performance: frame index cache + prefetch

```python
# 2-3x faster random seeks — scans keyframes once, caches to disk
reader = VideoReader("episode.mp4", cache_index=True)

# 2-4x faster sequential reads — background thread decodes ahead
reader = VideoReader("episode.mp4", prefetch=True)

# Both together — best for DataLoader workloads
reader = VideoReader("episode.mp4", cache_index=True, prefetch=True,
                     prefetch_ahead=32, prefetch_cache_size=256)

# Check prefetch stats
print(reader.prefetch_stats)
# {'cached_frames': 42, 'cache_size_limit': 256, 'prefetch_ahead': 32, 'cache_memory_mb': 12.3}

# Frame index is also useful standalone
from frameforge import VideoFrameIndex
index = VideoFrameIndex.build_or_load("episode.mp4")  # scans once, caches to .frameforge_cache/
print(f"{index.num_keyframes} keyframes, avg GOP {index.avg_gop_size:.0f}")
print(f"Frame 500 needs {index.frames_to_decode(500)} decodes from nearest keyframe")
```

### Run benchmarks

```bash
# Single backend
python -m frameforge.benchmark run --backend pyav --video sample.mp4

# Generate HTML report
python -m frameforge.benchmark report --results-dir benchmarks/results --output docs/benchmarks.html
```

## Backend comparison

| Backend | GPU Decode | Seeking | Thread-safe | Install |
|---------|-----------|---------|-------------|---------|
| PyAV | No | Accurate (slow for random) | Yes | `pip install av` |
| torchcodec | Yes (NVDEC) | Accurate | Yes | `pip install torchcodec` |
| decord | Yes (NVDEC) | Good (VFR issues) | No (use workers) | `pip install decord` |
| OpenCV | No | Approximate | Yes | `pip install opencv-python` |

## Where frameforge fits in the data flywheel

```
Collect → Encode → Store → Curate → Train → Evaluate → Deploy
           ▲                 ▲        ▲▲▲       ▲
           └─ VideoWriter    │    primary use    └─ VideoReader
                             └─ VideoReader         (rollout analysis)
                                (metadata scan,
                                 clip extraction)
```

frameforge is the **decode/load/sample layer** between storage and training. It doesn't replace your recording stack (ROS, RealSense SDK), your data platform (HuggingFace, LeRobot), or your inference runtime (ONNX, TensorRT).

### What it covers

| Stage | What frameforge does |
|-------|---------------------|
| **Training data loading** | Primary use case. Random access into compressed video, episode-aware sampling, worker-safe DataLoaders, backend benchmarking. Replaces pre-extracting frames to JPEG (10-30x storage savings). |
| **Data curation** | Scan metadata across 1000s of videos. Verify frame counts match annotations. Extract clips by index/timestamp. Multi-camera alignment checks. |
| **Post-collection encoding** | Re-encode with ML-friendly settings (small GOP, no B-frames, constant framerate). Downscale or transcode datasets. |
| **Offline evaluation** | Read specific rollout frames, compare predictions frame-by-frame, write prediction overlays back to video. |
| **Pipeline benchmarking** | Measure throughput/latency/memory per backend on your hardware before committing to a large training run. |

### What it doesn't cover

| Stage | Use instead |
|-------|-------------|
| Live camera capture | ROS2 image transport, GStreamer, RealSense SDK |
| On-robot inference | ONNX Runtime, TensorRT, custom C++ |
| Distributed training | NVIDIA DALI, Ray Data, WebDataset |
| Annotation / labeling | Label Studio, CVAT, Rerun |
| Dataset hosting | HuggingFace Datasets, LeRobot, Open X-Embodiment |

### Robotics-specific features

- **Egocentric video** — sample clips from wearable camera recordings with high motion and variable lighting
- **Manipulation demos** — `EpisodeSampler` respects task boundaries so you never sample across episodes
- **Multi-camera rigs** — `sync_readers_by_timestamp` for time-aligned reads across wrist/overhead/third-person cameras
- **Large-scale pretraining** — benchmark and pick the fastest decode backend for your GPU

## Landscape — how frameforge compares

Every robotics ML team needs video decode. Here's what exists and where frameforge fits.

| Library | What it is | Relationship to frameforge |
|---|---|---|
| **[torchcodec](https://github.com/meta-pytorch/torchcodec)** (~1K stars) | PyTorch's official low-level decoder. CUDA GPU decode. Actively maintained by Meta. Replacing torchvision IO. | **One of our backends.** frameforge wraps torchcodec and adds sampling, episodes, sync, fork safety on top. |
| **[decord](https://github.com/dmlc/decord)** (~2.5K stars) | C++ video loader. Was the ML standard. **Abandoned** — no commits in 3 years, 200+ open issues, segfaults in workers. | **One of our backends.** frameforge fixes decord's fork-safety issues via lazy init. Makes migration to torchcodec trivial. |
| **[NVIDIA DALI](https://github.com/NVIDIA/DALI)** (~5.7K stars) | GPU-accelerated data loading + augmentation pipeline. Linux/CUDA only. | **Complementary.** DALI handles GPU augmentation; frameforge handles robotics-aware decode/sample. Integration planned. |
| **[pytorchvideo](https://github.com/facebookresearch/pytorchvideo)** (~3.6K stars) | Model zoo + transforms for action recognition. **Dead** — no commits since 2023. | Different layer. pytorchvideo is models + transforms, not decode. |
| **torchvision IO** (part of torchvision) | `read_video()`, `VideoReader`. **Deprecated** as of v0.22, removal in v0.24. | frameforge replaces this with proper seeking, caching, and fork safety. |
| **[LeRobot](https://github.com/huggingface/lerobot)** (~23K stars) | Full robotics platform. Stores data as MP4. Uses torchcodec/pyav internally. | **Key integration target.** LeRobot's video decode layer is minimal. frameforge could serve as a faster, more capable decode backend. |
| **[Robo-DM](https://github.com/BerkeleyAutomation/robodm)** (~143 stars) | Cloud data management for robot datasets. EBML container, multi-format. | **Adjacent.** Robo-DM stores and manages; frameforge decodes and loads for training. |

### What no other library does

|  | frameforge | torchcodec | decord | DALI | LeRobot |
|---|:---:|:---:|:---:|:---:|:---:|
| Episode-aware sampling | **Yes** | No | No | No | At dataset level |
| Multi-camera sync | **Yes** | No | No | No | At format level |
| Pluggable backends | **Yes** (4) | Single | Single | Single | Hardcoded fallback |
| Fork-safe (lazy init + pickle) | **Yes** | Works in workers | **No** (segfaults) | Yes (own engine) | No special handling |
| Disk-cached frame index | **Yes** | In-memory only | In-memory only | Internal | No |
| Prefetch decode-ahead | **Yes** | No | No | Built-in | No |
| Decode benchmarking suite | **Yes** | Internal only | No | Profiling | No |
| Robotics-specific | **Yes** | No | No | No | **Yes** |

## Project structure

```
frameforge/
  backends/       # Pluggable decode backends (PyAV, torchcodec, decord, OpenCV)
  sampling/       # Temporal samplers + multi-camera sync
  dataloader/     # PyTorch Dataset/IterableDataset wrappers
  benchmark/      # Runner, configs, HTML report generator
notebooks/
  getting_started.ipynb  # Interactive walkthrough with widgets
docs/
  index.html             # Project landing page
  video-compression.html # Video formats knowledge base
  benchmarks.html        # Benchmark results
  architecture.md        # System design diagrams
```

## Roadmap

### Near-term
- [ ] **PyNvVideoCodec backend** — direct NVDEC/NVENC via NVIDIA Video Codec SDK, zero-copy GPU decode without torchcodec/decord overhead
- [ ] **LeRobot integration** — native frameforge backend for LeRobot's video dataset format, the emerging standard for robotics datasets
- [ ] **WebDataset / tar-based video sharding** — store videos in tar archives with index files for efficient distributed I/O

### Medium-term
- [ ] **Ray Data source** — distributed decode across a cluster for large-scale pretraining
- [ ] **NVIDIA DALI integration** — plug frameforge samplers into DALI pipelines for GPU-accelerated augmentation
- [ ] **Video-language model preprocessing** — frame-caption pair extraction, interleaved video-text batching
- [ ] **Streaming decode** — chunked HTTP range requests for reading video directly from S3/GCS without full download

### Long-term
- [ ] **Open X-Embodiment loader** — first-class support for the OXE dataset format (multi-robot, multi-task)
- [ ] **Hardware encode profiles** — preset encoding configs per camera model (RealSense D435, ZED 2, GoPro) with optimal GOP/bitrate/chroma
- [ ] **Annotation overlay engine** — render bounding boxes, keypoints, action labels directly onto decoded frames for debugging
