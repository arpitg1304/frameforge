# Data Packing — Design Document

## The Problem

Training VLA models from compressed video is bottlenecked by **random I/O**:

```
256 clips from 256 episode files
→ 256 file opens + 256 random seeks + 256 independent decodes
→ GPU idle 80% of the time
```

Sequential reads from a single file are 4-20x faster because the decoder stays warm, the OS prefetch buffer works, and there's zero seek overhead. The idea: **pre-pack selected clips into sequential shard videos** as an offline preprocessing step.

## Core Concept

```
Source episodes (raw data, kept as-is):
  episode_0001.mp4  →  300 frames, "pick up red cup"
  episode_0002.mp4  →  450 frames, "place cup on shelf"
  episode_0003.mp4  →  270 frames, "open drawer"
  ...

Packing (offline, one-time):
  Sample clips from episodes → re-encode sequentially into shard videos

Packed shards (training-optimized):
  shards/v1/shard_0000.mp4  →  [clip][clip][clip]...[clip]  (1000 clips)
  shards/v1/shard_0001.mp4  →  [clip][clip][clip]...[clip]
  shards/v1/manifest.json   →  clip-level metadata
```

Training reads shards sequentially. No seeking. Maximum throughput.

---

## Design Decisions

### 1. What goes into a clip?

A clip is a contiguous sequence of T frames from a single episode. The key parameters:

| Parameter | Description | Default | Why |
|-----------|-------------|---------|-----|
| `clip_length` | Frames per clip | 16 | Matches typical VLA input (ACT=16, Diffusion Policy=16, pi0=16-64) |
| `clip_stride` | Step between clip starts within an episode | `clip_length` (non-overlapping) | Overlapping clips (`stride < clip_length`) increase data but add redundancy |
| `clips_per_episode` | Max clips to extract per episode | `None` (all possible) | Cap this for very long episodes to avoid dataset imbalance |
| `sampler` | How clips are selected | `DenseSampler(stride=clip_length)` | Use `RandomSampler` for augmentation, `DenseSampler` for full coverage |
| `skip_short_episodes` | Skip episodes shorter than clip_length | `True` | Avoids padding edge cases |

**Non-overlapping dense (default)**:
```
Episode: [----clip 0----][----clip 1----][----clip 2----][--remainder--]
                                                          ↑ dropped
```

**Overlapping (stride < clip_length)**:
```
Episode: [----clip 0----]
              [----clip 1----]
                    [----clip 2----]
```

**Random sampling (N clips per episode)**:
```
Episode: ........[--clip--]......[--clip--]...[--clip--]..........
         random start positions, sorted by start frame
```

### 2. How are clips arranged in shards?

**Within a shard**: clips are written sequentially, one after another. Each clip starts on a keyframe (GOP=1 within the shard). No seeking needed — just read forward.

**Across shards**: clips are shuffled across episodes before being distributed to shards. Shard 0 doesn't contain only episode 0's clips — it has a random mix. This is critical for training because sequential reading of a shard should produce a diverse batch.

**Shard sizing options**:

| Strategy | Parameter | Tradeoff |
|----------|-----------|----------|
| By clip count | `clips_per_shard=1000` | Predictable shard count, variable file size |
| By file size | `max_shard_size_mb=500` | Predictable storage, variable clip count |
| By duration | `max_shard_duration_sec=300` | Predictable read time per shard |

Default: `clips_per_shard=1000` (simplest, predictable).

### 3. Multi-camera handling

Robotics rigs have multiple cameras. Clips from different cameras at the same timestep must stay together.

**Option A: Interleaved frames** (single shard per multi-cam clip)
```
shard.mp4: [wrist_f0][overhead_f0][wrist_f1][overhead_f1]...
```
Bad. Mixes resolutions, breaks standard decoders.

**Option B: Parallel shards** (one shard per camera, same clip order)
```
shards/v1/shard_0000_wrist.mp4:     [clip_a_wrist][clip_b_wrist]...
shards/v1/shard_0000_overhead.mp4:  [clip_a_overhead][clip_b_overhead]...
```
Good. Standard video files. Same clip at same index across camera shards. DataLoader reads from both in lockstep.

**Option C: Side-by-side tiling** (stitch cameras into one wide frame)
```
shard.mp4: [wrist|overhead] per frame, 2x width
```
Simple but inflexible. Couples camera resolutions. Can't add/remove cameras without re-packing.

**Decision: Option B (parallel shards).** Each camera gets its own shard set with identical clip ordering. The manifest maps clip index → (episode, start_frame, camera) across all camera shards.

### 4. What metadata goes in the manifest?

The manifest is the critical piece — it maps every clip in every shard to its source episode, frame range, and paired data (actions, language).

```json
{
  "version": 2,
  "pack_config": {
    "clip_length": 16,
    "clip_stride": 16,
    "clips_per_shard": 1000,
    "codec": "h264",
    "gop": 1,
    "resolution": [640, 480],
    "fps": 30,
    "seed": 42,
    "cameras": ["wrist", "overhead"],
    "created_at": "2026-04-08T10:30:00Z",
    "frameforge_version": "0.1.0",
    "source_hash": "sha256:abc123..."
  },
  "shards": [
    {
      "shard_id": 0,
      "files": {
        "wrist": "shard_0000_wrist.mp4",
        "overhead": "shard_0000_overhead.mp4"
      },
      "num_clips": 1000,
      "size_bytes": 52428800
    }
  ],
  "clips": [
    {
      "shard_id": 0,
      "clip_index": 0,
      "episode": "episode_0042",
      "episode_frame_start": 120,
      "episode_frame_end": 136,
      "language": "pick up the red cup",
      "action_labels_key": "episode_0042/actions/120:136"
    }
  ],
  "episodes": {
    "episode_0042": {
      "source_path": "data/episode_0042.mp4",
      "num_frames": 300,
      "fps": 30,
      "task": "pick_and_place"
    }
  }
}
```

Key fields:
- **`source_hash`**: SHA-256 of all source episode paths + sizes. Detects if source data changed since packing.
- **`seed`**: Random seed used for shuffling. Same seed + same source = identical shards. Reproducible.
- **`action_labels_key`**: Pointer to where action labels live (HDF5 path, Parquet row range, etc.). frameforge doesn't load actions itself — it provides the key so the user's Dataset can fetch them.
- **`episode_frame_start/end`**: Exact frame range from the source episode, for provenance.

### 5. Re-packing: what happens when you need to do it again?

**Scenarios that trigger re-packing**:
- New episodes added to the dataset
- Different clip length needed (switched from 16 to 64 for a new model)
- Different sampling strategy (dense → random with augmentation)
- Different resolution (downscale from 1080p to 480p)
- Bug found in an episode (need to exclude it)

**Design: versioned pack directories**

```
shards/
  v1/          ← first packing run
    manifest.json
    shard_0000_wrist.mp4
    shard_0000_overhead.mp4
    ...
  v2/          ← re-packed with different settings
    manifest.json
    shard_0000_wrist.mp4
    ...
```

Each packing run creates a new version directory. Old versions are kept (disk is cheap, re-encoding is expensive). Training config points to a specific version:

```python
dataset = ShardDataset("shards/v2/")
```

**The packer never modifies existing shards.** It always creates a new version directory. This is critical:
- Training runs are reproducible (point to v1, always get v1)
- No risk of corrupting data mid-training
- Can A/B test different packing strategies

### 6. Incremental packing (new episodes added)

When 500 new episodes arrive, you don't want to re-pack the entire 50K-episode dataset.

**Option A: Re-pack everything** — simplest, most shuffled, but expensive.

**Option B: Append-only shards** — pack only new episodes into new shards, add to manifest.
```
shards/v1/
  shard_0000.mp4  ← original 50K episodes
  shard_0049.mp4
  shard_0050.mp4  ← new 500 episodes (appended)
  shard_0050.mp4
  manifest.json   ← updated with new entries
```
Pro: Fast (only encode new data). Con: New shards aren't shuffled with old ones — later shards are all from new episodes.

**Option C: Manifest merge** — pack new episodes as a separate version, then create a merged manifest that references both.
```
shards/v1/       ← original
shards/v1_add1/  ← 500 new episodes
shards/v1_merged/
  manifest.json  ← points to shards in both v1/ and v1_add1/
```
Pro: No re-encoding, full shuffle at manifest level. Con: More complex file layout.

**Decision**: Support both A and C. Default is full re-pack (A) for simplicity. Advanced users can use manifest merge (C) for large datasets. The `pack` CLI should have `--incremental` flag.

### 7. Encoding settings for shard videos

Shards are training-optimized, not storage-optimized. Different priorities than source episodes:

| Setting | Source episodes | Packed shards | Why |
|---------|----------------|---------------|-----|
| GOP | 30-60 (compromise) | **1 (all-intra)** | Every clip starts on a keyframe. Zero seek cost. |
| B-frames | 0 (recommended) | **0** | No reordering complexity |
| CRF | 18 (near-lossless) | **Configurable, default 18** | User choice: quality vs size |
| Codec | h264 (compatible) | **h264** (or h265/av1 if user chooses) | h264 for max decode speed |
| Pixel format | yuv420p | **Same as source** | No unnecessary color conversion |
| Resolution | Source native | **Configurable** | Downscale at pack time saves decode work at training time |
| FPS | Source native | **Source or configurable** | Temporal downsampling can happen here too |

**Key insight**: GOP=1 makes shard files ~3-5x larger per frame than GOP=30. But shards only contain the clips you'll actually train on, not full episodes. If you extract 10% of frames from source episodes, shard storage is often *comparable* to source storage despite all-intra encoding.

Storage math example:
```
Source: 50K episodes × 300 frames avg × 1080p × CRF 18 × GOP 30
      = ~4 TB

Shards: 50K episodes × 30 clips avg × 16 frames × 1080p × CRF 18 × GOP 1
      = 24M frames (vs 15M source frames, but all-intra so ~4x per frame)
      = ~5 TB

Shards at 480p: same clips, downscaled 4x
      = ~1.3 TB  ← SMALLER than source
```

Downscaling at pack time is often the right move — your model sees 224x224 anyway.

### 8. Shuffling strategy

**Cross-episode shuffling** happens at packing time: clips from all episodes are mixed before distributing to shards.

**Cross-shard shuffling** happens at training time: DataLoader workers each read a different shard, and batch construction interleaves across workers.

**Epoch shuffling**: Between epochs, shuffle the shard order. Each worker gets a different shard sequence each epoch.

```python
class ShardDataset(IterableDataset):
    def __iter__(self):
        worker_info = get_worker_info()
        # Shuffle shard order per epoch
        shards = self.shards.copy()
        random.shuffle(shards)
        # Distribute shards to workers
        my_shards = shards[worker_info.id::worker_info.num_workers]
        for shard in my_shards:
            reader = VideoReader(shard.path, prefetch=True)
            for clip_start in range(0, shard.num_clips * clip_length, clip_length):
                yield reader[clip_start:clip_start + clip_length]
```

**Within-shard shuffling**: NOT done at read time (would require seeking, defeats the purpose). Shuffling within a shard happens at pack time. If you need different within-shard order, re-pack with a different seed.

### 9. Action labels and language pairing

frameforge packs video clips. But training needs action labels and language instructions paired with each clip. Two approaches:

**Option A: Metadata-only manifest** (recommended)
The manifest records `episode_id` + `frame_range` for each clip. The user's Dataset class uses this to look up actions from their own storage (HDF5, Parquet, Zarr, etc.).

```python
class VLAShardDataset(ShardDataset):
    def __init__(self, shard_dir, action_store):
        super().__init__(shard_dir)
        self.action_store = action_store  # user's HDF5/Parquet

    def __iter__(self):
        for clip, clip_meta in super().__iter__():
            actions = self.action_store.get(
                clip_meta["episode"],
                clip_meta["episode_frame_start"],
                clip_meta["episode_frame_end"],
            )
            yield clip, actions, clip_meta["language"]
```

**Option B: Co-packed actions** (advanced)
Pack action tensors alongside video in a sidecar file:
```
shard_0000_wrist.mp4      ← video
shard_0000_actions.npz    ← (N_clips, T, action_dim) array
shard_0000_language.json  ← list of N_clips strings
```

Pro: Single I/O path, no separate action store needed. Con: More complex packing, assumes action format.

**Decision**: Option A by default (manifest-only). Option B as an opt-in (`--pack-actions` flag). Keep frameforge focused on video; let the user handle action/language storage.

### 10. Fault tolerance

Packing 50K episodes takes hours. What if it crashes at episode 30K?

- **Checkpoint**: Write a progress file (`pack_progress.json`) after each shard is completed. On restart, skip completed shards.
- **Atomic shard writes**: Write to `shard_XXXX.mp4.tmp`, rename to `shard_XXXX.mp4` on completion. Incomplete shards are always `.tmp` and can be safely deleted.
- **Validation**: After packing, optionally verify each shard is readable and clip count matches manifest (`--verify` flag).

---

## API Design

### CLI

```bash
# Basic packing
frameforge pack \
  --episodes data/episodes/ \
  --output shards/v1/ \
  --clip-length 16 \
  --clips-per-shard 1000 \
  --seed 42

# With options
frameforge pack \
  --episodes data/episodes/ \
  --output shards/v2/ \
  --clip-length 64 \
  --clip-stride 32 \
  --clips-per-episode 50 \
  --resolution 480x360 \
  --codec h264 \
  --crf 20 \
  --cameras wrist,overhead \
  --num-workers 8 \
  --seed 42

# Verify packed shards
frameforge pack verify shards/v1/

# Info about a pack
frameforge pack info shards/v1/

# Incremental (append new episodes)
frameforge pack \
  --episodes data/new_episodes/ \
  --append-to shards/v1/ \
  --output shards/v1_add1/
```

### Python API

```python
from frameforge.packing import PackConfig, pack_shards, ShardDataset

# Configure
config = PackConfig(
    episode_paths=episode_paths,          # list of video paths
    episode_boundaries=boundaries,        # optional: {path: [start_frames]}
    output_dir="shards/v1/",
    clip_length=16,
    clip_stride=16,                       # non-overlapping
    clips_per_shard=1000,
    cameras=["wrist", "overhead"],        # multi-cam: parallel shards
    codec="h264",
    gop=1,                                # all-intra (default, don't change)
    crf=18,
    resolution=(480, 360),                # downscale at pack time
    fps=None,                             # None = keep source fps
    seed=42,
    num_workers=8,
    metadata={"dataset": "rh20t", "version": "2.0"},  # user metadata
)

# Pack (offline, takes minutes to hours)
result = pack_shards(config)
print(f"Packed {result.total_clips} clips into {result.num_shards} shards")

# Train (fast sequential reads)
dataset = ShardDataset(
    shard_dir="shards/v1/",
    camera="wrist",                       # which camera to load
    transform=preprocess,
)

loader = DataLoader(dataset, batch_size=256, num_workers=4)
```

---

## What frameforge Does vs Doesn't Do

| Responsibility | frameforge packing | User code |
|---|---|---|
| Extract clips from episodes | Yes | |
| Shuffle clips across episodes | Yes | |
| Encode into shard videos | Yes | |
| Write manifest with metadata | Yes | |
| Sequential shard reading | Yes | |
| Multi-camera parallel shards | Yes | |
| Action label storage/loading | **No** — provides episode + frame keys | Yes |
| Language instruction loading | **No** — stored in manifest as strings | Yes |
| HDF5/Parquet/Zarr integration | **No** | Yes |
| Distributed training coordination | **No** | Yes (via DDP / Ray) |

---

## Open Questions

1. **Should shards support random access at all?** If each clip starts on a keyframe (GOP=1), you *can* seek to any clip. This enables map-style Dataset access (not just IterableDataset). Worth supporting for debugging and evaluation.

2. **Compression format**: H.264 all-intra is simple. But JPEG-in-MKV (Motion JPEG) or FFV1 (lossless intra-only) might be better for all-intra encoding. H.264 intra mode is not as efficient as JPEG for single frames. Need benchmarks.

3. **Frame-level vs clip-level granularity**: Should each clip be a separate seek target? Or should we just write frames sequentially and use the manifest to compute byte offsets? The former is simpler; the latter is more flexible for variable-length clips.

4. **Memory-mapped reading**: For maximum speed, could we memory-map the shard file and decode frames directly from mapped memory? This bypasses file I/O entirely. PyAV supports reading from bytes buffers.

5. **Cloud storage**: If shards are on S3/GCS, sequential reads work well (single GET with range). But shard files should be sized for efficient cloud transfers (100-500 MB each). Too small = too many requests. Too large = slow startup.
