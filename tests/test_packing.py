"""Tests for data packing: pack episodes into shard videos and read them back."""

import json
from pathlib import Path

import pytest
import torch

from frameforge.packing import PackConfig, PackResult, ShardDataset, ShardStreamDataset, pack_shards

NUM_FRAMES = 60
WIDTH = 320
HEIGHT = 240


@pytest.fixture
def episode_dir(synthetic_video: Path, synthetic_video_pair: tuple[Path, Path], tmp_path: Path) -> Path:
    """Create a directory with 3 episode videos for packing tests."""
    import shutil

    ep_dir = tmp_path / "episodes"
    ep_dir.mkdir()
    shutil.copy(synthetic_video, ep_dir / "episode_000.mp4")
    shutil.copy(synthetic_video_pair[0], ep_dir / "episode_001.mp4")
    shutil.copy(synthetic_video_pair[1], ep_dir / "episode_002.mp4")
    return ep_dir


@pytest.fixture
def packed_dir(episode_dir: Path, tmp_path: Path) -> Path:
    """Pack episodes into shards and return the shard directory."""
    out = tmp_path / "shards_v1"
    config = PackConfig(
        episode_paths=sorted(episode_dir.glob("*.mp4")),
        output_dir=out,
        clip_length=8,
        clips_per_shard=5,
        seed=42,
        backend="pyav",
    )
    pack_shards(config)
    return out


class TestPackShards:
    def test_basic_pack(self, episode_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "shards"
        config = PackConfig(
            episode_paths=sorted(episode_dir.glob("*.mp4")),
            output_dir=out,
            clip_length=8,
            clips_per_shard=10,
            seed=42,
            backend="pyav",
        )
        result = pack_shards(config)

        assert result.num_shards > 0
        assert result.total_clips > 0
        assert result.total_frames == result.total_clips * 8
        assert result.manifest_path.exists()
        assert result.total_size_bytes > 0

    def test_manifest_structure(self, packed_dir: Path) -> None:
        manifest = json.loads((packed_dir / "manifest.json").read_text())

        assert manifest["version"] == 2
        assert "pack_config" in manifest
        assert "shards" in manifest
        assert "clips" in manifest
        assert manifest["pack_config"]["clip_length"] == 8
        assert manifest["pack_config"]["seed"] == 42

    def test_shard_files_exist(self, packed_dir: Path) -> None:
        manifest = json.loads((packed_dir / "manifest.json").read_text())
        for shard_info in manifest["shards"]:
            shard_path = packed_dir / shard_info["file"]
            assert shard_path.exists(), f"Missing shard: {shard_info['file']}"

    def test_clip_provenance(self, packed_dir: Path) -> None:
        manifest = json.loads((packed_dir / "manifest.json").read_text())
        for clip in manifest["clips"]:
            assert "episode" in clip
            assert "episode_frame_start" in clip
            assert "episode_frame_end" in clip
            assert clip["episode_frame_end"] - clip["episode_frame_start"] == 8

    def test_reproducible_with_seed(self, episode_dir: Path, tmp_path: Path) -> None:
        results = []
        for i in range(2):
            out = tmp_path / f"shards_seed_{i}"
            config = PackConfig(
                episode_paths=sorted(episode_dir.glob("*.mp4")),
                output_dir=out,
                clip_length=8,
                clips_per_shard=10,
                seed=42,
                backend="pyav",
            )
            pack_shards(config)
            manifest = json.loads((out / "manifest.json").read_text())
            results.append([c["episode"] for c in manifest["clips"]])

        assert results[0] == results[1], "Same seed should produce same clip order"

    def test_different_seed(self, episode_dir: Path, tmp_path: Path) -> None:
        orders = []
        for seed in [42, 99]:
            out = tmp_path / f"shards_s{seed}"
            config = PackConfig(
                episode_paths=sorted(episode_dir.glob("*.mp4")),
                output_dir=out,
                clip_length=8,
                clips_per_shard=100,
                seed=seed,
                backend="pyav",
            )
            pack_shards(config)
            manifest = json.loads((out / "manifest.json").read_text())
            orders.append([c["episode"] for c in manifest["clips"]])

        # Different seeds should (very likely) produce different order
        assert orders[0] != orders[1]

    def test_custom_clip_stride(self, episode_dir: Path, tmp_path: Path) -> None:
        # Overlapping clips with stride=4 and length=8
        out = tmp_path / "shards_overlap"
        config = PackConfig(
            episode_paths=sorted(episode_dir.glob("*.mp4")),
            output_dir=out,
            clip_length=8,
            clip_stride=4,
            clips_per_shard=100,
            seed=42,
            backend="pyav",
        )
        result = pack_shards(config)
        # Overlapping should produce more clips than non-overlapping
        assert result.total_clips > 0

    def test_clips_per_episode_cap(self, episode_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "shards_capped"
        config = PackConfig(
            episode_paths=sorted(episode_dir.glob("*.mp4")),
            output_dir=out,
            clip_length=8,
            clips_per_episode=2,
            clips_per_shard=100,
            seed=42,
            backend="pyav",
        )
        result = pack_shards(config)
        # 3 episodes × 2 clips each = 6 clips max
        assert result.total_clips <= 6

    def test_codec_mjpeg(self, episode_dir: Path, tmp_path: Path) -> None:
        out = tmp_path / "shards_mjpeg"
        config = PackConfig(
            episode_paths=sorted(episode_dir.glob("*.mp4")),
            output_dir=out,
            clip_length=8,
            clips_per_shard=5,
            codec="mjpeg",
            seed=42,
            backend="pyav",
        )
        result = pack_shards(config)
        assert result.num_shards > 0


class TestShardDataset:
    def test_len(self, packed_dir: Path) -> None:
        ds = ShardDataset(packed_dir, backend="pyav")
        manifest = json.loads((packed_dir / "manifest.json").read_text())
        assert len(ds) == len(manifest["clips"])
        ds.close()

    def test_getitem(self, packed_dir: Path) -> None:
        ds = ShardDataset(packed_dir, backend="pyav")
        clip = ds[0]
        assert isinstance(clip, torch.Tensor)
        assert clip.shape == (8, HEIGHT, WIDTH, 3)
        assert clip.dtype == torch.uint8
        ds.close()

    def test_all_clips_readable(self, packed_dir: Path) -> None:
        ds = ShardDataset(packed_dir, backend="pyav")
        for i in range(len(ds)):
            clip = ds[i]
            assert clip.shape[0] == 8
        ds.close()

    def test_with_transform(self, packed_dir: Path) -> None:
        ds = ShardDataset(
            packed_dir,
            backend="pyav",
            transform=lambda x: x.float() / 255.0,
        )
        clip = ds[0]
        assert clip.dtype == torch.float32
        assert clip.max() <= 1.0
        ds.close()

    def test_clip_metadata(self, packed_dir: Path) -> None:
        ds = ShardDataset(packed_dir, backend="pyav")
        meta = ds.get_clip_metadata(0)
        assert "episode" in meta
        assert "episode_frame_start" in meta
        ds.close()


class TestShardStreamDataset:
    def test_iteration(self, packed_dir: Path) -> None:
        ds = ShardStreamDataset(packed_dir, backend="pyav", prefetch=False)
        clips = list(ds)
        assert len(clips) > 0
        assert all(c.shape == (8, HEIGHT, WIDTH, 3) for c in clips)

    def test_iteration_count_matches_manifest(self, packed_dir: Path) -> None:
        manifest = json.loads((packed_dir / "manifest.json").read_text())
        ds = ShardStreamDataset(packed_dir, backend="pyav", prefetch=False)
        clips = list(ds)
        assert len(clips) == len(manifest["clips"])

    def test_with_transform(self, packed_dir: Path) -> None:
        ds = ShardStreamDataset(
            packed_dir,
            backend="pyav",
            prefetch=False,
            transform=lambda x: x.float() / 255.0,
        )
        clip = next(iter(ds))
        assert clip.dtype == torch.float32

    def test_set_epoch(self, packed_dir: Path) -> None:
        ds = ShardStreamDataset(packed_dir, backend="pyav", prefetch=False)
        ds.set_epoch(0)
        clips_e0 = [c.sum().item() for c in ds]
        ds.set_epoch(1)
        clips_e1 = [c.sum().item() for c in ds]
        # Same clips, possibly different order (shard shuffle)
        assert sorted(clips_e0) == sorted(clips_e1)
