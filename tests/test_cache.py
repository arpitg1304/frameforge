"""Tests for frame index cache and prefetch decode-ahead."""

from pathlib import Path

import pytest
import torch

from frameforge.cache import VideoFrameIndex
from frameforge.reader import VideoReader

NUM_FRAMES = 60
HEIGHT = 240
WIDTH = 320


class TestVideoFrameIndex:
    def test_build(self, synthetic_video: Path) -> None:
        index = VideoFrameIndex.build(synthetic_video)
        assert index.num_frames == NUM_FRAMES
        assert index.num_keyframes > 0
        assert index.avg_gop_size > 0

    def test_keyframe_pts(self, synthetic_video: Path) -> None:
        index = VideoFrameIndex.build(synthetic_video)
        # Frame 0 should be a keyframe
        assert index.is_keyframe(0)
        # Every frame should have a valid keyframe PTS
        for i in range(index.num_frames):
            kf_pts = index.keyframe_pts_for(i)
            assert kf_pts >= 0
            assert kf_pts <= index.frame_pts[i]

    def test_save_and_load(self, synthetic_video: Path, tmp_path: Path) -> None:
        index = VideoFrameIndex.build(synthetic_video)
        save_path = tmp_path / "test.fidx"
        index.save(save_path)
        assert save_path.exists()

        loaded = VideoFrameIndex.load(synthetic_video)
        # Will be None because the cache path is hash-based and we saved to tmp
        # So test explicit load by building and saving to the default location
        index2 = VideoFrameIndex.build_or_load(synthetic_video)
        assert index2.num_frames == index.num_frames

    def test_build_or_load_caches(self, synthetic_video: Path) -> None:
        # First call builds and saves
        index1 = VideoFrameIndex.build_or_load(synthetic_video)
        # Second call loads from cache
        index2 = VideoFrameIndex.build_or_load(synthetic_video)
        assert index1.num_frames == index2.num_frames
        assert (index1.frame_pts == index2.frame_pts).all()

    def test_frames_to_decode(self, synthetic_video: Path) -> None:
        index = VideoFrameIndex.build(synthetic_video)
        # Keyframe should require 1 frame to decode
        assert index.frames_to_decode(0) == 1
        # Non-keyframe should require > 1
        # Find a non-keyframe
        for i in range(1, index.num_frames):
            if not index.is_keyframe(i):
                assert index.frames_to_decode(i) > 1
                break


class TestReaderWithCacheIndex:
    def test_cache_index_flag(self, synthetic_video: Path) -> None:
        with VideoReader(synthetic_video, backend="pyav", cache_index=True) as r:
            frame = r[0]
            assert frame.shape == (HEIGHT, WIDTH, 3)
            assert r.frame_index is not None
            assert r.frame_index.num_frames == NUM_FRAMES

    def test_reads_match_without_cache(self, synthetic_video: Path) -> None:
        with VideoReader(synthetic_video, backend="pyav", cache_index=True) as r_cached:
            with VideoReader(synthetic_video, backend="pyav") as r_plain:
                for idx in [0, 10, 30, 59]:
                    f1 = r_cached[idx]
                    f2 = r_plain[idx]
                    assert torch.equal(f1, f2), f"Frame {idx} mismatch"


class TestReaderWithPrefetch:
    def test_prefetch_flag(self, synthetic_video: Path) -> None:
        with VideoReader(synthetic_video, backend="pyav", prefetch=True) as r:
            frame = r[0]
            assert frame.shape == (HEIGHT, WIDTH, 3)
            stats = r.prefetch_stats
            assert stats is not None
            assert stats["cached_frames"] >= 1

    def test_sequential_read(self, synthetic_video: Path) -> None:
        with VideoReader(synthetic_video, backend="pyav", prefetch=True, prefetch_ahead=8) as r:
            frames = []
            for i in range(20):
                frames.append(r[i])
            assert len(frames) == 20
            assert all(f.shape == (HEIGHT, WIDTH, 3) for f in frames)

    def test_clip_read(self, synthetic_video: Path) -> None:
        with VideoReader(synthetic_video, backend="pyav", prefetch=True) as r:
            clip = r[0:16]
            assert clip.shape == (16, HEIGHT, WIDTH, 3)

    def test_reads_match_without_prefetch(self, synthetic_video: Path) -> None:
        with VideoReader(synthetic_video, backend="pyav", prefetch=True) as r_pf:
            with VideoReader(synthetic_video, backend="pyav") as r_plain:
                for idx in [0, 10, 30, 59]:
                    f1 = r_pf[idx]
                    f2 = r_plain[idx]
                    assert torch.equal(f1, f2), f"Frame {idx} mismatch"

    def test_prefetch_and_cache_together(self, synthetic_video: Path) -> None:
        with VideoReader(
            synthetic_video, backend="pyav", cache_index=True, prefetch=True
        ) as r:
            clip = r[10:26]
            assert clip.shape == (16, HEIGHT, WIDTH, 3)
            assert r.frame_index is not None
            assert r.prefetch_stats["cached_frames"] >= 1


class TestPickleWithFeatures:
    def test_pickle_with_cache_and_prefetch(self, synthetic_video: Path) -> None:
        import pickle

        reader = VideoReader(
            synthetic_video, backend="pyav", cache_index=True, prefetch=True
        )
        _ = reader[0]

        data = pickle.dumps(reader)
        reader2 = pickle.loads(data)

        # Backend, prefetcher, and index should be None after unpickling
        assert reader2._backend is None
        assert reader2._prefetcher is None
        assert reader2._frame_index is None
        # But flags should persist
        assert reader2._cache_index is True
        assert reader2._prefetch is True

        # Should work after unpickling — lazy reinit
        frame = reader2[0]
        assert frame.shape == (HEIGHT, WIDTH, 3)

        reader.close()
        reader2.close()
