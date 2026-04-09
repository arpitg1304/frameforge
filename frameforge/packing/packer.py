"""Core packing logic: extract clips from episodes and write sequential shard videos."""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger

from frameforge.packing.config import PackConfig, PackResult
from frameforge.reader import VideoReader


@dataclass
class _ClipSpec:
    """Internal: specifies a clip to extract."""

    episode_path: str
    episode_id: str
    camera: str | None
    frame_start: int
    frame_end: int
    language: str | None = None


def _plan_clips(config: PackConfig) -> list[_ClipSpec]:
    """Plan which clips to extract from which episodes."""
    clips: list[_ClipSpec] = []

    # Determine paths per camera
    if isinstance(config.episode_paths, dict):
        # Multi-camera: use first camera to determine clip positions
        first_cam = config.cameras[0] if config.cameras else list(config.episode_paths.keys())[0]
        cam_paths = config.episode_paths
    else:
        first_cam = None
        cam_paths = {None: config.episode_paths}

    # Plan clip positions from the first (or only) camera
    ref_paths = cam_paths[first_cam]
    for path in ref_paths:
        path = Path(path)
        episode_id = path.stem

        try:
            with VideoReader(path, backend=config.backend) as reader:
                total = len(reader)
        except Exception as e:
            logger.warning("Skipping {} — {}", path, e)
            continue

        if config.skip_short_episodes and total < config.clip_length:
            logger.debug("Skipping {} — too short ({} < {})", episode_id, total, config.clip_length)
            continue

        # Generate clip start positions
        starts: list[int] = []
        pos = 0
        while pos + config.clip_length <= total:
            starts.append(pos)
            pos += config.clip_stride

        # Cap clips per episode
        if config.clips_per_episode is not None and len(starts) > config.clips_per_episode:
            rng = random.Random(config.seed + hash(episode_id))
            starts = sorted(rng.sample(starts, config.clips_per_episode))

        # Create clip specs for each camera
        cameras = config.cameras or [None]
        for start in starts:
            for cam in cameras:
                cam_path = str(cam_paths.get(cam, cam_paths.get(None, ref_paths))[ref_paths.index(path)])
                clips.append(_ClipSpec(
                    episode_path=cam_path,
                    episode_id=episode_id,
                    camera=cam,
                    frame_start=start,
                    frame_end=start + config.clip_length,
                ))

    return clips


def _compute_source_hash(config: PackConfig) -> str:
    """Hash source paths + sizes for change detection."""
    h = hashlib.sha256()
    if isinstance(config.episode_paths, dict):
        for cam, paths in sorted(config.episode_paths.items()):
            for p in sorted(paths, key=str):
                p = Path(p)
                h.update(f"{cam}:{p}:{p.stat().st_size}".encode())
    else:
        for p in sorted(config.episode_paths, key=str):
            p = Path(p)
            h.update(f"{p}:{p.stat().st_size}".encode())
    return h.hexdigest()[:16]


def _get_codec_settings(codec: str, crf: int) -> dict:
    """Return PyAV stream settings for the chosen codec."""
    if codec == "h264":
        return {
            "codec": "h264",
            "pix_fmt": "yuv420p",
            "options": {"crf": str(crf), "preset": "fast", "g": "1", "bf": "0"},
        }
    elif codec == "mjpeg":
        return {
            "codec": "mjpeg",
            "pix_fmt": "yuvj420p",
            "options": {"q:v": str(max(2, min(31, crf)))},
        }
    elif codec == "ffv1":
        return {
            "codec": "ffv1",
            "pix_fmt": "yuv420p",
            "options": {},
        }
    raise ValueError(f"Unsupported codec: {codec}")


def _write_shard(
    shard_path: Path,
    clips: list[_ClipSpec],
    config: PackConfig,
    codec_settings: dict,
) -> int:
    """Write a single shard video from a list of clip specs. Returns total frames written."""
    import av

    # Determine resolution from first clip if not configured
    if config.resolution:
        width, height = config.resolution
    else:
        with VideoReader(clips[0].episode_path, backend=config.backend) as r:
            meta = r.metadata
            width, height = meta["width"], meta["height"]

    # Determine FPS
    if config.fps:
        fps = config.fps
    else:
        with VideoReader(clips[0].episode_path, backend=config.backend) as r:
            fps = r.fps

    tmp_path = shard_path.parent / (shard_path.stem + "_tmp.mp4")
    container = av.open(str(tmp_path), mode="w")
    stream = container.add_stream(codec_settings["codec"], rate=int(round(fps)))
    stream.width = width
    stream.height = height
    stream.pix_fmt = codec_settings["pix_fmt"]
    for k, v in codec_settings["options"].items():
        stream.options[k] = v

    total_frames = 0
    current_episode = None
    reader = None

    for clip in clips:
        # Reuse reader if same episode
        if clip.episode_path != current_episode:
            if reader is not None:
                reader.close()
            reader = VideoReader(clip.episode_path, backend=config.backend)
            current_episode = clip.episode_path

        # Read clip frames
        clip_tensor = reader[clip.frame_start:clip.frame_end]

        for fi in range(clip_tensor.shape[0]):
            frame_np = clip_tensor[fi].numpy()

            # Resize if needed
            if config.resolution and (frame_np.shape[1] != width or frame_np.shape[0] != height):
                import cv2
                frame_np = cv2.resize(frame_np, (width, height), interpolation=cv2.INTER_AREA)

            av_frame = av.VideoFrame.from_ndarray(frame_np, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)
            total_frames += 1

    if reader is not None:
        reader.close()

    # Flush
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    # Atomic rename
    tmp_path.rename(shard_path)

    return total_frames


def pack_shards(config: PackConfig) -> PackResult:
    """Pack episode clips into sequential shard videos.

    This is the main entry point for data packing. It:
    1. Scans all episodes to plan clip extraction
    2. Shuffles clips across episodes (using config.seed)
    3. Distributes clips to shards
    4. Encodes each shard as a sequential video (GOP=1)
    5. Writes a manifest.json with full provenance

    Args:
        config: PackConfig with all packing parameters.

    Returns:
        PackResult with stats about the packing operation.
    """
    t0 = time.time()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    codec_settings = _get_codec_settings(config.codec, config.crf)

    # Plan clips
    logger.info("Planning clips from {} episodes ...",
                len(config.episode_paths) if not isinstance(config.episode_paths, dict)
                else len(list(config.episode_paths.values())[0]))
    all_clips = _plan_clips(config)
    logger.info("Planned {} total clips", len(all_clips))

    if not all_clips:
        raise RuntimeError("No clips could be extracted. Check episode paths and clip_length.")

    # Group by camera for multi-cam parallel shards
    cameras = config.cameras or [None]
    clips_by_camera: dict[str | None, list[_ClipSpec]] = {cam: [] for cam in cameras}
    for clip in all_clips:
        clips_by_camera[clip.camera].append(clip)

    # Shuffle within each camera (same order across cameras for sync)
    rng = random.Random(config.seed)
    # Generate a shared shuffle order from the first camera's clip count
    first_cam = cameras[0]
    n_clips_per_cam = len(clips_by_camera[first_cam])
    shuffle_order = list(range(n_clips_per_cam))
    rng.shuffle(shuffle_order)

    for cam in cameras:
        cam_clips = clips_by_camera[cam]
        clips_by_camera[cam] = [cam_clips[i] for i in shuffle_order]

    # Distribute to shards
    total_shards = 0
    total_clips = 0
    total_frames = 0
    total_bytes = 0
    shard_manifests: list[dict] = []
    clip_manifests: list[dict] = []

    for cam in cameras:
        cam_clips = clips_by_camera[cam]
        cam_suffix = f"_{cam}" if cam else ""

        # Split into shard-sized chunks
        for shard_idx in range(0, len(cam_clips), config.clips_per_shard):
            chunk = cam_clips[shard_idx:shard_idx + config.clips_per_shard]
            shard_name = f"shard_{shard_idx // config.clips_per_shard:04d}{cam_suffix}.mp4"
            shard_path = config.output_dir / shard_name

            logger.info("Writing {} ({} clips) ...", shard_name, len(chunk))
            n_frames = _write_shard(shard_path, chunk, config, codec_settings)

            shard_size = shard_path.stat().st_size
            total_shards += 1
            total_frames += n_frames
            total_bytes += shard_size

            # Only count clips once (not per camera)
            if cam == cameras[0]:
                total_clips += len(chunk)

            shard_manifests.append({
                "shard_id": shard_idx // config.clips_per_shard,
                "camera": cam,
                "file": shard_name,
                "num_clips": len(chunk),
                "num_frames": n_frames,
                "size_bytes": shard_size,
            })

            # Record clip metadata (once per clip position, not per camera)
            if cam == cameras[0]:
                for ci, clip in enumerate(chunk):
                    clip_manifests.append({
                        "shard_id": shard_idx // config.clips_per_shard,
                        "clip_index": ci,
                        "episode": clip.episode_id,
                        "episode_frame_start": clip.frame_start,
                        "episode_frame_end": clip.frame_end,
                    })

    # Build manifest
    manifest = {
        "version": 2,
        "pack_config": {
            "clip_length": config.clip_length,
            "clip_stride": config.clip_stride,
            "clips_per_shard": config.clips_per_shard,
            "codec": config.codec,
            "crf": config.crf if config.codec == "h264" else None,
            "resolution": list(config.resolution) if config.resolution else None,
            "fps": config.fps,
            "seed": config.seed,
            "cameras": config.cameras,
            "source_hash": _compute_source_hash(config),
            "frameforge_version": "0.1.0",
        },
        "stats": {
            "num_shards": total_shards,
            "total_clips": total_clips,
            "total_frames": total_frames,
            "total_size_bytes": total_bytes,
        },
        "shards": shard_manifests,
        "clips": clip_manifests,
    }

    if config.metadata:
        manifest["user_metadata"] = config.metadata

    manifest_path = config.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Manifest written to {}", manifest_path)

    elapsed = time.time() - t0

    result = PackResult(
        output_dir=config.output_dir,
        num_shards=total_shards,
        total_clips=total_clips,
        total_frames=total_frames,
        total_size_bytes=total_bytes,
        manifest_path=manifest_path,
        elapsed_sec=elapsed,
    )
    logger.info(result.summary())
    return result
