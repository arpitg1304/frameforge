"""CLI entry point: python -m frameforge.packing pack/verify/info"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loguru import logger


def cmd_pack(args: argparse.Namespace) -> None:
    from frameforge.packing import PackConfig, pack_shards

    # Collect episode paths
    episode_dir = Path(args.episodes)
    paths = sorted(episode_dir.glob("*.mp4"))
    if not paths:
        logger.error("No .mp4 files found in {}", episode_dir)
        sys.exit(1)

    # Parse resolution
    resolution = None
    if args.resolution:
        w, h = args.resolution.split("x")
        resolution = (int(w), int(h))

    # Parse cameras
    cameras = None
    if args.cameras:
        cameras = [c.strip() for c in args.cameras.split(",")]
        # For multi-cam, expect subdirectories per camera
        cam_paths: dict[str, list[Path]] = {}
        for cam in cameras:
            cam_dir = episode_dir / cam
            if cam_dir.is_dir():
                cam_paths[cam] = sorted(cam_dir.glob("*.mp4"))
            else:
                logger.error("Camera directory {} not found", cam_dir)
                sys.exit(1)
        episode_paths = cam_paths
    else:
        episode_paths = paths

    config = PackConfig(
        episode_paths=episode_paths,
        output_dir=args.output,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
        clips_per_episode=args.clips_per_episode,
        clips_per_shard=args.clips_per_shard,
        cameras=cameras,
        codec=args.codec,
        crf=args.crf,
        resolution=resolution,
        seed=args.seed,
        backend=args.backend,
        num_workers=args.num_workers,
    )

    result = pack_shards(config)
    print(result.summary())


def cmd_verify(args: argparse.Namespace) -> None:
    from frameforge.reader import VideoReader

    shard_dir = Path(args.shard_dir)
    manifest_path = shard_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error("No manifest.json in {}", shard_dir)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    errors = 0

    for shard_info in manifest["shards"]:
        shard_path = shard_dir / shard_info["file"]
        if not shard_path.exists():
            logger.error("Missing shard file: {}", shard_path)
            errors += 1
            continue

        try:
            with VideoReader(shard_path, backend="pyav") as r:
                actual = len(r)
                expected = shard_info["num_frames"]
                if actual != expected:
                    logger.warning(
                        "{}: expected {} frames, got {}",
                        shard_info["file"], expected, actual,
                    )
                    errors += 1
                else:
                    logger.info("{}: OK ({} frames)", shard_info["file"], actual)
        except Exception as e:
            logger.error("{}: {}", shard_info["file"], e)
            errors += 1

    if errors:
        logger.error("{} errors found", errors)
        sys.exit(1)
    else:
        print(f"All {len(manifest['shards'])} shards verified OK")


def cmd_info(args: argparse.Namespace) -> None:
    shard_dir = Path(args.shard_dir)
    manifest_path = shard_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error("No manifest.json in {}", shard_dir)
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    config = manifest["pack_config"]
    stats = manifest["stats"]

    print(f"Shard directory: {shard_dir}")
    print(f"  Clips:       {stats['total_clips']}")
    print(f"  Frames:      {stats['total_frames']}")
    print(f"  Shards:      {stats['num_shards']}")
    print(f"  Size:        {stats['total_size_bytes'] / (1024*1024):.1f} MB")
    print(f"  Clip length: {config['clip_length']} frames")
    print(f"  Clip stride: {config['clip_stride']} frames")
    print(f"  Codec:       {config['codec']}")
    print(f"  Resolution:  {config.get('resolution', 'source')}")
    print(f"  Cameras:     {config.get('cameras', 'single')}")
    print(f"  Seed:        {config['seed']}")
    print(f"  Source hash: {config.get('source_hash', 'unknown')}")

    # Episode stats
    episodes = set(c["episode"] for c in manifest["clips"])
    print(f"  Episodes:    {len(episodes)}")

    if manifest.get("user_metadata"):
        print(f"  Metadata:    {json.dumps(manifest['user_metadata'])}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="frameforge-pack",
        description="frameforge data packing — pack episode clips into shard videos",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- pack ---
    p = sub.add_parser("pack", help="Pack episodes into shard videos")
    p.add_argument("--episodes", required=True, help="Directory of episode .mp4 files")
    p.add_argument("--output", required=True, help="Output directory for shards")
    p.add_argument("--clip-length", type=int, default=16)
    p.add_argument("--clip-stride", type=int, default=None)
    p.add_argument("--clips-per-episode", type=int, default=None)
    p.add_argument("--clips-per-shard", type=int, default=1000)
    p.add_argument("--cameras", default=None, help="Comma-separated camera names")
    p.add_argument("--codec", default="h264", choices=["h264", "mjpeg", "ffv1"])
    p.add_argument("--crf", type=int, default=18)
    p.add_argument("--resolution", default=None, help="WxH, e.g. 480x360")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backend", default="pyav")
    p.add_argument("--num-workers", type=int, default=1)

    # --- verify ---
    v = sub.add_parser("verify", help="Verify shard integrity")
    v.add_argument("shard_dir", help="Path to shard directory")

    # --- info ---
    i = sub.add_parser("info", help="Show shard pack info")
    i.add_argument("shard_dir", help="Path to shard directory")

    args = parser.parse_args(argv)

    if args.command == "pack":
        cmd_pack(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "info":
        cmd_info(args)


if __name__ == "__main__":
    main()
