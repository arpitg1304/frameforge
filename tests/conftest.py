"""Shared fixtures: synthetic test videos generated with PyAV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

NUM_FRAMES = 60
WIDTH = 320
HEIGHT = 240
FPS = 30


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a short synthetic H.264 video for testing.

    Each frame is a solid colour that cycles through R/G/B so that
    frame-accuracy can be verified by checking pixel values.
    """
    import av

    path = tmp_path_factory.mktemp("videos") / "test_synthetic.mp4"

    container = av.open(str(path), mode="w")
    stream = container.add_stream("h264", rate=FPS)
    stream.width = WIDTH
    stream.height = HEIGHT
    stream.pix_fmt = "yuv420p"

    for i in range(NUM_FRAMES):
        # Cycle through R, G, B solid frames
        frame_data = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        channel = i % 3
        frame_data[:, :, channel] = 200
        av_frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
        for packet in stream.encode(av_frame):
            container.mux(packet)

    for packet in stream.encode():
        container.mux(packet)
    container.close()

    return path


@pytest.fixture(scope="session")
def synthetic_video_pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Create two synthetic videos (simulating multi-camera) with same frame count."""
    import av

    base = tmp_path_factory.mktemp("multi_cam")
    paths = []

    for cam_idx in range(2):
        path = base / f"cam{cam_idx}.mp4"
        container = av.open(str(path), mode="w")
        stream = container.add_stream("h264", rate=FPS)
        stream.width = WIDTH
        stream.height = HEIGHT
        stream.pix_fmt = "yuv420p"

        for i in range(NUM_FRAMES):
            frame_data = np.full((HEIGHT, WIDTH, 3), fill_value=(cam_idx + 1) * 50, dtype=np.uint8)
            av_frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
            for packet in stream.encode(av_frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)
        container.close()
        paths.append(path)

    return paths[0], paths[1]
