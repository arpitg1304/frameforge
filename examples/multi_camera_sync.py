"""Multi-camera synchronization for robotics setups."""

from frameforge.reader import VideoReader
from frameforge.sampling.sync import sync_readers_by_timestamp

# Open multiple camera feeds
readers = {
    "wrist_cam": VideoReader("wrist.mp4", backend="pyav"),
    "overhead_cam": VideoReader("overhead.mp4", backend="pyav"),
    "third_person": VideoReader("third_person.mp4", backend="pyav"),
}

# Get synchronized frames at specific timestamps
timestamps = [0.0, 0.5, 1.0, 1.5, 2.0]
synced = sync_readers_by_timestamp(readers, timestamps)

for cam_name, frames in synced.items():
    print(f"{cam_name}: {frames.shape}")  # (5, H, W, C)

for r in readers.values():
    r.close()
