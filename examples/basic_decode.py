"""Basic video decoding with frameforge."""

from frameforge import VideoReader

# Auto-select the best available backend
reader = VideoReader("sample.mp4")
print(f"Video: {reader.metadata}")

# Single frame
frame = reader[0]
print(f"Frame shape: {frame.shape}")  # (H, W, C)

# Clip via slicing
clip = reader[10:26]  # 16 frames
print(f"Clip shape: {clip.shape}")  # (16, H, W, C)

# Fancy indexing
selected = reader[[0, 15, 30, 45]]
print(f"Selected frames shape: {selected.shape}")  # (4, H, W, C)

reader.close()

# Or use as context manager
with VideoReader("sample.mp4", backend="pyav") as r:
    first_frame = r[0]
    print(f"First frame shape: {first_frame.shape}")
