"""frameforge — Video encoding/decoding utilities for robotics foundation model data pipelines."""

__version__ = "0.1.0"

from frameforge.cache import VideoFrameIndex
from frameforge.reader import VideoReader

__all__ = ["VideoReader", "VideoFrameIndex"]
