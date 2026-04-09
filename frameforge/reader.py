"""Unified VideoReader with pluggable backends and Pythonic indexing."""

from __future__ import annotations

from pathlib import Path
from typing import overload

import numpy as np
import torch
from loguru import logger

from frameforge.backends.base import AbstractBackend, BackendNotAvailable

# Backend priority for auto selection (most capable first)
_BACKEND_PRIORITY: list[tuple[str, str, str]] = [
    ("torchcodec", "frameforge.backends.torchcodec_backend", "TorchCodecBackend"),
    ("decord", "frameforge.backends.decord_backend", "DecordBackend"),
    ("pyav", "frameforge.backends.pyav_backend", "PyAVBackend"),
    ("opencv", "frameforge.backends.opencv_backend", "OpenCVBackend"),
]


def _load_backend(name: str, device: str = "cpu") -> AbstractBackend:
    """Instantiate a backend by name."""
    import importlib

    registry = {n: (mod, cls) for n, mod, cls in _BACKEND_PRIORITY}
    if name not in registry:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {list(registry.keys())}"
        )
    mod_path, cls_name = registry[name]
    module = importlib.import_module(mod_path)
    cls = getattr(module, cls_name)

    # Pass device to backends that accept it
    if name in ("torchcodec", "decord"):
        return cls(device=device)
    return cls()


def _auto_select_backend(device: str = "cpu") -> AbstractBackend:
    """Try backends in priority order, return the first that imports."""
    import importlib

    for name, mod_path, cls_name in _BACKEND_PRIORITY:
        try:
            module = importlib.import_module(mod_path)
            cls = getattr(module, cls_name)
            if name in ("torchcodec", "decord"):
                backend = cls(device=device)
            else:
                backend = cls()
            logger.info("Auto-selected backend: {}", name)
            return backend
        except BackendNotAvailable:
            continue

    raise BackendNotAvailable(
        "No video backend is available. Install at least one: "
        "pip install frameforge[pyav]  (recommended)"
    )


class VideoReader:
    """Unified video reader with pluggable backends and Pythonic indexing.

    Usage::

        reader = VideoReader("video.mp4")               # auto backend
        reader = VideoReader("video.mp4", backend="pyav")
        frame = reader[0]                                # single frame
        clip = reader[10:20]                             # slice → (T,H,W,C)
        frames = reader[[0, 5, 10]]                      # fancy index

    Performance options::

        # Enable frame index cache (2-3x faster random seeks)
        reader = VideoReader("video.mp4", cache_index=True)

        # Enable decode prefetching (2-4x faster sequential/semi-sequential)
        reader = VideoReader("video.mp4", prefetch=True)

        # Both together (best for DataLoader workloads)
        reader = VideoReader("video.mp4", cache_index=True, prefetch=True)

    Worker-safe: the underlying backend is lazily initialized on first access,
    so this object can be safely pickled and used across DataLoader workers
    without forked file handle issues.
    """

    def __init__(
        self,
        path: str | Path,
        backend: str = "auto",
        output: str = "torch",
        device: str = "cpu",
        cache_index: bool = False,
        prefetch: bool = False,
        prefetch_cache_size: int = 128,
        prefetch_ahead: int = 16,
    ) -> None:
        self._path = Path(path)
        self._backend_name = backend
        self._output = output
        self._device = device
        self._cache_index = cache_index
        self._prefetch = prefetch
        self._prefetch_cache_size = prefetch_cache_size
        self._prefetch_ahead = prefetch_ahead
        # Lazy init — backend is created on first use
        self._backend: AbstractBackend | None = None
        self._frame_index = None  # VideoFrameIndex, lazy
        self._prefetcher = None  # DecodePrefetcher, lazy

    def _ensure_backend(self) -> AbstractBackend:
        if self._backend is None:
            if self._backend_name == "auto":
                self._backend = _auto_select_backend(self._device)
            else:
                self._backend = _load_backend(self._backend_name, self._device)
            self._backend.open(self._path)

            # Initialize optional acceleration layers
            if self._cache_index:
                self._init_frame_index()
            if self._prefetch:
                self._init_prefetcher()

        return self._backend

    def _init_frame_index(self) -> None:
        """Build or load a cached frame index for fast seeking."""
        from frameforge.cache import VideoFrameIndex

        self._frame_index = VideoFrameIndex.build_or_load(self._path)
        logger.debug(
            "Frame index ready: {} frames, {} keyframes, avg GOP {:.1f}",
            self._frame_index.num_frames,
            self._frame_index.num_keyframes,
            self._frame_index.avg_gop_size,
        )

    def _init_prefetcher(self) -> None:
        """Start the background decode prefetcher (opens its own backend)."""
        from frameforge.prefetch import DecodePrefetcher

        assert self._backend is not None
        self._prefetcher = DecodePrefetcher(
            video_path=self._path,
            backend_name=self._backend_name if self._backend_name != "auto" else "pyav",
            device=self._device,
            cache_size=self._prefetch_cache_size,
            prefetch_ahead=self._prefetch_ahead,
        )
        self._prefetcher.start(num_frames=self._backend.num_frames)

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    @overload
    def __getitem__(self, idx: int) -> torch.Tensor | np.ndarray: ...
    @overload
    def __getitem__(self, idx: slice) -> torch.Tensor: ...
    @overload
    def __getitem__(self, idx: list[int]) -> torch.Tensor: ...

    def __getitem__(
        self, idx: int | slice | list[int]
    ) -> torch.Tensor | np.ndarray:
        backend = self._ensure_backend()

        if isinstance(idx, int):
            if idx < 0:
                idx += len(self)
            frame: np.ndarray | None = None
            if self._prefetcher is not None:
                frame = self._prefetcher.get_frame(idx)
            if frame is None:
                frame = backend.read_frame(idx)
                if self._prefetcher is not None:
                    self._prefetcher.put_frame(idx, frame)
            if self._output == "torch":
                return torch.from_numpy(frame)
            return frame

        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            clip = backend.read_clip(start, stop)
            # Populate prefetch cache with the decoded frames
            if self._prefetcher is not None:
                for i, fi in enumerate(range(start, stop)):
                    if i < clip.shape[0]:
                        self._prefetcher.put_frame(fi, clip[i].numpy())
            if step != 1:
                indices = list(range(0, stop - start, step))
                return clip[indices]
            return clip

        if isinstance(idx, list):
            frames_out: list[np.ndarray] = []
            for i in idx:
                frame = None
                if self._prefetcher is not None:
                    frame = self._prefetcher.get_frame(i)
                if frame is None:
                    frame = backend.read_frame(i)
                    if self._prefetcher is not None:
                        self._prefetcher.put_frame(i, frame)
                frames_out.append(frame)
            stacked = np.stack(frames_out)
            return torch.from_numpy(stacked)

        raise TypeError(f"Unsupported index type: {type(idx)}")

    def __len__(self) -> int:
        return self._ensure_backend().num_frames

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> dict:
        """Video metadata: fps, duration, resolution, codec."""
        return self._ensure_backend().metadata

    @property
    def fps(self) -> float:
        return self._ensure_backend().fps

    @property
    def num_frames(self) -> int:
        return self._ensure_backend().num_frames

    @property
    def duration_sec(self) -> float:
        return self._ensure_backend().duration_sec

    @property
    def frame_index(self):
        """Access the frame index (build/load if cache_index is enabled)."""
        if self._frame_index is None and self._cache_index:
            self._ensure_backend()
        return self._frame_index

    @property
    def prefetch_stats(self) -> dict | None:
        """Return prefetcher cache statistics, or None if prefetch is off."""
        if self._prefetcher is not None:
            return self._prefetcher.cache_stats
        return None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> VideoReader:
        self._ensure_backend()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.close()

    def close(self) -> None:
        """Release backend resources."""
        if self._prefetcher is not None:
            self._prefetcher.stop()
            self._prefetcher = None
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    def __repr__(self) -> str:
        extras = []
        if self._cache_index:
            extras.append("cache_index=True")
        if self._prefetch:
            extras.append("prefetch=True")
        extra_str = ", " + ", ".join(extras) if extras else ""
        return (
            f"VideoReader(path={self._path!r}, backend={self._backend_name!r}, "
            f"output={self._output!r}{extra_str})"
        )

    # ------------------------------------------------------------------
    # Pickling support for DataLoader workers
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_backend"] = None  # drop open file handles
        state["_prefetcher"] = None  # not picklable (thread)
        state["_frame_index"] = None  # will rebuild from cache in worker
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
