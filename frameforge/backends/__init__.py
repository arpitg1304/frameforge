"""Pluggable video decoding backends."""

from frameforge.backends.base import AbstractBackend, BackendNotAvailable

__all__ = ["AbstractBackend", "BackendNotAvailable"]
