"""Data packing — pre-pack episode clips into sequential shard videos for fast training."""

from frameforge.packing.config import PackConfig, PackResult
from frameforge.packing.packer import pack_shards
from frameforge.packing.dataset import ShardDataset, ShardStreamDataset

__all__ = [
    "PackConfig",
    "PackResult",
    "pack_shards",
    "ShardDataset",
    "ShardStreamDataset",
]
