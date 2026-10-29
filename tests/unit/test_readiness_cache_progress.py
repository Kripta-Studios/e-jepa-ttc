from __future__ import annotations

from scripts.build_recovery_readiness_v4 import (
    _manifest_compressed_cache_counts,
    _partial_compressed_cache_counts,
)


def test_partial_compressed_cache_counts_split_directories(tmp_path) -> None:
    root = tmp_path / "cache"
    (root / "train").mkdir(parents=True)
    (root / "validation").mkdir()
    (root / "train" / "shard-00000.pt.gz").write_bytes(b"train")
    (root / "train" / "shard-00001.pt.gz").write_bytes(b"train")
    (root / "train" / "shard-00001.meta.json").write_text("{}", encoding="utf-8")
    (root / "validation" / "shard-00000.pt.gz").write_bytes(b"validation")

    assert _partial_compressed_cache_counts(root) == {"train": 2, "validation": 1}


def test_manifest_compressed_cache_counts_uses_relative_split_path() -> None:
    manifest = {
        "shards": [
            {"path": "train/shard-00000.pt.gz"},
            {"path": "train/shard-00001.pt.gz"},
            {"path": "validation/shard-00000.pt.gz"},
            {"path": "other/shard-00000.pt.gz"},
            {"path": None},
        ]
    }

    assert _manifest_compressed_cache_counts(manifest) == {"train": 2, "validation": 1}
