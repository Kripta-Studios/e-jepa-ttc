"""Recovery must reject cache corruption and changes in causal input identity."""

import hashlib
import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/prepare_detection_event_panel.py"
spec = importlib.util.spec_from_file_location("prepare_detection_event_panel", SCRIPT)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_verified_cache_rejects_cutoff_role_and_file_changes(tmp_path: Path) -> None:
    path = tmp_path / "frame.npz"
    np.savez_compressed(path, voxel=np.zeros((10, 2, 2), dtype=np.float32))
    identity = {"sample_token": "seq:frame", "role": "train", "cutoff_us": 100000}
    receipt = {"identity": identity, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    module.write_json(path.with_suffix(".json"), receipt)
    assert module.verify_cached(path, identity) == receipt
    for change in ({"role": "selection"}, {"cutoff_us": 100001}):
        with pytest.raises(ValueError, match="differs"):
            module.verify_cached(path, identity | change)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="differs"):
        module.verify_cached(path, identity)


def test_interrupted_frame_without_receipt_is_not_admitted(tmp_path: Path) -> None:
    path = tmp_path / "frame.npz"
    path.write_bytes(b"interrupted")
    assert module.verify_cached(path, {"sample_token": "s:f"}) is None
