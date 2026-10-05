"""A complete count cannot replace exact row and content admission."""

import numpy as np
import pytest

from operational.efficient_context.common import digest
from operational.train40_system.seal_inputs import check_fragment


def test_fragment_identity_and_recipe(tmp_path):
    path = tmp_path / "shard.npz"
    np.savez_compressed(path, ordinals=np.arange(2), tokens=np.asarray(["a", "b"]))
    receipt = {"status": "VERIFIED", "rows": 2, "freeze_sha256": "recipe", "sha256": digest(path)}
    check_fragment(path, receipt, np.arange(2), np.asarray(["a", "b"]), "recipe")
    for indices, tokens, freeze in (
        (np.asarray([0, 2]), np.asarray(["a", "b"]), "recipe"),
        (np.arange(2), np.asarray(["a", "c"]), "recipe"),
        (np.arange(2), np.asarray(["a", "b"]), "changed"),
    ):
        with pytest.raises(ValueError):
            check_fragment(path, receipt, indices, tokens, freeze)


def test_nonfinite_teacher_cannot_be_sealed(tmp_path):
    path = tmp_path / "teacher.npz"
    target = np.zeros((1, 2, 6, 32, 32), np.float16)
    target[0, 0, 0, 0, 0] = np.nan
    np.savez_compressed(
        path,
        ordinals=np.asarray([0]),
        tokens=np.asarray(["a"]),
        relation_targets=target,
        relation_valid=np.ones_like(target, bool),
    )
    receipt = {"status": "VERIFIED", "rows": 1, "freeze_sha256": "recipe", "sha256": digest(path)}
    with pytest.raises(ValueError, match="finite"):
        check_fragment(path, receipt, np.asarray([0]), np.asarray(["a"]), "recipe")
