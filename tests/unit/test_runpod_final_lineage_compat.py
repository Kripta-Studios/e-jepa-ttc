"""Relocated role references retain byte and population identity checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts.runpod_final_lineage_compat import role_comparator


class IdentityMapping:
    def translate(self, value: str) -> str:
        return value


def test_copied_roles_require_exact_content_and_nonpath_fields(tmp_path: Path) -> None:
    roots = (tmp_path / "bundle", tmp_path / "final_inputs")
    roles = []
    for root in roots:
        path = root / "workspace/project/folds/rotation_2_producer.parquet"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"exact rows")
        roles.append(
            {
                "rotation": 2,
                "population": 52551,
                "group_ids": ["a", "b"],
                "rows_path": str(path),
                "rows_sha256": hashlib.sha256(b"exact rows").hexdigest(),
            }
        )
    compare = role_comparator(IdentityMapping(), roots)
    assert compare(*roles)
    assert not compare(roles[0], {**roles[1], "population": 52550})
    assert not compare(roles[0], {**roles[1], "group_ids": ["a", "c"]})
    different = roots[1] / "workspace/project/folds/other.parquet"
    different.write_bytes(b"exact rows")
    assert not compare(roles[0], {**roles[1], "rows_path": str(different)})
    Path(str(roles[1]["rows_path"])).write_bytes(b"corrupt rows")
    with pytest.raises(ValueError, match="content differs"):
        compare(*roles)
    outside = tmp_path / "outside.parquet"
    outside.write_bytes(b"exact rows")
    with pytest.raises(ValueError, match="escaped"):
        compare(roles[0], {**roles[0], "rows_path": str(outside)})
