from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from operational.sota_eval import population


def _row(anchor: int = 1_000_000) -> dict:
    return {
        "sequence_id": "sequence",
        "query_id": "sequence:00",
        "anchor_us": anchor,
        "windows_us": [
            [anchor - 300_000, anchor - 200_000],
            [anchor - 200_000, anchor - 100_000],
            [anchor - 100_000, anchor],
        ],
        "boxes_xyxy3": [[1.0, 2.0, 3.0, 4.0]] * 3,
        "square_xyxy": [0.0, 0.0, 5.0, 5.0],
        "bbox_sources": [
            {"frame_index": index, "timestamp_us": anchor - 300_000 + index, "sha256": "a" * 64}
            for index in range(3)
        ],
    }


def test_semantic_key_ignores_query_ordinal_but_not_anchor() -> None:
    first = _row()
    renamed = dict(first, query_id="renumbered:99")
    shifted = _row(anchor=1_000_001)

    assert population._semantic_key(first) == population._semantic_key(renamed)
    assert population._semantic_key(first) != population._semantic_key(shifted)


def test_immutable_writer_rejects_different_content(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    population._write_immutable(path, {"value": 1})
    population._write_immutable(path, {"value": 1})

    with pytest.raises(ValueError, match="immutable population artifact differs"):
        population._write_immutable(path, {"value": 2})


def test_fov_is_derived_from_intrinsics_without_claiming_lens_mm() -> None:
    fov = population._horizontal_fov_deg(
        np.asarray([1265.0, 1265.0, 640.0, 360.0]),
        np.asarray([1280, 720]),
    )

    assert fov == pytest.approx(53.7, abs=0.1)


def test_builder_freezes_all_rows_and_prior_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = tmp_path / "inventory.yaml"
    inventory.write_text("sequences: []\n", encoding="utf-8")
    prior_path = tmp_path / "prior.json"
    prior_path.write_text(json.dumps({"rows": [_row()]}), encoding="utf-8")
    data_root = tmp_path / "data"
    data_root.mkdir()

    def fake_build(
        _inventory: Path,
        _data_root: Path,
        _destination: Path,
        max_per_sequence: int,
    ) -> dict:
        assert max_per_sequence == population.ALL_FEASIBLE_LIMIT
        return {"rows": [_row(), _row(anchor=1_100_000)], "causal_exclusions": {}}

    monkeypatch.setattr(population, "build_manifest", fake_build)
    monkeypatch.setattr(
        population,
        "_audit_calibration",
        lambda _manifest, _inventory: {"status": "METADATA_ONLY_NO_TARGETS"},
    )

    freeze = population.build_population(
        inventory_path=inventory,
        data_root=data_root,
        prior_manifest_path=prior_path,
        output_dir=tmp_path / "out",
    )

    assert freeze["manifest_rows"] == 2
    assert freeze["targets_read"] is False
    manifest = json.loads((tmp_path / "out/QUERY_MANIFEST.json").read_text(encoding="utf-8"))
    snapshot = json.loads((tmp_path / "out/PRIOR_QUERY_SNAPSHOT.json").read_text(encoding="utf-8"))
    assert manifest["selection_mode"] == "ALL_CAUSALLY_FEASIBLE_NO_TARGETS"
    assert snapshot["status"] == "PASSED"
    assert snapshot["contained_semantic_queries"] == 1
