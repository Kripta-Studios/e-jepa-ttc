from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from operational.sota_eval import baselines


def _canonical(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _manifest(tmp_path: Path, *, forbidden_field: bool = False) -> Path:
    raw = tmp_path / "events.hdf5"
    raw.write_bytes(b"fixture")
    row = {
        "query_id": "sequence:00",
        "sequence_id": "sequence",
        "anchor_us": 300_000,
        "raw_path": str(raw),
        "raw_stat": {"size_bytes": raw.stat().st_size, "mtime_ns": raw.stat().st_mtime_ns},
        "windows_us": [[0, 100_000], [100_000, 200_000], [200_000, 300_000]],
        "boxes_xyxy3": [[10, 10, 20, 20], [11, 11, 22, 22], [12, 12, 24, 24]],
    }
    if forbidden_field:
        row["ttc_seconds"] = 3.0
    row["metadata_sha256"] = _canonical(row)
    payload = {
        "status": "LABEL_FREE_MANIFEST",
        "protocol": "fixture",
        "forbidden_assets": ["ttc.csv", "gt.hdf5", "distance", "depth", "navigation"],
        "rows": [row],
    }
    payload["rows_metadata_sha256"] = _canonical([row["metadata_sha256"]])
    path = tmp_path / "queries.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_runner_retains_failures_and_never_reads_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    events = np.asarray([[0.1, 12.0, 12.0, -1.0], [0.2, 13.0, 13.0, 1.0]])
    monkeypatch.setattr(
        baselines,
        "_load_query_events",
        lambda _row, _config: (
            events,
            (0, 0, 32, 32),
            np.asarray((0.1, 0.2, 0.3)),
            np.asarray(((10, 10, 20, 20), (11, 11, 22, 22), (12, 12, 24, 24))),
        ),
    )
    monkeypatch.setattr(
        baselines,
        "_predict_cmax",
        lambda *_args: (2.5, {"fixed": True}),
    )
    monkeypatch.setattr(
        baselines,
        "_predict_strttc",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("no normal flow")),
    )
    output = tmp_path / "output"
    summary = baselines.run_baselines(query_manifest=manifest, output_dir=output)
    assert summary["target_fields_opened"] == []
    assert summary["results"]["cmax"]["successful"] == 1
    assert summary["results"]["strttc"]["failed_retained"] == 1
    fragments = list(output.glob("*/fragments/*.json"))
    assert len(fragments) == 2
    assert {json.loads(path.read_text())["status"] for path in fragments} == {
        "SUCCESS",
        "FAILED_RETAINED",
    }


def test_resume_uses_signed_fragment_without_prediction_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(
        baselines,
        "_load_query_events",
        lambda *_args: (
            np.ones((2, 4)),
            (0, 0, 32, 32),
            np.asarray((0.1, 0.2, 0.3)),
            np.ones((3, 4)),
        ),
    )
    calls = 0

    def predict(*_args: object) -> tuple[float, dict[str, object]]:
        nonlocal calls
        calls += 1
        return 1.25, {}

    monkeypatch.setattr(baselines, "_predict_cmax", predict)
    output = tmp_path / "output"
    first = baselines.run_baselines(
        query_manifest=manifest, output_dir=output, methods=("cmax",)
    )
    second = baselines.run_baselines(
        query_manifest=manifest, output_dir=output, methods=("cmax",)
    )
    assert calls == 1
    assert first["fragment_sha256"] == second["fragment_sha256"]


def test_resume_rejects_tampered_fragment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _manifest(tmp_path)
    monkeypatch.setattr(
        baselines,
        "_load_query_events",
        lambda *_args: (
            np.ones((2, 4)),
            (0, 0, 32, 32),
            np.asarray((0.1, 0.2, 0.3)),
            np.ones((3, 4)),
        ),
    )
    monkeypatch.setattr(baselines, "_predict_cmax", lambda *_args: (1.0, {}))
    output = tmp_path / "output"
    baselines.run_baselines(query_manifest=manifest, output_dir=output, methods=("cmax",))
    fragment = next(output.glob("cmax/fragments/*.json"))
    document = json.loads(fragment.read_text())
    document["prediction_ttc_s"] = 99.0
    fragment.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical SHA"):
        baselines.run_baselines(query_manifest=manifest, output_dir=output, methods=("cmax",))


def test_manifest_rejects_target_field_before_prediction(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, forbidden_field=True)
    with pytest.raises(ValueError, match="forbidden target fields"):
        baselines.run_baselines(
            query_manifest=manifest,
            output_dir=tmp_path / "output",
            methods=("cmax",),
        )


def test_cmax_filters_events_by_interpolated_observed_boxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class Result:
        valid = True
        reason = "ok"
        inverse_ttc_per_s = 0.5
        ttc_seconds = 2.0
        contrast = 2.0
        null_contrast = 1.0
        relative_contrast_gain = 1.0
        survival_fraction = 1.0
        confidence = 0.5
        evaluations = 5

    def maximize(xy: np.ndarray, *_args: object, **kwargs: object) -> Result:
        captured["xy"] = xy
        captured["centers"] = kwargs["event_centers_xy"]
        return Result()

    import e_jepa_ttc.geometry.cmax as cmax

    monkeypatch.setattr(cmax, "maximize_radial_event_contrast", maximize)
    inside = np.column_stack(
        (
            np.linspace(0.1, 0.3, 1001),
            np.full(1001, 15.0),
            np.full(1001, 15.0),
            np.ones(1001),
        )
    )
    outside = np.asarray([[0.2, 100.0, 100.0, 1.0]])
    prediction, diagnostic = baselines._predict_cmax(
        np.concatenate((inside, outside)),
        (0, 0, 128, 128),
        np.asarray((0.1, 0.2, 0.3)),
        np.asarray(((10, 10, 20, 20), (10, 10, 20, 20), (10, 10, 20, 20))),
        baselines.BaselineConfig(),
    )
    assert prediction == 2.0
    assert diagnostic["object_event_count"] == 1001
    assert np.asarray(captured["xy"]).shape == (1001, 2)


def test_protocol_requires_nonlinear_strttc() -> None:
    with pytest.raises(ValueError, match="nonlinear"):
        baselines.BaselineConfig(strttc_nonlinear_refinement=False).validate()


def test_first_per_sequence_selection_is_fixed_before_prediction(tmp_path: Path) -> None:
    path = _manifest(tmp_path)
    document = json.loads(path.read_text())
    first = document["rows"][0]
    rows = []
    for query_id, sequence_id in (("a:00", "a"), ("a:01", "a"), ("b:00", "b")):
        row = dict(first, query_id=query_id, sequence_id=sequence_id)
        row.pop("metadata_sha256")
        row["metadata_sha256"] = _canonical(row)
        rows.append(row)
    document["rows"] = rows
    document["rows_metadata_sha256"] = _canonical(
        [row["metadata_sha256"] for row in rows]
    )
    path.write_text(json.dumps(document), encoding="utf-8")
    _, selected = baselines._validate_manifest(path, None, "first_per_sequence")
    assert [row["query_id"] for row in selected] == ["a:00", "b:00"]
