"""The global V9 archive overlay preserves compatibility evidence fail-closed."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from operational.runpod_migration import rgb_return_archive as v8
from operational.runpod_migration import rgb_return_archive_v9 as v9


def _sources(tmp_path: Path) -> list[Path]:
    repository = Path(__file__).resolve().parents[2]
    v9_source = tmp_path / "runpod_rgb_context_v9.py"
    v9_source.write_bytes(b"v9 fixture")
    return [repository / f"scripts/runpod_rgb_context_v{version}.py" for version in (6, 7, 8)] + [
        v9_source
    ]


def _sidecar(run: Path, sources: list[Path], *, parent_version: int = 7) -> Path:
    by_name = {path.name: path for path in sources}
    path = run / "RGB_OPERATIONAL_RECOVERY_V9.json"
    path.write_text(
        json.dumps(
            {
                "schema": v9.V9_SCHEMA,
                "status": "PASS",
                "source_sha256": v8.sha256(by_name["runpod_rgb_context_v9.py"]),
                "context_v6_sha256": v8.sha256(by_name["runpod_rgb_context_v6.py"]),
                "parent_context_sha256": v8.sha256(
                    by_name[f"runpod_rgb_context_v{parent_version}.py"]
                ),
                "parent_context_version": parent_version,
                "policy": v9.V9_POLICY,
                "original_markers_rewritten": False,
                "optimizer_updates_inferred": False,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_v9_binding_rejects_source_hash_tamper(tmp_path: Path) -> None:
    sources = _sources(tmp_path)
    run = tmp_path / "r2_rgb_downstream_v1"
    run.mkdir()
    sidecar = _sidecar(run, sources)
    value = json.loads(sidecar.read_text(encoding="utf-8"))
    value["source_sha256"] = "0" * 64
    sidecar.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="resolve exactly once"):
        v9._validate_v9(run, sources)


def test_global_overlay_requires_r2_r3_v9_and_keeps_r4_v8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _sources(tmp_path)
    downstream = {}
    for rotation in (2, 3, 4):
        run = tmp_path / f"r{rotation}_rgb_downstream_v1"
        run.mkdir()
        downstream[rotation] = run
    _sidecar(downstream[2], sources)
    _sidecar(downstream[3], sources)

    def base_collect(*_args: object, **_kwargs: object) -> tuple[dict[str, Any], dict[str, Any]]:
        records: dict[str, Any] = {}
        return {"files": records, "content_sha256": hashlib.sha256(b"").hexdigest()}, records

    monkeypatch.setattr(v8, "collect_rgb_return", base_collect)
    manifest, records = v9.collect_rgb_return(
        downstream,
        {rotation: tmp_path for rotation in (2, 3, 4)},
        {rotation: tmp_path for rotation in (2, 3, 4)},
        coverage_receipt=tmp_path,
        authorization_root=tmp_path,
        operational_sources=sources,
    )
    assert manifest["operational_v9_rotations"] == [2, 3]
    assert manifest["active_rgb_context_chain_by_rotation"]["2"] == [
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
    ]
    assert manifest["active_rgb_context_chain_by_rotation"]["3"] == [
        "V5",
        "V6",
        "V7",
        "V9",
    ]
    assert manifest["active_rgb_context_chain_by_rotation"]["4"][-1] == "V8"
    assert set(records) == {
        "roots/r2_downstream/RGB_OPERATIONAL_RECOVERY_V9.json",
        "roots/r3_downstream/RGB_OPERATIONAL_RECOVERY_V9.json",
    }
    _sidecar(downstream[4], sources, parent_version=8)
    with pytest.raises(ValueError, match="R4 must retain"):
        v9.collect_rgb_return(
            downstream,
            {rotation: tmp_path for rotation in (2, 3, 4)},
            {rotation: tmp_path for rotation in (2, 3, 4)},
            coverage_receipt=tmp_path,
            authorization_root=tmp_path,
            operational_sources=sources,
        )


def test_v9_overlay_rejects_singleton_selection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="global-only"):
        v9.collect_rgb_return(
            {4: tmp_path},
            {4: tmp_path},
            {4: tmp_path},
            coverage_receipt=tmp_path,
            authorization_root=tmp_path,
            operational_sources=[],
            rotations=(4,),
        )
