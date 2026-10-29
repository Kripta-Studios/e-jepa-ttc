from __future__ import annotations

from pathlib import Path

import pytest

from operational.efficient_context.common import digest
from operational.sota_eval.launch import materialize_manifest


def test_materialize_manifest_is_byte_exact_and_idempotent(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    full = tmp_path / "full"
    baseline.mkdir()
    source = baseline / "QUERY_MANIFEST.json"
    source.write_bytes(b'{"rows":[{"query_id":"q"}]}\n')

    first = materialize_manifest(baseline, full)
    second = materialize_manifest(baseline, full)

    assert first == second
    assert digest(full / "QUERY_MANIFEST.json") == digest(source)
    assert first["labels_read"] is False
    assert first["optimizer_updates"] == 0


def test_materialize_manifest_rejects_differing_existing_file(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline"
    full = tmp_path / "full"
    baseline.mkdir()
    full.mkdir()
    (baseline / "QUERY_MANIFEST.json").write_text('{"rows":[]}\n', encoding="utf-8")
    destination = full / "QUERY_MANIFEST.json"
    destination.write_text('{"rows":[1]}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Existing full manifest differs"):
        materialize_manifest(baseline, full)
    assert destination.read_text(encoding="utf-8") == '{"rows":[1]}\n'
