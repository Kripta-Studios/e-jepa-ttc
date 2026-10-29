from __future__ import annotations

import json

from scripts.record_eap_run_interruption import record_interruption


def test_interruption_artifact_records_no_completed_epoch_without_overwrite(tmp_path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "history.jsonl").write_text("", encoding="utf-8")

    first = record_interruption(
        output,
        command="python train.py",
        reason="operator stopped stale run",
        process_ids=[123],
    )
    second = record_interruption(
        output,
        command="python train.py",
        reason="second interruption",
        process_ids=[456],
    )

    assert first.name == "FAILURE.json"
    assert second.name != first.name
    payload = json.loads(first.read_text(encoding="utf-8"))
    assert payload["status"] == "interrupted"
    assert payload["epochs_completed"] == 0
    assert payload["metrics_available"] is False
    assert payload["checkpoint_paths"] == []
