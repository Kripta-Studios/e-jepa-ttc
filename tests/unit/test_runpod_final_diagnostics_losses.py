"""Loss extraction only reads durable checkpoint cursors and binds their exact bytes."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest
import torch

from scripts.runpod_final_diagnostics_losses import extract


def candidate(tmp_path: Path, losses: list[float], completed: int) -> Path:
    root = tmp_path / "candidate"
    root.mkdir()
    path = root / "HEAD.pt"
    torch.save({"completed_updates": completed, "cursor": {"losses": losses}}, path)
    (root / "PORTABLE_PATHS.json").write_text(
        json.dumps(
            {
                "components": {
                    "HEAD": {
                        "local_path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                }
            }
        )
    )
    return root


def test_partial_loss_cursor_does_not_invent_update_indices(tmp_path: Path) -> None:
    root = candidate(tmp_path, [2.0, 1.0], completed=3)
    result = extract(root, tmp_path / "out")
    assert result["components"]["HEAD"]["scientific_update_indices_verified"] is False
    with (tmp_path / "out/TRAINING_LOSS_BY_UPDATE.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["scientific_update"] for row in rows] == ["", ""]
    assert [row["recorded_loss_index"] for row in rows] == ["1", "2"]


def test_unchanged_extraction_reuses_bound_csv_without_torch_load(
    tmp_path: Path, monkeypatch
) -> None:
    root = candidate(tmp_path, [2.0, 1.0], completed=2)
    result = extract(root, tmp_path / "out")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("Checkpoint reloaded"))
    assert extract(root, tmp_path / "out") == result


def test_changed_checkpoint_is_rejected_before_loading(tmp_path: Path) -> None:
    root = candidate(tmp_path, [1.0], completed=1)
    extract(root, tmp_path / "out")
    with (root / "HEAD.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checkpoint differs"):
        extract(root, tmp_path / "out")
