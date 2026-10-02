"""Saved-output parity/fault tests. No models or optimizers are constructed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import atomic_bytes, repair_torn_journal, require_roots, resumable_hierarchical_losses

CAMPAIGN = ROOT / "artifacts/simplex_t/scientific_campaign"
DRAWS = (
    CAMPAIGN
    / "T6/CHECKPOINTED_WORK/analysis/analyses/T2/paired_uncertainty/HIERARCHICAL_DRAWS.jsonl"
)


@pytest.fixture(scope="module")
def saved():
    pub = json.loads((CAMPAIGN / "publication/T2_PREDICTIONS.json").read_text(encoding="utf-8"))
    pieces = [
        pd.read_parquet(CAMPAIGN / "publication" / r["path"])
        for key, r in pub["fits"].items()
        if key.startswith("T2/TPR-D1-H8-C160/")
    ]
    frame = pd.concat(pieces).sort_values("sample_token").reset_index(drop=True)
    frame = frame.rename(columns={"target_ttc": "target_ttc_s"})
    return frame, frame.loss.to_numpy(np.float64)[:, None]


@pytest.mark.parametrize("fault", ["before_fragment_commit", "after_draw_publication"])
def test_interruption_and_publication_equal_continuous(tmp_path, saved, fault):
    frame, losses = saved
    reference, report = resumable_hierarchical_losses(
        frame,
        losses,
        tmp_path / "continuous",
        lambda: None,
        draws_path=DRAWS,
        binding={"test": "saved_primary"},
    )
    failures = []

    def interrupt(point):
        if point == fault and not failures:
            failures.append(point)
            raise InterruptedError("injected interruption")

    with pytest.raises(InterruptedError):
        resumable_hierarchical_losses(
            frame,
            losses,
            tmp_path / "resumed",
            lambda: None,
            draws_path=DRAWS,
            binding={"test": "saved_primary"},
            publication_hook=interrupt,
        )
    resumed, resumed_report = resumable_hierarchical_losses(
        frame,
        losses,
        tmp_path / "resumed",
        lambda: None,
        draws_path=DRAWS,
        binding={"test": "saved_primary"},
    )
    assert np.array_equal(reference, resumed)
    assert report == resumed_report
    assert (tmp_path / "continuous/HIERARCHICAL_DRAWS.jsonl").read_bytes() == (
        tmp_path / "resumed/HIERARCHICAL_DRAWS.jsonl"
    ).read_bytes()
    a = [
        json.loads(p.read_text())
        for p in sorted((tmp_path / "continuous/.resume").glob("fragment_*/RECEIPT.json"))
    ]
    b = [
        json.loads(p.read_text())
        for p in sorted((tmp_path / "resumed/.resume").glob("fragment_*/RECEIPT.json"))
    ]
    assert a == b


def test_changed_hash_and_torn_log(tmp_path, saved):
    frame, losses = saved
    out = tmp_path / "output"
    out.mkdir()
    raw = DRAWS.read_bytes()
    atomic_bytes(out / "HIERARCHICAL_DRAWS.jsonl", raw[:100])
    resumable_hierarchical_losses(
        frame, losses, out, lambda: None, draws_path=DRAWS, binding={"freeze": "one"}
    )
    assert (out / ".resume/LEGACY_TORN_TAIL.json").exists()
    with pytest.raises(ValueError, match="publication changed"):
        resumable_hierarchical_losses(
            frame, losses, out, lambda: None, draws_path=DRAWS, binding={"freeze": "two"}
        )


def test_exact_root_error(tmp_path):
    absent = tmp_path / "missing_required_source"
    with pytest.raises(FileNotFoundError) as error:
        require_roots({"eap": absent})
    assert error.value.filename == str(absent)
    assert "required root stat (eap)" in str(error.value)


def test_windows_import_failure_is_distinct_from_corruption():
    from supervisor import import_memory_failure

    assert import_memory_failure("OSError: [WinError 1455] DLL import failed")
    assert not import_memory_failure("ValueError: checkpoint SHA256 differs")


def test_torn_journal_preserved_and_complete_corruption_rejected(tmp_path):
    path = tmp_path / "events.jsonl"
    raw = b'{"step":"draw256"}\n{"step":'
    path.write_bytes(raw)
    repair_torn_journal(path)
    assert path.read_bytes() == b'{"step":"draw256"}\n'
    assert next(tmp_path.glob("events.jsonl.torn_*")).read_bytes() == raw
    path.write_bytes(b"corrupt complete line\n")
    with pytest.raises(json.JSONDecodeError):
        repair_torn_journal(path)


def test_frozen_t2_numerical_parity(tmp_path, saved):
    from e_jepa_ttc.evaluation.risk_geometry_v10 import hierarchical_losses

    frame, losses = saved
    frozen, old = hierarchical_losses(frame, losses, tmp_path / "frozen", lambda: None)
    adapted, new = resumable_hierarchical_losses(
        frame,
        losses,
        tmp_path / "adapted",
        lambda: None,
        draws_path=DRAWS,
        binding={"test": "frozen_recipe"},
    )
    assert np.array_equal(frozen, adapted)
    assert old == new
    report = json.loads(
        (
            CAMPAIGN
            / "T6/CHECKPOINTED_WORK/analysis/analyses/T2/paired_uncertainty/PAIRED_UNCERTAINTY.json"
        ).read_text()
    )
    saved_matrix = np.load(
        CAMPAIGN
        / "T6/CHECKPOINTED_WORK/analysis/analyses/T2/paired_uncertainty/BOOTSTRAP_LOSSES.npy"
    )
    idx = report["column_order"].index("TPR-D1-H8-C160")
    assert np.max(np.abs(adapted[:, 0] - saved_matrix[:, idx])) <= 1e-10


def test_atomic_parquet_windows_flush_and_interrupted_replace(tmp_path, monkeypatch):
    import runtime

    path = tmp_path / "table.parquet"
    before = pd.DataFrame({"draw": [1, 2]})
    after = pd.DataFrame({"draw": [1, 2, 3]})
    before.to_parquet(path, index=False)
    old_bytes = path.read_bytes()
    original_replace = runtime.os.replace

    def interrupt(*args):
        raise InterruptedError("injected before atomic replacement")

    monkeypatch.setattr(runtime.os, "replace", interrupt)
    with pytest.raises(InterruptedError):
        runtime.atomic_parquet_write(after, path, pd.DataFrame.to_parquet, (), {"index": False})
    assert path.read_bytes() == old_bytes
    monkeypatch.setattr(runtime.os, "replace", original_replace)
    runtime.atomic_parquet_write(after, path, pd.DataFrame.to_parquet, (), {"index": False})
    pd.testing.assert_frame_equal(pd.read_parquet(path), after)


def test_json_graph_tuple_list_parity_preserves_numbers():
    from runtime import json_record

    graph = {"fraction_train_h8": (0.9918262237891046, 1.0, 0.9945678997987654)}
    stored = json.loads(json.dumps(graph))
    assert graph != stored
    assert json_record(graph) == stored
    assert tuple(stored["fraction_train_h8"]) == graph["fraction_train_h8"]
