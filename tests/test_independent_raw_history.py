from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from operational.independent_refit import raw_history
from operational.independent_refit.population import canonical_sha256

ROOT = Path("output/independent_phase/raw_history_index_v4")


@pytest.fixture(scope="module")
def actual_frame() -> pd.DataFrame:
    path = ROOT / "raw_history_index.parquet"
    if not path.exists():
        pytest.skip("real write-once raw-history V4 artifact is not present")
    return pd.read_parquet(path)


def test_real_index_rederives_from_fold_and_raw_sources() -> None:
    receipt = ROOT / "RECEIPT.json"
    if not receipt.exists():
        pytest.skip("real write-once raw-history V4 artifact is not present")
    assert raw_history.verify_raw_history_index(receipt) == {
        "status": "PASS",
        "queries": 53_047,
        "groups": 24,
        "roles": 12,
    }


@pytest.mark.parametrize(
    ("column", "mutation"),
    [
        ("event_history_anchor_us", lambda value: int(value) + 1),
        ("observation_source_start_us", lambda _value: -1),
        ("query_label_clock_us", lambda value: int(value) + 20_000),
        ("events_path", lambda _value: "../learned_h8/events.h5"),
    ],
)
def test_rejects_clock_underflow_label_shift_and_unsafe_source(
    actual_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    column: str,
    mutation: object,
) -> None:
    monkeypatch.setattr(raw_history, "QUERY_COUNT", 1)
    monkeypatch.setattr(raw_history, "GROUP_COUNT", 1)
    changed = actual_frame.iloc[:1].copy()
    changed.at[0, column] = mutation(changed.at[0, column])  # type: ignore[operator]
    with pytest.raises(ValueError):
        raw_history.validate_history_frame(changed)


def test_rejects_shifted_rgb_member(
    actual_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(raw_history, "QUERY_COUNT", 1)
    monkeypatch.setattr(raw_history, "GROUP_COUNT", 1)
    changed = actual_frame.iloc[:1].copy()
    members = list(changed.at[0, "rgb_member_paths"])
    members[1] = members[0]
    changed.at[0, "rgb_member_paths"] = members
    with pytest.raises(ValueError, match="bindings differ"):
        raw_history.validate_history_frame(changed)


def test_observation_packets_match_panel_cadence_law(actual_frame: pd.DataFrame) -> None:
    row = actual_frame.iloc[0]
    anchor = int(row["event_history_anchor_us"])
    observations = [[int(value) for value in pair] for pair in row["observation_windows_us"]]
    assert observations == [
        [anchor - 620_000 + 20_000 * index, anchor - 600_000 + 20_000 * index]
        for index in range(31)
    ]
    assert observations[:30] == [
        [anchor - 620_000 + 20_000 * index, anchor - 600_000 + 20_000 * index]
        for index in range(30)
    ]
    assert observations[-1] == [anchor - 20_000, anchor]
    assert all(
        left[1] == right[0] for left, right in zip(observations, observations[1:], strict=False)
    )


def test_rejects_forged_physical_source_pin(tmp_path: Path) -> None:
    original_path = ROOT / "RECEIPT.json"
    if not original_path.exists():
        pytest.skip("real write-once raw-history V4 artifact is not present")
    receipt = json.loads(original_path.read_text(encoding="utf-8"))
    forged = tmp_path / "SOURCE_RECEIPT.json"
    forged.write_text("{}\n", encoding="utf-8")
    receipt["source_receipt_path"] = str(forged.resolve())
    payload = dict(receipt)
    payload.pop("receipt_sha256")
    receipt["receipt_sha256"] = canonical_sha256(payload)
    candidate = tmp_path / "RECEIPT.json"
    candidate.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="pins"):
        raw_history.verify_raw_history_index(candidate)


def test_rejects_rehashed_semantic_receipt_mutation(tmp_path: Path) -> None:
    original_path = ROOT / "RECEIPT.json"
    if not original_path.exists():
        pytest.skip("real write-once raw-history V4 artifact is not present")
    receipt = json.loads(original_path.read_text(encoding="utf-8"))
    receipt["runtime_materialization"] = "ONLY_600MS"
    payload = dict(receipt)
    payload.pop("receipt_sha256")
    receipt["receipt_sha256"] = canonical_sha256(payload)
    candidate = tmp_path / "RECEIPT.json"
    candidate.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="admission state"):
        raw_history.verify_raw_history_index(candidate)


def test_training_loader_is_fail_closed() -> None:
    receipt = ROOT / "RECEIPT.json"
    if not receipt.exists():
        pytest.skip("real write-once raw-history V4 artifact is not present")
    with pytest.raises(RuntimeError, match="metadata-only"):
        raw_history.load_training_history(receipt)
