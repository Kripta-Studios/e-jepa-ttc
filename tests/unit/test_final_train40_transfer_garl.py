"""Integrity checks for resumable unscored Garl fragments."""

import pytest

from scripts.final_train40_transfer_garl import atomic, read, validate_fragment


def fragment():
    return {
        "query_id": "q1",
        "manifest_sha256": "m",
        "binding_sha256": "b",
        "targets_read": False,
        "optimizer_updates": 0,
        "status": "PREDICTED",
        "prediction_ttc": 1.25,
    }


def test_valid_prediction_roundtrip(tmp_path):
    path = tmp_path / "query.json"
    atomic(path, fragment())
    validate_fragment(read(path), {"query_id": "q1"}, "m", "b")
    assert not path.with_name(path.name + ".pending").exists()


@pytest.mark.parametrize(
    "key,value",
    [
        ("query_id", "q2"),
        ("manifest_sha256", "other"),
        ("binding_sha256", "other"),
        ("targets_read", True),
        ("optimizer_updates", 1),
        ("prediction_ttc", float("nan")),
        ("prediction_ttc", True),
        ("prediction_ttc", None),
        ("status", "SKIPPED"),
    ],
)
def test_resume_rejects_foreign_or_invalid_fragment(key, value):
    item = fragment()
    item[key] = value
    with pytest.raises(ValueError):
        validate_fragment(item, {"query_id": "q1"}, "m", "b")


def test_explicit_native_failure_keeps_population():
    item = fragment()
    item.update(status="NONFINITE_NATIVE_TTC", prediction_ttc=None)
    validate_fragment(item, {"query_id": "q1"}, "m", "b")
