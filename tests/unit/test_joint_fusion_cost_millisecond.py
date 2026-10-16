"""The amended cost tolerance preserves scientific outputs and rejects domain/sign drift."""

import copy

import pytest

from operational.joint_measurement import fusion_cost_millisecond as candidate
from operational.range_transfer.common import digest, write


def test_observed_discrepancy_is_accepted_but_strict_failure_is_preserved():
    result = candidate.equivalence(37.1013069152832, 37.10210037231445)
    assert result["parity"] and result["same_sign"] and result["valid_native_outputs"]
    assert not result["strict_parity"]
    assert result["absolute_difference"] == pytest.approx(0.00079345703125)


@pytest.mark.parametrize(
    ("value", "reference"),
    [
        (37.102, 37.1),
        (float("nan"), 1.0),
        (float("inf"), float("inf")),
        (0.1, 0.1001),
        (0.0, -0.0001),
        (-0.0001, 0.1001),
        (60.002, 60.0),
    ],
)
def test_invalid_or_larger_discrepancies_rejected(value, reference):
    assert not candidate.equivalence(value, reference)["parity"]


@pytest.mark.parametrize(("value", "reference"), [(1.0005, 1.0), (-1.0005, -1.0), (60.0, 60.0)])
def test_uniform_acceptance_across_ttc_signs(value, reference):
    assert candidate.equivalence(value, reference)["parity"]


def row():
    return {
        "binding": "old",
        "index": 17,
        "timings": {
            name: {
                "prediction": 37.1013069152832,
                "reference": 37.10210037231445,
                "seconds": 2.7,
                "parity": False,
            }
            for name in candidate.fusion_cost.NAMES
        },
    }


def test_annotation_does_not_mutate_original_or_change_measured_outputs():
    original = row()
    saved = copy.deepcopy(original)
    updated = candidate.annotate(original)
    assert original == saved
    for name, timing in updated["timings"].items():
        assert timing["parity"] and not timing["strict_parity"]
        for field in ("prediction", "reference", "seconds"):
            assert timing[field] == original["timings"][name][field]


def test_missing_route_rejected():
    original = row()
    original["timings"].pop("F_TRUE")
    with pytest.raises(ValueError, match="four paired"):
        candidate.annotate(original)


@pytest.mark.parametrize("corruption", ["index", "binding", "none"])
def test_reused_rows_require_exact_original_identity(monkeypatch, tmp_path, corruption):
    monkeypatch.setattr(candidate.historical, "OUT", tmp_path)
    binding = {"original": "binding"}
    write(tmp_path / "REGISTRATION.json", binding)
    original = row()
    original["binding"] = digest(binding)
    if corruption == "index":
        original["index"] = 18
    if corruption == "binding":
        original["binding"] = "changed"
    write(tmp_path / "row_000.json", original)
    if corruption == "none":
        rows = candidate.prior_rows({"indices": [17, 19]})
        assert len(rows) == 1 and rows[0][1]["index"] == 17
    else:
        with pytest.raises(ValueError, match="identity/position"):
            candidate.prior_rows({"indices": [17, 19]})
