"""Exact same-layout comparison must not hide small or structural changes."""

import json

import numpy as np
import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t import h16_qa_plan
from e_jepa_ttc.simplex_t.h16_qa_plan import require_exact_h16_arrays


def arrays():
    return {
        name: np.ones((16, 2), dtype=np.float32)
        for name in (
            "features145",
            "expert_ttc",
            "pair_features",
            "known",
            "observation_ids",
            "anchor_us",
            "available_us",
        )
    }


def test_identical_arrays_pass():
    require_exact_h16_arrays(arrays(), arrays())


@pytest.mark.parametrize("field", list(arrays()))
def test_one_ulp_difference_fails(field):
    observed = arrays()
    observed[field][0, 0] = np.nextafter(np.float32(1), np.float32(2))
    with pytest.raises(ValueError, match=field):
        require_exact_h16_arrays(arrays(), observed)


@pytest.mark.parametrize("change", ["missing", "extra", "dtype", "shape", "nan"])
def test_schema_or_nonfinite_change_fails(change):
    observed = arrays()
    if change == "missing":
        del observed["known"]
    elif change == "extra":
        observed["extra"] = observed["known"]
    elif change == "dtype":
        observed["known"] = observed["known"].astype(np.float64)
    elif change == "shape":
        observed["known"] = observed["known"].reshape(8, 4)
    else:
        observed["known"][0, 0] = np.nan
    with pytest.raises(ValueError):
        require_exact_h16_arrays(arrays(), observed)


@pytest.mark.parametrize("corrupt_receipt", [False, True])
def test_reference_preflight_never_admits_missing_or_corrupt_blocks(
    tmp_path, monkeypatch, corrupt_receipt
):
    base = tmp_path / "artifacts/simplex_t"
    dedup = base / "T1/query_context_dedup"
    index = base / "T1/query_context_index"
    cache = base / "T1/context_features_fp32"
    for path in (dedup, index, cache):
        path.mkdir(parents=True)
    np.savez(dedup / "outer0.npz", history=np.zeros((1, 16), dtype=np.int64))
    (dedup / "DEDUP_MANIFEST.json").write_text(
        json.dumps({"outputs": [{"path": "outer0.npz", "sha256": sha256(dedup / "outer0.npz")}]}),
        encoding="utf-8",
    )
    np.savez(index / "query_context_index.npz", valid=np.ones((1, 16), dtype=bool))
    query = {"query": 0, "family": 0, "reference_stem": "family00_query00000"}
    monkeypatch.setattr(
        h16_qa_plan,
        "plan_h16_replay_qa",
        lambda work: {"queries": [query], "reference_cache": "T1/context_features_fp32"},
    )
    if corrupt_receipt:
        (cache / "family00_query00000.json").write_text(
            json.dumps({"query": 1, "family": 0, "sha256": "0" * 64}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="receipt or payload"):
            h16_qa_plan.bind_h16_reference_receipts(tmp_path)
    else:
        result = h16_qa_plan.bind_h16_reference_receipts(tmp_path)
        assert result["status"] == "REFERENCES_INCOMPLETE"
        assert result["missing"] == [query]
        assert not result["scientific_admission"]
        assert not result["gpu_replay_performed"]
