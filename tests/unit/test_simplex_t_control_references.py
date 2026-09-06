"""Producer-bound control references reject changed input dependencies and closed roles."""

from copy import deepcopy

import numpy as np
import pytest

from e_jepa_ttc.simplex_t.control_references import control_references


def fixture():
    def index(tokens):
        n = len(tokens)
        valid = np.zeros((n, 16), bool)
        valid[:, -1] = True
        return {
            "tokens": np.array(tokens),
            "sequences": np.array(["seq"] * n),
            "base_windows_us": np.zeros((n, 3, 2), np.int64),
            "square_xyxy": np.ones((n, 4)),
            "anchor_us": np.ones(n, np.int64),
            "roi_available_us": np.ones(n, np.int64),
            "valid": valid,
            "lag_us": np.arange(16),
            "producer_family": np.zeros((3, n), np.int16),
        }

    indices = {"D0": index(["old"]), "D1": index(["extra"]), "DENSE": index(["old", "new"])}
    selection = {
        "tokens": ["new", "old"],
        "sequences": ["seq"],
        "sequence_family_sha256": {"seq": "a" * 64},
    }
    families = [{"outer_fold": 0, "role": "inner0", "family_sha256": "a" * 64}]
    return dict(pool="DENSE_OLD", selection=selection, outer=0, indices=indices, families=families)


def test_dense_reuses_old_only_after_exact_input_comparison():
    args = fixture()
    result = control_references(**args)
    assert result["tokens"].tolist() == ["new", "old"]
    assert result["source"].tolist() == [2, 0]
    assert result["query_row"].tolist() == [1, 0]


def test_diverse_has_unique_old_or_expansion_sources():
    args = fixture()
    args["pool"] = "DIVERSE_MATCHED"
    args["selection"]["tokens"] = ["extra", "old"]
    assert control_references(**args)["source"].tolist() == [1, 0]


@pytest.mark.parametrize("change", ["crop", "lag", "family", "inactive", "missing", "duplicate"])
def test_bad_reference_is_not_silently_redirected(change):
    args = deepcopy(fixture())
    if change == "crop":
        args["indices"]["DENSE"]["square_xyxy"][0, 0] = 99
    elif change == "lag":
        args["indices"]["DENSE"]["lag_us"][0] = 99
    elif change == "family":
        args["families"][0]["role"] = "outer_dev"
    elif change == "inactive":
        args["indices"]["DENSE"]["producer_family"][0, 1] = -1
    elif change == "missing":
        args["selection"]["tokens"] = ["unknown"]
    else:
        args["selection"]["tokens"] = ["old", "old"]
    with pytest.raises(ValueError):
        control_references(**args)
