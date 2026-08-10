from __future__ import annotations

import math

from e_jepa_ttc.training.causal_scale_eap import _is_better, _selection


def _metrics(first_mid: float, second_mid: float) -> dict[str, object]:
    finite = [value for value in (first_mid, second_mid) if math.isfinite(value)]
    return {
        "signed": {"failure_rate_pct": 10.0},
        "sequence_macro": {
            "sequence_macro_paper_MiD_overall": sum(finite) / len(finite),
            "per_sequence": {
                "seq-a": {"paper_MiD_overall": first_mid},
                "seq-b": {"paper_MiD_overall": second_mid},
            },
        },
    }


def test_selection_requires_finite_mid_for_every_sequence() -> None:
    incomplete = _selection(_metrics(100.0, float("nan")))

    assert incomplete["finite_sequence_count"] == 1.0
    assert incomplete["sequence_count"] == 2.0
    assert incomplete["complete_sequence_coverage"] == 0.0
    assert _is_better(incomplete, None) is False


def test_complete_sequence_candidate_remains_lexicographic() -> None:
    incumbent = _selection(_metrics(200.0, 220.0))
    improved = _selection(_metrics(180.0, 200.0))

    assert incumbent["complete_sequence_coverage"] == 1.0
    assert improved["complete_sequence_coverage"] == 1.0
    assert _is_better(improved, incumbent) is True
    assert _is_better(incumbent, improved) is False
