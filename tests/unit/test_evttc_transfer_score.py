"""External scoring must preserve invalid coverage and signed targets."""

import json

import numpy as np
import pytest

from operational.efficient_context.common import digest
from operational.evttc_transfer.run import measure_query
from operational.evttc_transfer.score import interpolate_label, metrics, run


def test_label_interpolation_never_extrapolates_or_crosses_nonfinite():
    table = np.array([[0, 1, 999, 999, -2], [1, 2, 999, 999, 2], [2, 3, 999, 999, np.nan]])
    assert interpolate_label(table, 0.9) is None
    assert interpolate_label(table, 1.5) == 0.0
    assert interpolate_label(table, 2.5) is None
    assert interpolate_label(table, 3) is None
    assert interpolate_label(table, 4) is None


def test_label_interpolation_rejects_ambiguous_time():
    with pytest.raises(ValueError, match="strictly increasing"):
        interpolate_label(np.array([[0, 1, 0, 0, 2], [1, 1, 0, 0, 3]]), 1)


def test_metrics_keeps_negative_labels_and_nonfinite_prediction_coverage():
    result = metrics(np.array([-1, 5, np.inf, 99]), np.array([-2, 3, 4, np.nan]))
    assert result["queries"] == 4
    assert result["labeled_queries"] == 3
    assert result["finite_predictions_on_labels"] == 2
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["mae_seconds"] == 1.5
    assert result["rmse_seconds"] == pytest.approx(np.sqrt(2.5))


@pytest.mark.parametrize("corrupt", ["seal_status", "binding", "prediction_binding"])
def test_score_rejects_broken_seal_before_labels(tmp_path, monkeypatch, corrupt):
    manifest = tmp_path / "QUERY_MANIFEST.json"
    manifest.write_text(json.dumps({"rows": [{"query_id": "query"}]}), encoding="utf-8")
    freeze = tmp_path / "INFERENCE_FREEZE.json"
    freeze.write_text("{}", encoding="utf-8")
    directory = tmp_path / "predictions"
    directory.mkdir()
    pred = directory / "query_00000.json"
    pred.write_text(
        json.dumps(
            {
                "query_id": "query",
                "binding_sha256": "wrong" if corrupt == "prediction_binding" else digest(freeze),
            }
        ),
        encoding="utf-8",
    )
    seal = dict(
        status="BROKEN" if corrupt == "seal_status" else "COMPLETE",
        binding_sha256="wrong" if corrupt == "binding" else digest(freeze),
        manifest_sha256=digest(manifest),
        queries=1,
        fragments={"predictions/query_00000.json": digest(pred)},
    )
    (tmp_path / "PREDICTIONS_SEALED.json").write_text(json.dumps(seal), encoding="utf-8")

    def no_labels(*args, **kwargs):
        pytest.fail("Labels must not be opened before prediction integrity checks")

    monkeypatch.setattr(np, "genfromtxt", no_labels)
    with pytest.raises(ValueError):
        run(tmp_path, tmp_path / "missing_inventory.yaml", tmp_path)


def test_missing_garl_endpoint_does_not_discard_own_predictions():
    class Model:
        def predict(self, *args, **kwargs):
            return {"H8_seed7": 1.0, "H8_seed13": 2.0, "H8_seed23": 3.0}

        def garl_predict(self, *args):
            pytest.fail("No Garl tensor is available")

    result = measure_query(
        {},
        Model(),
        lambda _: dict(
            own_events=np.zeros(1, np.float32),
            garl_events=None,
            delta_t_s=0.1,
            valid=np.ones(8, bool),
            garl_unavailable_reason="empty_sensor_endpoint",
        ),
    )
    assert result["ttc"]["H8_seed7"] == 1.0
    assert result["ttc"]["public_Garl_event_lhr"] is None
    assert result["unavailable_inputs"] == {"public_Garl_event_lhr": "empty_sensor_endpoint"}
