"""Synthetic CPU evaluation admission, coverage, resumability and export proofs."""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from operational.decision_recovery import common, evaluation, training


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "RUN", tmp_path / "v19")
    monkeypatch.setattr(common, "SOURCE_ROOT", tmp_path / "v18")
    return common.RUN


def rows():
    target = np.tile([0.5, 2.0, 4.0, 8.0, -2.0], 3)
    return pd.DataFrame(
        {
            "sample_token": [f"q{i}" for i in range(len(target))],
            "group_id": np.repeat(["a", "b", "c"], 5),
            "sequence_id": np.repeat(["a", "b", "c"], 5),
            "target_ttc": target,
            "role": "H",
        }
    )


def predictions():
    frame = rows()
    frame["prediction_ttc"] = frame.target_ttc * 1.1
    frame["raw_phase"] = -np.log1p(-0.1 / frame.prediction_ttc)
    return frame


@pytest.mark.parametrize("mutation", ["drop", "repeat", "group", "target", "role", "nan"])
def test_population_rejects_omission_identity_and_invalid_output(mutation):
    frame = predictions()
    if mutation == "drop":
        frame = frame.iloc[:-1]
    elif mutation == "repeat":
        frame.loc[1, "sample_token"] = frame.loc[0, "sample_token"]
    elif mutation == "group":
        frame.loc[0, "group_id"] = "other"
    elif mutation == "target":
        frame.loc[0, "target_ttc"] = 3
    elif mutation == "role":
        frame.loc[0, "role"] = "P"
    else:
        frame.loc[0, "prediction_ttc"] = np.nan
    with pytest.raises(ValueError):
        evaluation._check_frame(frame, rows(), "H")


def test_full_chunk_reassembly_and_metric_recalculation(sandbox):
    directory = training.destination("evaluation/predictions/H/L0")
    directory.mkdir(parents=True)
    frame = predictions()
    path = directory / "chunk_00000.parquet"
    frame.to_parquet(path, index=False)
    common.write_json(
        path.with_suffix(".json"),
        {"binding": "frozen", "start": 0, "stop": len(frame), "sha256": common.sha(path)},
    )
    receipt = evaluation._finish(directory, "frozen", rows(), "H")
    assert receipt["rows"] == len(frame) and receipt["optimizer_updates"] == 0
    _, restored = evaluation._completed(directory, "frozen", rows(), "H")
    pd.testing.assert_frame_equal(restored, frame)
    metric = common.read_json(directory / "METRICS.json")
    metric["overall_MiD"] += 1
    common.write_json(directory / "METRICS.json", metric)
    receipt["metrics_sha256"] = common.sha(directory / "METRICS.json")
    common.write_json(directory / "COMPLETE.json", receipt)
    with pytest.raises(ValueError, match="reproduce"):
        evaluation._completed(directory, "frozen", rows(), "H")


def test_tampered_chunk_or_wrong_binding_rejected(sandbox):
    directory = training.destination("evaluation/predictions/H/L0")
    directory.mkdir(parents=True)
    path = directory / "chunk_00000.parquet"
    predictions().to_parquet(path, index=False)
    common.write_json(
        path.with_suffix(".json"),
        {"binding": "one", "start": 0, "stop": len(rows()), "sha256": common.sha(path)},
    )
    with pytest.raises(ValueError, match="receipt"):
        evaluation._chunk(path, "other", 0, len(rows()), rows(), "H")
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="bytes"):
        evaluation._chunk(path, "one", 0, len(rows()), rows(), "H")


def endpoint_fixture(arm):
    directory = evaluation._directory(arm)
    versions = directory / "checkpoint_versions"
    versions.mkdir(parents=True)
    version = versions / "endpoint.pt"
    version.write_bytes(b"immutable checkpoint synthetic")
    (directory / "checkpoint_last.pt").write_bytes(version.read_bytes())
    common.write_json(training.destination("training/REGISTRATION.json"), {"test": True})
    identity = {
        "arm": arm,
        "seed": 7,
        "mode": "FULL",
        "endpoint": 12000,
        "population": "P",
        "parent_sha256": "parent",
        "identity_sha256": "identity",
        "registration_sha256": common.sha(training.destination("training/REGISTRATION.json")),
    }
    receipt = {
        "status": "COMPLETE",
        "scientific_endpoint": True,
        "complete_state": True,
        "completed_updates": 12000,
        "identity_sha256": "identity",
        "checkpoint_sha256": common.sha(version),
    }
    pointer = {
        "version": "endpoint.pt",
        "completed_updates": 12000,
        "checkpoint_sha256": common.sha(version),
    }
    journal = {
        "completed_updates": 12000,
        "durable_updates": 12000,
        "pending_update_upper": 0,
        "identity_sha256": "identity",
    }
    for name, value in (
        ("IDENTITY", identity),
        ("CHECKPOINT_RECEIPT", receipt),
        ("CHECKPOINT_POINTER", pointer),
        ("UPDATE_JOURNAL", journal),
    ):
        common.write_json(directory / f"{name}.json", value)
    return directory


def test_endpoint_requires_durable_complete_and_hashes(sandbox):
    directory = endpoint_fixture("L0")
    value = evaluation._endpoint("L0", {"parent_sha256": "parent"})
    assert value["checkpoint_sha256"] == common.sha(directory / "checkpoint_last.pt")
    journal = common.read_json(directory / "UPDATE_JOURNAL.json")
    journal["pending_update_upper"] = 1
    common.write_json(directory / "UPDATE_JOURNAL.json", journal)
    with pytest.raises(ValueError, match="incomplete"):
        evaluation._endpoint("L0", {"parent_sha256": "parent"})


def test_evaluate_orders_registration_before_roles_and_stops_on_partial(sandbox, monkeypatch):
    calls = []
    monkeypatch.setattr(training, "runtime_context", nullcontext)
    monkeypatch.setattr(training, "file_lock", lambda path: nullcontext())
    monkeypatch.setattr(evaluation, "_register", lambda: calls.append("registered") or {})

    def predict(arm, role, record):
        calls.append(role)
        return {"status": "PARTIAL", "role": role}

    monkeypatch.setattr(evaluation, "_predict_role", predict)
    monkeypatch.setattr(evaluation, "_export", lambda *a: pytest.fail("export on partial"))
    assert evaluation.evaluate("L0")["status"] == "PARTIAL"
    assert calls == ["registered", "P"]
    with pytest.raises(ValueError):
        evaluation.evaluate("LS")


def test_cpu_only_export_fixed_batch_replay_and_hashes(sandbox, monkeypatch):
    from e_jepa_ttc.simplex_t.phase import phase_to_ttc
    from operational.joint_measurement import data, geometry_export

    class Model(torch.nn.Module):
        def forward(self, events, delta):
            return {"raw_phase": events[:, 0] + delta.reshape(-1)}

    class Portable(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, events, delta):
            return phase_to_ttc(self.model(events, delta)["raw_phase"])

    class Dataset:
        def __init__(self, role):
            assert role == "P"

        def batch(self, indices):
            return SimpleNamespace(
                events=torch.tensor([[0.02 + indices[0] / 10000]]), delta_t_s=torch.tensor([[0.03]])
            )

        def close(self):
            pass

    monkeypatch.setattr(data, "SensorDataset", Dataset)
    monkeypatch.setattr(geometry_export, "PortableGeometry", Portable)
    monkeypatch.setattr(evaluation, "_load_model", lambda *args: Model())
    common.write_json(training.destination("evaluation/REGISTRATION.json"), {"test": True})
    record = {
        "endpoints": {"L0": {"checkpoint_sha256": "frozen"}},
        "export": {"P_rows": list(evaluation.EXPORT_ROWS), "batch_sizes": [1]},
    }
    receipt = evaluation._export("L0", record)
    directory = training.destination("evaluation/exports/L0")
    assert receipt["status"] == "COMPLETE"
    replay = common.read_json(directory / "REPLAY.json")
    assert len(replay["checks"]) == 4 and not replay["dynamic_batch_supported"]
    assert replay["optimizer_updates"] == 0
    assert evaluation._export("L0", record) == receipt
    assert evaluation.replay_export("L0")["status"] == "PASS"
    with (directory / "examples.npz").open("ab") as stream:
        stream.write(b"change")
    with pytest.raises(ValueError, match="Export bytes"):
        evaluation._export("L0", record)


def test_replication_trigger_checks_every_registered_safeguard():
    import copy

    a5 = {
        "overall_MiD": 100.0,
        "complete_prediction_and_mid_coverage": True,
        "bins": {
            name: {"group_macro_mid": 100.0} for name in ("crucial", "small", "large", "negative")
        },
        "diagnostics": {"mae_s": 5.0, "sign_error_rate": 0.05},
    }
    l0, lw = copy.deepcopy(a5), copy.deepcopy(a5)
    lw["overall_MiD"] = 96.0
    assert evaluation.replication_decision(a5, l0, lw)["enabled"]
    for column, value, reason in (
        ("mae_s", 5.11, "mae_max_2pct"),
        ("sign_error_rate", 0.056, "sign_max_0_5pp"),
    ):
        bad = copy.deepcopy(lw)
        bad["diagnostics"][column] = value
        decision = evaluation.replication_decision(a5, l0, bad)
        assert not decision["enabled"] and reason in decision["failed_checks"]
    bad = copy.deepcopy(lw)
    bad["bins"]["negative"]["group_macro_mid"] = 110.1
    assert not evaluation.replication_decision(a5, l0, bad)["enabled"]
    bad = copy.deepcopy(lw)
    bad["overall_MiD"] = 98.0
    assert not evaluation.replication_decision(a5, l0, bad)["enabled"]
