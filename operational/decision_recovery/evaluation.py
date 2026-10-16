"""Full P/H endpoint evaluation and fixed-batch exports, isolated inside V19."""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from . import common, training
from .router import _bootstrap, development_gate

if TYPE_CHECKING:
    from operational.joint_measurement.geometry_model import GeometricPairMeasurement

ARMS = ("L0", "LW")
POPULATIONS = {"P": 53890, "H": 18422}
EXPORT_ROWS = (0, 17, 83, 159)
BATCH = 4
CHUNK = 256


def _directory(arm: str, seed: int = 7) -> Path:
    if arm not in ARMS or seed != 7:
        raise ValueError("Only registered L0/LW seed7 endpoints are admissible")
    return training.destination(f"training/fits/FULL-{arm}-S{seed}")


def _freeze(relative: str, value: dict) -> dict:
    return training._freeze(f"evaluation/{relative}", value)


def _endpoint(arm: str, registration: dict) -> dict:
    """Verify physical checkpoint bytes and durable boundaries without loading CUDA."""
    directory = _directory(arm)
    receipt = common.read_json(directory / "CHECKPOINT_RECEIPT.json")
    pointer = common.read_json(directory / "CHECKPOINT_POINTER.json")
    journal = common.read_json(directory / "UPDATE_JOURNAL.json")
    identity = common.read_json(directory / "IDENTITY.json")
    registration_sha = common.sha(training.destination("training/REGISTRATION.json"))
    version = (directory / "checkpoint_versions" / pointer["version"]).resolve()
    if not version.is_relative_to((directory / "checkpoint_versions").resolve()):
        raise ValueError("Checkpoint version escaped its fit directory")
    if (
        receipt["status"] != "COMPLETE"
        or not receipt["scientific_endpoint"]
        or not receipt["complete_state"]
        or receipt["completed_updates"] != training.ENDPOINT
        or pointer["completed_updates"] != training.ENDPOINT
        or journal["completed_updates"] != training.ENDPOINT
        or journal["durable_updates"] != training.ENDPOINT
        or journal["pending_update_upper"] != 0
        or identity["arm"] != arm
        or identity["seed"] != 7
        or identity["mode"] != "FULL"
        or identity["endpoint"] != training.ENDPOINT
        or identity["population"] != "P"
        or identity["parent_sha256"] != registration["parent_sha256"]
        or identity["registration_sha256"] != registration_sha
        or journal["identity_sha256"] != receipt["identity_sha256"]
        or identity["identity_sha256"] != receipt["identity_sha256"]
        or pointer["checkpoint_sha256"] != receipt["checkpoint_sha256"]
        or common.sha(version) != receipt["checkpoint_sha256"]
        or common.sha(directory / "checkpoint_last.pt") != receipt["checkpoint_sha256"]
    ):
        raise ValueError("Endpoint is incomplete, changed, or not the registered scientific fit")
    return {
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "identity_sha256": common.sha(directory / "IDENTITY.json"),
        "receipt_sha256": common.sha(directory / "CHECKPOINT_RECEIPT.json"),
        "journal_sha256": common.sha(directory / "UPDATE_JOURNAL.json"),
    }


def _register() -> dict:
    """Freeze both endpoints and evaluation sources before any H forward."""
    from .sensor_runtime import runtime_contract

    registration = training.registered()
    endpoints = {arm: _endpoint(arm, registration) for arm in ARMS}
    _freeze("ENDPOINTS.json", endpoints)
    paths = [
        Path(__file__),
        common.ROOT / "tests/unit/test_decision_recovery_evaluation.py",
        Path(common.__file__),
        Path(training.__file__),
        common.ROOT / "operational/decision_recovery/router.py",
        common.ROOT / "operational/decision_recovery/sensor_runtime.py",
        common.ROOT / "tests/unit/test_decision_recovery_sensor_runtime.py",
        *[
            common.ROOT / "operational/joint_measurement" / name
            for name in (
                "geometry_export.py",
                "geometry_model.py",
                "data.py",
                "verify_rgb_endpoint.py",
                "parallel_training_sensor.py",
                "fast_runtime.py",
                "prefetched_sensor.py",
            )
        ],
        common.ROOT / "operational/range_transfer/analyze.py",
        common.ROOT / "operational/rgb_port/evaluate.py",
        common.ROOT / "src/e_jepa_ttc/simplex_t/phase.py",
    ]
    sources = {path.relative_to(common.ROOT).as_posix(): common.sha(path) for path in paths}
    source_commit = training.committed_sources(sources)
    prior = training.destination("evaluation/REGISTRATION.json")
    if prior.exists():
        source_commit = common.read_json(prior)["source_commit"]
    record = {
        "schema": "v19_full_loss_evaluation_v1",
        "endpoints": endpoints,
        "training_registration_sha256": common.sha(
            training.destination("training/REGISTRATION.json")
        ),
        "sources": sources,
        "source_commit": source_commit,
        "populations": POPULATIONS,
        "batch_size": BATCH,
        "chunk_size": CHUNK,
        "device": "CUDA_FP32_no_TF32",
        "sensor_runtime": runtime_contract()["evaluation"],
        "transfer": "persistent FP32 buffers; synchronous exact copies; actual tail size",
        "transfer_parity": "first evaluated batch per scope: exact raw phase and emitted TTC",
        "roles": ["P", "H"],
        "export": {
            "device": "CPU",
            "batch_sizes": [1],
            "P_rows": list(EXPORT_ROWS),
            "rtol": 1e-5,
            "atol": 1e-5,
            "outputs": ["TTC", "emitted_phase"],
        },
        "bootstrap": {
            "replicates": 2000,
            "seed": 12345,
            "scope": "descriptive_fixed_predictions_group_bootstrap",
        },
        "cost_complete": False,
        "system_promotion": False,
    }
    return _freeze("REGISTRATION.json", record)


def _load_model(arm: str, record: dict) -> GeometricPairMeasurement:
    """Inspect optimizer/sampler continuity on CPU and load the exact frozen endpoint."""
    import torch

    from operational.joint_measurement.verify_rgb_endpoint import inspect_payload

    directory = _directory(arm)
    payload = torch.load(directory / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    identity = common.read_json(directory / "IDENTITY.json")
    receipt = common.read_json(directory / "CHECKPOINT_RECEIPT.json")
    journal = common.read_json(directory / "UPDATE_JOURNAL.json")
    if (
        payload["identity"] != identity
        or payload["identity_sha256"] != receipt["identity_sha256"]
        or payload["recovery_upper"] != journal["recovery_upper"]
        or common.sha(directory / "checkpoint_last.pt")
        != record["endpoints"][arm]["checkpoint_sha256"]
    ):
        raise ValueError("Endpoint payload identity changed")
    model = training.make_model()
    proof = inspect_payload(
        model,
        payload,
        updates=training.ENDPOINT,
        population=POPULATIONS["P"],
        batch_size=32,
        seed=7,
        frozen=False,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    proof.update(endpoint=record["endpoints"][arm], optimizer_updates=0)
    common.write_json(training.destination(f"evaluation/verification/{arm}.json"), proof)
    return model.eval().requires_grad_(False)


def _check_frame(frame: pd.DataFrame, rows: pd.DataFrame, role: str) -> None:
    if len(frame) != len(rows) or frame.sample_token.duplicated().any():
        raise ValueError("Incomplete or duplicate evaluation population")
    for column in common.IDENTITY:
        if not np.array_equal(frame[column], rows[column]):
            raise ValueError(f"Evaluation population identity mismatch: {column}")
    if set(frame.role) != {role}:
        raise ValueError("Evaluation role changed")
    values = frame[["prediction_ttc", "raw_phase"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or not ((values[:, 0] < 0) | (values[:, 0] > 0.1)).all():
        raise ValueError(
            "Invalid output retained as failed evidence; cannot mark coverage complete"
        )


def _chunk(
    path: Path, binding: str, begin: int, stop: int, rows: pd.DataFrame, role: str
) -> pd.DataFrame:
    receipt = common.read_json(path.with_suffix(".json"))
    if (
        receipt["binding"] != binding
        or receipt["start"] != begin
        or receipt["stop"] != stop
        or common.sha(path) != receipt["sha256"]
    ):
        raise ValueError("Evaluation chunk receipt or bytes changed")
    frame = pd.read_parquet(path)
    _check_frame(frame, rows.iloc[begin:stop], role)
    return frame


def _completed(
    directory: Path, binding: str, rows: pd.DataFrame, role: str
) -> tuple[dict, pd.DataFrame]:
    receipt = common.read_json(directory / "COMPLETE.json")
    if (
        receipt["status"] != "COMPLETE"
        or receipt["binding"] != binding
        or receipt["rows"] != len(rows)
        or common.sha(directory / "predictions.parquet") != receipt["sha256"]
        or common.sha(directory / "METRICS.json") != receipt["metrics_sha256"]
    ):
        raise ValueError("Completed predictions or metrics changed")
    frame = pd.read_parquet(directory / "predictions.parquet")
    _check_frame(frame, rows, role)
    if common.score(frame, frame.prediction_ttc.to_numpy()) != common.read_json(
        directory / "METRICS.json"
    ):
        raise ValueError("Metrics do not reproduce from complete frozen predictions")
    return receipt, frame


def _finish(directory: Path, binding: str, rows: pd.DataFrame, role: str) -> dict:
    frames = [
        _chunk(
            directory / f"chunk_{begin // CHUNK:05d}.parquet",
            binding,
            begin,
            min(begin + CHUNK, len(rows)),
            rows,
            role,
        )
        for begin in range(0, len(rows), CHUNK)
    ]
    frame = pd.concat(frames, ignore_index=True)
    _check_frame(frame, rows, role)
    pending, final = directory / "predictions.pending", directory / "predictions.parquet"
    frame.to_parquet(pending, index=False)
    os.replace(pending, final)
    common.write_json(
        directory / "METRICS.json", common.score(frame, frame.prediction_ttc.to_numpy())
    )
    receipt = {
        "status": "COMPLETE",
        "binding": binding,
        "rows": len(frame),
        "sha256": common.sha(final),
        "metrics_sha256": common.sha(directory / "METRICS.json"),
        "optimizer_updates": 0,
    }
    common.write_json(directory / "COMPLETE.json", receipt)
    return _completed(directory, binding, rows, role)[0]


def _role_binding(arm: str, role: str, record: dict) -> tuple[Path, str, pd.DataFrame]:
    from operational.range_transfer.common import digest

    if role not in POPULATIONS:
        raise ValueError("Only registered P/H populations are allowed")
    manifest = training.manifest_path(role)
    rows = training.role_rows(role)
    complete = common.read_json(training.cache_root(role) / "COMPLETE.json")
    sensor_binding = common.read_json(training.cache_root(role) / "BINDING.json")
    if (
        len(rows) != POPULATIONS[role]
        or complete["status"] != "COMPLETE"
        or complete["rows"] != len(rows)
        or sensor_binding["manifest_sha256"] != common.sha(manifest)
        or complete["binding"] != digest(sensor_binding)
    ):
        raise ValueError("Full native sensor/manifest population is required")
    identity = {
        "arm": arm,
        "seed": 7,
        "role": role,
        "endpoint": record["endpoints"][arm],
        "evaluation_registration_sha256": common.sha(
            training.destination("evaluation/REGISTRATION.json")
        ),
        "manifest_sha256": common.sha(manifest),
        "sensor_complete_sha256": common.sha(training.cache_root(role) / "COMPLETE.json"),
        "rows_sha256": common.read_json(manifest)["rows_sha256"],
    }
    _freeze(f"predictions/{role}/{arm}/IDENTITY.json", identity)
    return training.destination(f"evaluation/predictions/{role}/{arm}"), digest(identity), rows


def _predict_role(arm: str, role: str, record: dict) -> dict:
    import torch

    from e_jepa_ttc.simplex_t.phase import phase_to_ttc
    from operational.joint_measurement.train import setup

    from .sensor_runtime import make_evaluation_sensors

    directory, key, rows = _role_binding(arm, role, record)
    if (directory / "COMPLETE.json").exists():
        return _completed(directory, key, rows, role)[0]
    dataset = make_evaluation_sensors(role)
    model = None
    try:
        if (
            not dataset.rows[common.IDENTITY]
            .reset_index(drop=True)
            .equals(rows[common.IDENTITY].reset_index(drop=True))
        ):
            raise ValueError("Loader metadata differs from admitted population")
        with training.gpu_charge(f"V19_EVALUATE_{arm}_{role}", reserve=900) as deadline:
            setup(7)
            model = _load_model(arm, record).cuda()
            events_buffer, delta_buffer = None, None
            transfer_checked = False
            with torch.inference_mode():
                for begin in range(0, len(rows), CHUNK):
                    stop = min(begin + CHUNK, len(rows))
                    path = directory / f"chunk_{begin // CHUNK:05d}.parquet"
                    if path.with_suffix(".json").exists():
                        _chunk(path, key, begin, stop, rows, role)
                        continue
                    if time.perf_counter() >= deadline:
                        return {"status": "PARTIAL", "arm": arm, "role": role, "next_row": begin}
                    training.require_resources()
                    predictions, phases = [], []
                    for start in range(begin, stop, BATCH):
                        if time.perf_counter() >= deadline:
                            return {
                                "status": "PARTIAL",
                                "arm": arm,
                                "role": role,
                                "next_row": begin,
                            }
                        batch = dataset.batch(list(range(start, min(start + BATCH, stop))))
                        if events_buffer is None:
                            events_buffer = torch.empty(
                                (BATCH, *batch.events.shape[1:]),
                                dtype=batch.events.dtype,
                                device="cuda",
                            )
                            delta_buffer = torch.empty(
                                BATCH, dtype=batch.delta_t_s.dtype, device="cuda"
                            )
                        count = len(batch.events)
                        events_buffer[:count].copy_(batch.events)
                        delta_buffer[:count].copy_(batch.delta_t_s.reshape(-1))
                        raw = model(events_buffer[:count], delta_buffer[:count])["raw_phase"]
                        if not transfer_checked:
                            reference_raw = model(
                                batch.events.cuda(), batch.delta_t_s.cuda().reshape(-1)
                            )["raw_phase"]
                            if not torch.equal(raw, reference_raw) or not torch.equal(
                                phase_to_ttc(raw), phase_to_ttc(reference_raw)
                            ):
                                raise ValueError("Persistent input transfer changed native forward")
                            common.write_json(
                                directory / f"TRANSFER_PARITY_{begin:06d}.json",
                                {
                                    "status": "PASS",
                                    "start": start,
                                    "rows": count,
                                    "raw_phase_exact": True,
                                    "emitted_ttc_exact": True,
                                    "binding": key,
                                    "optimizer_updates": 0,
                                },
                            )
                            transfer_checked = True
                        predictions.extend(phase_to_ttc(raw).cpu().double().tolist())
                        phases.extend(raw.cpu().double().tolist())
                    frame = rows.iloc[begin:stop][common.IDENTITY].copy()
                    frame["prediction_ttc"], frame["raw_phase"] = predictions, phases
                    pending = path.with_suffix(".pending")
                    frame.to_parquet(pending, index=False)
                    os.replace(pending, path)
                    _check_frame(frame, rows.iloc[begin:stop], role)
                    common.write_json(
                        path.with_suffix(".json"),
                        {"binding": key, "start": begin, "stop": stop, "sha256": common.sha(path)},
                    )
                    print(f"V19 EVALUATE {arm} {role} {stop}/{len(rows)}", flush=True)
        return _finish(directory, key, rows, role)
    finally:
        dataset.shutdown()
        del model
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def _export(arm: str, record: dict) -> dict:
    import torch
    from torch import nn

    from e_jepa_ttc.simplex_t.phase import phase_to_ttc, ttc_to_phase
    from operational.joint_measurement.data import SensorDataset
    from operational.joint_measurement.geometry_export import PortableGeometry

    directory = training.destination(f"evaluation/exports/{arm}")
    binding = {
        "endpoint": record["endpoints"][arm],
        "contract": record["export"],
        "evaluation_registration_sha256": common.sha(
            training.destination("evaluation/REGISTRATION.json")
        ),
    }
    _freeze(f"exports/{arm}/BINDING.json", binding)
    if (directory / "COMPLETE.json").exists():
        receipt = common.read_json(directory / "COMPLETE.json")
        if any(common.sha(directory / name) != value for name, value in receipt["files"].items()):
            raise ValueError("Export bytes changed")
        return receipt

    class Outputs(nn.Module):
        def __init__(self, model: GeometricPairMeasurement) -> None:
            super().__init__()
            self.portable = PortableGeometry(model)

        def forward(
            self, events: torch.Tensor, delta: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            prediction = self.portable(events, delta)
            return prediction, ttc_to_phase(prediction)

    dataset = SensorDataset("P")
    try:
        torch.set_num_threads(1)
        model = _load_model(arm, record).cpu()
        inputs = []
        for index in EXPORT_ROWS:
            batch = dataset.batch([index])
            inputs.append((batch.events, batch.delta_t_s.reshape(1, 1)))
        wrapper = Outputs(model).eval()
        with torch.inference_mode():
            targets = []
            for example in inputs:
                prediction = phase_to_ttc(model(*example)["raw_phase"])
                target = (prediction, ttc_to_phase(prediction))
                if not all(
                    torch.equal(a, b) for a, b in zip(wrapper(*example), target, strict=True)
                ):
                    raise ValueError("Portable eager adapter changed the native outputs")
                targets.append(target)
            traced = torch.jit.trace(wrapper, inputs[0], check_inputs=inputs[1:], strict=True)
            path = directory / "model.torchscript.pt"
            with path.open("wb") as stream:
                torch.jit.save(traced, stream)
            with path.open("rb") as stream:
                loaded = torch.jit.load(stream, map_location="cpu").eval()
            checks = []
            for index, (example, expected) in enumerate(zip(inputs, targets, strict=True)):
                actual = loaded(*example)
                for value, target in zip(actual, expected, strict=True):
                    torch.testing.assert_close(value, target, rtol=1e-5, atol=1e-5)
                checks.append(
                    {
                        "P_row": EXPORT_ROWS[index],
                        "expected": [v.tolist() for v in expected],
                        "actual": [v.tolist() for v in actual],
                        "maxabs": [
                            float((a - b).abs().max())
                            for a, b in zip(actual, expected, strict=True)
                        ],
                    }
                )
            np.savez(
                directory / "examples.npz",
                events=np.concatenate([x[0].numpy() for x in inputs]),
                delta_t_s=np.concatenate([x[1].numpy() for x in inputs]),
            )
        common.write_json(
            directory / "REPLAY.json",
            {
                "status": "PASS",
                "checks": checks,
                "device": "CPU",
                "optimizer_updates": 0,
                "rtol": 1e-5,
                "atol": 1e-5,
                "eager_adapter_bit_exact": True,
                "batch_size": 1,
                "dynamic_batch_supported": False,
                "phase_output": "emitted_phase_not_raw_phase",
            },
        )
        names = ("BINDING.json", "model.torchscript.pt", "examples.npz", "REPLAY.json")
        receipt = {
            "status": "COMPLETE",
            "files": {name: common.sha(directory / name) for name in names},
            "includes_sensor_preprocessing": False,
            "system_promotion": False,
        }
        common.write_json(directory / "COMPLETE.json", receipt)
        return receipt
    finally:
        dataset.close()


def evaluate(arm: str, seed: int = 7) -> dict:
    """Evaluate one fixed arm in resumable scopes; no optimizer updates or V access."""
    _directory(arm, seed)
    with (
        training.file_lock(training.destination("training/DRIVER.lock")),
        training.runtime_context(),
    ):
        record = _register()
        results = {}
        for role in POPULATIONS:
            results[role] = _predict_role(arm, role, record)
            if results[role]["status"] != "COMPLETE":
                return results[role]
        export = _export(arm, record)
        replay = replay_export(arm, seed)
        value = {
            "status": "COMPLETE",
            "arm": arm,
            "seed": seed,
            "predictions": results,
            "export": export,
            "export_replay": replay,
            "optimizer_updates": 0,
            "system_promotion": False,
        }
        common.write_json(training.destination(f"evaluation/{arm}_COMPLETE.json"), value)
        return value


def replay_export(arm: str, seed: int = 7) -> dict:
    """Replay saved input examples on CPU without reopening historical producers."""
    import torch

    _directory(arm, seed)
    directory = training.destination(f"evaluation/exports/{arm}")
    receipt = common.read_json(directory / "COMPLETE.json")
    if receipt["status"] != "COMPLETE" or any(
        common.sha(directory / name) != expected for name, expected in receipt["files"].items()
    ):
        raise ValueError("Export receipt or payload changed")
    checks = common.read_json(directory / "REPLAY.json")["checks"]
    with (directory / "model.torchscript.pt").open("rb") as stream:
        model = torch.jit.load(stream, map_location="cpu").eval()
    with np.load(directory / "examples.npz", allow_pickle=False) as data, torch.inference_mode():
        if len(data["events"]) != len(EXPORT_ROWS) or len(checks) != len(EXPORT_ROWS):
            raise ValueError("Export example coverage changed")
        for index, check in enumerate(checks):
            actual = model(
                torch.from_numpy(data["events"][index : index + 1].copy()),
                torch.from_numpy(data["delta_t_s"][index : index + 1].copy()),
            )
            for value, expected in zip(actual, check["expected"], strict=True):
                torch.testing.assert_close(
                    value, torch.tensor(expected, dtype=value.dtype), rtol=1e-5, atol=1e-5
                )
    return {
        "status": "PASS",
        "examples": len(checks),
        "device": "CPU",
        "optimizer_updates": 0,
        "raw_phase_exported": False,
    }


def replication_decision(a5: dict, l0: dict, lw: dict) -> dict:
    """Apply the registered loss-screen replication trigger, without a new sweep."""
    development = development_gate(a5, lw)
    checks = {"A5_development_eligible": development["eligible"]}
    if l0["overall_MiD"] is None or lw["overall_MiD"] is None:
        checks["complete_comparison"] = False
    else:
        checks["aggregate_3pct_vs_L0"] = lw["overall_MiD"] <= 0.97 * l0["overall_MiD"]
        for name in ("crucial", "small", "large", "negative"):
            old, new = l0["bins"][name]["group_macro_mid"], lw["bins"][name]["group_macro_mid"]
            checks[f"band_{name}_max_10pct"] = (
                old is not None and new is not None and new <= 1.1 * old
            )
        checks["sign_max_0_5pp"] = (
            lw["diagnostics"]["sign_error_rate"] <= l0["diagnostics"]["sign_error_rate"] + 0.005
        )
        checks["mae_max_2pct"] = lw["diagnostics"]["mae_s"] <= 1.02 * l0["diagnostics"]["mae_s"]
    enabled = all(checks.values())
    return {
        "status": "REPLICATION_ENABLED" if enabled else "REPLICATION_NOT_ENABLED",
        "enabled": enabled,
        "checks": checks,
        "failed_checks": [k for k, v in checks.items() if not v],
        "scope": "H development numerical trigger; no system promotion",
        "replicate_control_and_candidate": enabled,
        "seeds_if_enabled": [13, 23] if enabled else [],
        "cost_complete": False,
        "system_promotion": False,
    }


def report() -> dict[str, Any]:
    """Recalculate all full-cohort comparisons; eligibility is numerical, cost pending."""
    from operational.range_transfer.analyze import gate

    record = _register()
    comparisons, bindings = {}, {}
    for role in POPULATIONS:
        tables = {"A5": common.load_population(role, ["A5"])["A5"]}
        bindings[role] = {"A5": common.source_bindings(role, ["A5"])}
        for arm in ARMS:
            directory, key, rows = _role_binding(arm, role, record)
            receipt, tables[arm] = _completed(directory, key, rows, role)
            _check_frame(tables[arm], tables["A5"], role)
            bindings[role][arm] = receipt
        scored = {
            name: common.score(table, table.prediction_ttc.to_numpy())
            for name, table in tables.items()
        }
        results = {}
        for reference, candidate in (("A5", "L0"), ("A5", "LW"), ("L0", "LW")):
            results[f"{candidate}_vs_{reference}"] = {
                "system_numeric_gates": gate(scored[reference], scored[candidate]),
                "development_gate": development_gate(scored[reference], scored[candidate]),
                "bootstrap": _bootstrap(
                    tables[reference],
                    tables[reference].prediction_ttc.to_numpy(),
                    tables[candidate].prediction_ttc.to_numpy(),
                ),
            }
        comparisons[role] = {"metrics": scored, "contrasts": results}
    result = {
        "status": "FULL_P_H_NUMERICAL_REPORT",
        "comparisons": comparisons,
        "bindings": bindings,
        "evaluation_registration_sha256": common.sha(
            training.destination("evaluation/REGISTRATION.json")
        ),
        "scope": (
            "P training; H reused development; group bootstrap is descriptive, "
            "not independent confirmation"
        ),
        "seed": 7,
        "system_promotion": False,
        "cost_complete": False,
        "claim": "Numerical eligibility only; complete sensor cost and confirmation remain pending",
    }
    h = comparisons["H"]["metrics"]
    result["replication_decision"] = replication_decision(h["A5"], h["L0"], h["LW"])
    common.write_json(training.destination("evaluation/REPORT.json"), result)
    lines = [
        "# V19 FULL-L0 / FULL-LW: evaluación P/H",
        "",
        "P es entrenamiento; H es desarrollo reutilizado. Bootstrap emparejado por grupos "
        "de adquisición: 2.000 réplicas, semilla 12345, descriptivo sobre predicciones fijas.",
        "No es confirmación independiente ni promoción. Coste sensorial completo pendiente.",
        "",
    ]
    for role in POPULATIONS:
        lines += [
            f"## {role}",
            "",
            "| Modelo | MiD | Crítico | Pequeño | Largo | Negativo | MAE s | Error signo |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, metric in comparisons[role]["metrics"].items():
            values = [
                metric["overall_MiD"],
                *[
                    metric["bins"][band]["group_macro_mid"]
                    for band in ("crucial", "small", "large", "negative")
                ],
                metric["diagnostics"]["mae_s"],
                metric["diagnostics"]["sign_error_rate"],
            ]
            display = ["no definido" if value is None else f"{value:.6f}" for value in values]
            lines.append(f"| {name} | " + " | ".join(display) + " |")
        lines.append("")
    decision = result["replication_decision"]
    lines += [
        f"Decisión de réplica: **{decision['status']}**.",
        "",
        "Comprobaciones fallidas: " + (", ".join(decision["failed_checks"]) or "ninguna") + ".",
        "",
        "REPORT.json conserva todas las métricas, gates, intervalos y hashes, incluidos "
        "los resultados negativos. El export admite batch1 y emite TTC y fase emitida; "
        "la fase raw permanece en las tablas de predicciones.",
    ]
    training.destination("evaluation/REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result
