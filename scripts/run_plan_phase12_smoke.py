"""Run the bounded, fixture-only preparation smoke for PLAN.md Phase 12.

This runner deliberately exercises the existing robustness, calibration,
low-label-manifest, and deployment-export APIs without training a model or
opening any external dataset.  Every numerical value in the resulting JSON is
either returned by an existing API on a synthetic fixture or is a runtime
equivalence check.  The artifact is diagnostic evidence only and is not a
dataset result.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402
from e_jepa_ttc.data.types import EventBatch  # noqa: E402
from e_jepa_ttc.evaluation.calibration import (  # noqa: E402
    fit_conformal_interval,
    fit_temperature_scaler,
    interval_metrics,
)
from e_jepa_ttc.evaluation.robustness import evaluate_robustness  # noqa: E402
from e_jepa_ttc.models.object_jepa import ObjectCentricEventJEPA, ObjectJEPAConfig  # noqa: E402
from e_jepa_ttc.models.tiny_cnn import TinyCNNRegressor  # noqa: E402
from e_jepa_ttc.representations.corruptions import (  # noqa: E402
    EventCorruptionSpec,
    corrupt_event_batch,
)
from e_jepa_ttc.runtime.benchmark import benchmark_object_ttc_model  # noqa: E402
from e_jepa_ttc.runtime.export import export_object_ttc_onnx  # noqa: E402

DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "audits" / "plan_phase12_smoke_v1"
LOW_LABEL_FRACTIONS = (1.0, 0.25, 0.10, 0.05, 0.01)


class _SyntheticEventDataset(Dataset[dict[str, object]]):
    """Small deterministic raw-event fixture used by the robustness API."""

    def __init__(self, size: int = 6) -> None:
        if size <= 0:
            raise ValueError("Synthetic dataset size must be positive.")
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, object]:
        if index < 0 or index >= self.size:
            raise IndexError(index)
        base = np.arange(128, dtype=np.int32)
        timestamps = np.arange(128, dtype=np.int64) * 500
        return {
            "events": EventBatch(
                x=(base * 3 + index) % 32,
                y=(base * 5 + 2 * index) % 32,
                t_us=timestamps,
                polarity=np.where((base + index) % 2, 1, -1).astype(np.int8),
                width=32,
                height=32,
                sequence_id=f"phase12-robust-{index}",
                t_start_us=0,
                t_end_us=64_000,
            ),
            "sample_id": f"phase12-robust-{index}",
        }


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return result.stdout.decode("utf-8").strip()


def _git_diff_sha256() -> str:
    result = subprocess.run(
        ["git", "diff", "--binary"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return _sha256_bytes(result.stdout)


def _event_fingerprint(events: EventBatch) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "width": events.width,
                "height": events.height,
                "sequence_id": events.sequence_id,
                "t_start_us": events.t_start_us,
                "t_end_us": events.t_end_us,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    for array in (events.x, events.y, events.t_us, events.polarity):
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    paths = (
        "src/e_jepa_ttc/evaluation/robustness.py",
        "src/e_jepa_ttc/evaluation/calibration.py",
        "scripts/create_low_label_manifest.py",
        "src/e_jepa_ttc/runtime/export.py",
        "src/e_jepa_ttc/runtime/benchmark.py",
    )
    return {path: _sha256_file(ROOT / path) for path in paths}


def _fixture_events(dataset: _SyntheticEventDataset, index: int) -> EventBatch:
    value = dataset[index]["events"]
    if not isinstance(value, EventBatch):
        raise TypeError("Synthetic robustness fixture did not return EventBatch.")
    return value


def _run_robustness_smoke(seed: int) -> tuple[dict[str, Any], list[Path]]:
    dataset = _SyntheticEventDataset()
    before = {
        str(index): _event_fingerprint(_fixture_events(dataset, index))
        for index in range(len(dataset))
    }
    specs = [
        EventCorruptionSpec(kind="none", severity=0.0, seed=seed),
        EventCorruptionSpec(kind="event_dropout", severity=0.5, seed=seed),
        EventCorruptionSpec(kind="timestamp_jitter_us", severity=1_000.0, seed=seed),
        EventCorruptionSpec(kind="temporal_window_fraction", severity=0.8, seed=seed),
    ]
    contract_checks: list[dict[str, Any]] = []
    for spec in specs:
        for index in range(len(dataset)):
            original = _fixture_events(dataset, index)
            corrupted = corrupt_event_batch(original, spec, seed_offset=index)
            monotonic = bool(corrupted.t_us.size < 2 or np.all(np.diff(corrupted.t_us) >= 0))
            in_bounds = bool(
                np.all((corrupted.x >= 0) & (corrupted.x < corrupted.width))
                and np.all((corrupted.y >= 0) & (corrupted.y < corrupted.height))
            )
            same_sequence = corrupted.sequence_id == original.sequence_id
            no_future_timestamp = corrupted.t_us.size == 0 or (
                int(corrupted.t_us.max()) < corrupted.t_end_us
            )
            if not (monotonic and in_bounds and same_sequence and no_future_timestamp):
                raise AssertionError(f"Corruption contract failed for {spec.kind}, sample {index}.")
            contract_checks.append(
                {
                    "kind": spec.kind,
                    "sample_index": index,
                    "timestamp_monotonic": monotonic,
                    "coordinates_in_bounds": in_bounds,
                    "sequence_preserved": same_sequence,
                    "no_future_timestamp": no_future_timestamp,
                }
            )

    torch.manual_seed(seed)
    model = TinyCNNRegressor(in_channels=10, width=8)
    result = evaluate_robustness(
        model=model,
        dataset=dataset,
        device=torch.device("cpu"),
        corruptions=specs,
    )
    after = {
        str(index): _event_fingerprint(_fixture_events(dataset, index))
        for index in range(len(dataset))
    }
    source_unchanged = before == after
    if not source_unchanged:
        raise AssertionError("Robustness evaluation mutated the source fixture.")
    finite = all(
        int(summary["finite_prediction_count"]) == len(dataset)
        for summary in result["results"].values()
    )
    if not finite:
        raise AssertionError("Robustness smoke produced a non-finite prediction.")
    return (
        {
            "status": "passed",
            "fixture": "synthetic_raw_events",
            "metric_scope": "diagnostic_fixture_only",
            "api_result": result,
            "raw_event_contract_checks": len(contract_checks),
            "source_fixture_unchanged": source_unchanged,
            "all_predictions_finite": finite,
        },
        [],
    )


def _run_calibration_smoke(seed: int) -> dict[str, Any]:
    del seed  # The arrays are deterministic by construction; no random fit is used.
    calibration_ids = np.asarray([f"calibration-{index:02d}" for index in range(12)])
    evaluation_ids = np.asarray([f"holdout-{index:02d}" for index in range(12)])
    disjoint = not set(calibration_ids.tolist()).intersection(evaluation_ids.tolist())
    if not disjoint:
        raise AssertionError("Calibration and holdout IDs overlap.")

    calibration_target = np.linspace(0.5, 1.6, calibration_ids.size, dtype=np.float64)
    calibration_mean = calibration_target + np.linspace(-0.08, 0.08, calibration_ids.size)
    calibration_std = np.linspace(0.08, 0.14, calibration_ids.size, dtype=np.float64)
    conformal = fit_conformal_interval(
        calibration_target,
        calibration_mean,
        calibration_std,
        coverage=0.9,
        min_support=8,
    )

    holdout_target = np.linspace(0.55, 1.65, evaluation_ids.size, dtype=np.float64)
    holdout_mean = holdout_target + np.linspace(0.05, -0.05, evaluation_ids.size)
    holdout_std = np.linspace(0.09, 0.15, evaluation_ids.size, dtype=np.float64)
    lower, upper = conformal.interval(holdout_mean, holdout_std)
    interval = interval_metrics(holdout_target, lower, upper)

    calibration_logits = np.linspace(-2.0, 2.0, calibration_ids.size, dtype=np.float64)
    calibration_labels = (calibration_target < 1.1).astype(np.float64)
    scaler = fit_temperature_scaler(
        calibration_logits,
        calibration_labels,
        min_support=8,
    )
    holdout_logits = np.linspace(-1.8, 2.2, evaluation_ids.size, dtype=np.float64)
    holdout_probabilities = scaler.probabilities(holdout_logits)
    if not np.isfinite(holdout_probabilities).all():
        raise AssertionError("Temperature scaling produced non-finite probabilities.")

    return {
        "status": "passed",
        "fixture": "synthetic_calibration_and_holdout",
        "fit_split": "calibration",
        "evaluation_split": "synthetic_holdout",
        "calibration_count": int(calibration_ids.size),
        "evaluation_count": int(evaluation_ids.size),
        "calibration_ids_sha256": _sha256_bytes("\n".join(calibration_ids).encode("utf-8")),
        "evaluation_ids_sha256": _sha256_bytes("\n".join(evaluation_ids).encode("utf-8")),
        "disjoint_ids": disjoint,
        "conformal": {
            "support_status": conformal.support_status,
            "calibration_count": conformal.calibration_count,
            "minimum_support": conformal.minimum_support,
            "coverage_nominal": conformal.coverage,
            "scale_fitted_on_calibration": conformal.scale,
            "holdout_interval_metrics": interval,
        },
        "temperature": {
            "support_status": scaler.support_status,
            "calibration_count": scaler.calibration_count,
            "minimum_support": scaler.minimum_support,
            "temperature_fitted_on_calibration": scaler.temperature,
            "holdout_probability_mean": float(np.mean(holdout_probabilities)),
        },
    }


def _canonical_low_label_hash(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    encoded = json.dumps(
        unsigned,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _fraction_filename(fraction: float, seed: int) -> str:
    return f"evttc_frac{int(round(fraction * 100))}_seed{seed}.json"


def _run_low_label_smoke(output_dir: Path, seed: int) -> tuple[dict[str, Any], list[Path]]:
    cache_path = output_dir / "low_label_fixture.npz"
    train_sequences = np.repeat(
        np.asarray([f"phase12-train-{index}" for index in range(4)]),
        100,
    )
    validation_sequences = np.repeat(
        np.asarray([f"phase12-validation-{index}" for index in range(2)]),
        40,
    )
    split = np.concatenate(
        (
            np.full(train_sequences.size, "train"),
            np.full(validation_sequences.size, "validation"),
        )
    )
    sequence_ids = np.concatenate((train_sequences, validation_sequences))
    sample_ids = np.asarray([f"phase12-sample-{index:04d}" for index in range(split.size)])
    np.savez_compressed(cache_path, split=split, sequence_id=sequence_ids, sample_id=sample_ids)

    manifest_dir = output_dir / "low_label_manifests"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "create_low_label_manifest.py"),
        "--cache",
        str(cache_path),
        "--output-dir",
        str(manifest_dir),
        "--seeds",
        str(seed),
        "--fractions",
        *(str(fraction) for fraction in LOW_LABEL_FRACTIONS),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Low-label manifest smoke failed:\n"
            f"stdout={completed.stdout}\n"
            f"stderr={completed.stderr}"
        )

    train_indices = np.flatnonzero(split == "train")
    validation_indices = np.flatnonzero(split == "validation")
    train_sequence_set = set(sequence_ids[train_indices].tolist())
    validation_sequence_set = set(sequence_ids[validation_indices].tolist())
    sequence_split_disjoint = not train_sequence_set.intersection(validation_sequence_set)
    if not sequence_split_disjoint:
        raise AssertionError("Synthetic low-label train/validation sequences overlap.")

    manifests: dict[str, dict[str, Any]] = {}
    artifact_paths = [cache_path]
    ordered_sets: list[set[int]] = []
    counts: dict[str, int] = {}
    for fraction in LOW_LABEL_FRACTIONS:
        path = manifest_dir / _fraction_filename(fraction, seed)
        if not path.is_file():
            raise FileNotFoundError(f"Missing low-label manifest: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("sha256") != _canonical_low_label_hash(payload):
            raise AssertionError(f"Low-label manifest hash mismatch: {path}")
        indices = [int(value) for value in payload["global_indices"]]
        selected = set(indices)
        if len(indices) != len(selected):
            raise AssertionError(f"Duplicate low-label indices: {path}")
        if not selected.issubset(set(int(value) for value in train_indices)):
            raise AssertionError(f"Low-label manifest selected non-train rows: {path}")
        if set(payload.get("sequence_ids", [])) != set(sequence_ids[indices].tolist()):
            raise AssertionError(f"Low-label sequence IDs do not match indices: {path}")
        manifests[str(fraction)] = {
            "path": path.relative_to(output_dir).as_posix(),
            "sha256": _sha256_file(path),
            "declared_payload_sha256": payload["sha256"],
            "selected_count": len(selected),
            "sequence_count": len(set(sequence_ids[indices].tolist())),
            "fraction": float(payload["fraction"]),
        }
        ordered_sets.append(selected)
        counts[str(fraction)] = len(selected)
        artifact_paths.append(path)

    for larger, smaller in zip(ordered_sets[:-1], ordered_sets[1:], strict=True):
        if not smaller.issubset(larger) or len(smaller) >= len(larger):
            raise AssertionError("Low-label manifests are not strictly nested.")

    return (
        {
            "status": "passed",
            "fixture": "synthetic_npz_train_validation",
            "manifest_generator": "scripts/create_low_label_manifest.py",
            "seed": seed,
            "train_count": int(train_indices.size),
            "validation_count": int(validation_indices.size),
            "train_sequence_count": len(train_sequence_set),
            "validation_sequence_count": len(validation_sequence_set),
            "sequence_split_disjoint": sequence_split_disjoint,
            "selected_rows_are_train_only": True,
            "full_train_reference_count": int(train_indices.size),
            "fractions": manifests,
            "counts": counts,
        },
        artifact_paths,
    )


def _export_inputs(model: ObjectCentricEventJEPA, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    return {
        "context_events": torch.randn(1, 3, 4, 16, 16),
        "context_boxes": torch.tensor([[[[0.2, 0.2, 0.8, 0.8]]] * 3]),
        "context_object_mask": torch.ones(1, 3, 1, dtype=torch.bool),
        "context_sampling_boxes": torch.tensor([[[[0.0, 0.0, 1.0, 1.0]]] * 3]),
        "context_ego_actions": torch.zeros(1, 3, model.config.action_dim),
        "context_ego_action_mask": torch.zeros(1, 3, dtype=torch.bool),
    }


def _run_export_smoke(output_dir: Path, seed: int) -> tuple[dict[str, Any], list[Path]]:
    torch.manual_seed(seed)
    model = ObjectCentricEventJEPA(
        ObjectJEPAConfig(
            in_channels=4,
            embedding_dim=16,
            feature_dim=16,
            predictor_depth=1,
            predictor_heads=4,
            dropout=0.0,
            pre_cropped_events=True,
        )
    ).eval()
    inputs = _export_inputs(model, seed)
    export_dir = output_dir / "export"
    metadata = export_object_ttc_onnx(model, inputs, output_dir=export_dir)
    latency = benchmark_object_ttc_model(
        model,
        inputs,
        device="cpu",
        warmup_iterations=1,
        measured_iterations=2,
    )
    required = (
        "model.onnx",
        "model_metadata.json",
        "normalization.json",
        "example_input.npz",
        "example_output.json",
    )
    paths = [export_dir / name for name in required]
    if any(not path.is_file() or path.stat().st_size <= 0 for path in paths):
        raise AssertionError("ONNX smoke did not produce all required local artifacts.")
    if metadata.get("verified_with_onnxruntime_cpu") is not True:
        raise AssertionError("ONNX export was not verified with the CPU runtime.")
    if metadata.get("batch_size_contract") != 1 or metadata.get("non_batch_axes_fixed") is not True:
        raise AssertionError("ONNX export metadata violates the fixed-input contract.")
    maximum_errors = metadata.get("maximum_absolute_error", {})
    if not maximum_errors or not all(
        np.isfinite(float(value)) for value in maximum_errors.values()
    ):
        raise AssertionError("ONNX equivalence errors are missing or non-finite.")
    return (
        {
            "status": "passed",
            "fixture": "synthetic_object_jepa_batch_size_one",
            "metric_scope": "local_runtime_equivalence_and_cpu_latency",
            "metadata": metadata,
            "latency": latency,
            "required_files": [path.relative_to(output_dir).as_posix() for path in paths],
        },
        paths,
    )


def _artifact_records(output_dir: Path, paths: Iterator[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.is_file():
            raise FileNotFoundError(f"Expected smoke artifact does not exist: {resolved}")
        records.append(
            {
                "path": resolved.relative_to(output_dir.resolve()).as_posix(),
                "sha256": _sha256_file(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )
    return sorted(records, key=lambda item: str(item["path"]))


def run_smoke(output_dir: str | Path, *, seed: int = 7) -> dict[str, Any]:
    """Run all bounded Phase 12 preparation checks and write one signed JSON."""

    if seed < 0:
        raise ValueError("seed must be non-negative.")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)
    torch.manual_seed(seed)

    robustness, robustness_paths = _run_robustness_smoke(seed)
    calibration = _run_calibration_smoke(seed)
    low_label, low_label_paths = _run_low_label_smoke(output, seed)
    export, export_paths = _run_export_smoke(output, seed)
    no_leakage = {
        "robustness_source_fixture_unchanged": robustness["source_fixture_unchanged"],
        "calibration_holdout_ids_disjoint": calibration["disjoint_ids"],
        "low_label_sequence_splits_disjoint": low_label["sequence_split_disjoint"],
        "low_label_rows_train_only": low_label["selected_rows_are_train_only"],
        "test_split_not_opened": True,
        "evttc_labels_not_opened": True,
        "external_data_not_opened": True,
        "codabench_not_contacted": True,
    }
    if not all(no_leakage.values()):
        raise AssertionError(f"Phase 12 smoke leakage audit failed: {no_leakage}")

    report_path = output / "phase12_smoke.json"
    artifact_paths = [*robustness_paths, *low_label_paths, *export_paths]
    payload: dict[str, Any] = {
        "artifact_type": "plan_phase12_smoke_v1",
        "schema_version": "1.0",
        "status": "passed",
        "evidence_type": "synthetic_fixture_and_local_artifact_smoke",
        "seed": seed,
        "metrics_are_not_real_dataset_results": True,
        "scope": {
            "external_data": False,
            "network": False,
            "training": False,
            "codabench": False,
            "evttc_used_for_selection": False,
            "metric_interpretation": "fixture diagnostics or runtime equivalence only",
        },
        "provenance": {
            "git_commit": _git_value("rev-parse", "HEAD"),
            "git_diff_sha256": _git_diff_sha256(),
            "runner_sha256": _sha256_file(Path(__file__).resolve()),
            "api_source_sha256": _source_hashes(),
            "generated_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        },
        "checks": {
            "robustness": robustness,
            "calibration": calibration,
            "low_label": low_label,
            "export": export,
            "no_leakage": no_leakage,
        },
        "artifacts": _artifact_records(output, iter(artifact_paths)),
    }
    sign_artifact(payload)
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if not verify_artifact_hash(payload):
        raise AssertionError("Phase 12 smoke artifact signature did not verify.")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    try:
        payload = run_smoke(args.output_dir, seed=args.seed)
    except Exception as exc:  # pragma: no cover - CLI boundary diagnostics
        print(f"Phase 12 smoke failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output": (Path(args.output_dir) / "phase12_smoke.json").as_posix(),
                "status": payload["status"],
                "artifact_sha256": payload["artifact_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
