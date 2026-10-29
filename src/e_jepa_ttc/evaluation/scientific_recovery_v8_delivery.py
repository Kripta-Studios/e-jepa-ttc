"""Fail-closed V8 delivery evaluation primitives.

The functions in this module intentionally operate on caller-provided, already
authorised samples.  They never discover a dataset path, make a network request,
or select a model.  This keeps robustness, calibration, latency and export checks
usable on the outer-development split while making accidental access to sealed
evaluation roots an explicit error in the command-line entry points.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from torch import nn

from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.data.benchmark10_guard import assert_no_sealed_benchmark_paths
from e_jepa_ttc.data.types import EventBatch
from e_jepa_ttc.representations.corruptions import EventCorruptionSpec, corrupt_event_batch
from e_jepa_ttc.reproducibility import environment_snapshot

SEALED_V8_PATH_MARKERS = (
    "public_validation",
    "private_test",
    "evttc_test",
    "evttc-official",
    "codabench",
    "coda_bench",
)


@dataclass(frozen=True)
class V8RobustnessSpec:
    """One preregistered target-preserving event perturbation."""

    kind: str
    intensity: float | str
    seed: int
    target_preserved: bool = True


def v8_robustness_specs(seed: int) -> tuple[V8RobustnessSpec, ...]:
    """Return the exact V8 §15 perturbation matrix in stable order."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    numeric = (
        ("event_dropout", (0.1, 0.3, 0.5, 0.7)),
        ("timestamp_jitter_us", (50.0, 200.0, 1000.0)),
        ("background_event_rate", (0.01, 0.05, 0.1)),
        ("hot_pixel_fraction", (0.001, 0.005)),
        ("dead_pixel_fraction", (0.01, 0.05)),
        ("temporal_window_scale", (0.5, 0.75, 1.25, 1.5)),
        ("spatial_crop_fraction", (0.9, 0.75)),
    )
    result = [V8RobustnessSpec(kind, value, seed) for kind, values in numeric for value in values]
    result.extend(
        (
            V8RobustnessSpec("polarity_drop", "positive", seed),
            V8RobustnessSpec("polarity_drop", "negative", seed),
        )
    )
    return tuple(result)


def assert_v8_delivery_paths_safe(paths: Sequence[str | Path]) -> None:
    """Reject sealed V8 inputs without opening, globbing, or enumerating them."""

    assert_no_sealed_benchmark_paths(paths)
    rejected = [
        str(path)
        for path in paths
        if any(marker in str(path).replace("\\", "/").lower() for marker in SEALED_V8_PATH_MARKERS)
    ]
    if rejected:
        raise ValueError("V8 delivery may not open sealed evaluation paths: " + ", ".join(rejected))


def _as_event_sample(sample: object) -> tuple[EventBatch, float | None, str]:
    if isinstance(sample, EventBatch):
        return sample, None, sample.sequence_id
    if not isinstance(sample, Mapping):
        raise TypeError("V8 robustness samples must be EventBatch values or mappings")
    events = sample.get("events", sample.get("context_events"))
    if not isinstance(events, EventBatch):
        raise TypeError("V8 robustness mappings require EventBatch under events or context_events")
    target = sample.get("target_ttc", sample.get("ttc_seconds", sample.get("target")))
    if target is not None and not isinstance(target, (int, float, np.integer, np.floating)):
        raise TypeError("V8 robustness target must be a numeric TTC value when present")
    identifier = str(sample.get("token_id", sample.get("sample_id", events.sequence_id)))
    return events, None if target is None else float(target), identifier


def _model_prediction(output: object) -> tuple[float, float | None]:
    """Extract a scalar TTC and optional log variance from common output forms."""

    if isinstance(output, torch.Tensor):
        values, log_variance = output, None
    elif isinstance(output, tuple):
        if not output or not isinstance(output[0], torch.Tensor):
            raise TypeError("Tuple model output must begin with a prediction tensor")
        values = output[0]
        log_variance = (
            output[1] if len(output) > 1 and isinstance(output[1], torch.Tensor) else None
        )
    elif isinstance(output, Mapping):
        values = output.get("ttc_mean_seconds", output.get("prediction", output.get("pred")))
        log_variance = output.get("ttc_log_variance", output.get("log_variance"))
    else:
        values = getattr(output, "ttc_mean_seconds", getattr(output, "prediction", None))
        log_variance = getattr(output, "ttc_log_variance", getattr(output, "log_variance", None))
    if not isinstance(values, torch.Tensor) or values.numel() != 1:
        raise TypeError("V8 delivery predictor must return exactly one scalar TTC prediction")
    if log_variance is not None and (
        not isinstance(log_variance, torch.Tensor) or log_variance.numel() != 1
    ):
        raise TypeError("V8 delivery log variance must be scalar when provided")
    return float(values.detach().reshape(-1)[0].cpu()), (
        None if log_variance is None else float(log_variance.detach().reshape(-1)[0].cpu())
    )


TemporalHistoryProvider = Callable[[EventBatch, float], EventBatch]
Representation = Callable[[EventBatch], torch.Tensor]


def apply_v8_robustness(
    events: EventBatch,
    spec: V8RobustnessSpec,
    *,
    sample_index: int,
    temporal_history_provider: TemporalHistoryProvider | None = None,
) -> EventBatch:
    """Apply one V8 corruption without mutating its input event batch.

    Longer temporal windows cannot be fabricated from a short context.  For a
    scale above one the caller must supply a raw-history provider which rereads a
    causal prefix ending at the same endpoint.  The TTC target is deliberately
    unchanged in all branches because these are observation corruptions, not time
    rescalings of the underlying world.
    """

    if spec.kind == "polarity_drop":
        if spec.intensity not in {"positive", "negative"}:
            raise ValueError("polarity_drop intensity must be positive or negative")
        kind = f"polarity_drop_{spec.intensity}"
        return corrupt_event_batch(
            events, EventCorruptionSpec(kind=kind, seed=spec.seed), seed_offset=sample_index
        )
    if not isinstance(spec.intensity, (float, int)):
        raise TypeError(f"{spec.kind} requires a numeric intensity")
    intensity = float(spec.intensity)
    if spec.kind == "temporal_window_scale" and intensity > 1.0:
        if temporal_history_provider is None:
            raise ValueError(
                "temporal_window_scale > 1 requires temporal_history_provider; "
                "a longer context may only be reread from causal raw history"
            )
        expanded = temporal_history_provider(events, intensity)
        if not isinstance(expanded, EventBatch):
            raise TypeError("temporal_history_provider must return EventBatch")
        if expanded.sequence_id != events.sequence_id or expanded.t_end_us != events.t_end_us:
            raise ValueError("expanded temporal context must keep sequence and endpoint unchanged")
        if expanded.t_start_us > events.t_start_us:
            raise ValueError("expanded temporal context must not shorten the source window")
        if expanded.t_us.size and int(expanded.t_us.max()) > expanded.t_end_us:
            raise ValueError("expanded temporal context contains future events")
        return expanded
    return corrupt_event_batch(
        events,
        EventCorruptionSpec(kind=spec.kind, severity=intensity, seed=spec.seed),
        seed_offset=sample_index,
    )


def evaluate_v8_robustness(
    model: nn.Module,
    samples: Sequence[object],
    representation: Representation,
    *,
    device: str | torch.device = "cpu",
    seed: int = 7,
    temporal_history_provider: TemporalHistoryProvider | None = None,
) -> dict[str, Any]:
    """Run the frozen V8 matrix and report target preservation and uncertainty shift."""

    if not samples:
        raise ValueError("V8 robustness requires at least one sample")
    target_device = torch.device(device)
    specifications = v8_robustness_specs(seed)
    baseline = V8RobustnessSpec("none", 0.0, seed)
    all_specs = (baseline, *specifications)
    original_fingerprints = [_event_fingerprint(_as_event_sample(sample)[0]) for sample in samples]
    model = model.to(target_device)
    previous_training = model.training
    model.eval()
    results: list[dict[str, Any]] = []
    try:
        for spec in all_specs:
            predictions: list[float] = []
            uncertainty: list[float] = []
            targets: list[float] = []
            target_ids: list[str] = []
            event_counts: list[int] = []
            for index, sample in enumerate(samples):
                events, target, identifier = _as_event_sample(sample)
                corrupted = (
                    events
                    if spec.kind == "none"
                    else apply_v8_robustness(
                        events,
                        spec,
                        sample_index=index,
                        temporal_history_provider=temporal_history_provider,
                    )
                )
                encoded = representation(corrupted)
                if not isinstance(encoded, torch.Tensor) or encoded.ndim < 1:
                    raise TypeError("V8 robustness representation must return a tensor")
                with torch.inference_mode():
                    prediction, log_variance = _model_prediction(
                        model(encoded.unsqueeze(0).to(target_device))
                    )
                predictions.append(prediction)
                uncertainty.append(
                    np.nan if log_variance is None else float(np.exp(0.5 * log_variance))
                )
                event_counts.append(corrupted.num_events)
                if target is not None and np.isfinite(target):
                    targets.append(target)
                    target_ids.append(identifier)
            values = np.asarray(predictions, dtype=np.float64)
            summary: dict[str, Any] = {
                "kind": spec.kind,
                "intensity": spec.intensity,
                "seed": spec.seed,
                "target_preserved": spec.target_preserved,
                "sample_count": len(predictions),
                "target_count": len(targets),
                "target_ids_sha256": _hash_strings(target_ids),
                "finite_prediction_fraction": float(np.isfinite(values).mean()),
                "mean_event_count": float(np.mean(event_counts)),
                "mean_predicted_uncertainty_s": _finite_mean(np.asarray(uncertainty)),
            }
            if len(targets) == len(predictions):
                error = values - np.asarray(targets, dtype=np.float64)
                summary["mae_s"] = float(np.mean(np.abs(error)))
                summary["rmse_s"] = float(np.sqrt(np.mean(np.square(error))))
            results.append(summary)
    finally:
        model.train(previous_training)
    after_fingerprints = [_event_fingerprint(_as_event_sample(sample)[0]) for sample in samples]
    if original_fingerprints != after_fingerprints:
        raise AssertionError("V8 robustness mutated its source event samples")
    clean = results[0]
    clean_uncertainty = float(clean["mean_predicted_uncertainty_s"])
    clean_mae = clean.get("mae_s")
    for summary in results[1:]:
        summary["uncertainty_delta_s"] = _finite_difference(
            float(summary["mean_predicted_uncertainty_s"]), clean_uncertainty
        )
        if isinstance(clean_mae, float) and isinstance(summary.get("mae_s"), float):
            summary["mae_delta_s"] = float(summary["mae_s"] - clean_mae)
    return {
        "schema_version": "v8_delivery_robustness_v1",
        "seed": seed,
        "source_events_unchanged": True,
        "targets_are_observation_preserved": True,
        "results": results,
    }


def _check_calibration_scope(scope: str) -> None:
    if scope not in {"train", "inner_oof"}:
        raise ValueError(
            "calibrator fit scope must be exactly 'train' or 'inner_oof'; outer-dev is forbidden"
        )


def evaluate_v8_calibration(
    fit_records: Mapping[str, np.ndarray],
    evaluation_records: Mapping[str, np.ndarray],
    *,
    fit_scope: str,
    risk_threshold_s: float = 1.0,
) -> dict[str, Any]:
    """Fit only on train/inner-OOF data and assess V8 uncertainty and risk metrics."""

    _check_calibration_scope(fit_scope)
    if risk_threshold_s <= 0.0:
        raise ValueError("risk_threshold_s must be positive")
    fit = _calibration_arrays(fit_records, label="fit")
    evaluate = _calibration_arrays(evaluation_records, label="evaluation")
    fit_ids = set(fit["sample_id"].tolist())
    evaluation_ids = set(evaluate["sample_id"].tolist())
    if fit_ids.intersection(evaluation_ids):
        raise ValueError("calibration fit and evaluation sample IDs must be disjoint")
    scale = _standardized_residual_scale(fit["target"], fit["prediction"], fit["std"])
    logits = fit["risk_logit"]
    labels = (fit["target"] <= risk_threshold_s).astype(np.float64)
    temperature = _fit_temperature(logits, labels)
    std = evaluate["std"]
    error = evaluate["prediction"] - evaluate["target"]
    probabilities = _sigmoid(evaluate["risk_logit"] / temperature)
    risk_labels = (evaluate["target"] <= risk_threshold_s).astype(np.int64)
    intervals: dict[str, dict[str, float]] = {}
    for coverage in (0.5, 0.8, 0.95):
        multiplier = _conformal_multiplier(fit["target"], fit["prediction"], fit["std"], coverage)
        radius = multiplier * std
        intervals[f"{int(round(coverage * 100))}%"] = {
            "nominal_coverage": coverage,
            "empirical_coverage": float(np.mean(np.abs(error) <= radius)),
            "mean_width_s": float(np.mean(2.0 * radius)),
            "multiplier": multiplier,
        }
    risk = _risk_metrics(risk_labels, probabilities, evaluate["target"])
    result: dict[str, Any] = {
        "schema_version": "v8_delivery_calibration_v1",
        "fit_scope": fit_scope,
        "fit_count": int(fit["target"].size),
        "evaluation_count": int(evaluate["target"].size),
        "fit_ids_sha256": _hash_strings(sorted(fit_ids)),
        "evaluation_ids_sha256": _hash_strings(sorted(evaluation_ids)),
        "fit_evaluation_disjoint": True,
        "conformal_standardized_residual_scale": scale,
        "temperature": temperature,
        "regression_nll": _gaussian_nll(error, std),
        "absolute_error_quantiles_s": {
            "p50": float(np.quantile(np.abs(error), 0.50)),
            "p80": float(np.quantile(np.abs(error), 0.80)),
            "p90": float(np.quantile(np.abs(error), 0.90)),
            "p95": float(np.quantile(np.abs(error), 0.95)),
        },
        "intervals": intervals,
        "risk": risk,
    }
    return result


def benchmark_v8_delivery(
    model: nn.Module,
    read_sample: Callable[[], Any],
    tensorize: Callable[[Any, torch.device], tuple[tuple[Any, ...], int]],
    *,
    device: str | torch.device = "cpu",
    warmup_iterations: int = 20,
    measured_iterations: int = 100,
) -> dict[str, Any]:
    """Measure reading, tensorisation, inference and total batch-one latency separately."""

    if warmup_iterations < 0 or measured_iterations <= 0:
        raise ValueError("warmup_iterations must be non-negative and measured_iterations positive")
    target = torch.device(device)
    model = model.to(target).eval()

    def sync() -> None:
        if target.type == "cuda":
            torch.cuda.synchronize(target)

    def one_iteration(measure: bool) -> tuple[float, float, float, float, int]:
        sync()
        total_start = time.perf_counter()
        start = total_start
        raw = read_sample()
        read_s = time.perf_counter() - start
        start = time.perf_counter()
        args, events = tensorize(raw, target)
        tensor_s = time.perf_counter() - start
        start = time.perf_counter()
        with torch.inference_mode():
            model(*args)
        sync()
        inference_s = time.perf_counter() - start
        total_s = time.perf_counter() - total_start
        return read_s, tensor_s, inference_s, total_s, events

    for _ in range(warmup_iterations):
        one_iteration(False)
    if target.type == "cuda":
        torch.cuda.reset_peak_memory_stats(target)
    stages = ([], [], [], [])
    event_counts: list[int] = []
    for _ in range(measured_iterations):
        read_s, tensor_s, inference_s, total_s, events = one_iteration(True)
        for values, value in zip(stages, (read_s, tensor_s, inference_s, total_s), strict=True):
            values.append(value)
        event_counts.append(events)
    process = psutil.Process(os.getpid())
    total_seconds = float(np.sum(stages[3]))
    return {
        "schema_version": "v8_delivery_benchmark_v1",
        "device": str(target),
        "batch_size": 1,
        "eval_mode": True,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "stages": {
            name: _duration_summary(values)
            for name, values in zip(
                ("read", "tensorization", "inference", "total"), stages, strict=True
            )
        },
        "windows_per_second": float(measured_iterations / total_seconds),
        "events_per_second": float(np.sum(event_counts) / total_seconds),
        "mean_events_per_window": float(np.mean(event_counts)),
        "ram_rss_bytes": int(process.memory_info().rss),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(target))
        if target.type == "cuda"
        else 0,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "hardware": environment_snapshot(),
        "explicit_scope": (
            "batch-one evaluation; stages are reported separately and total includes all stages"
        ),
    }


class _DenseTTCExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, input_names: tuple[str, ...]) -> None:
        super().__init__()
        self.model = model
        self.input_names = input_names

    def forward(self, *values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.model(*values)
        if isinstance(output, tuple):
            if len(output) < 3 or not all(isinstance(value, torch.Tensor) for value in output[:3]):
                raise TypeError(
                    "Dense ONNX tuple output must contain TTC, log variance and risk logits"
                )
            return output[0], output[1], output[2]
        mean = getattr(output, "ttc_mean_seconds", None)
        log_variance = getattr(output, "ttc_log_variance", None)
        risk = getattr(output, "collision_logits", None)
        if not all(isinstance(value, torch.Tensor) for value in (mean, log_variance, risk)):
            raise TypeError(
                "Dense ONNX model output must expose TTC, log variance and collision logits"
            )
        return mean, log_variance, risk


def export_v8_dense_onnx(
    model: nn.Module,
    example_inputs: Mapping[str, torch.Tensor],
    *,
    output_dir: str | Path,
    state_adapter_disclosure: Mapping[str, Any],
    normalization: Mapping[str, Any],
    opset_version: int = 18,
) -> dict[str, Any]:
    """Atomically export dense batch-one V8 inference and prove CPU ONNX parity."""

    if not example_inputs:
        raise ValueError("Dense ONNX export requires at least one input")
    if not state_adapter_disclosure:
        raise ValueError("Dense ONNX export requires an explicit state-adapter disclosure")
    names = tuple(example_inputs)
    tensors = tuple(value.detach().cpu().float() for value in example_inputs.values())
    if any(value.shape[0] != 1 for value in tensors):
        raise ValueError("V8 deployment ONNX export requires batch size one for every input")
    destination = Path(output_dir)
    if destination.exists():
        if any(destination.iterdir()):
            raise FileExistsError("Refusing to overwrite an existing ONNX delivery directory")
        destination.rmdir()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    ) as temporary:
        output = Path(temporary)
        wrapper = _DenseTTCExportWrapper(model.cpu().eval(), names).eval()
        with torch.inference_mode():
            reference = wrapper(*tensors)
        onnx_path = output / "model.onnx"
        torch.onnx.export(
            wrapper,
            tensors,
            onnx_path,
            input_names=list(names),
            output_names=["ttc_mean_seconds", "ttc_log_variance", "collision_logits"],
            opset_version=opset_version,
            dynamo=True,
            external_data=False,
        )
        import onnx
        import onnxruntime

        exported = onnx.load(onnx_path)
        onnx.checker.check_model(exported)
        session = onnxruntime.InferenceSession(
            onnx_path.as_posix(), providers=["CPUExecutionProvider"]
        )
        onnx_inputs = {name: value.numpy() for name, value in zip(names, tensors, strict=True)}
        actual = session.run(
            ["ttc_mean_seconds", "ttc_log_variance", "collision_logits"], onnx_inputs
        )
        errors: dict[str, float] = {}
        for name, expected, observed in zip(
            ("ttc_mean_seconds", "ttc_log_variance", "collision_logits"),
            reference,
            actual,
            strict=True,
        ):
            error = float(np.max(np.abs(expected.detach().numpy() - observed)))
            errors[name] = error
            if not np.allclose(expected.detach().numpy(), observed, rtol=1e-4, atol=1e-5):
                raise RuntimeError(f"ONNX CPU parity failed for {name}: maximum error {error}")
        np.savez_compressed(output / "example_input.npz", **onnx_inputs)
        (output / "example_output.json").write_text(
            json.dumps(
                {
                    name: np.asarray(value).tolist()
                    for name, value in zip(
                        ("ttc_mean_seconds", "ttc_log_variance", "collision_logits"),
                        actual,
                        strict=True,
                    )
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        file_hashes = {
            name: _hash_file(output / name)
            for name in ("model.onnx", "example_input.npz", "example_output.json")
        }
        metadata: dict[str, Any] = {
            "schema_version": "v8_dense_onnx_v1",
            "opset_version": opset_version,
            "batch_size_contract": 1,
            "input_shapes": {
                name: list(value.shape) for name, value in zip(names, tensors, strict=True)
            },
            "output_shapes": {
                name: list(np.asarray(value).shape)
                for name, value in zip(
                    ("ttc_mean_seconds", "ttc_log_variance", "collision_logits"),
                    actual,
                    strict=True,
                )
            },
            "state_adapter_disclosure": dict(state_adapter_disclosure),
            "normalization_contract": dict(normalization),
            "maximum_absolute_error": errors,
            "verified_with_onnxruntime_cpu": True,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "file_sha256": file_hashes,
        }
        sign_artifact(metadata)
        (output / "model_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        normalization_payload = dict(normalization)
        normalization_payload["state_adapter_disclosure"] = dict(state_adapter_disclosure)
        (output / "normalization.json").write_text(
            json.dumps(normalization_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shutil.copytree(output, destination)
    return metadata


def _calibration_arrays(records: Mapping[str, np.ndarray], *, label: str) -> dict[str, np.ndarray]:
    required = ("sample_id", "target", "prediction", "std", "risk_logit")
    missing = [name for name in required if name not in records]
    if missing:
        raise ValueError(f"{label} calibration records lack fields: {missing}")
    result = {name: np.asarray(records[name]) for name in required}
    size = result["target"].size
    if size < 2 or any(values.size != size for values in result.values()):
        raise ValueError(
            f"{label} calibration records must be aligned and contain at least two rows"
        )
    for name in ("target", "prediction", "std", "risk_logit"):
        result[name] = np.asarray(result[name], dtype=np.float64).reshape(-1)
    result["sample_id"] = np.asarray(result["sample_id"], dtype=str).reshape(-1)
    if len(set(result["sample_id"].tolist())) != size:
        raise ValueError(f"{label} sample_id values must be unique")
    if not all(
        np.isfinite(result[name]).all() for name in ("target", "prediction", "std", "risk_logit")
    ):
        raise ValueError(f"{label} calibration records must be finite")
    if np.any(result["std"] <= 0.0):
        raise ValueError(f"{label} standard deviations must be positive")
    return result


def _conformal_multiplier(
    target: np.ndarray, prediction: np.ndarray, std: np.ndarray, coverage: float
) -> float:
    scores = np.abs(target - prediction) / std
    rank = min(scores.size, int(np.ceil((scores.size + 1) * coverage)))
    return float(np.partition(scores, rank - 1)[rank - 1])


def _standardized_residual_scale(
    target: np.ndarray, prediction: np.ndarray, std: np.ndarray
) -> float:
    return _conformal_multiplier(target, prediction, std, 0.9)


def _fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    candidates = np.geomspace(0.05, 10.0, 400)
    losses = [
        float(np.mean(np.logaddexp(0.0, logits / temp) - labels * logits / temp))
        for temp in candidates
    ]
    return float(candidates[int(np.argmin(losses))])


def _gaussian_nll(error: np.ndarray, std: np.ndarray) -> float:
    return float(np.mean(0.5 * (error / std) ** 2 + np.log(std) + 0.5 * np.log(2.0 * np.pi)))


def _risk_metrics(
    labels: np.ndarray, probabilities: np.ndarray, ttc: np.ndarray
) -> dict[str, float]:
    prediction = probabilities >= 0.5
    positives = labels == 1
    negatives = ~positives
    true_positive = int(np.sum(prediction & positives))
    false_negative = int(np.sum(~prediction & positives))
    true_negative = int(np.sum(~prediction & negatives))
    false_positive = int(np.sum(prediction & negatives))
    return {
        "ece_10_bins": _ece(labels, probabilities, bins=10),
        "brier": float(np.mean((probabilities - labels) ** 2)),
        "auroc": _auroc(labels, probabilities),
        "auprc": _average_precision(labels, probabilities),
        "precision": _safe_ratio(true_positive, true_positive + false_positive),
        "recall": _safe_ratio(true_positive, true_positive + false_negative),
        "false_negative_rate": _safe_ratio(false_negative, true_positive + false_negative),
        "expected_warning_lead_time_s": float(np.mean(ttc[prediction & positives]))
        if true_positive
        else float("nan"),
        "true_negative_count": float(true_negative),
    }


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not positive.size or not negative.size:
        return float("nan")
    comparisons = positive[:, None] - negative[None, :]
    return float((np.sum(comparisons > 0) + 0.5 * np.sum(comparisons == 0)) / comparisons.size)


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    positives = int(np.sum(labels == 1))
    if not positives:
        return float("nan")
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels)
    precision = cumulative / np.arange(1, labels.size + 1)
    return float(np.sum(precision[sorted_labels == 1]) / positives)


def _ece(labels: np.ndarray, probabilities: np.ndarray, *, bins: int) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    values = 0.0
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (
            (probabilities < edges[index + 1])
            if index + 1 < bins
            else (probabilities <= edges[index + 1])
        )
        if np.any(mask):
            values += float(mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean()))
    return values


def _duration_summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64) * 1000.0
    return {
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p90_ms": float(np.percentile(array, 90)),
        "p99_ms": float(np.percentile(array, 99)),
    }


def _hash_strings(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _event_fingerprint(events: EventBatch) -> str:
    digest = hashlib.sha256()
    for values in (events.x, events.y, events.t_us, events.polarity):
        digest.update(np.asarray(values).tobytes())
    digest.update(f"{events.sequence_id}|{events.t_start_us}|{events.t_end_us}".encode())
    return digest.hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def _finite_difference(value: float, baseline: float) -> float:
    return float(value - baseline) if np.isfinite(value) and np.isfinite(baseline) else float("nan")


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    bounded = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-bounded))


__all__ = [
    "SEALED_V8_PATH_MARKERS",
    "V8RobustnessSpec",
    "apply_v8_robustness",
    "assert_v8_delivery_paths_safe",
    "benchmark_v8_delivery",
    "evaluate_v8_calibration",
    "evaluate_v8_robustness",
    "export_v8_dense_onnx",
    "v8_robustness_specs",
]
