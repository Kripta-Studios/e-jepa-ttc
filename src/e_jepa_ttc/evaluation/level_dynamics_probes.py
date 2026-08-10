"""Frozen, sequence-disjoint probes for Dense Level--Dynamics embeddings.

Probe fitting is intentionally separate from SSL training.  The functions take
an already exported embedding/metadata artifact, fit on the declared train role,
and evaluate once on validation.  They never accept a model, parquet path, or
EvTTC asset, and R² is reported only as a diagnostic.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

NUMERIC_TARGETS: tuple[str, ...] = (
    "expansion",
    "expansion_ratio",
    "log_height_ratio",
    "ttc_seconds",
    "log_ttc_seconds",
    "event_count",
    "event_rate",
    "timestamp_s",
    "horizon_s",
)
CATEGORICAL_TARGETS: tuple[str, ...] = ("sequence_id", "track_id")
TTC_TARGETS: frozenset[str] = frozenset({"ttc_seconds", "log_ttc_seconds"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def _require_hash(value: object, name: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(text):
        raise ValueError(f"{name} must be an exact 64-hex SHA-256 hash.")
    return text


def _require_commit(value: object) -> str:
    text = str(value).strip().lower()
    if not _COMMIT_RE.fullmatch(text):
        raise ValueError("code_commit must be an exact 40-hex git commit identifier.")
    return text


def _array(value: object, *, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim == 0:
        result = result.reshape(1)
    return result


def _as_str_array(value: object, *, name: str, length: int) -> np.ndarray:
    result = _array(value, name=name).astype(str)
    if result.shape[0] != length:
        raise ValueError(f"{name} length {result.shape[0]} does not match embeddings {length}.")
    return result


def _validate_embeddings(embeddings: object) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError("Frozen embeddings must be a finite [N,D] array with N>=2.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Frozen embeddings contain non-finite values.")
    return array


def _metadata_arrays(metadata: Mapping[str, Any], n: int) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key, value in metadata.items():
        array = _array(value, name=key)
        if array.shape[0] != n:
            raise ValueError(f"Metadata field {key!r} has length {array.shape[0]}, expected {n}.")
        arrays[str(key)] = array
    if "sequence_id" not in arrays:
        raise ValueError("Probe metadata requires sequence_id for split safety.")
    arrays["sequence_id"] = arrays["sequence_id"].astype(str)
    if "track_id" in arrays:
        arrays["track_id"] = arrays["track_id"].astype(str)
    return arrays


def _split_masks(metadata: Mapping[str, np.ndarray], n: int) -> tuple[np.ndarray, np.ndarray]:
    if "split" not in metadata:
        raise ValueError("Probe metadata requires a split field with train/validation roles.")
    split = metadata["split"].astype(str)
    train = split == "train"
    validation = np.isin(split, np.asarray(["validation", "val"], dtype=str))
    if not train.any() or not validation.any():
        raise ValueError("Frozen probes require non-empty train and validation rows.")
    train_sequences = set(metadata["sequence_id"][train].tolist())
    validation_sequences = set(metadata["sequence_id"][validation].tolist())
    overlap = sorted(train_sequences & validation_sequences)
    if overlap:
        raise ValueError(f"Probe train/validation sequence overlap: {overlap[:5]}")
    if np.intersect1d(np.flatnonzero(train), np.flatnonzero(validation)).size:
        raise ValueError("Probe train and validation masks overlap.")
    return train, validation


def _fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    if alpha <= 0.0:
        raise ValueError("ridge alpha must be positive.")
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    standardized = (x - mean) / scale
    design = np.concatenate([np.ones((x.shape[0], 1)), standardized], axis=1)
    regularizer = np.eye(design.shape[1], dtype=np.float64) * alpha
    regularizer[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + regularizer, design.T @ y)
    return coef, np.concatenate([mean[None], scale[None]], axis=0)


def _predict_ridge(x: np.ndarray, state: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    coef, stats = state
    standardized = (x - stats[0]) / stats[1]
    design = np.concatenate([np.ones((x.shape[0], 1)), standardized], axis=1)
    return design @ coef


def _fit_numpy_multinomial(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    steps: int = 300,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Small deterministic softmax fallback when scikit-learn is unavailable."""

    del seed  # zero initialization is deterministic and avoids hidden RNG state.
    classes = np.unique(labels.astype(str))
    if classes.size < 2:
        raise ValueError("Multinomial fallback requires at least two classes.")
    encoded = np.searchsorted(classes, labels.astype(str))
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    design = np.concatenate(
        [np.ones((x.shape[0], 1), dtype=np.float64), (x - mean) / scale], axis=1
    )
    weights = np.zeros((design.shape[1], classes.size), dtype=np.float64)
    one_hot = np.eye(classes.size, dtype=np.float64)[encoded]
    learning_rate = 0.2 / max(1.0, np.sqrt(float(design.shape[1])))
    for _ in range(int(steps)):
        logits = design @ weights
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        gradient = design.T @ (probabilities - one_hot) / max(1, design.shape[0])
        gradient[1:] += 1e-4 * weights[1:]
        weights -= learning_rate * gradient
    return classes, np.asarray((mean, scale), dtype=np.float64), weights


def _numpy_multinomial_predict(
    x: np.ndarray, state: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    classes, stats, weights = state
    mean, scale = stats
    design = np.concatenate([np.ones((x.shape[0], 1)), (x - mean) / scale], axis=1)
    logits = design @ weights
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return classes[np.argmax(probabilities, axis=1)], probabilities


def _r2(y_true: np.ndarray, prediction: np.ndarray) -> float | None:
    if y_true.size < 2:
        return None
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    if denominator <= 1e-12:
        return None
    return float(1.0 - np.sum((y_true - prediction) ** 2) / denominator)


def _effective_rank(embeddings: np.ndarray) -> float:
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    if not np.any(energy > 0.0):
        return 0.0
    probabilities = energy / energy.sum()
    return float(np.exp(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-30)))))


def embedding_diagnostics(embeddings: object) -> dict[str, float]:
    """Return detached effective-rank, duplication and variance diagnostics."""

    array = _validate_embeddings(embeddings)
    unique = np.unique(array, axis=0).shape[0]
    return {
        "effective_rank": _effective_rank(array),
        "duplication_rate": float(1.0 - unique / array.shape[0]),
        "mean_variance": float(np.var(array, axis=0).mean()),
        "mean_norm": float(np.linalg.norm(array, axis=1).mean()),
    }


def _canonical_hash(value: object) -> str:
    """Hash JSON-compatible diagnostic metadata deterministically."""

    def normalize(item: object) -> object:
        if isinstance(item, Mapping):
            return {str(key): normalize(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        if isinstance(item, np.ndarray):
            return normalize(item.tolist())
        if isinstance(item, np.generic):
            return item.item()
        return item

    encoded = json.dumps(
        normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metadata_scalar(metadata: Mapping[str, np.ndarray], key: str, index: int) -> object:
    value = metadata[key][index]
    if isinstance(value, np.generic):
        return value.item()
    return value.tolist() if isinstance(value, np.ndarray) else value


def _identity_metadata_arrays(metadata: Mapping[str, Any], n: int) -> dict[str, np.ndarray]:
    arrays = _metadata_arrays(metadata, n)
    if "timestamp_s" in arrays:
        timestamps = arrays["timestamp_s"].astype(np.float64)
    elif "timestamp_us" in arrays:
        timestamps = arrays["timestamp_us"].astype(np.float64) / 1_000_000.0
    else:
        raise ValueError("Identity diagnostics require timestamp_s or timestamp_us metadata.")
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("Identity diagnostic timestamps must be finite.")
    arrays["_timestamp_s"] = timestamps
    if "track_id" not in arrays:
        arrays["track_id"] = np.full(n, "", dtype=str)
    else:
        arrays["track_id"] = arrays["track_id"].astype(str)
    return arrays


def _duplicate_identity_keys(metadata: Mapping[str, np.ndarray], n: int) -> np.ndarray:
    paths = metadata.get("events_path")
    windows = metadata.get("event_windows_us")
    if paths is None or windows is None:
        # Without the frozen full-frame identity there is no basis for claiming
        # duplicate inseparability; each row is therefore its own atomic unit.
        return np.asarray([f"row:{index}" for index in range(n)], dtype=str)
    return np.asarray(
        [
            _canonical_hash(
                {
                    "events_path": _metadata_scalar(metadata, "events_path", index),
                    "event_windows_us": _metadata_scalar(metadata, "event_windows_us", index),
                }
            )
            for index in range(n)
        ],
        dtype=str,
    )


def _build_identity_folds(
    metadata: Mapping[str, np.ndarray],
    *,
    n_folds: int,
    guard_gap_s: float,
) -> tuple[list[dict[str, Any]], str, str]:
    """Build deterministic contiguous temporal folds with duplicate-safe groups."""

    if n_folds < 2:
        raise ValueError("Identity diagnostics require at least two temporal folds.")
    if guard_gap_s < 0.0:
        raise ValueError("Identity diagnostic guard gap must be non-negative.")
    n = metadata["sequence_id"].shape[0]
    duplicate_keys = _duplicate_identity_keys(metadata, n)
    groups_by_sequence: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, sequence in enumerate(metadata["sequence_id"].astype(str)):
        groups_by_sequence[str(sequence)][str(duplicate_keys[index])].append(index)
    fold_by_index = np.full(n, -1, dtype=np.int64)
    for sequence in sorted(groups_by_sequence):
        groups = list(groups_by_sequence[sequence].values())
        groups.sort(
            key=lambda values: (
                float(np.min(metadata["_timestamp_s"][values])),
                min(values),
            )
        )
        chunks = np.array_split(np.arange(len(groups)), n_folds)
        for fold, chunk in enumerate(chunks):
            for group_index in chunk.tolist():
                fold_by_index[np.asarray(groups[group_index], dtype=np.int64)] = fold
    if np.any(fold_by_index < 0):
        raise RuntimeError("Identity fold construction left rows unassigned.")
    row_order = [
        {
            "index": int(index),
            "sequence_id": str(metadata["sequence_id"][index]),
            "timestamp_s": float(metadata["_timestamp_s"][index]),
            "duplicate_identity": str(duplicate_keys[index]),
        }
        for index in range(n)
    ]
    row_order_hash = _canonical_hash(row_order)
    folds: list[dict[str, Any]] = []
    for fold in range(n_folds):
        test_indices = np.flatnonzero(fold_by_index == fold)
        if test_indices.size == 0:
            folds.append(
                {
                    "fold": fold,
                    "test_indices": [],
                    "train_indices": [],
                    "guard_gap_s": float(guard_gap_s),
                }
            )
            continue
        test_set = set(int(index) for index in test_indices.tolist())
        train_indices: list[int] = []
        for index in range(n):
            if index in test_set:
                continue
            sequence = str(metadata["sequence_id"][index])
            test_times = metadata["_timestamp_s"][
                test_indices[metadata["sequence_id"][test_indices].astype(str) == sequence]
            ]
            if test_times.size and np.any(
                np.abs(float(metadata["_timestamp_s"][index]) - test_times) <= guard_gap_s
            ):
                continue
            train_indices.append(index)
        folds.append(
            {
                "fold": fold,
                "test_indices": [int(index) for index in test_indices.tolist()],
                "train_indices": train_indices,
                "guard_gap_s": float(guard_gap_s),
            }
        )
    fold_hash = _canonical_hash(
        {
            "fold_by_index": fold_by_index.tolist(),
            "folds": folds,
            "guard_gap_s": float(guard_gap_s),
        }
    )
    return folds, row_order_hash, fold_hash


def _classification_metrics(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    probabilities: np.ndarray,
    classes: np.ndarray,
) -> dict[str, float]:
    accuracy = float(np.mean(labels_pred == labels_true))
    chance = 1.0 / max(1, classes.size)
    try:
        from sklearn.metrics import (  # pyright: ignore[reportMissingImports]
            balanced_accuracy_score,
            log_loss,
        )

        balanced = float(balanced_accuracy_score(labels_true, labels_pred))
        loss = float(log_loss(labels_true, probabilities, labels=classes.tolist()))
    except ImportError:  # pragma: no cover - minimal environments
        recalls: list[float] = []
        class_indices = {str(label): index for index, label in enumerate(classes.tolist())}
        for label in classes.astype(str):
            mask = labels_true.astype(str) == label
            recalls.append(
                float(np.mean(labels_pred[mask].astype(str) == label)) if mask.any() else 0.0
            )
        balanced = float(np.mean(recalls)) if recalls else 0.0
        truth_indices = np.asarray(
            [class_indices.get(str(label), 0) for label in labels_true], dtype=np.int64
        )
        loss = float(
            -np.mean(
                np.log(
                    np.maximum(probabilities[np.arange(len(truth_indices)), truth_indices], 1e-12)
                )
            )
        )
    return {
        "balanced_accuracy": balanced,
        "log_loss": loss,
        "accuracy": accuracy,
        "chance_corrected_accuracy": float((accuracy - chance) / max(1e-12, 1.0 - chance)),
    }


def _fit_predict_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit deterministic logistic regression with a NumPy softmax fallback."""

    try:
        from sklearn.linear_model import LogisticRegression  # pyright: ignore[reportMissingImports]
    except ImportError:  # pragma: no cover - minimal environments
        classes, stats, weights = _fit_numpy_multinomial(x_train, y_train, seed=seed)
        predictions, probabilities = _numpy_multinomial_predict(x_test, (classes, stats, weights))
        return classes, predictions, probabilities
    model = LogisticRegression(
        max_iter=500,
        random_state=int(seed),
        solver="lbfgs",
    )
    model.fit(x_train, y_train)
    return model.classes_, model.predict(x_test), model.predict_proba(x_test)


def _identity_fold_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    folds: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    target: str,
) -> dict[str, Any]:
    """Evaluate a fixed logistic identity probe over the frozen temporal folds."""

    observed: list[dict[str, float]] = []
    null: list[dict[str, float]] = []
    unavailable: list[dict[str, Any]] = []
    valid_labels = ~np.isin(
        labels.astype(str), np.asarray(["", "nan", "none", "null", "invalid", "-1"])
    )
    for fold in folds:
        test_indices = np.asarray(fold.get("test_indices", []), dtype=np.int64)
        train_indices = np.asarray(fold.get("train_indices", []), dtype=np.int64)
        train_indices = train_indices[valid_labels[train_indices]]
        test_indices = test_indices[valid_labels[test_indices]]
        if train_indices.size < 2 or test_indices.size < 1:
            unavailable.append({"fold": int(fold["fold"]), "reason": "insufficient_rows"})
            continue
        train_labels = labels[train_indices].astype(str)
        test_labels = labels[test_indices].astype(str)
        classes = np.unique(train_labels)
        unseen = sorted(set(test_labels.tolist()) - set(classes.tolist()))
        if classes.size < 2 or unseen:
            unavailable.append(
                {
                    "fold": int(fold["fold"]),
                    "reason": "unseen_validation_classes" if unseen else "single_train_class",
                    "unseen_validation_classes": unseen,
                }
            )
            continue
        classes_observed, observed_pred, probabilities = _fit_predict_classifier(
            embeddings[train_indices], train_labels, embeddings[test_indices], seed=seed
        )
        observed.append(
            _classification_metrics(test_labels, observed_pred, probabilities, classes_observed)
        )
        permutation_rng = np.random.default_rng(int(seed))
        permuted = permutation_rng.permutation(train_labels)
        if np.unique(permuted).size < 2 and train_labels.size >= 2:
            permuted = train_labels.copy()
            permuted[[0, 1]] = permuted[[1, 0]]
        null_classes, null_pred, null_probabilities = _fit_predict_classifier(
            embeddings[train_indices], permuted, embeddings[test_indices], seed=seed
        )
        null.append(
            _classification_metrics(
                test_labels,
                null_pred,
                null_probabilities,
                null_classes,
            )
        )
    if not observed:
        return {
            "target": target,
            "status": "unavailable",
            "diagnostic_only": True,
            "fold_count": len(folds),
            "available_fold_count": 0,
            "unavailable_folds": unavailable,
            "metrics": {
                "macro_balanced_accuracy": None,
                "macro_log_loss": None,
                "macro_chance_corrected_accuracy": None,
                "null_macro_balanced_accuracy": None,
                "null_macro_log_loss": None,
                "null_macro_chance_corrected_accuracy": None,
            },
        }
    metric_names = tuple(observed[0])
    macro = {name: float(np.mean([item[name] for item in observed])) for name in metric_names}
    null_macro = {
        f"null_{name}": float(np.mean([item[name] for item in null])) for name in metric_names
    }
    return {
        "target": target,
        "status": "available",
        "diagnostic_only": True,
        "fold_count": len(folds),
        "available_fold_count": len(observed),
        "unavailable_folds": unavailable,
        "metrics": {
            "macro_balanced_accuracy": macro["balanced_accuracy"],
            "macro_log_loss": macro["log_loss"],
            "macro_chance_corrected_accuracy": macro["chance_corrected_accuracy"],
            "null_macro_balanced_accuracy": null_macro["null_balanced_accuracy"],
            "null_macro_log_loss": null_macro["null_log_loss"],
            "null_macro_chance_corrected_accuracy": null_macro["null_chance_corrected_accuracy"],
        },
    }


def run_identity_shortcut_diagnostics(
    embeddings: object,
    metadata: Mapping[str, Any],
    *,
    checkpoint_hash: str,
    manifest_hash: str,
    config_hash: str,
    code_commit: str,
    seed: int = 7,
    context_s: float = 0.2,
    max_horizon_s: float = 0.3,
    guard_gap_s: float | None = None,
    n_folds: int = 3,
) -> dict[str, Any]:
    """Run diagnostic-only identity probes on fixed duplicate-safe time blocks.

    This artifact is intentionally separate from semantic probes and is never
    considered for SSL model selection or promotion.
    """

    checkpoint_hash = _require_hash(checkpoint_hash, "checkpoint_hash")
    manifest_hash = _require_hash(manifest_hash, "manifest_hash")
    config_hash = _require_hash(config_hash, "config_hash")
    code_commit = _require_commit(code_commit)
    array = _validate_embeddings(embeddings)
    arrays = _identity_metadata_arrays(metadata, array.shape[0])
    guard = max(float(context_s) + float(max_horizon_s), float(guard_gap_s or 0.0))
    folds, row_order_hash, fold_hash = _build_identity_folds(
        arrays, n_folds=int(n_folds), guard_gap_s=guard
    )
    sequence_result = _identity_fold_probe(
        array,
        arrays["sequence_id"].astype(str),
        folds,
        seed=seed,
        target="sequence_id",
    )
    per_sequence_track: dict[str, dict[str, Any]] = {}
    for sequence in sorted(np.unique(arrays["sequence_id"]).tolist()):
        indices = np.flatnonzero(arrays["sequence_id"] == sequence)
        sequence_folds = []
        index_set = set(int(index) for index in indices.tolist())
        for fold in folds:
            sequence_folds.append(
                {
                    **fold,
                    "test_indices": [
                        int(index) for index in fold["test_indices"] if int(index) in index_set
                    ],
                    "train_indices": [
                        int(index) for index in fold["train_indices"] if int(index) in index_set
                    ],
                }
            )
        per_sequence_track[str(sequence)] = _identity_fold_probe(
            array,
            arrays["track_id"].astype(str),
            sequence_folds,
            seed=seed,
            target=f"track_id|sequence={sequence}",
        )
    row_order = [
        {
            "index": int(index),
            "sequence_id": str(arrays["sequence_id"][index]),
            "track_id": str(arrays["track_id"][index]),
            "timestamp_s": float(arrays["_timestamp_s"][index]),
        }
        for index in range(array.shape[0])
    ]
    result: dict[str, Any] = {
        "artifact_type": "identity_shortcut_diagnostics_v1",
        "schema_version": "identity_shortcut_diagnostics_v1",
        "evidence_type": "diagnostic_only_identity_probe",
        "code_commit": code_commit,
        "protocol_version": "identity_shortcut_diagnostics_v1",
        "protocol_sha256": _canonical_hash(
            {"artifact_type": "identity_shortcut_diagnostics_v1", "fold_type": "contiguous"}
        ),
        "created_at": "diagnostic_time_not_recorded",
        "diagnostic_only": True,
        "excluded_from_ssl_selection_and_promotion": True,
        "ssl_labels_used": False,
        "seed": int(seed),
        "hparams": {
            "classifier": "logistic_regression",
            "max_iter": 500,
            "solver": "lbfgs",
            "context_s": float(context_s),
            "max_horizon_s": float(max_horizon_s),
            "guard_gap_s": float(guard),
            "n_folds": int(n_folds),
            "fold_type": "contiguous_temporal_blocks",
            "duplicate_identity": "events_path,event_windows_us",
            "null_policy": "single_seed_fixed_label_permutation",
            "null_seed": int(seed),
        },
        "checkpoint_hash": str(checkpoint_hash),
        "manifest_hash": str(manifest_hash),
        "config_hash": config_hash,
        "row_order_hash": row_order_hash,
        "fold_hash": fold_hash,
        "row_order": row_order,
        "folds": folds,
        "probes": {
            "sequence_id_across_sequences": sequence_result,
            "track_id_conditionally_within_sequence": per_sequence_track,
        },
    }
    result["artifact_sha256"] = _canonical_hash(result)
    result["signature"] = _canonical_hash(result)
    return result


def write_identity_diagnostics(result: Mapping[str, Any], output_json: str | Path) -> None:
    """Atomically write a signed identity-shortcut diagnostic artifact."""

    if result.get("artifact_type") != "identity_shortcut_diagnostics_v1":
        raise ValueError("Identity diagnostic artifact_type is required.")
    signature = result.get("signature")
    if signature != _canonical_hash(
        {key: value for key, value in result.items() if key != "signature"}
    ):
        raise ValueError("Identity diagnostic signature mismatch.")
    destination = Path(output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)


@dataclass(frozen=True)
class FrozenProbeResult:
    """JSON-serializable result for one frozen probe target."""

    target: str
    family: str
    diagnostic_only: bool
    fit_count: int
    validation_count: int
    metrics: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "family": self.family,
            "diagnostic_only": self.diagnostic_only,
            "fit_count": self.fit_count,
            "validation_count": self.validation_count,
            "metrics": dict(self.metrics),
        }


def _numeric_probe(
    embeddings: np.ndarray,
    metadata: Mapping[str, np.ndarray],
    target: str,
    train: np.ndarray,
    validation: np.ndarray,
    *,
    alpha: float,
) -> FrozenProbeResult:
    if target not in metadata:
        raise ValueError(f"Requested numeric probe target {target!r} is absent from metadata.")
    y = metadata[target].astype(np.float64)
    train_valid = train & np.isfinite(y)
    val_valid = validation & np.isfinite(y)
    if train_valid.sum() < 2 or val_valid.sum() < 1:
        raise ValueError(f"Numeric probe {target!r} lacks finite train/validation rows.")
    state = _fit_ridge(embeddings[train_valid], y[train_valid], alpha)
    train_prediction = _predict_ridge(embeddings[train_valid], state)
    val_prediction = _predict_ridge(embeddings[val_valid], state)
    sequence_metrics: dict[str, dict[str, float]] = {}
    for sequence in sorted(np.unique(metadata["sequence_id"][val_valid]).tolist()):
        sequence_mask = val_valid & (metadata["sequence_id"] == sequence)
        if not sequence_mask.any():
            continue
        sequence_prediction = _predict_ridge(embeddings[sequence_mask], state)
        sequence_error = sequence_prediction - y[sequence_mask]
        sequence_metrics[str(sequence)] = {
            "mae": float(np.mean(np.abs(sequence_error))),
            "rmse": float(np.sqrt(np.mean(sequence_error**2))),
            "count": float(sequence_mask.sum()),
        }
    sequence_mae = [item["mae"] for item in sequence_metrics.values()]
    sequence_rmse = [item["rmse"] for item in sequence_metrics.values()]
    metrics = {
        "train_mae": float(np.mean(np.abs(train_prediction - y[train_valid]))),
        "validation_mae": float(np.mean(np.abs(val_prediction - y[val_valid]))),
        "validation_rmse": float(np.sqrt(np.mean((val_prediction - y[val_valid]) ** 2))),
        "validation_r2_diagnostic": _r2(y[val_valid], val_prediction),
        "validation_sequence_count": int(len(sequence_metrics)),
        "validation_mae_macro_by_sequence": float(np.mean(sequence_mae)) if sequence_mae else None,
        "validation_rmse_macro_by_sequence": float(np.mean(sequence_rmse))
        if sequence_rmse
        else None,
        "validation_by_sequence": sequence_metrics,
    }
    return FrozenProbeResult(
        target=target,
        family="numeric_ridge",
        diagnostic_only=target in TTC_TARGETS,
        fit_count=int(train_valid.sum()),
        validation_count=int(val_valid.sum()),
        metrics=metrics,
    )


def _categorical_probe(
    embeddings: np.ndarray,
    metadata: Mapping[str, np.ndarray],
    target: str,
    train: np.ndarray,
    validation: np.ndarray,
    *,
    seed: int,
) -> FrozenProbeResult:
    if target not in metadata:
        raise ValueError(f"Requested categorical probe target {target!r} is absent from metadata.")
    labels = metadata[target].astype(str)
    valid_labels = ~np.isin(labels, np.asarray(["", "nan", "none", "-1", "invalid"], dtype=str))
    train_valid = train & valid_labels
    val_valid = validation & valid_labels
    if target == "track_id":
        # Invalid track IDs are explicitly excluded rather than turned into a
        # learnable sentinel class.
        valid_track = ~np.isin(
            labels,
            np.asarray(["", "nan", "none", "null", "invalid", "-1"], dtype=str),
        )
        train_valid &= valid_track
        val_valid &= valid_track
    classes = np.unique(labels[train_valid])
    if val_valid.sum() < 1:
        raise ValueError(f"Categorical probe {target!r} needs at least one valid validation row.")
    if train_valid.sum() < 1:
        return FrozenProbeResult(
            target=target,
            family="categorical_logistic",
            diagnostic_only=True,
            fit_count=0,
            validation_count=int(val_valid.sum()),
            metrics={
                "status": "unavailable",
                "train_accuracy": None,
                "validation_accuracy": None,
                "class_count": 0,
                "unavailable_reason": "no_train_classes",
                "unseen_validation_classes": sorted(set(labels[val_valid].tolist())),
            },
        )
    unseen_validation = sorted(set(labels[val_valid].tolist()) - set(classes.tolist()))
    if classes.size < 2 or unseen_validation:
        reason = "unseen_validation_classes" if unseen_validation else "single_train_class"
        return FrozenProbeResult(
            target=target,
            family="categorical_logistic",
            diagnostic_only=True,
            fit_count=int(train_valid.sum()),
            validation_count=int(val_valid.sum()),
            metrics={
                "status": "unavailable",
                "train_accuracy": None,
                "validation_accuracy": None,
                "class_count": int(classes.size),
                "unavailable_reason": reason,
                "unseen_validation_classes": unseen_validation,
            },
        )
    # A deterministic multinomial logistic probe; a local import keeps the
    # frozen artifact path usable without importing sklearn at package import.
    try:
        from sklearn.linear_model import LogisticRegression  # pyright: ignore[reportMissingImports]
    except ImportError:  # pragma: no cover - exercised on minimal CPU installs
        classes_fallback, stats_fallback, weights_fallback = _fit_numpy_multinomial(
            embeddings[train_valid], labels[train_valid], seed=seed
        )
        train_pred, _ = _numpy_multinomial_predict(
            embeddings[train_valid], (classes_fallback, stats_fallback, weights_fallback)
        )
        val_pred, _ = _numpy_multinomial_predict(
            embeddings[val_valid], (classes_fallback, stats_fallback, weights_fallback)
        )
    else:
        model = LogisticRegression(max_iter=500, random_state=seed, solver="lbfgs")
        model.fit(embeddings[train_valid], labels[train_valid])
        train_pred = model.predict(embeddings[train_valid])
        val_pred = model.predict(embeddings[val_valid])
    metrics = {
        "status": "available",
        "train_accuracy": float(np.mean(train_pred == labels[train_valid])),
        "validation_accuracy": float(np.mean(val_pred == labels[val_valid])),
        "class_count": float(classes.size),
    }
    return FrozenProbeResult(
        target=target,
        family="categorical_logistic",
        diagnostic_only=False,
        fit_count=int(train_valid.sum()),
        validation_count=int(val_valid.sum()),
        metrics=metrics,
    )


def run_level_dynamics_probes(
    embeddings: object,
    metadata: Mapping[str, Any],
    *,
    checkpoint_hash: str,
    manifest_hash: str,
    config_hash: str,
    code_commit: str,
    seed: int = 7,
    numeric_targets: Sequence[str] | None = None,
    categorical_targets: Sequence[str] | None = None,
    ridge_alpha: float = 1.0,
) -> dict[str, Any]:
    """Fit/evaluate all requested frozen probes and return compact diagnostics."""

    checkpoint_hash = _require_hash(checkpoint_hash, "checkpoint_hash")
    manifest_hash = _require_hash(manifest_hash, "manifest_hash")
    config_hash = _require_hash(config_hash, "config_hash")
    code_commit = _require_commit(code_commit)
    array = _validate_embeddings(embeddings)
    metadata_arrays = _metadata_arrays(metadata, array.shape[0])
    train, validation = _split_masks(metadata_arrays, array.shape[0])
    numeric = tuple(numeric_targets or NUMERIC_TARGETS)
    categorical = tuple(categorical_targets or CATEGORICAL_TARGETS)
    results: list[FrozenProbeResult] = []
    for target in numeric:
        if target in metadata_arrays:
            results.append(
                _numeric_probe(array, metadata_arrays, target, train, validation, alpha=ridge_alpha)
            )
    for target in categorical:
        if target in metadata_arrays:
            # Invalid track IDs are valid metadata but not a valid probe target;
            # skip only when no valid IDs exist in either split.
            try:
                results.append(
                    _categorical_probe(array, metadata_arrays, target, train, validation, seed=seed)
                )
            except ValueError:
                if target != "track_id":
                    raise
    row_order = [
        {
            "index": int(index),
            "sequence_id": str(metadata_arrays["sequence_id"][index]),
            "split": str(metadata_arrays.get("split", np.asarray([""] * array.shape[0]))[index]),
        }
        for index in range(array.shape[0])
    ]
    result: dict[str, Any] = {
        "artifact_type": "dense_level_dynamics_frozen_probes_v1",
        "schema_version": "dense_level_dynamics_frozen_probes_v1",
        "protocol_version": "dense_level_dynamics_frozen_probes_v1",
        "protocol_sha256": _canonical_hash(
            {"fit_policy": "train_fit_validation_sequence_disjoint"}
        ),
        "checkpoint_hash": checkpoint_hash,
        "manifest_hash": manifest_hash,
        "config_hash": config_hash,
        "code_commit": code_commit,
        "row_order_hash": _canonical_hash(row_order),
        "hparams": {"seed": int(seed), "ridge_alpha": float(ridge_alpha)},
        "selection_evidence": True,
        "diagnostic_only": False,
        "probe_stage": "post_ssl_evaluation",
        "seed": int(seed),
        "fit_policy": "train_fit_validation_evaluate_sequence_disjoint",
        "ssl_labels_used": False,
        "diagnostics": embedding_diagnostics(array),
        "level_scale_retention": {
            "expansion_probe_present": any(
                result.target in {"expansion", "expansion_ratio", "log_height_ratio"}
                for result in results
            ),
            "targets": [
                result.target
                for result in results
                if result.target in {"expansion", "expansion_ratio", "log_height_ratio"}
            ],
        },
        "probes": [result.as_dict() for result in results],
    }
    result["artifact_sha256"] = _canonical_hash(result)
    result["signature"] = _canonical_hash(result)
    return result


def fit_frozen_probe(
    embeddings: object,
    metadata: Mapping[str, Any],
    target: str,
    *,
    fit_split: str = "train",
    evaluate_split: str = "validation",
    seed: int = 7,
    ridge_alpha: float = 1.0,
) -> FrozenProbeResult:
    """Fit one probe; explicitly reject fitting on validation rows."""

    if fit_split != "train" or evaluate_split not in {"validation", "val"}:
        raise ValueError("Frozen probes may fit only on train and evaluate only on validation.")
    array = _validate_embeddings(embeddings)
    arrays = _metadata_arrays(metadata, array.shape[0])
    train, validation = _split_masks(arrays, array.shape[0])
    if target in NUMERIC_TARGETS:
        return _numeric_probe(array, arrays, target, train, validation, alpha=ridge_alpha)
    if target in CATEGORICAL_TARGETS:
        return _categorical_probe(array, arrays, target, train, validation, seed=seed)
    raise ValueError(f"Unknown frozen probe target: {target!r}")


def load_embedding_metadata_artifact(path: str | Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load a compact ``.npz`` or JSON embedding/metadata artifact."""

    artifact = Path(path)
    if artifact.suffix.lower() == ".npz":
        with np.load(artifact, allow_pickle=False) as values:
            if "embeddings" not in values:
                raise ValueError("Embedding artifact requires an 'embeddings' array.")
            embeddings = np.asarray(values["embeddings"])
            metadata = {key: np.asarray(values[key]) for key in values.files if key != "embeddings"}
        return _validate_embeddings(embeddings), metadata
    value = json.loads(artifact.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or "embeddings" not in value or "metadata" not in value:
        raise ValueError("JSON probe artifact requires embeddings and metadata fields.")
    return _validate_embeddings(value["embeddings"]), dict(value["metadata"])


def write_probe_outputs(
    result: Mapping[str, Any], output_json: str | Path, output_csv: str | Path | None = None
) -> None:
    """Write compact JSON and optional one-row-per-probe CSV outputs atomically."""

    signature = result.get("signature")
    if signature != _canonical_hash(
        {key: value for key, value in result.items() if key != "signature"}
    ):
        raise ValueError("Frozen probe artifact signature mismatch.")

    destination = Path(output_json)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)
    if output_csv is not None:
        csv_destination = Path(output_csv)
        csv_destination.parent.mkdir(parents=True, exist_ok=True)
        rows = result.get("probes", [])
        fields = ["target", "family", "diagnostic_only", "fit_count", "validation_count", "metrics"]
        csv_temp = csv_destination.with_suffix(csv_destination.suffix + ".tmp")
        with csv_temp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(row[key], sort_keys=True)
                        if key == "metrics"
                        else row.get(key)
                        for key in fields
                    }
                )
        csv_temp.replace(csv_destination)


__all__ = [
    "CATEGORICAL_TARGETS",
    "FrozenProbeResult",
    "NUMERIC_TARGETS",
    "TTC_TARGETS",
    "embedding_diagnostics",
    "fit_frozen_probe",
    "load_embedding_metadata_artifact",
    "run_identity_shortcut_diagnostics",
    "run_level_dynamics_probes",
    "write_identity_diagnostics",
    "write_probe_outputs",
]
