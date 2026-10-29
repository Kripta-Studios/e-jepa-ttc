#!/usr/bin/env python
"""Evaluate the preregistered train-only constant control on campaign v5.

The full-population scorer is streaming and compact: it retains only global
match flags and denominator counts, never frame images, full GT objects, PR
point dictionaries, or per-match diagnostics.  AP is computed once globally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.detection.campaign_data import _training_reason  # noqa: E402
from e_jepa_ttc.detection.evaluation_controls import (  # noqa: E402
    fit_train_constant_prior,
)
from e_jepa_ttc.detection.evaluation_ground_truth import (  # noqa: E402
    EvaluationBox,
    EvaluationGroundTruth,
    build_evaluation_ground_truth,
    domain_reason,
    parse_box,
)
from e_jepa_ttc.detection.evaluation_protocol import EvaluationProtocol  # noqa: E402

SCHEMA = "phase_ttc_detection_campaign_controls_v1"
DEFAULT_OUTPUT = Path("output/campaign_controls/v5_train_constant_v1")
EXPECTED_PROTOCOL_SHA256 = "10da30cf8f08247d48d975f3650cb02f1b43375fe0e841a5389dfb3b5d7b4a5e"
EXPECTED_INPUT_SHA256 = {
    "campaign_receipt": "4ba97b0b0ec19936c2cdfe9fa718e5f79e30bb3348d67df0026b4267d82f309a",
    "campaign_manifest": "9beeb6a3c5b76bf994808eee0b3f0dfa2f7339f139bd120dbfa6318e414945f1",
    "media_admission": "c625eec068727b5df15f527483b9bf343f6b566fc19eba20449066de2390eae8",
    "population_audit": "b8d5d5fec171e2af73a360fd13e8466757baf801b15c571728e31cb107d42673",
    "full_selection": "e4303685e39f635a47b76921b9c5f38821e8715f3aa5f202c73a2c32dae1a497",
    "candidate_split": "91a13c5ae7f6e40ee5fdb4a62da481eba5ffff2b8209ea1110befe3ad7cdc903",
}
EXPECTED_CANDIDATE_SPLIT_SOURCE_SHA256 = (
    "91a13c5ae7f6e40ee5fdb4a62da481eba5ffff2b8209ea1110befe3ad7cdc903"
)
MIB = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Hash one file incrementally."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash canonical JSON."""
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object and reject duplicate keys."""

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    """Write strict, readable JSON."""
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _xy(left: EvaluationBox, right: EvaluationBox) -> float:
    return math.hypot(
        left.translation_m[0] - right.translation_m[0],
        left.translation_m[1] - right.translation_m[1],
    )


def _ap101(flags: Sequence[int], count: int) -> float | None:
    if count == 0:
        return None
    correct = 0
    curve: list[tuple[float, float]] = []
    for rank, flag in enumerate(flags, 1):
        correct += flag
        curve.append((correct / count, correct / rank))
    return sum(
        max((precision for recall, precision in curve if recall >= step / 100), default=0.0)
        for step in range(101)
    ) / 101


def score_constant_streaming(
    frames: Iterable[tuple[str, EvaluationGroundTruth]],
    priors: Sequence[Mapping[str, object]],
    *,
    protocol: EvaluationProtocol,
    max_rss_bytes: int | None = None,
    resource_sample_interval: int = 128,
) -> dict[str, object]:
    """Compute exact global AP for constant priors with bounded frame residency."""
    parsed = []
    for index, row in enumerate(priors):
        box, score = parse_box(row, source_index=index, classes=protocol.classes, prediction=True)
        assert score is not None
        parsed.append((score, index, box))
    parsed.sort(key=lambda item: (-item[0], item[1]))
    if len(parsed) > protocol.max_predictions_per_frame:
        raise ValueError("Constant priors exceed protocol proposal cap")
    class_names = [box.class_name for _, _, box in parsed]
    if len(class_names) != len(set(class_names)):
        raise ValueError("Constant control requires exactly zero or one prior per class")
    if resource_sample_interval < 1:
        raise ValueError("resource_sample_interval must be positive")

    flags: dict[str, dict[float, list[int]]] = {
        category: {threshold: [] for threshold in protocol.distance_thresholds_m}
        for category in protocol.classes
    }
    ignored_predictions: dict[str, Counter[float]] = {
        category: Counter() for category in protocol.classes
    }
    eligible_by_class: Counter[str] = Counter()
    ignored_by_class: Counter[str] = Counter()
    raw_by_class: Counter[str] = Counter()
    ignored_reasons: Counter[str] = Counter()
    class_agnostic_tp: Counter[float] = Counter()
    gt_hashes: dict[str, str] = {}
    outside_domain: Counter[str] = Counter()
    scored_by_class: Counter[str] = Counter()
    sample_count = raw_count = eligible_count = ignored_count = 0
    previous_token: str | None = None
    measured_peak_rss = 0
    measured_tree_rss = 0

    for token, frame in frames:
        if previous_token is not None and token <= previous_token:
            raise ValueError("Streaming frames must have unique strictly sorted tokens")
        previous_token = token
        if frame.protocol_sha256 != protocol.sha256:
            raise ValueError("Ground truth protocol mismatch")
        if frame.raw_count != len(frame.eligible) + len(frame.ignored):
            raise ValueError("Ground-truth denominator identity failed")
        sample_count += 1
        if max_rss_bytes is not None and (
            sample_count == 1 or sample_count % resource_sample_interval == 0
        ):
            process_peak, tree_current = _resource_rss_bytes()
            measured_peak_rss = max(measured_peak_rss, process_peak)
            measured_tree_rss = max(measured_tree_rss, tree_current)
            if max(process_peak, tree_current) > max_rss_bytes:
                raise MemoryError(
                    "Observed process/tree RSS exceeds the preregistered resource gate"
                )
        raw_count += frame.raw_count
        eligible_count += len(frame.eligible)
        ignored_count += len(frame.ignored)
        gt_hashes[token] = frame.sha256
        ignored_reasons.update(dict(frame.ignored_by_reason))
        raw_by_class.update(box.class_name for box in (*frame.eligible, *frame.ignored))
        eligible_by_class.update(box.class_name for box in frame.eligible)
        ignored_by_class.update(box.class_name for box in frame.ignored)

        admitted: list[tuple[float, int, EvaluationBox]] = []
        for score, index, box in parsed:
            reason = domain_reason(
                box,
                intrinsic=np.asarray(frame.intrinsic),
                event_from_ego=np.asarray(frame.event_from_ego),
                image_wh=frame.image_wh,
                protocol=protocol,
            )
            if reason is None:
                admitted.append((score, index, box))
                scored_by_class[box.class_name] += 1
            else:
                outside_domain[reason] += 1

        for _, _, prediction in admitted:
            category = prediction.class_name
            for threshold in protocol.distance_thresholds_m:
                eligible = [
                    target
                    for target in frame.eligible
                    if target.class_name == category and _xy(prediction, target) < threshold
                ]
                if eligible:
                    flags[category][threshold].append(1)
                else:
                    ignored = [
                        target
                        for target in frame.ignored
                        if target.class_name == category and _xy(prediction, target) < threshold
                    ]
                    if ignored:
                        ignored_predictions[category][threshold] += 1
                    else:
                        flags[category][threshold].append(0)

        for threshold in protocol.distance_thresholds_m:
            assigned: set[int] = set()
            for _, _, prediction in admitted:
                candidates = [
                    (_xy(prediction, target), target.source_index)
                    for target in frame.eligible
                    if target.source_index not in assigned and _xy(prediction, target) < threshold
                ]
                if candidates:
                    _, source_index = min(candidates)
                    assigned.add(source_index)
            class_agnostic_tp[threshold] += len(assigned)

    if sample_count == 0:
        raise ValueError("Cannot evaluate an empty population")
    per_class: dict[str, object] = {}
    supported: list[float] = []
    for category in protocol.classes:
        class_aps = []
        thresholds = {}
        count = eligible_by_class[category]
        for threshold in protocol.distance_thresholds_m:
            threshold_flags = flags[category][threshold]
            ap = _ap101(threshold_flags, count)
            if ap is not None:
                class_aps.append(ap)
            tp = sum(threshold_flags)
            thresholds[str(threshold)] = {
                "AP101": ap,
                "TP": tp,
                "FP": len(threshold_flags) - tp,
                "FN": count - tp,
                "ignored_prediction_count": ignored_predictions[category][threshold],
            }
        mean_ap = sum(class_aps) / len(class_aps) if class_aps else None
        if mean_ap is not None:
            supported.append(mean_ap)
        per_class[category] = {
            "raw_ground_truth_count": raw_by_class[category],
            "eligible_ground_truth_count": count,
            "ignored_ground_truth_count": ignored_by_class[category],
            "ground_truth_identity_check": (
                raw_by_class[category] == count + ignored_by_class[category]
            ),
            "prediction_count_after_domain_filter": scored_by_class[category],
            "AP101_mean_thresholds": mean_ap,
            "thresholds": thresholds,
        }
    return {
        "schema": "phase_ttc_detection_evaluation_result_v2_compact_constant",
        "profile": protocol.profile,
        "protocol_sha256": protocol.sha256,
        "official_eap_score": False,
        "compact_output": True,
        "global_ap_not_chunk_average": True,
        "sample_count": sample_count,
        "population": {
            "raw_ground_truth_count": raw_count,
            "eligible_ground_truth_count": eligible_count,
            "ignored_ground_truth_count": ignored_count,
            "ignored_by_reason": dict(sorted(ignored_reasons.items())),
            "identity_check": raw_count == eligible_count + ignored_count,
            "ground_truth_sha256_by_token": gt_hashes,
        },
        "predictions": {
            "input_count": sample_count * len(parsed),
            "capped_before_domain_filter": 0,
            "outside_domain_count": sum(outside_domain.values()),
            "outside_domain_by_reason": dict(sorted(outside_domain.items())),
            "scored_count": sum(scored_by_class.values()),
            "identity_check": sample_count * len(parsed)
            == sum(outside_domain.values()) + sum(scored_by_class.values()),
        },
        "class_agnostic_recall": {
            "eligible_ground_truth_count": eligible_count,
            "thresholds": {
                str(threshold): {
                    "TP": class_agnostic_tp[threshold],
                    "FN": eligible_count - class_agnostic_tp[threshold],
                    "recall": (
                        class_agnostic_tp[threshold] / eligible_count
                        if eligible_count
                        else None
                    ),
                }
                for threshold in protocol.distance_thresholds_m
            },
        },
        "supported_class_count": len(supported),
        "dev_AP101_supported_classes": sum(supported) / len(supported) if supported else None,
        "dev_AP101_all_eight_absent_zero": sum(supported) / len(protocol.classes),
        "per_class": per_class,
        "omitted_for_bounded_memory": [
            "PR point arrays",
            "per-prediction ignored records",
            "diagnostic matched-pair records",
        ],
        "resource_monitor": {
            "sample_interval_frames": resource_sample_interval,
            "maximum_process_peak_rss_bytes": measured_peak_rss or None,
            "maximum_owned_tree_current_rss_bytes": measured_tree_rss or None,
            "gate_bytes": max_rss_bytes,
        },
    }


def require_frozen_completed_predictions(path: Path) -> dict[str, Any]:
    """Keep comparison unavailable until the actual completed R0 schema is reviewed."""
    if not path.is_file():
        raise FileNotFoundError(path)
    raise NotImplementedError(
        "R0 comparison is unavailable until the actual completed summary, evaluation hash, "
        "token map and GT source contract are reviewed"
    )


def _verify_inputs(args: argparse.Namespace) -> dict[str, Any]:
    protocol = EvaluationProtocol.default()
    if protocol.sha256 != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("The local v2 protocol changed")
    paths = {
        "campaign_receipt": args.campaign_receipt,
        "campaign_manifest": args.campaign_manifest,
        "media_admission": args.media_admission,
        "population_audit": args.population_audit,
        "full_selection": args.full_selection,
        "candidate_split": args.candidate_split,
        "media_table": args.data_root / "data/train.parquet",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    values = {name: load_json(path) for name, path in paths.items() if name != "media_table"}
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    for name, expected in EXPECTED_INPUT_SHA256.items():
        if hashes[name] != expected:
            raise ValueError(f"Frozen v5 input SHA-256 mismatch: {name}")
    manifest = values["campaign_manifest"]
    receipt = values["campaign_receipt"]
    admission = values["media_admission"]
    audit = values["population_audit"]
    full = values["full_selection"]
    split = values["candidate_split"]
    if (
        manifest["candidate_split_sha256"] != EXPECTED_CANDIDATE_SPLIT_SOURCE_SHA256
        or full["candidate_split_sha256"] != EXPECTED_CANDIDATE_SPLIT_SOURCE_SHA256
        or audit["candidate_split_sha256"] != EXPECTED_CANDIDATE_SPLIT_SOURCE_SHA256
    ):
        raise ValueError("Campaign artifacts do not share the frozen candidate split")
    if receipt["status"] != "PASS":
        raise ValueError("Campaign v5 receipt is not PASS")
    if receipt["new_manifest_sha256"] != hashes["campaign_manifest"]:
        raise ValueError("Campaign v5 receipt does not bind its manifest")
    if receipt["media_admission_sha256"] != hashes["media_admission"]:
        raise ValueError("Campaign v5 receipt does not bind media admission")
    if receipt["population_audit_sha256"] != hashes["population_audit"]:
        raise ValueError("Campaign v5 receipt does not bind population audit")
    if receipt["full_selection_sha256"] != hashes["full_selection"]:
        raise ValueError("Campaign v5 receipt does not bind full selection")
    if manifest["media_admission"]["receipt_sha256"] != hashes["media_admission"]:
        raise ValueError("Campaign manifest does not bind the media admission receipt")
    if hashes["media_table"] != split["media_sha256"]:
        raise ValueError("Canonical train metadata hash mismatch")
    if full["source_media_sha256"] != hashes["media_table"]:
        raise ValueError("Full-selection population binds different metadata")
    if admission["status"] != "PASS" or len(admission["final_media"]) != 28054:
        raise ValueError("Final media admission is incomplete")
    train_tokens = manifest["roles"]["train"]["sample_tokens"]
    monitor_tokens = manifest["roles"]["selection"]["sample_tokens"]
    full_tokens = full["sample_tokens"]
    if [len(train_tokens), len(monitor_tokens), len(full_tokens)] != [4096, 1024, 23958]:
        raise ValueError("Frozen campaign population sizes differ")
    if set(train_tokens) & set(full_tokens):
        raise ValueError("Train and selection tokens overlap")
    if not set(monitor_tokens) <= set(full_tokens):
        raise ValueError("Monitor is not a subset of full selection")
    if set(admission["final_media"]) != set(train_tokens) | set(full_tokens):
        raise ValueError("Media admission tokens differ from campaign populations")
    train_sequences = set(split["assignments"]["train"])
    selection_sequences = set(split["assignments"]["selection"])
    if set(manifest["roles"]["train"]["sequence_ids"]) != train_sequences:
        raise ValueError("Campaign train sequences differ from canonical split ownership")
    if set(manifest["roles"]["selection"]["sequence_ids"]) != selection_sequences:
        raise ValueError("Campaign monitor sequences differ from canonical split ownership")
    if set(full["sequence_ids"]) != selection_sequences:
        raise ValueError("Full selection sequences differ from canonical split ownership")
    for token in train_tokens:
        if admission["final_media"][token]["sequence_id"] not in train_sequences:
            raise ValueError(f"Train token leaves its canonical sequence role: {token}")
    for token in full_tokens:
        if admission["final_media"][token]["sequence_id"] not in selection_sequences:
            raise ValueError(f"Selection token leaves its canonical sequence role: {token}")

    label_pins = {row["path"]: row for row in audit["label_files"]}
    if len(label_pins) != 40:
        raise ValueError("Expected forty pinned public-train label files")
    for relative, receipt in sorted(label_pins.items()):
        path = args.data_root.joinpath(*relative.split("/"))
        if not path.is_file() or path.stat().st_size != receipt["bytes"]:
            raise ValueError(f"Pinned label size mismatch: {relative}")
        if sha256_file(path) != receipt["sha256"]:
            raise ValueError(f"Pinned label SHA-256 mismatch: {relative}")
    return {
        "protocol": protocol,
        "values": values,
        "inputs_sha256": hashes,
        "train_tokens": tuple(train_tokens),
        "monitor_tokens": tuple(monitor_tokens),
        "full_tokens": tuple(full_tokens),
        "all_label_files_verified": len(label_pins),
    }


def _media_by_sequence(data_root: Path, tokens: set[str]) -> dict[str, list[dict[str, Any]]]:
    columns = ["sample_token", "sequence_id", "K_event", "T_event_ego", "split"]
    table = pq.read_table(
        data_root / "data/train.parquet",
        columns=columns,
        filters=[("sample_token", "in", sorted(tokens))],
    )
    rows = table.to_pylist()
    if len(rows) != len(tokens) or {str(row["sample_token"]) for row in rows} != tokens:
        raise ValueError("Metadata query did not return the exact token population")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] != "train":
            raise ValueError("Control population leaves public train")
        grouped[str(row["sequence_id"])].append(row)
    return grouped


def _labels_for_sequence(data_root: Path, sequence: str) -> dict[str, list[dict[str, Any]]]:
    path = data_root / "data/train" / sequence / "labels.parquet"
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pq.read_table(path).to_pylist():
        if row["split"] != "train" or row["sequence_id"] != sequence:
            raise ValueError(f"Label identity mismatch: {sequence}")
        grouped[str(row["sample_token"])].append(row)
    return grouped


def _fit_priors(
    data_root: Path,
    train_tokens: Sequence[str],
    *,
    protocol: EvaluationProtocol,
) -> list[dict[str, object]]:
    grouped_media = _media_by_sequence(data_root, set(train_tokens))
    train_truth: dict[str, list[dict[str, object]]] = {}
    for sequence in sorted(grouped_media):
        labels = _labels_for_sequence(data_root, sequence)
        for media in grouped_media[sequence]:
            token = str(media["sample_token"])
            admitted = []
            for annotation in labels.get(token, []):
                if _training_reason(annotation, media) is None:
                    raw = annotation["bbox_3d_ego"]
                    admitted.append(
                        {
                            "class_name": str(annotation["category"]),
                            "translation_m": [float(value) for value in raw[:3]],
                            "size_lwh_m": [float(raw[3]), float(raw[5]), float(raw[4])],
                            "yaw_rad": float(raw[6]),
                        }
                    )
            train_truth[token] = admitted
    return fit_train_constant_prior(train_truth, protocol=protocol)


def _frames(
    data_root: Path,
    tokens: Sequence[str],
    *,
    media_admission: Mapping[str, Any],
    protocol: EvaluationProtocol,
) -> Iterable[tuple[str, EvaluationGroundTruth]]:
    token_set = set(tokens)
    media_by_sequence = _media_by_sequence(data_root, token_set)
    indexed_media = {
        str(row["sample_token"]): row for rows in media_by_sequence.values() for row in rows
    }
    labels_by_sequence = {
        sequence: _labels_for_sequence(data_root, sequence) for sequence in media_by_sequence
    }
    final_media = media_admission["final_media"]
    for token in sorted(tokens):
        media = indexed_media[token]
        sequence = str(media["sequence_id"])
        receipt = final_media[token]
        if receipt["sequence_id"] != sequence or receipt["role"] not in {
            "train",
            "full_selection",
        }:
            raise ValueError(f"Media admission identity mismatch: {token}")
        image_wh = tuple(receipt["image_wh"])
        if len(receipt["member_sha256"]) != 64 or image_wh != (1280, 720):
            raise ValueError(f"Media admission lacks verified geometry/hash: {token}")
        yield token, build_evaluation_ground_truth(
            labels_by_sequence[sequence].get(token, []),
            intrinsic=media["K_event"],
            event_from_ego=media["T_event_ego"],
            image_wh=image_wh,  # type: ignore[arg-type]
            protocol=protocol,
        )


def _resource_rss_bytes() -> tuple[int, int]:
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError("psutil is required to enforce the preregistered RSS gate") from error
    process = psutil.Process(os.getpid())
    memory = process.memory_info()
    if not hasattr(memory, "peak_wset"):
        raise RuntimeError("This admission requires Windows peak_wset RSS measurement")
    tree_current = int(memory.rss)
    for child in process.children(recursive=True):
        try:
            tree_current += int(child.memory_info().rss)
        except psutil.Error:
            continue
    return int(memory.peak_wset), tree_current


def render(report: Mapping[str, Any]) -> str:
    """Render a result summary from machine-readable output."""
    lines = [
        "# Campaign controls v5",
        "",
        "Prior constante ajustado exclusivamente con los 4.096 frames de train v5. "
        "No se consultaron etiquetas de monitor o selección para ajustar el control.",
        "",
        "| Population | Frames | Global dev AP101 | Eligible GT | Peak RSS bytes |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("monitor1024", "full23958"):
        result = report["populations"][name]
        metrics = result["metrics"]
        lines.append(
            f"| {name} | {metrics['sample_count']} | "
            f"{metrics['dev_AP101_supported_classes']} | "
            f"{metrics['population']['eligible_ground_truth_count']} | "
            f"{result['peak_rss_bytes']} |"
        )
    lines.extend(
        [
            "",
            "La AP se calcula globalmente; no es una media de AP por chunks. El scorer es el "
            "protocolo local v2, no el scorer oficial. El resultado completo conserva hashes GT "
            "por token, pero omite curvas PR y pares diagnósticos para mantener memoria acotada.",
            "",
            "La permutación, bootstrap y comparación con R0 permanecen bloqueados hasta disponer "
            "del artefacto R0 completado real y revisar su summary, hashes, token map y fuente GT.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    """Validate all pins, write preregistration, then run the constant control."""
    if args.output.exists():
        raise FileExistsError("Use a new output directory")
    args.output.mkdir(parents=True)
    verified = _verify_inputs(args)
    protocol: EvaluationProtocol = verified["protocol"]
    preregistration = {
        "schema": "phase_ttc_detection_campaign_controls_preregistration_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "control": "train-only constant category prior",
        "fit_population": "v5 train4096 trainer-admitted legacy profile",
        "evaluation_populations": ["v5 monitor1024", "v5 full_selection23958"],
        "global_ap_not_chunk_average": True,
        "protocol_sha256": protocol.sha256,
        "official_eap_score": False,
        "maximum_process_tree_rss_bytes": args.max_rss_bytes,
        "inputs_sha256": verified["inputs_sha256"],
        "all_label_files_verified": verified["all_label_files_verified"],
        "future_comparison_gate": (
            "unimplemented pending review of actual completed R0 summary, hashes, token map "
            "and GT source"
        ),
    }
    write_json(args.output / "preregistration.json", preregistration)
    priors = _fit_priors(args.data_root, verified["train_tokens"], protocol=protocol)
    fit_peak, fit_tree = _resource_rss_bytes()
    if max(fit_peak, fit_tree) > args.max_rss_bytes:
        raise MemoryError("Prior fitting exceeded the preregistered resource gate")
    populations = {}
    admission = verified["values"]["media_admission"]
    for name, tokens in (
        ("monitor1024", verified["monitor_tokens"]),
        ("full23958", verified["full_tokens"]),
    ):
        started = time.perf_counter()
        metrics = score_constant_streaming(
            _frames(
                args.data_root,
                tokens,
                media_admission=admission,
                protocol=protocol,
            ),
            priors,
            protocol=protocol,
            max_rss_bytes=args.max_rss_bytes,
        )
        peak, tree = _resource_rss_bytes()
        if max(peak, tree) > args.max_rss_bytes:
            raise MemoryError("Final process/tree RSS exceeds preregistered gate")
        populations[name] = {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_rss_bytes": peak,
            "owned_tree_current_rss_bytes": tree,
            "metrics": metrics,
        }
    report = {
        "schema": SCHEMA,
        "status": "PASS",
        "protocol_sha256": protocol.sha256,
        "official_eap_score": False,
        "constant_prior": {
            "fit_source": "v5 train4096 trainer-admitted labels only",
            "fit": "component medians, circular yaw mean, class train-frame frequency score",
            "priors": priors,
            "boxes_per_frame": len(priors),
        },
        "populations": populations,
        "provenance": {
            "inputs_sha256": verified["inputs_sha256"],
            "preregistration_sha256": sha256_file(args.output / "preregistration.json"),
            "script_sha256": sha256_file(Path(__file__)),
            "all_label_files_verified": verified["all_label_files_verified"],
        },
        "pending": [
            "R0 observed comparison",
            "prediction derangement",
            "paired sequence bootstrap",
        ],
        "limitations": [
            "Local v2 development AP, not official eAP",
            "Full compact output omits PR point arrays and per-match diagnostics",
            "Constant control is fitted only on trainer-admitted legacy train labels",
        ],
    }
    write_json(args.output / "constant_controls.json", report)
    (args.output / "CAMPAIGN_CONTROLS.md").write_text(render(report), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("E:/eAP_dataset"))
    parser.add_argument(
        "--campaign-receipt",
        type=Path,
        default=Path("output/campaign_data/d4096_v5_readable/receipt.json"),
    )
    parser.add_argument(
        "--campaign-manifest",
        type=Path,
        default=Path("output/campaign_data/d4096_v5_readable/d4096_manifest.json"),
    )
    parser.add_argument(
        "--media-admission",
        type=Path,
        default=Path("output/campaign_data/d4096_v5_readable/media_admission.json"),
    )
    parser.add_argument(
        "--population-audit",
        type=Path,
        default=Path("output/campaign_data/d4096_v5_readable/population_audit.json"),
    )
    parser.add_argument(
        "--full-selection",
        type=Path,
        default=Path("output/campaign_data/full_selection_v1/full_selection_tokens.json"),
    )
    parser.add_argument(
        "--candidate-split",
        type=Path,
        default=Path("data/splits/eap_detection_candidate_v1.json"),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-rss-bytes", type=int, default=1024 * MIB)
    parser.add_argument(
        "--observed-predictions",
        type=Path,
        help="reserved; comparison remains unavailable pending actual R0 contract review",
    )
    args = parser.parse_args()
    try:
        if args.observed_predictions is not None:
            require_frozen_completed_predictions(args.observed_predictions)
        run(args)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
