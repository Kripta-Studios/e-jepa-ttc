#!/usr/bin/env python3
"""Execute the v4.31 diagnostic using v4.30's frozen event-only operator.

No TTC field is loaded.  The only optimization is the inherited v4.30 local
projection distillation stage, restricted to the explicitly selected adaptation rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema
import numpy as np
import torch
import yaml
from torch.nn import functional

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from e_jepa_ttc.data.object_event_v4_31 import (  # noqa: E402, I001
    ADAPT_SEQUENCES,
    AtomicDirectory,
    scientific_metadata,
    strict_json,
)
from e_jepa_ttc.evaluation.object_event_v4_31 import (  # noqa: E402
    THRESHOLDS,
    chunked_controls,
    causal_decision,
    control_metrics,
    gate,
    pearson,
    posterior_stability_by_seed,
    radial_spectrum,
    sequence_swap_gate,
)
from e_jepa_ttc.models.object_event_v4_30 import ObjectEventTTCV430, ObjectEventV430Config  # noqa: E402
from scripts.analyze_object_event_v4_30_stable_similarity import (  # noqa: E402
    _build_teacher_consensus_cache,
    _head_state_hash,
    _stage1,
)
from scripts.preflight_object_event_v4_31 import run as preflight  # noqa: E402
from scripts.train_e_jepa_object_event_v4_12 import _load_backbone  # noqa: E402


@dataclass(frozen=True)
class _Split:
    events: torch.Tensor
    sequence_ids: tuple[str, ...]
    delta_t_s: np.ndarray


def _offsets() -> dict[int, np.ndarray]:
    """Locked v4.30 displacement support grids, expressed in base pixels."""
    result: dict[int, np.ndarray] = {}
    for scale in (1, 2, 4):
        axis = np.arange(-scale, scale + 1, dtype=np.float32) * scale
        yy, xx = np.meshgrid(axis, axis, indexing="ij")
        result[scale] = np.stack((xx.reshape(-1), yy.reshape(-1)), axis=1)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _gate_thresholds(raw: dict[str, Any]) -> dict[str, float]:
    """Flatten the single validated YAML threshold source for the pure gate API."""
    source = raw.get("thresholds")
    if not isinstance(source, dict):
        raise ValueError("config thresholds must be a mapping")
    try:
        slope = source["slope"]
        swap = source["swap"]
        sequence_swap = source["sequence_swap"]
        if not (
            isinstance(slope, list)
            and len(slope) == 2
            and isinstance(swap, dict)
            and isinstance(sequence_swap, dict)
        ):
            raise ValueError
        values = {
            "js_median_max": source["js_median_max"],
            "js_p95_max": source["js_p95_max"],
            "displacement_p95_max": source["displacement_p95_max"],
            "nonempty_min": source["nonempty_min"],
            "valid_energy_fraction_min": source["valid_energy_fraction"],
            "high_band_cv_p95_max": source["high_band_cv_p95_max"],
            "effective_rank_min": source["effective_rank_min"],
            "analytic_pearson_min": source["analytic_pearson_min"],
            "slope_min": slope[0],
            "slope_max": slope[1],
            "sign_min": source["sign_min"],
            "oddness_median_max": source["oddness_median_max"],
            "oddness_p95_max": source["oddness_p95_max"],
            "identity_p95_max": source["identity_p95_max"],
            "leakage_p95_max": source["leakage_p95_max"],
            "swap_corr_max": swap["corr_max"],
            "swap_flip_min": swap["flip_min"],
            "swap_coverage_min": swap["coverage_min"],
            "sequence_swap_corr_max": sequence_swap["corr_max"],
            "sequence_swap_flip_min": sequence_swap["flip_min"],
            "sequence_swap_coverage_min": sequence_swap["coverage_min"],
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("config thresholds do not satisfy the v4.31 gate contract") from exc
    result = {key: float(value) for key, value in values.items()}
    if set(result) != set(THRESHOLDS) or not all(np.isfinite(value) for value in result.values()):
        raise ValueError("config thresholds are incomplete or nonfinite")
    return result


def _validate_summary(summary: dict[str, Any]) -> None:
    """Validate the runtime object before atomically publishing strict JSON."""
    schema = json.loads(
        (ROOT / "schemas/object_event_v4_31_audit_v1.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.validate(summary, schema)


def _read_rows(cache: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in (cache / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _load_split(cache: Path, *, full: bool) -> tuple[_Split, np.ndarray, np.ndarray]:
    expected = 4096 if full else 512
    values = np.load(cache / "events.npy", mmap_mode="r")
    delta = np.load(cache / "delta_t_s.npy", mmap_mode="r")
    rows = _read_rows(cache)
    if (
        values.shape != (expected, 3, 12, 128, 128)
        or len(rows) != expected
        or len(delta) != expected
    ):
        raise ValueError("sanitized cache arrays/rows do not have the mode-locked length")
    sequence_ids = tuple(str(row["sequence_id"]) for row in rows)
    adapt = np.asarray(
        [index for index, item in enumerate(sequence_ids) if item in ADAPT_SEQUENCES]
    )
    audit = np.asarray(
        [index for index, item in enumerate(sequence_ids) if item not in ADAPT_SEQUENCES]
    )
    if len(adapt) != len(audit) or not len(adapt):
        raise ValueError("cache must contain equal nonempty adaptation and audit partitions")
    if set(adapt).intersection(audit):
        raise RuntimeError("adaptation/audit index overlap")
    events = torch.from_numpy(values)
    # The frozen v4.30 v48 route consumes 64x64 tensors; preserve scalar planes.
    dense = functional.interpolate(
        events[:, :, :10].float().reshape(-1, 10, 128, 128), size=(64, 64), mode="area"
    )
    dense = dense.reshape(len(events), 3, 10, 64, 64)
    scalar = events[:, :, 10:12, :1, :1].float().expand(-1, -1, -1, 64, 64)
    return _Split(torch.cat((dense, scalar), dim=2), sequence_ids, np.asarray(delta)), adapt, audit


def _stage2_metrics(stage2: Path | None) -> tuple[dict[str, Any], bool, bool]:
    if stage2 is None:
        return {"available": False, "reason": "stage2_not_supplied"}, False, False
    manifest = json.loads((stage2 / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("artifact_type") != "object_event_v4_31_stage2_v1":
        raise ValueError("stage2 input is not the v4.31 sanitized artifact")
    required = {"row_index", "log_eta", "endpoint_swap_log_eta", "unknown", "row_identity"}
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("stage2 manifest lacks output hashes")
    per_seed: dict[str, dict[str, float | bool | None]] = {}
    evidence_complete = True
    for seed in (7, 13, 23):
        output_path = stage2 / f"seed_{seed}.npz"
        output_meta = outputs.get(str(seed))
        if not isinstance(output_meta, dict) or output_meta.get("sha256") != _sha256(output_path):
            raise ValueError("stage2 output hash differs from manifest")
        with np.load(output_path, allow_pickle=False) as values:
            if set(values.files) != required:
                raise ValueError("stage2 artifact contains non-allowlisted fields")
            row_index = np.asarray(values["row_index"], dtype=np.int64)
            base = np.asarray(values["log_eta"], dtype=np.float32)
            swap = np.asarray(values["endpoint_swap_log_eta"], dtype=np.float32)
            unknown = np.asarray(values["unknown"])
            identities = np.asarray(values["row_identity"])
        if len(row_index) != 2048 or not np.array_equal(row_index, np.arange(2048)):
            raise ValueError("stage2 evidence must retain independent OOF rows 0..2047")
        if (
            base.shape != (2048,)
            or swap.shape != (2048,)
            or unknown.shape != (2048,)
            or identities.shape != (2048,)
            or identities.dtype.kind not in {"U", "S"}
            or not all(str(item) for item in identities)
            or len(set(str(item) for item in identities)) != 2048
            or not np.isfinite(base).all()
            or not np.isfinite(swap).all()
            or not (unknown.dtype == np.bool_ or np.isin(unknown, (0, 1)).all())
        ):
            raise ValueError(
                "stage2 arrays have an invalid shape, finite contract, or unknown mask"
            )
        valid = ~unknown.astype(bool, copy=False)
        covered = valid & (np.abs(base) >= 0.005)
        corr = pearson(base[covered], swap[covered])
        flip = (
            float(np.mean(np.sign(base[covered]) != np.sign(swap[covered])))
            if covered.any()
            else None
        )
        passed = bool(
            corr is not None
            and corr <= -0.80
            and flip is not None
            and flip >= 0.90
            and float(covered.mean()) >= 0.25
        )
        per_seed[str(seed)] = {
            "corr": corr,
            "flip": flip,
            "coverage": float(covered.mean()),
            "valid_fraction": float(valid.mean()),
            "passed": passed,
        }
        evidence_complete = (
            evidence_complete and bool(valid.any()) and corr is not None and flip is not None
        )
    median = {
        key: float(np.median([float(per_seed[str(seed)][key] or 0.0) for seed in (7, 13, 23)]))
        for key in ("corr", "flip", "coverage")
    }
    median["passed"] = bool(
        median["corr"] <= -0.80 and median["flip"] >= 0.90 and median["coverage"] >= 0.25
    )
    all_gates_pass = bool(
        median["passed"] and all(bool(value["passed"]) for value in per_seed.values())
    )
    return (
        {"available": True, "per_seed": per_seed, "median": median},
        evidence_complete,
        all_gates_pass,
    )


def _sequence_controls(control: dict[str, Any]) -> dict[str, dict[str, float | None]]:
    result: dict[str, dict[str, float | None]] = {}
    ids = np.asarray(control["sequence_id"])
    for sequence in sorted(set(ids.tolist())):
        mask = ids == sequence
        sub: dict[str, Any] = {
            "base": np.asarray(control["base"])[mask],
            "sequence_id": ids[mask].tolist(),
        }
        for key, value in control.items():
            if key in {"base", "sequence_id"}:
                continue
            sub[key] = {
                "prediction": np.asarray(value["prediction"])[mask],
                "unknown": np.asarray(value["unknown"])[mask],
            }
        result[sequence] = control_metrics(sub)
    return result


def _projected_spectra(
    model: ObjectEventTTCV430, events: torch.Tensor, batch_size: int
) -> list[dict[str, float | bool]]:
    """Capture actual [B,3,C,H,W] local-projection output and remove the hook."""
    captured: list[torch.Tensor] = []

    def save_output(_: torch.nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        captured.append(output.detach().cpu())

    handle = model.local_projection.register_forward_hook(save_output)
    try:
        with torch.no_grad():
            for start in range(0, len(events), batch_size):
                chunk = events[start : start + batch_size]
                parameter = next(model.parameters(), None)
                if parameter is not None:
                    chunk = chunk.to(parameter.device)
                model(chunk)
    finally:
        handle.remove()
    projected = torch.cat(captured, dim=0)
    if projected.shape[0] != len(events) * 3:
        raise RuntimeError("local_projection hook did not capture one map per audit timepoint")
    projected = projected.reshape(len(events), 3, *projected.shape[1:])
    return [radial_spectrum(item[2]) for item in projected]


def _metrics_for_seed(
    control: dict[str, Any], spectra: list[dict[str, float | bool]]
) -> tuple[dict[str, float | None], dict[str, dict[str, float | None]]]:
    metrics: dict[str, float | None] = {
        "js_median": None,
        "js_p95": None,
        "displacement_p95": None,
        "nonempty_fraction": float(np.mean([item["valid_energy"] for item in spectra])),
        "valid_energy_fraction": float(np.mean([item["valid_energy"] for item in spectra])),
        "high_band_cv_p95": None,
        "effective_rank": float(np.median([float(item["effective_rank"]) for item in spectra])),
    }
    metrics.update(control_metrics(control))
    return metrics, _sequence_controls(control)


def _median_metrics(per_seed: dict[str, dict[str, float | None]]) -> dict[str, float | None]:
    keys = next(iter(per_seed.values())).keys()
    output: dict[str, float | None] = {}
    for key in keys:
        values: list[float] = []
        for item in per_seed.values():
            value = item[key]
            if value is not None:
                values.append(float(value))
        output[key] = (
            float(np.median(values))
            if len(values) == len(per_seed) and all(np.isfinite(value) for value in values)
            else None
        )
    return output


def analyze(
    config: Path,
    cache: Path,
    stage2: Path | None,
    output: Path,
    *,
    full: bool,
    force: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run real frozen-model diagnostic forwards; full mode requires usable stage2."""
    started = time.perf_counter()
    raw = yaml.safe_load(config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("config must be a mapping")
    thresholds = _gate_thresholds(raw)
    pf = preflight(config, cache, full=full)
    split, adapt, audit = _load_split(cache, full=full)
    v430_path = ROOT / raw["v430"]["config"]
    v48_path = ROOT / raw["v430"]["v48_config"]
    if _sha256(v430_path) != raw["v430"].get("config_sha256"):
        raise ValueError("v4.30 config SHA mismatch")
    if _sha256(v48_path) != raw["v430"].get("v48_config_sha256"):
        raise ValueError("v4.8 backbone config SHA mismatch")
    if _sha256(ROOT / raw["v430"]["summary"]) != raw["v430"]["summary_sha256"]:
        raise ValueError("v4.30 summary SHA mismatch")
    v430 = yaml.safe_load(v430_path.read_text(encoding="utf-8"))
    device_value = torch.device(device)
    checkpoints = {
        int(seed): ROOT / path for seed, path in raw["adaptation"]["checkpoints"].items()
    }
    if set(checkpoints) != {7, 13, 23} or not all(path.is_file() for path in checkpoints.values()):
        raise FileNotFoundError("all real v4.22 frozen geometry checkpoints are required")
    hashes = {str(seed): _sha256(path) for seed, path in checkpoints.items()}
    if hashes != {str(seed): raw["checkpoint_sha256"].get(seed) for seed in (7, 13, 23)}:
        raise ValueError("frozen checkpoint SHA mismatch")
    consensus = _build_teacher_consensus_cache(
        split,
        checkpoints,
        checkpoint_hashes=hashes,
        v48_config=ROOT / raw["v430"]["v48_config"],
        cfg=v430["arms"]["stable_multiscale_similarity"],
        device=device_value,
    )
    histories: dict[str, Any] = {}
    models: dict[str, Any] = {}
    posterior: list[dict[int, np.ndarray]] = []
    for seed in (7, 13, 23):
        backbone, _ = _load_backbone(
            v48_config_path=ROOT / raw["v430"]["v48_config"], checkpoint_path=checkpoints[seed]
        )
        initial_cfg = {
            key: value
            for key, value in v430["arms"]["stable_multiscale_similarity"].items()
            if key != "batch_size"
        }
        initial_model = ObjectEventTTCV430(backbone, ObjectEventV430Config(**initial_cfg))
        initial = _head_state_hash(initial_model.local_projection.state_dict())
        del initial_model
        model, history, held = _stage1(
            split,
            audit,
            checkpoints[seed],
            consensus_cache=consensus,
            v48_config=ROOT / raw["v430"]["v48_config"],
            cfg=v430["arms"]["stable_multiscale_similarity"],
            train_cfg=v430["train"],
            stabilization_cfg=v430["stabilization"],
            seed=seed,
            fold_seed=int(raw["adaptation"]["shuffle_seed_offset"]),
            device=device_value,
        )
        final = _head_state_hash(model.local_projection.state_dict())
        loss = [float(item["distill_kl"]) for item in history]
        histories[str(seed)] = {
            "history": history,
            "initial_head_sha256": initial,
            "final_head_sha256": final,
            "loss_ratio_last3_first3": float(np.mean(loss[-3:]) / np.mean(loss[:3])),
        }
        if (
            not np.isfinite(loss).all()
            or histories[str(seed)]["loss_ratio_last3_first3"] > 0.90
            or initial == final
        ):
            raise FloatingPointError("adaptation loss gate failed")
        models[str(seed)] = model
        posterior.append(held)
    posterior_by_seed = {seed: value for seed, value in zip((7, 13, 23), posterior, strict=True)}
    per_seed_stability, joint_stability, stability_counts = posterior_stability_by_seed(
        posterior_by_seed,
        _offsets(),
    )
    per_seed: dict[str, dict[str, float | None]] = {}
    sequence_metrics: dict[str, Any] = {}
    controls: dict[str, Any] = {}
    spectra_by_seed: dict[str, list[dict[str, float | bool]]] = {}
    audit_events = split.events[audit]
    audit_ids = [split.sequence_ids[index] for index in audit]
    for seed, model in models.items():
        control = chunked_controls(model, audit_events, audit_ids, batch_size=8)
        controls[seed] = control
        spectra_by_seed[seed] = _projected_spectra(model, audit_events, batch_size=8)
        per_seed[seed], sequence_metrics[seed] = _metrics_for_seed(control, spectra_by_seed[seed])
        per_seed[seed]["sequence_swap_all_pass"] = float(
            all(sequence_swap_gate(value, thresholds) for value in sequence_metrics[seed].values())
        )
    high = np.asarray(
        [
            [float(item["high_fraction"]) for item in spectra_by_seed[seed]]
            for seed in ("7", "13", "23")
        ]
    )
    high_cv = np.std(high, axis=0) / np.maximum(np.mean(high, axis=0), 1e-12)
    for seed, values in per_seed.items():
        values["high_band_cv_p95"] = float(np.percentile(high_cv, 95))
        values.update(per_seed_stability[int(seed)])
    median = _median_metrics(per_seed)
    gates = {seed: gate(values, thresholds) for seed, values in per_seed.items()}
    median_gate = gate(median, thresholds)
    stage2_metrics, stage2_evidence_complete, stage2_pass = _stage2_metrics(stage2)
    metric_evidence_complete = all(item.get("finite", False) for item in gates.values()) and bool(
        median_gate.get("finite", False)
    )
    evidence_complete = metric_evidence_complete and (not full or stage2_evidence_complete)
    all_gates_pass = (
        all(item["passed"] for item in gates.values())
        and median_gate["passed"]
        and all(per_seed[seed].get("sequence_swap_all_pass") == 1.0 for seed in per_seed)
        and (not full or stage2_pass)
    )
    decision = causal_decision(
        complete=evidence_complete,
        stability_pass=all(item.get("stability", False) for item in gates.values()),
        spectrum_pass=all(item.get("spectrum", False) for item in gates.values()),
        operator_pass=all(
            item.get("equivariance", False)
            and item.get("invariance", False)
            and item.get("reversal", False)
            and item.get("zero", False)
            and per_seed[seed].get("sequence_swap_all_pass") == 1.0
            for seed, item in gates.items()
        ),
        stage2_pass=True if not full else stage2_pass,
        diagnostic=not full,
    )
    summary = {
        "artifact_type": "object_event_v4_31_audit_v1",
        **scientific_metadata(
            artifact_type="object_event_v4_31_audit_v1",
            evidence_type="causal_operator_audit",
            protocol_version="object_event_v4_31_train_only_v1",
            protocol_sha256=str(raw["split_sha256"]),
            artifact_sha256=hashlib.sha256(
                strict_json(
                    {
                        "config": _sha256(config),
                        "cache": pf["count"],
                        "teacher": consensus.consensus_cache_sha256,
                    }
                ).encode("utf-8")
            ).hexdigest(),
        ),
        "status": "not_issued_diagnostic"
        if not full
        else (
            "completed"
            if all_gates_pass
            else ("completed_gate_failed" if evidence_complete else "invalid_incomplete")
        ),
        "selectable": False,
        "mode": "full" if full else "diagnostic",
        "preflight": pf,
        "seeds": [7, 13, 23],
        "adaptation": histories,
        "teacher_consensus": {
            "sha256": consensus.consensus_cache_sha256,
            "rows": consensus.row_count,
            "batches": consensus.teacher_backbone_forward_batches,
        },
        "metrics": {
            "per_seed": per_seed,
            "median": median,
            "joint_stability": joint_stability,
            "stability_counts": stability_counts,
            "sequence": sequence_metrics,
        },
        "gates": {
            "per_seed": gates,
            "median": median_gate,
            "evidence_complete": evidence_complete,
            "all_gates_pass": all_gates_pass,
            "thresholds": thresholds,
        },
        "stage2": stage2_metrics,
        "causal_decision": decision,
        "timing": {
            "elapsed_s": time.perf_counter() - started,
            "forwards": consensus.teacher_backbone_forward_batches,
            "batches": 0,
            "ram_bytes": None,
            "vram_bytes": None,
            "throughput_rows_s": None,
        },
        "opened_paths": pf["opened_paths"]
        + [str(v430_path), *(str(value) for value in checkpoints.values())],
        "forbidden_access": [],
    }
    _validate_summary(summary)
    with AtomicDirectory(output, force=force) as stage:
        (stage / "summary.json").write_text(strict_json(summary), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/experiment/e_jepa_garl_object_event_operator_audit_v4_31.yaml",
    )
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--stage2", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        print(
            strict_json(
                analyze(
                    args.config,
                    args.cache,
                    args.stage2,
                    args.output_dir,
                    full=args.full,
                    force=args.force,
                    device=args.device,
                )
            )
        )
    except Exception as exc:
        print(
            strict_json(
                {
                    "artifact_type": "object_event_v4_31_audit_v1",
                    "status": "invalid_incomplete",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ),
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
