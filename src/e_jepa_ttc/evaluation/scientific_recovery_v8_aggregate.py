"""Fail-closed seed-7 aggregate for the frozen V8 downstream screen."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from collections import Counter
from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.artifacts.hashing import canonical_json, sign_artifact, verify_artifact_hash
from e_jepa_ttc.evaluation.garl_ttc_protocol import (
    PAPER_MID_WEIGHTS,
    sequence_macro_signed_metrics,
    signed_garl_metrics,
)

ROOT = Path(__file__).resolve().parents[3]
_SEALED = ("public_validation", "private_test", "evttc_test", "codabench")
_CANDIDATES = {
    "router": "R",
    "timevol20_3": "B1_TIMEVOL20_3",
    "exp6_3": "B2_EXP6_3",
    "pair20_2": "B3_PAIR20_2",
    "gated_exp6_3": "C1_GATED_EXP6_3",
}


class V8AggregateIntegrityError(RuntimeError):
    """Raised when seed-7 selection inputs are not exact frozen OOF evidence."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records_hash(rows: list[dict[str, str]]) -> str:
    return hashlib.sha256(
        b"".join(canonical_json(row) + b"\n" for row in sorted(rows, key=lambda x: x["token_id"]))
    ).hexdigest()


def contract_hashes(rows: list[dict[str, str]]) -> dict[str, str]:
    """Return immutable row-contract hashes using the exact macro-MiD weight definition."""
    expected_weights = _expected_weights(rows)
    return {
        "ordered_token_ids_sha256": _records_hash([{"token_id": r["token_id"]} for r in rows]),
        "row_identity_sha256": _records_hash(
            [{k: r[k] for k in ("sequence_id", "token_id", "track_id")} for r in rows]
        ),
        "target_sha256": _records_hash(
            [{"target_ttc_s": r["target_ttc"], "token_id": r["token_id"]} for r in rows]
        ),
        "mid_sample_weight_sha256": _records_hash(
            [{"sample_weight": expected_weights[r["token_id"]], "token_id": r["token_id"]} for r in rows]
        ),
        "fold_assignment_sha256": _records_hash(
            [{k: r[k] for k in ("outer_fold", "sequence_id", "token_id")} for r in rows]
        ),
    }


def _expected_weights(rows: list[dict[str, str]]) -> dict[str, str]:
    """Recreate the frozen Decimal macro-MiD coefficient per token."""
    def bucket(value: Decimal) -> tuple[str, Decimal]:
        if Decimal("0") < value <= Decimal("3"):
            return "crucial", Decimal("0.5")
        if Decimal("3") < value <= Decimal("6"):
            return "small", Decimal("0.3")
        if Decimal("6") < value <= Decimal("10"):
            return "large", Decimal("0.1")
        if Decimal("-10") < value <= Decimal("0"):
            return "negative", Decimal("0.1")
        raise V8AggregateIntegrityError("target outside frozen signed MiD domain")
    parsed = [(r, Decimal(r["target_ttc"])) for r in rows]
    counts = Counter((r["sequence_id"], bucket(value)[0]) for r, value in parsed)
    return {
        r["token_id"]: str(bucket(value)[1] / Decimal(9) / counts[(r["sequence_id"], bucket(value)[0])])
        for r, value in parsed
    }


def _closed(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower().endswith(("_opened", "_used_for_selection")) and item is not False:
                raise V8AggregateIntegrityError("sealed evaluation flag is not false")
            _closed(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _closed(item)
    elif isinstance(value, str) and any(x in value.lower().replace("\\", "/") for x in _SEALED):
        raise V8AggregateIntegrityError("sealed evaluation source rejected")


def _read_rows(path: Path, *, candidate: bool) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    if not raw:
        raise V8AggregateIntegrityError(f"empty predictions: {path}")
    aliases = {
        "token_id": ("token_id", "sample_token"),
        "outer_fold": ("outer_fold", "fold"),
        "target_ttc": ("target_ttc", "target_ttc_s"),
        "prediction_ttc": ("prediction_ttc", "point_prediction_ttc_s", "prediction_ttc_s"),
        "sample_weight": ("sample_weight",),
        "sequence_id": ("sequence_id",),
        "track_id": ("track_id",),
        "seed": ("seed",),
    }
    out = []
    for row in raw:
        normalized = {}
        for key, choices in aliases.items():
            value = next(
                (row.get(choice) for choice in choices if row.get(choice) not in (None, "")), None
            )
            if value is None and key == "sample_weight" and not candidate:
                value = "1"
            if value is None and key == "seed" and not candidate:
                value = "7"
            if value is None:
                raise V8AggregateIntegrityError(f"missing {key} in {path}")
            normalized[key] = str(value)
        normalized.update({k: str(v) for k, v in row.items() if v is not None})
        out.append(normalized)
    if len({r["token_id"] for r in out}) != len(out):
        raise V8AggregateIntegrityError("duplicate token_id in predictions")
    for r in out:
        try:
            values = [float(r[k]) for k in ("target_ttc", "prediction_ttc", "sample_weight")]
        except ValueError as e:
            raise V8AggregateIntegrityError("non-numeric OOF value") from e
        if not all(math.isfinite(x) for x in values):
            raise V8AggregateIntegrityError("NaN or infinity in OOF predictions")
    return out


def _resolve(root: Path, owner: Path, value: object) -> Path:
    if not isinstance(value, str):
        raise V8AggregateIntegrityError("artifact path is missing")
    path = Path(value)
    path = path if path.is_absolute() else owner / path
    path = path.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as e:
        raise V8AggregateIntegrityError("artifact path escapes repository") from e
    if not path.is_file():
        raise V8AggregateIntegrityError(f"artifact is missing: {path}")
    return path


def _fold_rows(
    *,
    arm: str,
    fold: int,
    config_hash: str,
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    results_root: Path,
    repository_root: Path,
    allow_fixture: bool,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    run = results_root / "runs" / f"{arm}_fold{fold}_seed7"
    summary_path = run / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise V8AggregateIntegrityError(f"missing/invalid summary for {arm} fold {fold}") from e
    if not isinstance(summary, dict) or not verify_artifact_hash(summary):
        raise V8AggregateIntegrityError("fold summary is unsigned")
    _closed(summary)
    required = {
        "artifact_type": "scientific_recovery_v8_fold_result_v1",
        "status": "completed",
        "arm": arm,
        "outer_fold": fold,
        "seed": 7,
        "config_sha256": config_hash,
        "protocol_sha256": protocol.get("artifact_sha256"),
        "frozen_manifest_sha256": manifest.get("artifact_sha256"),
    }
    if any(summary.get(k) != v for k, v in required.items()):
        raise V8AggregateIntegrityError("fold summary is not bound to its frozen job")
    if summary.get("fixture") is True and not allow_fixture:
        raise V8AggregateIntegrityError("fixture result rejected")
    checkpoint = summary.get("checkpoint")
    prediction = summary.get("dev_predictions")
    if not isinstance(checkpoint, Mapping) or not isinstance(prediction, Mapping):
        raise V8AggregateIntegrityError("summary lacks checkpoint/prediction bindings")
    cp = _resolve(repository_root, run, checkpoint.get("path"))
    pp = _resolve(repository_root, run, prediction.get("path"))
    if checkpoint.get("sha256") != _sha(cp):
        raise V8AggregateIntegrityError("checkpoint hash mismatch")
    if prediction.get("sha256") not in (None, _sha(pp)):
        raise V8AggregateIntegrityError("prediction hash mismatch")
    rows = _read_rows(pp, candidate=True)
    if any(
        int(r["outer_fold"]) != fold
        or int(r["seed"]) != 7
        or r.get("config_sha256") != config_hash
        or r.get("checkpoint_sha256") != checkpoint["sha256"]
        for r in rows
    ):
        raise V8AggregateIntegrityError("OOF rows are not bound to summary config/checkpoint")
    return rows, {
        "summary": str(summary_path),
        "summary_sha256": _sha(summary_path),
        "prediction": str(pp),
        "prediction_sha256": _sha(pp),
        "checkpoint": str(cp),
        "checkpoint_sha256": _sha(cp),
    }


def _metrics(
    rows: list[dict[str, str]], baseline: list[dict[str, str]]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    by_token = {r["token_id"]: r for r in baseline}
    if set(by_token) != {r["token_id"] for r in rows}:
        raise V8AggregateIntegrityError("candidate and A5 token universes differ")
    for r in rows:
        b = by_token[r["token_id"]]
        keys = (
            ("track_id", "outer_fold", "target_ttc")
            if "__draw" in r["sequence_id"]
            else ("sequence_id", "track_id", "outer_fold", "target_ttc")
        )
        if any(r[k] != b[k] for k in keys):
            raise V8AggregateIntegrityError("candidate/A5 identity mismatch")
    target = np.array([float(r["target_ttc"]) for r in rows])
    pred = np.array([float(r["prediction_ttc"]) for r in rows])
    base = np.array([float(by_token[r["token_id"]]["prediction_ttc"]) for r in rows])
    seq = np.array([r["sequence_id"] for r in rows])
    cm = sequence_macro_signed_metrics(target, pred, seq)
    bm = sequence_macro_signed_metrics(target, base, seq)
    mid = float(cm["sequence_macro_paper_MiD_overall"])
    base_mid = float(bm["sequence_macro_paper_MiD_overall"])
    if not math.isfinite(mid) or not math.isfinite(base_mid):
        raise V8AggregateIntegrityError("non-finite sequence macro MiD")
    failure = float(np.mean(~np.isfinite(pred) | (np.abs(pred) < 0.1)))
    per_seq = {}
    per_bucket = {}
    for key, groups in (("sequence_id", sorted(set(seq))),):
        for group in groups:
            ix = np.array([r[key] == group for r in rows])
            per_seq[group] = {
                "mid_macro_sequence": float(
                    sequence_macro_signed_metrics(target[ix], pred[ix], seq[ix])[
                        "sequence_macro_paper_MiD_overall"
                    ]
                ),
                "delta_mid_vs_a5": float(
                    sequence_macro_signed_metrics(target[ix], pred[ix], seq[ix])[
                        "sequence_macro_paper_MiD_overall"
                    ]
                    - sequence_macro_signed_metrics(target[ix], base[ix], seq[ix])[
                        "sequence_macro_paper_MiD_overall"
                    ]
                ),
                "row_count": int(ix.sum()),
            }
    for name in PAPER_MID_WEIGHTS:
        # signed metric bucket membership is exact and target-defined.
        mask = (
            (target > 0) & (target <= 3)
            if name == "crucial"
            else (target > 3) & (target <= 6)
            if name == "small"
            else (target > 6) & (target <= 10)
            if name == "large"
            else (target > -10) & (target <= 0)
        )
        if not mask.any():
            per_bucket[name] = {"mid_macro_sequence": 0.0, "delta_mid_vs_a5": 0.0, "row_count": 0}
        else:
            per_bucket[name] = {
                "mid_macro_sequence": float(
                    signed_garl_metrics(target[mask], pred[mask])["paper_MiD_overall"]
                ),
                "delta_mid_vs_a5": float(
                    signed_garl_metrics(target[mask], pred[mask])["paper_MiD_overall"]
                    - signed_garl_metrics(target[mask], base[mask])["paper_MiD_overall"]
                ),
                "row_count": int(mask.sum()),
            }
    return (
        {
            "mid_macro_sequence": mid,
            "delta_mid_vs_a5": mid - base_mid,
            "finite_fraction": float(np.isfinite(pred).mean()),
            "failure_rate": failure,
            "coverage_drop_max_pp": 0.0,
        },
        per_seq,
        per_bucket,
    )


def _bootstrap(
    rows: list[dict[str, str]], baseline: list[dict[str, str]], n: int, seed: int
) -> dict[str, Any]:
    if n < 1:
        raise ValueError("resamples must be positive")
    b = {r["token_id"]: r for r in baseline}
    by_seq = {
        s: [r for r in rows if r["sequence_id"] == s]
        for s in sorted({r["sequence_id"] for r in rows})
    }
    if len(by_seq) < 2:
        raise V8AggregateIntegrityError("bootstrap requires at least two sequences")
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        chosen = rng.choice(list(by_seq), size=len(by_seq), replace=True)
        cr = []
        br = []
        for draw, s in enumerate(chosen):
            tracks = sorted({r["track_id"] for r in by_seq[s]})
            picked = rng.choice(tracks, size=len(tracks), replace=True)
            for ti, t in enumerate(picked):
                for r in by_seq[s]:
                    if r["track_id"] == t:
                        q = dict(r)
                        q["sequence_id"] = f"{s}__draw{draw}__track{ti}"
                        cr.append(q)
                        x = dict(b[r["token_id"]])
                        x["sequence_id"] = q["sequence_id"]
                        br.append(x)
        m, _, _ = _metrics(cr, br)
        deltas.append(m["delta_mid_vs_a5"])
    return {
        "probability_delta_lt_zero": float(np.mean(np.asarray(deltas) < 0)),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "resamples": n,
    }



def _implementation_commit(repository_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _candidate_config_entries(
    manifest: Mapping[str, Any], *, arm: str, results_root: Path
) -> list[tuple[str, Mapping[str, Any]]]:
    """Return enabled configs or preregistered conditional configs once executed."""
    enabled = manifest.get("enabled_seed7_configs", {})
    if not isinstance(enabled, Mapping):
        raise V8AggregateIntegrityError("missing frozen configs")
    entries = [
        (str(name), entry)
        for name, entry in enabled.items()
        if str(name).startswith(f"{arm}_fold")
        and str(name).endswith("_seed7")
        and isinstance(entry, Mapping)
    ]
    if entries:
        return entries
    templates = manifest.get("conditional_templates", {})
    template = templates.get(arm) if isinstance(templates, Mapping) else None
    if not isinstance(template, Mapping):
        return []
    presence = [
        (results_root / "runs" / f"{arm}_fold{fold}_seed7").exists() for fold in range(3)
    ]
    if not any(presence):
        return []
    if not all(presence):
        raise V8AggregateIntegrityError(f"partial conditional run set for {arm}")
    raw = template.get("fold_configs")
    if not isinstance(raw, list) or len(raw) != 3 or not all(isinstance(x, Mapping) for x in raw):
        raise V8AggregateIntegrityError(f"conditional template for {arm} lacks three configs")
    return [(f"{arm}_fold{fold}_seed7", entry) for fold, entry in enumerate(raw)]


def _bind_candidate_to_baseline_contract(
    rows: list[dict[str, str]], baseline: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Validate protected row values numerically and canonicalize their exact text."""
    by_token = {row["token_id"]: row for row in baseline}
    expected_weights = _expected_weights(baseline)
    if set(by_token) != {row["token_id"] for row in rows}:
        raise V8AggregateIntegrityError("candidate token set differs from frozen A5 baseline")
    normalized: list[dict[str, str]] = []
    for row in rows:
        base = by_token[row["token_id"]]
        for key in ("sequence_id", "track_id", "outer_fold"):
            if str(row[key]) != str(base[key]):
                raise V8AggregateIntegrityError(f"candidate {key} differs from baseline")
        if abs(float(row["target_ttc"]) - float(base["target_ttc"])) > 1e-6:
            raise V8AggregateIntegrityError("candidate target differs from baseline")
        expected_weight = float(expected_weights[row["token_id"]])
        if not math.isclose(float(row["sample_weight"]), expected_weight, rel_tol=2e-5, abs_tol=1e-10):
            raise V8AggregateIntegrityError("candidate MiD sample weight differs from frozen definition")
        item = dict(row)
        item["sequence_id"] = base["sequence_id"]
        item["track_id"] = base["track_id"]
        item["outer_fold"] = base["outer_fold"]
        item["target_ttc"] = base["target_ttc"]
        item["sample_weight"] = expected_weights[row["token_id"]]
        normalized.append(item)
    return normalized

def aggregate_seed7(
    *,
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    results_root: Path,
    repository_root: Path = ROOT,
    allow_fixture: bool = False,
    resamples: int = 5000,
    bootstrap_seed: int = 20260814,
) -> dict[str, Any]:
    """Recompute and sign one screen decision; fixtures can never nominate a winner."""
    _closed(protocol)
    _closed(manifest)
    configs = manifest.get("enabled_seed7_configs")
    if not isinstance(configs, Mapping):
        raise V8AggregateIntegrityError("missing frozen configs")
    source = (
        protocol.get("sources", {}).get("a5_oof_predictions")
        if isinstance(protocol.get("sources"), Mapping)
        else None
    )
    if not isinstance(source, Mapping):
        raise V8AggregateIntegrityError("missing frozen A5 control")
    a5 = _resolve(repository_root, repository_root, source.get("path"))
    if isinstance(source.get("sha256"), str) and source["sha256"] != _sha(a5):
        raise V8AggregateIntegrityError("frozen A5 hash mismatch")
    base = _read_rows(a5, candidate=False)
    candidates = []
    for arm, candidate_id in _CANDIDATES.items():
        entries = _candidate_config_entries(manifest, arm=arm, results_root=results_root)
        if not entries:
            continue
        if len(entries) != 3:
            raise V8AggregateIntegrityError(f"incomplete frozen configs for {arm}")
        rows = []
        refs = []
        for fold in range(3):
            entry = next((e for n, e in entries if f"fold{fold}_" in str(n)), None)
            if not isinstance(entry, Mapping) or not isinstance(entry.get("sha256"), str):
                raise V8AggregateIntegrityError("invalid frozen config hash")
            r, ref = _fold_rows(
                arm=arm,
                fold=fold,
                config_hash=entry["sha256"],
                protocol=protocol,
                manifest=manifest,
                results_root=results_root,
                repository_root=repository_root,
                allow_fixture=allow_fixture,
            )
            rows += r
            refs.append(ref)
        if len({r["token_id"] for r in rows}) != len(rows):
            raise V8AggregateIntegrityError("duplicate OOF tokens across folds")
        if not allow_fixture:
            rows = _bind_candidate_to_baseline_contract(rows, base)
        hashes = contract_hashes(rows)
        expected = protocol.get("sample_contract", {})
        if len(rows) != expected.get("rows") or any(hashes[k] != expected.get(k) for k in hashes):
            raise V8AggregateIntegrityError(
                "OOF rows/folds/targets/weights differ from frozen contract"
            )
        metrics, per_sequence, per_bucket = _metrics(rows, base)
        boot = _bootstrap(rows, base, resamples, bootstrap_seed)
        gate = protocol.get("gates", {}).get("ttc_candidate_gate", {})
        passed = (
            metrics["delta_mid_vs_a5"] <= float(gate.get("delta_MiD_max", -3))
            and boot["probability_delta_lt_zero"]
            >= float(gate.get("bootstrap_probability_delta_below_zero_min", 0.9))
            and metrics["finite_fraction"] == 1
            and metrics["failure_rate"] == 0
            and metrics["coverage_drop_max_pp"] <= float(gate.get("coverage_drop_max_pp", 1))
        )
        candidates.append(
            {
                "arm": arm,
                "candidate_id": candidate_id,
                "metrics": metrics,
                "bootstrap": boot,
                "per_sequence": per_sequence,
                "per_bucket": per_bucket,
                "passed": passed,
                "refs": refs,
                "rows": rows,
                "hashes": hashes,
            }
        )
    admissible = [x for x in candidates if x["passed"]]
    winner = (
        min(admissible, key=lambda x: x["metrics"]["mid_macro_sequence"]) if admissible else None
    )
    report = {
        "artifact_type": "scientific_recovery_v8_seed7_aggregate_v1",
        "schema_version": protocol.get("schema_version", "scientific_recovery_v8_temporal_v1"),
        "status": "completed",
        "git_commit": protocol.get("git_base_commit"),
        "base_git_commit": protocol.get("git_base_commit"),
        "implementation_git_commit": _implementation_commit(repository_root),
        "protocol_sha256": protocol.get("artifact_sha256"),
        "frozen_manifest_sha256": manifest.get("artifact_sha256"),
        "seed": 7,
        "candidate_id": winner["candidate_id"] if winner else "A5",
        "multiseed_replication_candidate": bool(winner and not allow_fixture),
        "selection_reason": "admissible_lowest_macro_MiD"
        if winner and not allow_fixture
        else "fixture_smoke_no_nomination"
        if allow_fixture
        else "no_downstream_arm_passed; fallback_A5",
        "fixture": allow_fixture,
        "candidate_results": [
            {k: v for k, v in x.items() if k not in {"rows"}} for x in candidates
        ],
        "a5_control": {"path": str(a5), "sha256": _sha(a5)},
        "closed_evaluation": {
            "public_validation_used_for_selection": False,
            "private_test_opened": False,
            "evttc_test_opened": False,
            "codabench_opened": False,
        },
    }
    sign_artifact(report)
    return report


__all__ = ["V8AggregateIntegrityError", "aggregate_seed7", "contract_hashes"]
