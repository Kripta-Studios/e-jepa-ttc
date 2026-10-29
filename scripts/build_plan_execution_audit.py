"""Build a machine-readable, evidence-backed PLAN.md execution matrix."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from e_jepa_ttc.utils.io import read_structured, write_structured

EXPECTED_BASELINE_VARIANTS = {"event_only", "visual_only", "rgbe_late_fusion"}
EXPECTED_BASELINE_SEEDS = {7, 13, 23}


def _safe_read(path: Path) -> dict[str, Any] | None:
    """Read one evidence object without making the audit fragile to a bad run."""

    try:
        value = read_structured(path)
    except (OSError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _baseline_phase_state(
    repo_root: Path,
) -> tuple[str, list[str], list[str]]:
    """Classify the official baseline matrix without treating a screen as full."""

    evidence = [
        "configs/experiment/garl_baseline_suite_v1.yaml",
        "scripts/run_garl_baseline_suite_v1.py",
        "scripts/execute_garl_baseline_suite_v1.py",
    ]
    root = repo_root / "artifacts/runs/garl_baseline_training_v1"
    matrix_path = root / "matrix.json"
    failure_paths = sorted(root.rglob("FAILURE*.json")) if root.is_dir() else []
    for path in [matrix_path, *failure_paths]:
        if path.is_file():
            evidence.append(path.relative_to(repo_root).as_posix())

    cache_root = repo_root / "artifacts/runs/garl_release_cache_training_v1"
    cache_matrix_candidates = [
        cache_root / "matrix.json",
        *sorted(
            repo_root.glob("artifacts/runs/garl_release_cache_matrix_full*/matrix.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        ),
    ]
    cache_payload_candidates = [
        (path, _safe_read(path)) for path in cache_matrix_candidates if path.is_file()
    ]
    cache_matrix_path, cache_matrix = next(
        (
            (path, payload)
            for path, payload in cache_payload_candidates
            if payload is not None
            and payload.get("status") == "completed"
            and payload.get("full_matrix") is True
        ),
        next(
            iter(cache_payload_candidates),
            (cache_root / "matrix.json", None),
        ),
    )
    cache_metrics_path = repo_root / "artifacts/metrics/garl_release_cache_training_v1_signed.json"
    cache_metrics = _safe_read(cache_metrics_path) if cache_metrics_path.is_file() else None
    for path in (cache_matrix_path, cache_metrics_path):
        if path.is_file():
            evidence.append(path.relative_to(repo_root).as_posix())
    cache_runs = cache_matrix.get("runs") if cache_matrix else None
    cache_run_keys = (
        {
            (run.get("variant"), run.get("seed"))
            for run in cache_runs
            if isinstance(run, dict)
            and isinstance(run.get("variant"), str)
            and isinstance(run.get("seed"), int)
        }
        if isinstance(cache_runs, list)
        else set()
    )
    cache_expected = {
        (variant, seed)
        for variant in EXPECTED_BASELINE_VARIANTS
        for seed in EXPECTED_BASELINE_SEEDS
    }
    cache_full = bool(
        cache_matrix
        and cache_matrix.get("status") == "completed"
        and cache_matrix.get("max_batches") is None
        and cache_matrix.get("full_matrix") is True
        and cache_matrix.get("bbox_protocol") == "P0_oracle_bbox_roi"
        and cache_run_keys == cache_expected
        and isinstance(cache_runs, list)
        and len(cache_runs) == len(cache_expected)
        and all(
            isinstance(run, dict)
            and run.get("status") == "completed"
            and isinstance(run.get("validation_metrics"), dict)
            for run in cache_runs
        )
    )
    cache_metrics_green = bool(
        cache_metrics
        and cache_metrics.get("artifact_type") == "garl_release_cache_training_metrics_v1"
        and cache_metrics.get("status") == "pass"
        and cache_metrics.get("expected_run_count") == len(cache_expected)
        and cache_metrics.get("observed_metric_count") == len(cache_expected)
        and cache_metrics.get("missing") == []
        and cache_metrics.get("test_used_for_selection") is False
        and cache_metrics.get("evttc_used_for_selection") is False
    )
    if cache_full and cache_metrics_green:
        return "verified", evidence, []

    matrix = _safe_read(matrix_path) if matrix_path.is_file() else None
    if matrix is None:
        return (
            "not_executed",
            evidence,
            [
                "official baseline matrix with variants event_only/visual_only/rgbe_late_fusion",
                "seeds 7/13/23 and full (not max-batches) training",
                "signed validation metrics generated from the completed checkpoints",
            ],
        )

    runs = matrix.get("runs")
    run_keys: set[tuple[str, int]] = set()
    all_completed = True
    if isinstance(runs, list):
        for run in runs:
            if not isinstance(run, dict):
                all_completed = False
                continue
            variant = run.get("variant")
            seed = run.get("seed")
            if isinstance(variant, str) and isinstance(seed, int):
                run_keys.add((variant, seed))
            all_completed &= run.get("status") == "completed"
    else:
        all_completed = False

    expected = {
        (variant, seed)
        for variant in EXPECTED_BASELINE_VARIANTS
        for seed in EXPECTED_BASELINE_SEEDS
    }
    full_matrix = (
        matrix.get("status") == "completed"
        and matrix.get("max_batches") is None
        and matrix.get("full_matrix") is True
        and run_keys == expected
        and all_completed
        and isinstance(runs, list)
        and all(
            isinstance(run, dict)
            and isinstance(run.get("validation_evaluation"), dict)
            and run["validation_evaluation"].get("status") == "completed"
            for run in runs
        )
    )
    signed_metrics_path = repo_root / "artifacts/metrics/garl_baseline_training_v1_signed.json"
    signed_metrics = _safe_read(signed_metrics_path) if signed_metrics_path.is_file() else None
    if signed_metrics is not None:
        evidence.append(signed_metrics_path.relative_to(repo_root).as_posix())
    metrics_green = bool(
        signed_metrics
        and signed_metrics.get("artifact_type")
        in {"garl_baseline_metrics_v1", "garl_signed_metrics_v1"}
        and str(signed_metrics.get("status", "")).lower() in {"pass", "passed", "verified"}
        and signed_metrics.get("expected_run_count") == len(expected)
        and signed_metrics.get("observed_metric_count") == len(expected)
        and signed_metrics.get("missing") == []
        and signed_metrics.get("test_used_for_selection") is False
        and signed_metrics.get("evttc_used_for_selection") is False
    )
    if full_matrix and metrics_green:
        return "verified", evidence, []
    missing = []
    if not full_matrix:
        missing.append("complete 9-run official matrix with no max-batches bound")
    if not metrics_green:
        missing.append("signed validation metrics generated from the completed checkpoints")
    return "partial", evidence, missing


def _ssl_phase_state(
    repo_root: Path,
    gates: dict[str, Any],
) -> tuple[str, list[str], list[str]]:
    """Classify SSL-Pure full from terminal summaries/failures, never from a screen."""

    evidence = [
        "artifacts/runs/eap_ssl_smoke_multiwindow_current_seed7/metrics.json",
        "artifacts/runs/eap_ssl_cache_capacity_smoke_current_seed7/metrics.json",
        "artifacts/runs/eap_ssl_train40_screen_seed7_workers2/metrics.json",
        "artifacts/runs/eap_ssl_multiwindow_perf_screen_128_workers4_current_seed7/metrics.json",
    ]
    runs_root = repo_root / "artifacts/runs"
    full_metrics = sorted(runs_root.glob("eap_ssl_train40_full*/metrics.json"))
    full_failures = sorted(runs_root.glob("eap_ssl_train40_full*/FAILURE*.json"))
    full_history = sorted(runs_root.glob("eap_ssl_train40_full*/history.jsonl"))
    candidates = [*full_metrics, *full_failures, *full_history]
    for path in candidates:
        if path.is_file():
            evidence.append(path.relative_to(repo_root).as_posix())

    def is_complete(path: Path) -> bool:
        payload = _safe_read(path)
        if payload is None:
            return False
        config = payload.get("trainer_config")
        return bool(
            payload.get("pretraining_regime") == "eap_ssl"
            and int(payload.get("epochs_completed", 0)) >= 1
            and isinstance(config, dict)
            and config.get("max_train_samples") is None
            and config.get("max_validation_samples") is None
            and payload.get("train_window_count") == 16_384
            and payload.get("validation_window_count") == 4_096
        )

    complete_summary = any(is_complete(path) for path in full_metrics)
    if gates.get("ssl_full_train40_green") is True and complete_summary:
        return "verified", evidence, []
    if full_metrics or full_failures or full_history:
        return (
            "partial",
            evidence,
            [
                "terminal SSL-Pure full summary with one completed epoch and no sampled limits",
                "readiness gate ssl_full_train40_green",
            ],
        )
    return (
        "not_executed",
        evidence,
        [
            "SSL-Pure full train40 terminal summary or preserved FAILURE.json",
        ],
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _phase(
    phase: str,
    status: str,
    evidence: list[str],
    missing: list[str],
) -> dict[str, Any]:
    return {
        "phase": phase,
        "status": status,
        "claim_allowed": status == "verified",
        "evidence": evidence,
        "missing_or_blocked": missing,
    }


def build_audit(repo_root: Path, readiness_path: Path) -> dict[str, Any]:
    readiness = read_structured(readiness_path)
    gates = readiness.get("gates", {})

    def green(name: str) -> bool:
        return gates.get(name) is True

    def exists(relative: str) -> bool:
        return (repo_root / relative).is_file()

    baseline_status, baseline_evidence, baseline_missing = _baseline_phase_state(repo_root)
    ssl_status, ssl_evidence, ssl_missing = _ssl_phase_state(repo_root, gates)

    phases = [
        _phase(
            "Fase 0 — saneamiento y contratos",
            "verified" if green("ci_green") and green("artifact_contract_green") else "partial",
            [
                "artifacts/audits/recovery_v4/quality_gate.json",
                "schemas/jepa_pretrain_run_v4.schema.json",
                "artifacts/audit/recovery_v3/frozen_protocol.json",
            ],
            []
            if green("ci_green") and green("artifact_contract_green")
            else ["CI/contratos todavía no están verdes"],
        ),
        _phase(
            "Fase 1 — oracle oficial reproducido",
            "partial",
            [
                "artifacts/audits/garl_release_v1/audit.json",
                "artifacts/parity/garl_preprocessing_v2/parity.json",
                "artifacts/parity/garl_model_v1/parity.json",
                "artifacts/official/garl_release_inference_local_smoke_v1/submission_validation.json",
            ],
            ["Tabla VI completa y proximidad oficial 10.60 % no están demostradas"],
        ),
        _phase(
            "Fase 2 — cache Garl v4",
            "verified" if green("full_train40_cache_gzip_green") else "partial",
            [
                "artifacts/cache/garlttc_lhr_v4_smoke_workers4/manifest.json",
                "artifacts/cache/garlttc_lhr_v4_pilot_4096_workers4/manifest.json",
                "artifacts/audits/garlttc_rotating_cache_plan_v1/plan.json",
                "artifacts/cache/garlttc_lhr_v4_smoke_gzip_workers1/manifest.json",
                "artifacts/audits/garlttc_lhr_v4_smoke_gzip_workers1_audit.json",
                "artifacts/cache/garlttc_lhr_v4_full_train40_gzip_v5/build_state.json",
                "artifacts/cache/garlttc_lhr_v4_full_train40_gzip_v5/manifest.json",
                "artifacts/audits/garlttc_lhr_v4_full_train40_gzip_v5_audit.json",
            ],
            []
            if green("full_train40_cache_gzip_green")
            else [
                "cache full gzip no materializada o sin auditoría PASS",
                "artifacts/cache/garlttc_lhr_v4_full_train40_v1/FAILURE.json",
            ],
        ),
        _phase(
            "Fase 3 — baseline local exacto",
            baseline_status,
            baseline_evidence,
            baseline_missing,
        ),
        _phase(
            "Fase 4 — JEPA smoke sintético/eAP pequeño",
            "verified" if green("phase4_jepa_gate_green") else "partial",
            [
                "artifacts/runs/carla_jepa_overfit_32_current_seed7/metrics.json",
                "artifacts/figures/carla_jepa_overfit_32_current_v1/embedding_health.json",
            ],
            [] if green("phase4_jepa_gate_green") else ["gate JEPA de fase 4"],
        ),
        _phase(
            "Fase 5 — pretraining eAP SSL-Pure",
            ssl_status,
            ssl_evidence,
            ssl_missing,
        ),
        _phase(
            "Fase 6 — Tubelet LHR supervised",
            "partial",
            ["artifacts/runs/garl_v4_lhr_smoke_seed7_shardlocal_modern/summary.json"],
            ["comparación JEPA/scratch equivalente en protocolo full"],
        ),
        _phase(
            "Fase 7 — dense block-causal",
            "partial",
            ["artifacts/benchmarks/highres_architecture_screen_v1.json"],
            ["confirmación eAP completa y comparación de runtime"],
        ),
        _phase(
            "Fase 7B — high-resolution/KDA",
            "partial",
            [
                "artifacts/audits/patch_resolution_v1/patch_resolution_audit.json",
                "artifacts/benchmarks/highres_token_scaling_v1.json",
                "artifacts/benchmarks/highres_real_screen_v1_gpu_workers4/screen.json",
                "artifacts/benchmarks/highres_matrix_aggregate_v1.json",
            ],
            ["KDA no promovido: regression_in_short_screen; falta confirmación de candidato"],
        ),
        _phase(
            "Fase 8 — RGBE y foreground",
            "partial",
            [
                "artifacts/cache/garlttc_lhr_v4_smoke_rgb_workers1/manifest.json",
                "artifacts/runs/garl_v4_lhr_rgb_smoke_seed7/summary.json",
                "artifacts/runs/garl_v4_lhr_foreground_smoke_seed7/FAILURE.json",
            ],
            ["RGBE/foreground oficiales full; guard de máscara foreground falló como esperado"],
        ),
        _phase(
            "Fase 9 — congelación",
            "not_executed",
            ["artifacts/audit/recovery_v3/frozen_protocol.json"],
            ["no existe freeze final de checkpoint/config candidato"],
        ),
        _phase(
            "Fase 10 — zero-shot Tabla VI",
            "partial",
            [
                "artifacts/runs/garl_v4_lhr_rgb_smoke_seed7/zero_shot_validation.json",
                "artifacts/runs/garl_v4_lhr_rgb_smoke_seed7/zero_shot_aggregate.json",
                "scripts/evaluate_garl_evttc_table_vi.py",
                "scripts/predict_garl_evttc_table_vi.py",
                "scripts/table_vi_label_free.py",
                "tests/unit/test_table_vi_predict_runner.py",
                "data/protocols/garl_evttc_table_vi_v1.yaml",
            ],
            [
                "token coverage completa y checkpoint final no demostrados",
                "ejecución label-free completa sobre las tres secuencias aún no demostrada",
            ],
        ),
        _phase(
            "Fase 11 — CodaBench",
            "blocked_authorization",
            ["artifacts/official/garl_release_inference_local_smoke_v1/submission_validation.json"],
            ["envío externo prohibido sin autorización explícita"],
        ),
        _phase(
            "Fase 12 — robustez, export y report",
            "partial",
            [
                "artifacts/metrics/robustness_synthetic_smoke_current_v1.json",
                "artifacts/audits/plan_phase12_smoke_v1/phase12_smoke.json",
                "artifacts/demos/runtime_smoke_current_v1/runtime_smoke_metrics.json",
                "artifacts/tables/regenerable_report/report_manifest.json",
            ],
            ["robustez/calibración/low-label reales y export final del candidato"],
        ),
    ]
    for item in phases:
        item["evidence_exists"] = [relative for relative in item["evidence"] if exists(relative)]
        item["missing_evidence_files"] = [
            relative for relative in item["evidence"] if not exists(relative)
        ]

    return {
        "artifact_type": "plan_execution_audit_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "git_commit": _commit(repo_root),
        "plan_path": "PLAN.md",
        "readiness_path": readiness_path.relative_to(repo_root).as_posix(),
        "readiness_sha256": _sha256(readiness_path),
        "plan_declared_complete": False,
        "phases": phases,
        "summary": {
            "verified": sum(item["status"] == "verified" for item in phases),
            "partial": sum(item["status"] == "partial" for item in phases),
            "blocked_authorization": sum(
                item["status"] == "blocked_authorization" for item in phases
            ),
            "not_executed": sum(item["status"] == "not_executed" for item in phases),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--readiness",
        type=Path,
        default=Path("artifacts/audits/recovery_v4/readiness.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/audits/plan_execution/PLAN_AUDIT.json"),
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    readiness = (repo_root / args.readiness).resolve()
    output = (repo_root / args.output).resolve()
    audit = build_audit(repo_root, readiness)
    write_structured(output, audit)
    print(audit["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
