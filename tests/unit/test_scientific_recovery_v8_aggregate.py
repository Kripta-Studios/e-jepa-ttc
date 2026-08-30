from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash
from e_jepa_ttc.evaluation.scientific_recovery_v8_aggregate import (
    V8AggregateIntegrityError,
    aggregate_seed7,
    contract_hashes,
)


def _rows(prediction: float) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for fold, (sequence, track, target) in enumerate(
        (("s0", "t0", "2"), ("s1", "t1", "4"), ("s2", "t2", "7"))
    ):
        result.append(
            {
                "token_id": f"{sequence}_0",
                "sequence_id": sequence,
                "track_id": track,
                "outer_fold": str(fold),
                "seed": "7",
                "target_ttc": target,
                "sample_weight": "1",
                "prediction_ttc": str(prediction),
                "prediction_log_variance": "0",
                "finite": "True",
                "failure_reason": "",
                "event_count": "1",
                "event_rate": "1",
                "support_ms": "1",
                "model_name": "timevol20_3",
                "config_sha256": "a" * 64,
                "checkpoint_sha256": hashlib.sha256(b"x").hexdigest(),
            }
        )
    return result


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fixture(
    tmp_path: Path, *, duplicate: bool = False
) -> tuple[dict[str, object], dict[str, object], Path]:
    candidate = _rows(3.0)
    if duplicate:
        candidate.append(dict(candidate[0]))
    baseline = _rows(5.0)
    hashes = contract_hashes(candidate)
    protocol = {
        "artifact_sha256": "p" * 64,
        "git_base_commit": "base",
        "sample_contract": {
            "rows": len(candidate),
            "sequences": 3,
            **hashes,
            "row_count_contract": {
                "total": len(candidate),
                "by_outer_fold": {str(i): 1 for i in range(3)},
                "by_sequence": {f"s{i}": 1 for i in range(3)},
                "by_bucket": {"crucial": 1, "small": 1, "large": 1, "negative": 0},
            },
            "fold_definitions": [{"fold": i, "dev_sequence_ids": [f"s{i}"]} for i in range(3)],
        },
        "closed_evaluation": {
            "public_validation_used_for_selection": False,
            "private_test_opened": False,
            "evttc_test_opened": False,
            "codabench_opened": False,
        },
        "gates": {
            "ttc_candidate_gate": {
                "delta_MiD_max": -3.0,
                "bootstrap_probability_delta_below_zero_min": 0.9,
                "finite_fraction_required": 1.0,
                "failure_rate_required": 0.0,
                "coverage_drop_max_pp": 1.0,
            }
        },
        "training_contract": {"screen_seed": 7},
    }
    manifest = {
        "artifact_sha256": "m" * 64,
        "enabled_seed7_configs": {
            f"timevol20_3_fold{i}_seed7": {"sha256": "a" * 64} for i in range(3)
        },
        "model_configs": {},
        "closed_evaluation": protocol["closed_evaluation"],
    }
    root = tmp_path / "results"
    _write_csv(tmp_path / "a5.csv", baseline)
    protocol["sources"] = {"a5_oof_predictions": {"path": str(tmp_path / "a5.csv")}}
    for fold in range(3):
        run = root / "runs" / f"timevol20_3_fold{fold}_seed7"
        prediction_path = run / "dev_predictions.csv"
        _write_csv(prediction_path, [candidate[fold]])
        checkpoint = run / "model_best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"x")
        summary = sign_artifact(
            {
                "artifact_type": "scientific_recovery_v8_fold_result_v1",
                "status": "completed",
                "arm": "timevol20_3",
                "outer_fold": fold,
                "seed": 7,
                "config_sha256": "a" * 64,
                "checkpoint": {"path": "model_best.pt", "sha256": hashlib.sha256(b"x").hexdigest()},
                "dev_predictions": {"path": "dev_predictions.csv"},
                "protocol_sha256": "p" * 64,
                "frozen_manifest_sha256": "m" * 64,
                "fixture": True,
                "closed_evaluation": protocol["closed_evaluation"],
            }
        )
        (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return protocol, manifest, root


def test_fixture_smoke_is_signed_but_cannot_nominate(tmp_path: Path) -> None:
    protocol, manifest, root = _fixture(tmp_path)
    report = aggregate_seed7(
        protocol=protocol,
        manifest=manifest,
        results_root=root,
        repository_root=tmp_path,
        allow_fixture=True,
        resamples=25,
    )
    assert verify_artifact_hash(report)
    assert report["fixture"] is True
    assert report["multiseed_replication_candidate"] is False


def test_protocol_sealed_authorization_name_is_not_an_opened_source(tmp_path: Path) -> None:
    protocol, manifest, root = _fixture(tmp_path)
    protocol["external_confirmation"] = {
        "name": "single_user_authorized_sealed_public_validation",
        "requires_user_authorization": True,
        "selection_or_tuning_forbidden": True,
    }
    report = aggregate_seed7(
        protocol=protocol,
        manifest=manifest,
        results_root=root,
        repository_root=tmp_path,
        allow_fixture=True,
        resamples=25,
    )
    assert verify_artifact_hash(report)
    assert report["fixture"] is True


def test_opened_public_validation_flag_still_fails_closed(tmp_path: Path) -> None:
    protocol, manifest, root = _fixture(tmp_path)
    protocol["closed_evaluation"]["public_validation_used_for_selection"] = True
    with pytest.raises(V8AggregateIntegrityError, match="sealed evaluation flag is not false"):
        aggregate_seed7(
            protocol=protocol,
            manifest=manifest,
            results_root=root,
            repository_root=tmp_path,
            allow_fixture=True,
            resamples=5,
        )


def test_documented_unknown_support_empty_prediction_does_not_abort(tmp_path: Path) -> None:
    protocol, manifest, root = _fixture(tmp_path)
    path = root / "runs" / "timevol20_3_fold0_seed7" / "dev_predictions.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows[0]["prediction_ttc"] = ""
    rows[0]["prediction_log_variance"] = ""
    rows[0]["finite"] = "False"
    rows[0]["failure_reason"] = "no_known_causal_support"
    _write_csv(path, rows)
    report = aggregate_seed7(
        protocol=protocol,
        manifest=manifest,
        results_root=root,
        repository_root=tmp_path,
        allow_fixture=True,
        resamples=25,
    )
    assert verify_artifact_hash(report)
    candidate = report["candidate_results"][0]
    assert candidate["metrics"]["finite_fraction"] < 1.0
    assert candidate["passed"] is False
    assert report["candidate_id"] == "A5"


def test_empty_prediction_without_failure_reason_still_fails_closed(tmp_path: Path) -> None:
    protocol, manifest, root = _fixture(tmp_path)
    path = root / "runs" / "timevol20_3_fold0_seed7" / "dev_predictions.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows[0]["prediction_ttc"] = ""
    rows[0]["finite"] = "False"
    rows[0]["failure_reason"] = ""
    _write_csv(path, rows)
    with pytest.raises(V8AggregateIntegrityError, match="missing prediction_ttc"):
        aggregate_seed7(
            protocol=protocol,
            manifest=manifest,
            results_root=root,
            repository_root=tmp_path,
            allow_fixture=True,
            resamples=5,
        )


def test_duplicate_tokens_fail_closed(tmp_path: Path) -> None:
    protocol, manifest, root = _fixture(tmp_path)
    path = root / "runs" / "timevol20_3_fold0_seed7" / "dev_predictions.csv"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(path.read_text(encoding="utf-8").splitlines()[1] + "\n")
    with pytest.raises(V8AggregateIntegrityError, match="duplicate"):
        aggregate_seed7(
            protocol=protocol,
            manifest=manifest,
            results_root=root,
            repository_root=tmp_path,
            allow_fixture=True,
            resamples=5,
        )


def test_identical_signed_seed7_aggregate_reuses_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from e_jepa_ttc.evaluation import scientific_recovery_v8_aggregate as module

    protocol, manifest, root = _fixture(tmp_path)
    calls = {"n": 0}
    original = module._bootstrap

    def counted(*args: object, **kwargs: object) -> dict[str, object]:
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_bootstrap", counted)
    first = module.aggregate_seed7(
        protocol=protocol,
        manifest=manifest,
        results_root=root,
        repository_root=tmp_path,
        allow_fixture=True,
        resamples=25,
    )
    output = tmp_path / "aggregate_seed7.json"
    output.write_text(json.dumps(first, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert calls["n"] == 1
    reused = module.aggregate_seed7(
        protocol=protocol,
        manifest=manifest,
        results_root=root,
        repository_root=tmp_path,
        allow_fixture=True,
        resamples=25,
        existing_output=output,
    )
    assert reused["artifact_sha256"] == first["artifact_sha256"]
    assert calls["n"] == 1


def test_changed_prediction_hash_does_not_reuse_bootstrap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from e_jepa_ttc.evaluation import scientific_recovery_v8_aggregate as module

    protocol, manifest, root = _fixture(tmp_path)
    first = module.aggregate_seed7(
        protocol=protocol,
        manifest=manifest,
        results_root=root,
        repository_root=tmp_path,
        allow_fixture=True,
        resamples=25,
    )
    output = tmp_path / "aggregate_seed7.json"
    output.write_text(json.dumps(first, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    path = root / "runs" / "timevol20_3_fold0_seed7" / "dev_predictions.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    rows[0]["prediction_ttc"] = "9.0"
    _write_csv(path, rows)
    summary_path = root / "runs" / "timevol20_3_fold0_seed7" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["dev_predictions"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary.pop("artifact_sha256", None)
    sign_artifact(summary)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    calls = {"n": 0}
    original = module._bootstrap

    def counted(*args: object, **kwargs: object) -> dict[str, object]:
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_bootstrap", counted)
    second = module.aggregate_seed7(
        protocol=protocol,
        manifest=manifest,
        results_root=root,
        repository_root=tmp_path,
        allow_fixture=True,
        resamples=25,
        existing_output=output,
    )
    assert calls["n"] == 1
    assert second["artifact_sha256"] != first["artifact_sha256"]
