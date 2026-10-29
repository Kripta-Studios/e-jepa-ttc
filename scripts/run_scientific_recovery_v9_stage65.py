"""Execute the deterministic strict-nested Stage 65 full-regret fallback on CPU."""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
import yaml

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from e_jepa_ttc.artifacts.campaign_session import CampaignLock, append_transition  # noqa: E402
from e_jepa_ttc.artifacts.hashing import compute_file_hash, verify_artifact_hash  # noqa: E402
from e_jepa_ttc.artifacts.stage63_65 import sign_stage63_65_artifact  # noqa: E402
from e_jepa_ttc.artifacts.training_authorization import (  # noqa: E402
    bind_output_files,
    read_signed,
    validate_training_authorization,
    verify_output_bindings,
)
from e_jepa_ttc.data.nested_provenance import (  # noqa: E402
    validate_exact_oof_union,
    validate_frozen_teacher_dependency,
    validate_nested_split,
    validate_uninitialized_producer_contract,
)
from e_jepa_ttc.data.stage61_pair_feature_cache import load_feature_cache  # noqa: E402
from e_jepa_ttc.evaluation.nested_router import (  # noqa: E402
    bind_expert_oof_to_trainer_point_ttc,
)
from e_jepa_ttc.evaluation.stage61_nested_pair_router import (  # noqa: E402
    build_router_features,
)
from e_jepa_ttc.evaluation.stage63_65 import (  # noqa: E402
    evaluate_gate,
    paired_hierarchical_bootstrap,
    scientific_mid_per_row,
    strict_macro_mass,
    strict_score,
    validate_campaign_universe,
)
from e_jepa_ttc.models.full_regret_router import FullRegretRidge  # noqa: E402
from e_jepa_ttc.models.three_expert_router import (  # noqa: E402
    BASE8_FEATURES,
    PHASE17_FEATURES,
)
from e_jepa_ttc.training.campaign_budget import (  # noqa: E402
    CampaignBudget,
    check_resource_margins,
)
from e_jepa_ttc.training.incremental_residual import (  # noqa: E402
    deterministic_sequence_grouped_schedule,
)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha(path: Path) -> str:
    return compute_file_hash(str(path))


def _load_experts(a5_path: Path, c2f_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    a5 = (
        pd.read_csv(a5_path, dtype={"token_id": str}).sort_values("token_id").reset_index(drop=True)
    )
    c2f = (
        pd.read_csv(c2f_path, dtype={"token_id": str})
        .sort_values("token_id")
        .reset_index(drop=True)
    )
    if a5_path.name == "expert_oof.csv":
        a5 = _bound_routing_points(a5, a5_path.parent)[0]
    if c2f_path.name == "expert_oof.csv":
        c2f = _bound_routing_points(c2f, c2f_path.parent)[0]
    identity = ["token_id", "sequence_id", "track_id", "target_ttc", "sample_weight"]
    if len(a5) == 0 or len(a5) != len(c2f):
        raise ValueError("A5/C2F nested tables have different row counts")
    for column in identity[:3]:
        if not a5[column].astype(str).equals(c2f[column].astype(str)):
            raise ValueError(f"A5/C2F nested identity mismatch: {column}")
    if not np.array_equal(
        a5[["target_ttc", "sample_weight"]].to_numpy(np.float64),
        c2f[["target_ttc", "sample_weight"]].to_numpy(np.float64),
    ):
        raise ValueError("A5/C2F target or weight mismatch")
    for frame in (a5, c2f):
        if not np.isfinite(frame["prediction_ttc"].to_numpy(np.float64)).all():
            raise ValueError("nested expert contains a non-finite point prediction")
    return a5, c2f


def _bound_routing_points(frame: pd.DataFrame, root: Path) -> tuple[pd.DataFrame, list[Path]]:
    """Use the producer's own signed point output, never another router's winner."""
    artifact = read_signed(root / "expert_artifact.json")
    summary_path = root / "train/summary.json"
    summary = read_signed(summary_path)
    if summary["checkpoint"]["sha256"] != artifact["checkpoint"]["sha256"]:
        raise ValueError("routing point producer differs from nested expert")
    point_path = (root / "train" / summary["predictions"]["path"]).resolve()
    if not point_path.is_relative_to((root / "train").resolve()):
        raise ValueError("routing point prediction path escapes producer")
    if _sha(point_path) != summary["predictions"]["sha256"]:
        raise ValueError("routing point prediction SHA mismatch")
    return bind_expert_oof_to_trainer_point_ttc(frame, pd.read_csv(point_path)), [
        summary_path,
        point_path,
    ]


def _load_pair(path: Path, tokens: pd.Series) -> pd.DataFrame:
    pair = pd.read_csv(path, dtype={"token_id": str}).set_index("token_id")
    if pair.index.duplicated().any() or set(tokens.astype(str)) != set(pair.index.astype(str)):
        raise ValueError("PAIR nested table token identity mismatch")
    return pair.loc[tokens.astype(str)].reset_index()


def _expert_matrix(a5: pd.DataFrame, c2f: pd.DataFrame, pair: pd.DataFrame) -> np.ndarray:
    value = np.column_stack(
        (
            a5["prediction_ttc"].to_numpy(np.float64),
            c2f["prediction_ttc"].to_numpy(np.float64),
            pair["prediction_ttc"].to_numpy(np.float64),
        )
    )
    if not np.isfinite(value).all():
        raise ValueError("expert prediction matrix is non-finite")
    return value


def _loss_matrix(target: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [scientific_mid_per_row(target, predictions[:, index]) for index in range(3)]
    )


def _save_fit(
    path: Path, fit: FullRegretRidge, *, feature_order: tuple[str, ...]
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=fit.mean,
        scale=fit.scale,
        coefficient=fit.coefficient,
        intercept=fit.intercept,
        ridge=np.asarray(fit.ridge),
        feature_order=np.asarray(feature_order),
    )
    return {"path": str(path), "sha256": _sha(path), "feature_order": list(feature_order)}


def _load_fit(path: Path) -> FullRegretRidge:
    with np.load(path, allow_pickle=False) as value:
        ridge = float(value["ridge"])
        return FullRegretRidge(
            value["mean"], value["scale"], value["coefficient"], value["intercept"], ridge
        )


def _verify_fit_record(record: dict[str, Any], output: Path, lock_sha: str) -> None:
    outer, name = record["outer_fold"], record["model"]
    if outer not in (0, 1, 2) or name not in ("S65-RISK8", "S65-RISK17"):
        raise ValueError("unknown Stage65 fit identity")
    path = output / f"outer{outer}/{name}.npz"
    if (
        record["training_lock_sha256"] != lock_sha
        or Path(record["path"]).resolve() != path.resolve()
    ):
        raise ValueError("Stage65 fit lock/path identity mismatch")
    order = BASE8_FEATURES if name == "S65-RISK8" else PHASE17_FEATURES
    if record["feature_order"] != list(order) or _sha(path) != record["sha256"]:
        raise ValueError("Stage65 frozen fit identity mismatch")
    fit = _load_fit(path)
    if (
        fit.ridge != 0.01
        or fit.mean.shape != (len(order),)
        or fit.scale.shape != fit.mean.shape
        or fit.coefficient.shape != (len(order), 2)
        or fit.intercept.shape != (2,)
        or np.any(fit.scale <= 0)
        or any(
            not np.isfinite(array).all()
            for array in (fit.mean, fit.scale, fit.coefficient, fit.intercept)
        )
    ):
        raise ValueError("Stage65 frozen fit is numerically invalid")


def _prediction_frame(
    metadata: pd.DataFrame, prediction: np.ndarray, selected: np.ndarray
) -> pd.DataFrame:
    result = pd.DataFrame(
        {
            "sample_token": metadata["token_id"].astype(str),
            "sequence_id": metadata["sequence_id"].astype(str),
            "track_id": metadata["track_id"].astype(str),
            "target_ttc_s": metadata["target_ttc"].to_numpy(np.float64),
            "prediction_ttc_s": np.asarray(prediction, dtype=np.float64),
            "selected_expert": np.asarray(selected, dtype=np.int64),
        }
    )
    result["scientific_mid_per_row"] = scientific_mid_per_row(
        result["target_ttc_s"].to_numpy(), result["prediction_ttc_s"].to_numpy()
    )
    result["failure"] = False
    return result


def _validate_nested_sources(
    reference_router: Path, stage61_root: Path, frozen_teacher_audit_path: Path
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    dependencies = {frozen_teacher_audit_path.resolve()}
    canonical = pd.read_csv(stage61_root.parent / "feature_cache/outer0_final.metadata.csv")
    teacher_audit = read_signed(frozen_teacher_audit_path)
    dependencies.add(Path(teacher_audit["teacher_manifest_path"]).resolve())
    if (
        _sha(Path(teacher_audit["teacher_manifest_path"]))
        != teacher_audit["teacher_manifest_sha256"]
    ):
        raise ValueError("Stage65 teacher dependency changed since its audit")
    for outer in range(3):
        for expert in ("a5", "c2f"):
            parts: list[pd.DataFrame] = []
            for role in ("inner0", "inner1", "inner2", "outer_dev"):
                root = reference_router / f"outer_fold{outer}_seed7" / expert / role
                artifact_path = root / "expert_artifact.json"
                protocol_path = root / "nested_protocol.json"
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
                if not verify_artifact_hash(artifact) or not verify_artifact_hash(protocol):
                    raise ValueError(f"nested {expert} signature mismatch: {root}")
                checkpoint = root / "train" / "model_best.pt"
                if _sha(checkpoint) != artifact["checkpoint"]["sha256"]:
                    raise ValueError(f"nested {expert} checkpoint mismatch: {root}")
                predictions_path = root / "expert_oof.csv"
                if (
                    _sha(predictions_path) != artifact["oof_csv"]["sha256"]
                    or _sha(protocol_path) != artifact["nested_contract"]["sha256"]
                ):
                    raise ValueError(f"nested {expert} output/protocol identity mismatch")
                frame = pd.read_csv(predictions_path)
                relationship = validate_nested_split(
                    protocol,
                    artifact,
                    canonical,
                    frame,
                    outer_fold=outer,
                    inner_fold=None if role == "outer_dev" else int(role[-1]),
                )
                effective = root / "effective_config.yaml"
                if set(frame.config_sha256) != {_sha(effective)}:
                    raise ValueError("nested producer configuration is not bound by output rows")
                initialization = validate_uninitialized_producer_contract(
                    torch.load(checkpoint, map_location="cpu", weights_only=False),
                    yaml.safe_load(effective.read_text(encoding="utf-8")),
                    protocol,
                )
                validate_frozen_teacher_dependency(initialization, teacher_audit)
                relationship["ancestry_validated"] = True
                # The historical merger explicitly bound routing TTC to the
                # same producer's signed trainer point output before CSV export.
                frame, point_sources = _bound_routing_points(frame, root)
                summary_path, point_path = point_sources
                dependencies.update(
                    path.resolve()
                    for path in (
                        artifact_path,
                        protocol_path,
                        checkpoint,
                        predictions_path,
                        effective,
                        *point_sources,
                    )
                )
                relationship["routing_point_summary_sha256"] = _sha(summary_path)
                relationship["routing_point_csv_sha256"] = _sha(point_path)
                if role != "outer_dev":
                    parts.append(frame)
                records.append(
                    {
                        "outer_fold": outer,
                        "expert": expert.upper(),
                        "role": role,
                        "checkpoint_sha256": artifact["checkpoint"]["sha256"],
                        "protocol_sha256": _sha(protocol_path),
                        "split_validation": relationship,
                        "initialization_validation": initialization,
                    }
                )
            aggregate_path = (
                reference_router / f"outer_fold{outer}_seed7" / expert / "inner_oof.csv"
            )
            # Reproduce the historical concat -> CSV -> parse operation exactly,
            # not a floating-point tolerance that could hide changed predictions.
            serialized_union = pd.read_csv(
                io.StringIO(
                    pd.concat(parts, ignore_index=True)
                    .sort_values("token_id", kind="stable")
                    .to_csv(index=False, lineterminator="\n")
                )
            )
            validate_exact_oof_union(pd.read_csv(aggregate_path), [serialized_union])
            dependencies.add(aggregate_path.resolve())
        pair_parts = []
        for role in ("inner0", "inner1", "inner2", "outer_dev"):
            if role == "outer_dev":
                path = stage61_root / f"outer_fold{outer}_seed7" / "pair" / "outer_dev_oof.csv"
                checkpoint_column = "checkpoint_sha256"
            else:
                path = stage61_root / f"outer_fold{outer}_seed7" / "pair" / role / "expert_oof.csv"
                checkpoint_column = "checkpoint_sha256"
            frame = pd.read_csv(path)
            if role != "outer_dev":
                pair_parts.append(frame)
            values = np.unique(np.asarray(frame[checkpoint_column]).astype(str))
            if (
                len(values) != 1
                or not Path(
                    stage61_root
                    / f"outer_fold{outer}_seed7"
                    / "pair"
                    / role
                    / "train"
                    / "model_last.pt"
                ).is_file()
            ):
                raise ValueError(f"PAIR producer identity is incomplete: {path}")
            checkpoint = (
                stage61_root
                / f"outer_fold{outer}_seed7"
                / "pair"
                / role
                / "train"
                / "model_last.pt"
            )
            if _sha(checkpoint) != values[0]:
                raise ValueError(f"PAIR producer SHA mismatch: {checkpoint}")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            identity = payload["identity"]
            inner = None if role == "outer_dev" else int(role[-1])
            expected_role = "outer_dev" if inner is None else "inner_oof"
            if (
                identity["outer_fold"] != outer
                or identity.get("inner_fold") != inner
                or identity["role"] != expected_role
            ):
                raise ValueError("PAIR checkpoint producer fold/role mismatch")
            feature_name = f"outer{outer}_final" if inner is None else f"outer{outer}_inner{inner}"
            _, metadata, feature_manifest = load_feature_cache(
                stage61_root.parent / "feature_cache" / f"{feature_name}.npz"
            )
            dependencies.update(
                path.resolve()
                for path in (
                    path,
                    checkpoint,
                    *(
                        stage61_root.parent / "feature_cache" / f"{feature_name}{suffix}"
                        for suffix in (".npz", ".manifest.json", ".metadata.csv")
                    ),
                )
            )
            if feature_manifest["artifact_sha256"] != identity["feature_cache_artifact_sha256"]:
                raise ValueError("PAIR ancestor feature cache differs from checkpoint identity")
            ancestor = reference_router / f"outer_fold{outer}_seed7/a5/{role}"
            ancestor_artifact = json.loads((ancestor / "expert_artifact.json").read_text())
            nested = json.loads((ancestor / "nested_protocol.json").read_text())
            if (
                feature_manifest["identity"]["a5_checkpoint_sha256"]
                != ancestor_artifact["checkpoint"]["sha256"]
            ):
                raise ValueError("PAIR features do not come from the same nested A5 producer")
            split = nested["folds"][0]
            train_meta = metadata.loc[metadata.sequence_id.isin(split["train_sequence_ids"])]
            dev_meta = metadata.loc[metadata.sequence_id.isin(split["dev_sequence_ids"])]
            if (
                set(frame.token_id) != set(dev_meta.sample_token)
                or frame.token_id.duplicated().any()
            ):
                raise ValueError("PAIR predictions differ from the nested dev token universe")
            paired = frame.set_index("token_id").loc[dev_meta.sample_token]
            for field in ("sequence_id", "track_id", "outer_fold"):
                if paired[field].astype(str).tolist() != dev_meta[field].astype(str).tolist():
                    raise ValueError(f"PAIR row identity differs from canonical {field}")
            config = identity["config"]
            _, schedule_sha = deterministic_sequence_grouped_schedule(
                train_meta.sequence_id.astype(str).tolist(),
                train_meta.sample_token.astype(str).tolist(),
                seed=config["seed"],
                batch_size=config["batch_size"],
                updates=config["update_budget"],
            )
            if schedule_sha != identity["batch_schedule_sha256"]:
                raise ValueError("PAIR training schedule does not bind the excluded-dev token set")
            if payload["completed_updates"] != config["update_budget"]:
                raise ValueError("PAIR checkpoint did not reach its fixed endpoint")
            records.append(
                {
                    "outer_fold": outer,
                    "expert": "PAIR",
                    "role": role,
                    "checkpoint_sha256": values[0],
                    "protocol_sha256": None,
                    "nested_a5_ancestor_sha256": ancestor_artifact["checkpoint"]["sha256"],
                    "train_token_schedule_sha256": schedule_sha,
                    "exact_dev_universe_validated": True,
                }
            )
        validate_exact_oof_union(
            pd.read_csv(stage61_root / f"outer_fold{outer}_seed7/pair/inner_oof.csv"), pair_parts
        )
        dependencies.add((stage61_root / f"outer_fold{outer}_seed7/pair/inner_oof.csv").resolve())
        dependencies.update(
            (stage61_root / f"outer_fold{outer}_seed7/router" / name).resolve()
            for name in ("R2_signature.json", "R2_outer_oof.csv")
        )
    if len(records) != 36:
        raise AssertionError("Stage 65 requires 12 A5, 12 C2F and 12 PAIR identities")
    return {
        "status": "passed",
        "producer_count": len(records),
        "producers": records,
        "input_bindings": {
            str(path): {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha(path)}
            for path in sorted(dependencies)
        },
        "aggregate_validation": "exact_historical_point_bind_and_csv_serialization",
    }


def _ce17_replay(
    *, signature_path: Path, features: pd.DataFrame, predictions: np.ndarray, expected_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    signature = json.loads(signature_path.read_text(encoding="utf-8"))
    if not verify_artifact_hash(signature) or signature["feature_order"] != list(PHASE17_FEATURES):
        raise ValueError("CE17 replay signature is invalid")
    x = features.to_numpy(np.float64)
    z = (x - np.asarray(signature["scaler_mean"])) / np.asarray(signature["scaler_scale"])
    logits = z @ np.asarray(signature["coef"]).T + np.asarray(signature["intercept"])
    selected = np.argmax(logits, axis=1)
    prediction = predictions[np.arange(len(predictions)), selected]
    expected = pd.read_csv(expected_path, dtype={"sample_token": str}).sort_values("sample_token")
    order = np.argsort(features.index.to_numpy())
    del order  # Feature rows are already aligned to sorted expert tables.
    expected_prediction = expected["prediction_ttc_s"].to_numpy(np.float64)
    if len(expected_prediction) != len(prediction) or not np.allclose(
        expected_prediction, prediction, rtol=0, atol=1e-12
    ):
        raise ValueError("CE17 replay does not reproduce the frozen R2 outer predictions")
    return prediction, selected


def run(args: argparse.Namespace) -> dict[str, Any]:
    torch.set_num_threads(8)
    validate_training_authorization(
        args.output_root,
        _REPOSITORY_ROOT,
        stage=65,
        invocation={
            "stage61_worktree": str(args.stage61_worktree.resolve()),
            "output_root": str(args.output_root.resolve()),
            "reference_root": str(args.reference_root.resolve()),
            "frozen_teacher_audit": str(args.frozen_teacher_audit.resolve()),
        },
    )
    if (Path(__file__).resolve().parents[1] / "CAMPAIGN_INTEGRITY_HOLD.json").exists():
        raise RuntimeError("Campaign integrity hold: Stage 65 fitting is disabled")
    began = time.perf_counter()
    if args.max_hours != 1.0:
        raise ValueError("Stage65 must use the one-hour CPU budget")
    budget = CampaignBudget(args.output_root / "budgets/stage65.json", hours=1.0)

    def resource_check() -> None:
        budget.check()
        check_resource_margins(args.output_root)

    output = args.output_root / "stage65"
    result_path = output / "STAGE65_RESULT.json"
    if result_path.is_file():
        if not args.resume:
            raise FileExistsError("Stage 65 result exists; --resume is required")
        prior = read_signed(result_path)
        if prior.get("training_lock_sha256") != _sha(args.output_root / "TRAINING_LOCK.json"):
            raise ValueError("Stage65 result belongs to a different training lock")
        verify_output_bindings(output, prior)
        return prior
    resource_check()
    output.mkdir(parents=True, exist_ok=args.resume)
    reference_router = (
        args.reference_root / "artifacts" / "scientific_recovery_v8" / "results" / "router"
    )
    stage61_root = (
        args.stage61_worktree / "artifacts" / "scientific_recovery_v9_stage61_stage62" / "stage61"
    )
    provenance = _validate_nested_sources(reference_router, stage61_root, args.frozen_teacher_audit)
    _atomic_json(output / "STAGE65_SOURCE_PROVENANCE.json", provenance)
    fit_records: list[dict[str, Any]] = []
    freeze_path = output / "ALL_RIDGE_FITS_FROZEN.json"
    lock_sha = _sha(args.output_root / "TRAINING_LOCK.json")
    if freeze_path.is_file():
        frozen = read_signed(freeze_path)
        if frozen["training_lock_sha256"] != lock_sha or frozen["status"] != "frozen":
            raise ValueError("Stage65 frozen endpoint belongs to a different lock")
        fit_records = frozen["fits"]
        expected = {(outer, name) for outer in range(3) for name in ("S65-RISK8", "S65-RISK17")}
        if (
            len(fit_records) != 6
            or {(r["outer_fold"], r["model"]) for r in fit_records} != expected
        ):
            raise ValueError("Stage65 frozen endpoint fit inventory is incomplete")
        for record in fit_records:
            _verify_fit_record(record, output, lock_sha)
    # Fit every outer model before opening any corresponding outer-dev score.
    for outer in () if freeze_path.is_file() else range(3):
        append_transition(
            args.output_root / "RUN_LEDGER.jsonl", "stage65_outer_fit_running", outer_fold=outer
        )
        resource_check()
        a5, c2f = _load_experts(
            reference_router / f"outer_fold{outer}_seed7" / "a5" / "inner_oof.csv",
            reference_router / f"outer_fold{outer}_seed7" / "c2f" / "inner_oof.csv",
        )
        pair = _load_pair(
            stage61_root / f"outer_fold{outer}_seed7" / "pair" / "inner_oof.csv",
            cast(pd.Series, a5["token_id"]),
        )
        predictions = _expert_matrix(a5, c2f, pair)
        target = a5["target_ttc"].to_numpy(np.float64)
        losses = _loss_matrix(target, predictions)
        mass = strict_macro_mass(target, a5["sequence_id"].astype(str).to_numpy())
        base8, phase17 = build_router_features(a5, c2f, predictions[:, 2])
        for name, features, order in (
            ("S65-RISK8", base8, BASE8_FEATURES),
            ("S65-RISK17", phase17, PHASE17_FEATURES),
        ):
            fit_path = output / f"outer{outer}" / f"{name}.npz"
            receipt = fit_path.with_suffix(".fit.json")
            if fit_path.exists() or receipt.exists():
                record = read_signed(receipt)
            else:
                fit = FullRegretRidge.fit(features.to_numpy(np.float64), losses, mass)
                resource_check()
                record = sign_stage63_65_artifact(
                    {
                        "outer_fold": outer,
                        "model": name,
                        "training_lock_sha256": lock_sha,
                        **_save_fit(fit_path, fit, feature_order=order),
                    },
                    evidence_type="train_only_frozen_router_fit",
                )
                _atomic_json(receipt, record)
            _verify_fit_record(record, output, lock_sha)
            fit_records.append(record)
    if not freeze_path.is_file():
        _atomic_json(
            freeze_path,
            sign_stage63_65_artifact(
                {
                    "status": "frozen",
                    "fits": fit_records,
                    "training_lock_sha256": lock_sha,
                },
                evidence_type="all_router_fits_frozen_before_evaluation",
            ),
        )
    arms: dict[str, list[pd.DataFrame]] = {
        "S65-RISK8": [],
        "S65-RISK17": [],
        "S65-CE17-REPLAY": [],
    }
    for outer in range(3):
        resource_check()
        append_transition(
            args.output_root / "RUN_LEDGER.jsonl",
            "stage65_outer_evaluating",
            outer_fold=outer,
            all_fits_frozen=True,
        )
        a5, c2f = _load_experts(
            reference_router / f"outer_fold{outer}_seed7" / "a5" / "outer_dev" / "expert_oof.csv",
            reference_router / f"outer_fold{outer}_seed7" / "c2f" / "outer_dev" / "expert_oof.csv",
        )
        pair = _load_pair(
            stage61_root / f"outer_fold{outer}_seed7" / "pair" / "outer_dev_oof.csv",
            cast(pd.Series, a5["token_id"]),
        )
        predictions = _expert_matrix(a5, c2f, pair)
        base8, phase17 = build_router_features(a5, c2f, predictions[:, 2])
        for name, features in (("S65-RISK8", base8), ("S65-RISK17", phase17)):
            fit = _load_fit(output / f"outer{outer}" / f"{name}.npz")
            selected = fit.select(features.to_numpy(np.float64))
            chosen = predictions[np.arange(len(predictions)), selected]
            frame = _prediction_frame(a5, chosen, selected)
            frame["outer_fold"] = outer
            arms[name].append(frame)
        ce_prediction, ce_selected = _ce17_replay(
            signature_path=stage61_root
            / f"outer_fold{outer}_seed7"
            / "router"
            / "R2_signature.json",
            features=phase17,
            predictions=predictions,
            expected_path=stage61_root / f"outer_fold{outer}_seed7" / "router" / "R2_outer_oof.csv",
        )
        ce_frame = _prediction_frame(a5, ce_prediction, ce_selected)
        ce_frame["outer_fold"] = outer
        arms["S65-CE17-REPLAY"].append(ce_frame)
    aggregate: dict[str, pd.DataFrame] = {}
    canonical = pd.read_csv(stage61_root / "aggregate_seed7/R2_oof.csv")
    for name, parts in arms.items():
        frame = pd.concat(parts).sort_values("sample_token").reset_index(drop=True)
        if len(frame) != 8192:
            raise ValueError(f"Stage 65 {name} does not cover 8192 tokens")
        validate_campaign_universe(frame, canonical)
        frame.to_csv(output / f"{name}_oof.csv", index=False, lineterminator="\n")
        aggregate[name] = frame
    router_r = pd.read_csv(
        stage61_root / "aggregate_seed7" / "RouterR_oof.csv", dtype={"sample_token": str}
    )
    if "outer_fold" not in router_r:
        router_r = router_r.merge(
            canonical[["sample_token", "outer_fold"]], on="sample_token", validate="one_to_one"
        )
    validate_campaign_universe(router_r, canonical)
    references = {
        "S61-R2": aggregate["S65-CE17-REPLAY"],
        "S65-RISK8": aggregate["S65-RISK8"],
        "RouterR": router_r,
    }
    thresholds = {
        "S61-R2": -1.0,
        "S65-RISK8": -1.0,
        "RouterR": -3.0,
    }
    comparisons: dict[str, Any] = {}
    hashes: set[str] = set()
    for name, reference in references.items():
        bootstrap = paired_hierarchical_bootstrap(
            aggregate["S65-RISK17"], reference, resource_check=resource_check
        )
        hashes.add(bootstrap.draws_sha256)
        comparisons[name] = evaluate_gate(bootstrap, point_delta_lte=thresholds[name])
    if len(hashes) != 1:
        raise AssertionError("Stage 65 comparisons did not reuse bootstrap draws")
    gates_passed = all(value["passed"] for value in comparisons.values())
    diagnostics: dict[str, Any] = {}
    for name, frame in aggregate.items():
        diagnostics[name] = {
            "score": strict_score(frame),
            "selection_fraction": frame["selected_expert"].value_counts(normalize=True).to_dict(),
        }
    result = {
        "artifact_type": "scientific_recovery_v9_stage65_full_regret_result_v1",
        "training_lock_sha256": _sha(args.output_root / "TRAINING_LOCK.json"),
        "status": "RISK_ROUTER_DEV_CANDIDATE" if gates_passed else "RISK_ROUTER_NEGATIVE",
        "gates_passed": gates_passed,
        "comparisons": comparisons,
        "scores_and_diagnostics": diagnostics,
        "source_provenance": provenance,
        "all_fits_frozen_before_outer_evaluation": True,
        "elapsed_seconds": time.perf_counter() - began,
        "output_bindings": bind_output_files(
            output,
            [
                output / "ALL_RIDGE_FITS_FROZEN.json",
                *(
                    output / f"outer{outer}/{arm}.npz"
                    for outer in range(3)
                    for arm in ("S65-RISK8", "S65-RISK17")
                ),
                *(output / f"{arm}_oof.csv" for arm in arms),
            ],
        ),
    }
    resource_check()
    result = sign_stage63_65_artifact(result, evidence_type="nested_outer_dev")
    _atomic_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--stage61-worktree", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frozen-teacher-audit", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify nested source identities without fitting or scoring",
    )
    args = parser.parse_args()
    if args.preflight_only:
        args.output_root.mkdir(parents=True, exist_ok=True)
        with CampaignLock(args.output_root / "CAMPAIGN_LOCK.json"):
            proof = _validate_nested_sources(
                args.reference_root / "artifacts/scientific_recovery_v8/results/router",
                args.stage61_worktree / "artifacts/scientific_recovery_v9_stage61_stage62/stage61",
                args.frozen_teacher_audit,
            )
            proof = sign_stage63_65_artifact(proof, evidence_type="nested_source_preflight")
            destination = args.output_root / "stage65/PREREQUISITES_MANIFEST.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError("preserve the prior source audit; use a new output root")
            _atomic_json(destination, proof)
        print(json.dumps({"status": proof["status"], "producer_count": proof["producer_count"]}))
        return
    if (_REPOSITORY_ROOT / "CAMPAIGN_INTEGRITY_HOLD.json").exists():
        raise RuntimeError("Campaign integrity hold: no campaign writes or fitting authorized")
    with CampaignLock(args.output_root / "CAMPAIGN_LOCK.json"):
        result = run(args)
    print(json.dumps({"status": result["status"]}))


if __name__ == "__main__":
    main()
