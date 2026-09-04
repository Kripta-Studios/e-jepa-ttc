"""Execute the deterministic strict-nested Stage 65 full-regret fallback on CPU."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import compute_file_hash, verify_artifact_hash  # noqa: E402
from e_jepa_ttc.artifacts.stage63_65 import sign_stage63_65_artifact  # noqa: E402
from e_jepa_ttc.evaluation.stage61_nested_pair_router import (  # noqa: E402
    build_router_features,
)
from e_jepa_ttc.evaluation.stage63_65 import (  # noqa: E402
    evaluate_gate,
    paired_hierarchical_bootstrap,
    scientific_mid_per_row,
    strict_macro_mass,
    strict_score,
)
from e_jepa_ttc.models.full_regret_router import FullRegretRidge  # noqa: E402
from e_jepa_ttc.models.three_expert_router import (  # noqa: E402
    BASE8_FEATURES,
    PHASE17_FEATURES,
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


def _validate_nested_sources(reference_router: Path, stage61_root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for outer in range(3):
        for expert in ("a5", "c2f"):
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
                records.append(
                    {
                        "outer_fold": outer,
                        "expert": expert.upper(),
                        "role": role,
                        "checkpoint_sha256": artifact["checkpoint"]["sha256"],
                        "protocol_sha256": _sha(protocol_path),
                    }
                )
        for role in ("inner0", "inner1", "inner2", "outer_dev"):
            if role == "outer_dev":
                path = stage61_root / f"outer_fold{outer}_seed7" / "pair" / "outer_dev_oof.csv"
                checkpoint_column = "checkpoint_sha256"
            else:
                path = stage61_root / f"outer_fold{outer}_seed7" / "pair" / role / "expert_oof.csv"
                checkpoint_column = "checkpoint_sha256"
            frame = pd.read_csv(path)[[checkpoint_column]]
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
            records.append(
                {
                    "outer_fold": outer,
                    "expert": "PAIR",
                    "role": role,
                    "checkpoint_sha256": values[0],
                    "protocol_sha256": None,
                }
            )
    if len(records) != 36:
        raise AssertionError("Stage 65 requires 12 A5, 12 C2F and 12 PAIR identities")
    return {"status": "passed", "producer_count": len(records), "producers": records}


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
    began = time.perf_counter()
    output = args.output_root / "stage65"
    result_path = output / "STAGE65_RESULT.json"
    if result_path.is_file():
        if not args.resume:
            raise FileExistsError("Stage 65 result exists; --resume is required")
        return json.loads(result_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=args.resume)
    reference_router = (
        args.reference_root / "artifacts" / "scientific_recovery_v8" / "results" / "router"
    )
    stage61_root = (
        args.stage61_worktree / "artifacts" / "scientific_recovery_v9_stage61_stage62" / "stage61"
    )
    provenance = _validate_nested_sources(reference_router, stage61_root)
    _atomic_json(output / "STAGE65_SOURCE_PROVENANCE.json", provenance)
    fit_records: list[dict[str, Any]] = []
    # Fit every outer model before opening any corresponding outer-dev score.
    for outer in range(3):
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
            fit = FullRegretRidge.fit(features.to_numpy(np.float64), losses, mass)
            fit_path = output / f"outer{outer}" / f"{name}.npz"
            record = _save_fit(fit_path, fit, feature_order=order)
            fit_records.append({"outer_fold": outer, "model": name, **record})
    _atomic_json(output / "ALL_RIDGE_FITS_FROZEN.json", {"status": "frozen", "fits": fit_records})
    arms: dict[str, list[pd.DataFrame]] = {
        "S65-RISK8": [],
        "S65-RISK17": [],
        "S65-CE17-REPLAY": [],
    }
    for outer in range(3):
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
    for name, parts in arms.items():
        frame = pd.concat(parts).sort_values("sample_token").reset_index(drop=True)
        if len(frame) != 8192:
            raise ValueError(f"Stage 65 {name} does not cover 8192 tokens")
        frame.to_csv(output / f"{name}_oof.csv", index=False, lineterminator="\n")
        aggregate[name] = frame
    router_r = pd.read_csv(
        stage61_root / "aggregate_seed7" / "RouterR_oof.csv", dtype={"sample_token": str}
    )
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
        bootstrap = paired_hierarchical_bootstrap(aggregate["S65-RISK17"], reference)
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
        "status": "RISK_ROUTER_DEV_CANDIDATE" if gates_passed else "RISK_ROUTER_NEGATIVE",
        "gates_passed": gates_passed,
        "comparisons": comparisons,
        "scores_and_diagnostics": diagnostics,
        "source_provenance": provenance,
        "all_fits_frozen_before_outer_evaluation": True,
        "elapsed_seconds": time.perf_counter() - began,
    }
    if result["elapsed_seconds"] > args.max_hours * 3600:
        result["status"] = "STAGE65_RESOURCE_BLOCKED"
    result = sign_stage63_65_artifact(result, evidence_type="nested_outer_dev")
    _atomic_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--stage61-worktree", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-hours", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    result = run(parser.parse_args())
    print(json.dumps({"status": result["status"]}))


if __name__ == "__main__":
    main()
