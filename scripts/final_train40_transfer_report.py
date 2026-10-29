"""Score two sealed, frozen EvTTC routes without importing models or running inference."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import shutil
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OWN = "PHASE_ONLY_TRAIN40_S7"
GARL = "Garl_full"
MODELS = [OWN, GARL, "E_CTX_TRAIN40", "R_CTX_TRAIN40"]
HISTORICAL = "HISTORICAL_AUTHOR46"
EXTENDED = "EXTENDED_UNIFORM_16"
GARL_WEIGHT = "e96a613a4fb877a1969d57ab562cadba89961fb202f5f2f2f0658f333a0d443e"
GARL_CONFIG = "c7dedc9b93d32a416b8e92a6b474cc5a17ffd96b2cdaae3a4554e3cc0ee28454"
MODEL_COMPONENTS = {
    "E_A5",
    "E_C2F",
    "E_CTX_MATCHED",
    "PAIR_E",
    "PAIR_R",
    "R_A5",
    "R_C2F",
    "R_CTX",
    "PHASE_ONLY_EVENT_S7_TRAIN40",
    "event_normalizer",
    "rgb_normalizer",
    "MODEL_SEAL.json",
    "OOF_RECIPE.json",
}


def sha(path: Path) -> str:
    """Hash a local evidence file without loading model tensors."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path: Path) -> Any:
    """Read strict JSON; reject duplicate keys and non-standard numbers."""

    def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid(value: str) -> None:
        raise ValueError(f"Non-standard JSON number: {value}")

    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=unique, parse_constant=invalid
    )


def write(path: Path, value: Any) -> None:
    """Write reproducible tables with strict finite JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def legacy_functions() -> tuple[dict[str, Any], dict[str, str]]:
    """Reuse the exact historical scoring functions without model-heavy imports."""
    path = ROOT / "scripts/post_v18_transfer_report.py"
    source = path.read_text(encoding="utf-8")
    names = {"match_target", "summary", "tables", "contrasts"}
    nodes = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in nodes} != names or len(nodes) != len(names):
        raise ValueError("Historical scoring functions changed structure")
    namespace: dict[str, Any] = {"np": np, "pd": pd, "Any": Any}
    tree = ast.Module(body=cast(list[ast.stmt], nodes), type_ignores=[])
    exec(compile(tree, str(path), "exec"), namespace)
    return namespace, {
        "source_sha256": sha(path),
        "selected_ast_sha256": hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
    }


def prediction(value: Any) -> float | None:
    """Keep explicit missing outputs; never coerce strings or booleans into scores."""
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Prediction must be a finite scalar or explicit null")
    return float(value)


def no_targets(value: dict[str, Any]) -> None:
    """Require the prediction phase to declare zero training and no TTC access."""
    if value.get("targets_read") is not False or value.get("optimizer_updates") != 0:
        raise ValueError("Prediction ancestry read targets or updated weights")


def sealed(directory: Path, manifest_sha: str, ids: set[str]) -> tuple[dict, dict, dict]:
    """Verify full coverage, exact bytes, ancestry and safe fragment paths."""
    seal = read(directory / "PREDICTIONS_SEALED.json")
    freeze_path = directory / "INFERENCE_FREEZE.json"
    freeze = read(freeze_path)
    binding = sha(freeze_path)
    no_targets(seal)
    no_targets(freeze)
    if (
        seal.get("schema") != "final_train40_transfer_predictions_v1"
        or seal.get("status") != "COMPLETE"
        or seal.get("manifest_sha256") != manifest_sha
        or freeze.get("manifest_sha256") != manifest_sha
        or seal.get("binding_sha256") != binding
        or seal.get("queries") != len(ids)
        or len(seal.get("fragments", {})) != len(ids)
    ):
        raise ValueError("Incomplete or incompatible prediction seal")
    rows: dict[str, dict] = {}
    resolved: set[Path] = set()
    for name, digest in seal["fragments"].items():
        path = (directory / name).resolve()
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or not path.is_relative_to(directory.resolve())
            or path in resolved
        ):
            raise ValueError("Unsafe or aliased prediction path")
        resolved.add(path)
        if sha(path) != digest:
            raise ValueError(f"Prediction fragment changed: {name}")
        row = read(path)
        no_targets(row)
        if (
            row.get("binding_sha256") != binding
            or row.get("manifest_sha256") != manifest_sha
            or row.get("query_id") not in ids
            or row["query_id"] in rows
        ):
            raise ValueError("Prediction query/ancestry mismatch")
        rows[row["query_id"]] = row
    if set(rows) != ids:
        raise ValueError("Prediction query coverage differs")
    return rows, freeze, seal


def verify_inputs(
    manifest_path: Path, registration_path: Path, own_dir: Path, garl_dir: Path, candidate: Path
) -> tuple[dict, dict, dict, dict]:
    """Finish both prediction and model identity gates before any GT file access."""
    manifest, registration = read(manifest_path), read(registration_path)
    digest = sha(manifest_path)
    no_targets(manifest)
    no_targets(registration)
    rows = manifest["rows"]
    ids = {row["query_id"] for row in rows}
    physical = {(row["raw_path"], row["anchor_us"]) for row in rows}
    if (
        not rows
        or len(ids) != len(rows)
        or len(physical) != len(rows)
        or registration.get("manifest_sha256") != digest
        or registration.get("queries") != len(rows)
    ):
        raise ValueError("Manifest duplicate queries or registration mismatch")
    for row in rows:
        if (
            not row.get("physical_sequence_id")
            or not row.get("strata")
            or not set(row["strata"]) <= {HISTORICAL, EXTENDED}
        ):
            raise ValueError("Invalid physical sequence or cohort stratum")
    own, a, _ = sealed(own_dir, digest, ids)
    garl, b, _ = sealed(garl_dir, digest, ids)
    portable = read(candidate / "PORTABLE_PATHS.json")
    # PORTABLE_PATHS also inventories old test12 outputs and export files. They
    # are preserved evidence, but are not inputs consumed by this inference.
    components = {key: portable["components"][key]["sha256"] for key in MODEL_COMPONENTS}
    if (
        a.get("model_name") != OWN
        or a.get("seed") != 7
        or a.get("precision") != "float32"
        or a.get("modality") != "RGB+EVENTS"
        or a.get("model_seal_sha256") != sha(candidate / "MODEL_SEAL.json")
        or a.get("model_hashes") != components
        or a.get("model", {}).get("portable_paths_sha256") != sha(candidate / "PORTABLE_PATHS.json")
        or a.get("checkpoint_selection") is not False
    ):
        raise ValueError("Final candidate binding differs from preserved candidate")
    if (
        b.get("model_name") != GARL
        or b.get("checkpoint_sha256") != GARL_WEIGHT
        or b.get("config_sha256") != GARL_CONFIG
        or b.get("precision") != "float32"
        or b.get("registration_sha256") != sha(registration_path)
        or b.get("native_delta_t_s") != 0.1
        or b.get("clipping") is not None
    ):
        raise ValueError("Garl native full-model binding differs")
    for query_id in ids:
        values = own[query_id].get("predictions", {})
        if set(values) != set(MODELS) - {GARL}:
            raise ValueError("Own output model set differs")
        for value in values.values():
            prediction(value)
        if "prediction_ttc" not in garl[query_id]:
            raise ValueError("Garl prediction field is missing")
        value = prediction(garl[query_id]["prediction_ttc"])
        if value is None and not garl[query_id].get("failure_reason"):
            raise ValueError("Missing Garl prediction has no recorded reason")
    provenance = {
        "manifest_sha256": digest,
        "registration_sha256": sha(registration_path),
        "model_bindings": {"own": a, "garl": b},
        "prediction_seals": {
            "own": sha(own_dir / "PREDICTIONS_SEALED.json"),
            "garl": sha(garl_dir / "PREDICTIONS_SEALED.json"),
        },
        "report_source_sha256": sha(Path(__file__)),
    }
    return manifest, own, garl, provenance


def band(value: float) -> str:
    """Descriptive RTE ranges, explicitly distinct from official MiD boundaries."""
    if value < 0:
        return "negative"
    if value == 0 or not math.isfinite(value):
        return "invalid"
    if value < 3:
        return "positive_lt3"
    if value < 6:
        return "positive_3_to_lt6"
    return "positive_6_to_10" if value <= 10 else "positive_gt10"


def join_targets(
    manifest: dict, own: dict, garl: dict, scorer: dict
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read public labels only after the caller completed verify_inputs."""
    tables_cache, hashes, result = {}, {}, []
    for row in manifest["rows"]:
        path = Path(row["ttc_csv_path"])
        key = str(path.resolve())
        if key not in tables_cache:
            if path.is_file():
                payload = path.read_bytes()
                hashes[key] = hashlib.sha256(payload).hexdigest()
                tables_cache[key] = np.genfromtxt(io.BytesIO(payload), dtype=np.float64, ndmin=2)
            else:
                tables_cache[key] = None
        target = (
            {
                "target_ttc": None,
                "target_timestamp_s": None,
                "target_delta_s": None,
                "target_exclusion": "TTC_FILE_MISSING",
            }
            if tables_cache[key] is None
            else scorer["match_target"](tables_cache[key], row["anchor_us"] / 1e6)
        )
        query_id = row["query_id"]
        result.append(
            {
                "query_id": query_id,
                "group_id": row["physical_sequence_id"],
                "scenario_family": row.get("scenario_family", "UNKNOWN"),
                "target_type": row.get("target_type", "UNKNOWN"),
                "speed_bucket": row.get("speed_bucket", "UNKNOWN"),
                "cohort": "historical46" if HISTORICAL in row["strata"] else "extended_new",
                "also_extended": EXTENDED in row["strata"],
                "anchor_s": row["anchor_us"] / 1e6,
                **target,
                **own[query_id]["predictions"],
                GARL: garl[query_id]["prediction_ttc"],
                "garl_failure": garl[query_id].get("failure_reason"),
                "rgb_available": own[query_id].get("rgb_available"),
                "rgb_unavailable_reason": own[query_id].get("rgb_unavailable_reason"),
                "own_seconds": json.dumps(own[query_id].get("seconds", {}), sort_keys=True),
                "garl_seconds": json.dumps(garl[query_id].get("seconds", {}), sort_keys=True),
            }
        )
    frame = pd.DataFrame(result)
    frame["ttc_band"] = frame.target_ttc.map(lambda x: band(float(x)) if pd.notna(x) else "invalid")
    frame["target_sign"] = frame.target_ttc.map(
        lambda x: "negative" if x < 0 else "positive" if x > 0 else "invalid"
    )
    return frame, hashes


def grouped_metrics(frame: pd.DataFrame, scorer: dict) -> dict:
    """Keep complete and conditional metrics separate and equal-sequence macro explicit."""
    if frame.empty:
        return {"status": "NO_VALID_TARGETS", "queries": 0}
    result = scorer["tables"](frame, MODELS)
    macro = {}
    for model in MODELS:
        groups = [value[model] for value in result["per_group"].values()]
        metrics = [value["complete_cohort"] for value in groups]
        complete = all(value is not None for value in metrics)
        macro[model] = {
            "sequences": len(groups),
            "complete_cohort": (
                {key: float(np.mean([value[key] for value in metrics])) for key in metrics[0]}
                if complete
                else None
            ),
        }
    result["equal_sequence_macro"] = macro
    result["contrasts"] = scorer["contrasts"](frame, [(OWN, GARL)])
    for contrast in result["contrasts"]:
        contrast["independent_sequences"] = int(frame.group_id.nunique())
        contrast["CI_informative"] = frame.group_id.nunique() >= 2
    common = frame.loc[np.isfinite(frame[[OWN, GARL]].to_numpy(dtype=float)).all(axis=1)]
    result["paired_common_finite"] = {
        "queries": len(common),
        "excluded_queries": len(frame) - len(common),
        "selection": "Conditional on both finite predictions; not the full cohort",
        "metrics": scorer["tables"](common, [OWN, GARL]) if len(common) else None,
        "contrasts": scorer["contrasts"](common, [(OWN, GARL)]) if len(common) else [],
        "CI_informative": common.group_id.nunique() >= 2,
    }
    return result


def build_report(
    manifest_path: Path,
    registration: Path,
    own_dir: Path,
    garl_dir: Path,
    candidate: Path,
    output: Path,
    *,
    figure: bool = True,
) -> dict:
    """Validate both routes first, then produce all descriptive scores and provenance."""
    for protected in (candidate, own_dir, garl_dir):
        if output.resolve().is_relative_to(protected.resolve()):
            raise ValueError("Report must not modify sealed predictions or candidate closure")
    manifest, own, garl, provenance = verify_inputs(
        manifest_path, registration, own_dir, garl_dir, candidate
    )
    scorer, source = legacy_functions()
    frame, labels = join_targets(manifest, own, garl, scorer)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "QUERIES.csv", index=False)
    result: dict[str, Any] = {
        "status": "COMPLETE",
        "provenance": provenance,
        "legacy_metric_source": source,
        "label_sha256": labels,
        "sensor_queries": len(frame),
        "sensor_sequences": int(frame.group_id.nunique()),
        "manifest_exclusions": manifest.get("excluded", []),
        "cohorts": {},
        "optimizer_updates": 0,
        "official_evaluation": False,
        "checkpoint_selection": False,
    }
    flat = []
    for cohort in ("historical46", "extended_new"):
        selected = frame.loc[frame.cohort == cohort]
        valid = selected.loc[selected.target_exclusion.isna()]
        report = {
            "sensor_queries": len(selected),
            "scored_queries": len(valid),
            "target_exclusions": selected.target_exclusion.dropna().value_counts().to_dict(),
            "metrics": grouped_metrics(valid, scorer),
            "stratified": {},
        }
        for column in ("scenario_family", "target_type", "speed_bucket", "target_sign", "ttc_band"):
            report["stratified"][column] = {
                str(key): grouped_metrics(part, scorer)
                for key, part in valid.groupby(column, sort=True)
            }
        result["cohorts"][cohort] = report
        for group, part in valid.groupby("group_id", sort=True):
            for model in MODELS:
                item = scorer["summary"](part, model)
                flat.append(
                    {
                        "cohort": cohort,
                        "sequence": group,
                        "model": model,
                        "queries": item["queries"],
                        "coverage": item["coverage"],
                        "complete": item["complete_cohort"] is not None,
                        **item["finite_only_conditional"],
                    }
                )
    sequence_frame = pd.DataFrame(flat)
    sequence_frame.to_csv(output / "PER_SEQUENCE.csv", index=False)
    methodology = {
        "GT_gate": (
            "Both complete seals, exact query IDs, manifest SHA and model bindings before TTC reads"
        ),
        "GT_matching": (
            "Nearest public ttc.csv row, maximum 50 ms, earlier timestamp wins ties; "
            "zero/nonfinite excluded"
        ),
        "RTE_pct": (
            "100 * abs(prediction-target) / abs(target); signed nonzero targets, no clipping"
        ),
        "macro": "Equal weight per physical sequence; pooled gives equal weight per query",
        "bootstrap": (
            "2000 whole physical sequence draws seed 7, macro paired difference, exploratory"
        ),
        "cohorts": (
            "historical46 contains shared anchors; extended_new excludes all overlaps; "
            "no merged primary score"
        ),
        "bands": (
            "negative<0; (0,3); [3,6); [6,10]; (10,infinity). "
            "Descriptive RTE bins, NOT official MiD bins"
        ),
        "missing_predictions": (
            "Coverage retained; complete-cohort score withheld; "
            "finite and paired subsets labeled conditional"
        ),
        "inputs": (
            "Own RGB+events uses native causal sensor history up to 650 ms; "
            "Garl uses two RGB frames and two event segments dt 0.1 s"
        ),
        "ROI": (
            "Public annotated boxes and declared adapters, not detector outputs; "
            "shared query ROI differs from native per-anchor ROI"
        ),
        "transfer": (
            "EvTTC previously observed; descriptive transfer, "
            "not blind confirmation or exact paper Table VI reproduction"
        ),
        "timing": (
            "Raw instrumented wall seconds include preprocessing/caches; "
            "no isolated kernel or head-to-head latency claim"
        ),
        "precision": (
            "Both routes use float32 tensors. Own explicitly disables matmul/cuDNN TF32; "
            "Garl preserves backend defaults. Not a matched backend precision benchmark."
        ),
        "scope": (
            "No official benchmark score, no SOTA claim, no new fitting or checkpoint selection"
        ),
    }
    write(output / "METHODOLOGY.json", methodology)
    write(output / "RESULTS.json", result)
    lines = [
        "# Frozen TRAIN40 versus Garl: EvTTC transfer",
        "",
        "Descriptive evaluation on previously observed EvTTC; not an official score or SOTA claim.",
        "",
    ]
    for cohort, report in result["cohorts"].items():
        lines += [
            f"## {cohort}",
            "",
            f"{report['scored_queries']}/{report['sensor_queries']} queries have valid TTC.",
            "",
            "| Model | Coverage | Pooled RTE % | Pooled MAE s |",
            "|---|---:|---:|---:|",
        ]
        for model, item in report["metrics"].get("pooled", {}).items():
            score = item["complete_cohort"]
            rte = f"{score['RTE_pct']:.4f}" if score else "withheld"
            mae = f"{score['MAE_s']:.4f}" if score else "withheld"
            lines.append(
                f"| {model} | {item['finite_predictions']}/{item['queries']} | {rte} | {mae} |"
            )
        lines.append("")
        paired = report["metrics"].get("paired_common_finite", {})
        if paired.get("metrics"):
            lines += [
                f"Paired common finite subset: {paired['queries']} queries; "
                f"{paired['excluded_queries']} excluded because a route has no finite output. "
                "These are conditional scores, not full-cohort scores.",
                "",
                "| Model | Paired pooled RTE % | Paired pooled MAE s |",
                "|---|---:|---:|",
            ]
            for model, item in paired["metrics"]["pooled"].items():
                score = item["complete_cohort"]
                lines.append(f"| {model} | {score['RTE_pct']:.4f} | {score['MAE_s']:.4f} |")
            lines.append("")
        for contrast in report["metrics"].get("contrasts", []):
            if contrast.get("status") == "INCOMPLETE":
                lines.append(
                    "Full-cohort paired contrast withheld because predictions are missing."
                )
            else:
                delta = contrast["RTE_pct"]
                lines.append(
                    f"Equal-sequence macro RTE difference (TRAIN40 minus Garl): "
                    f"{delta['point']:.4f} percentage points; whole-sequence bootstrap "
                    f"95% interval [{delta['CI95'][0]:.4f}, {delta['CI95'][1]:.4f}]. "
                    "Exploratory; negative favors TRAIN40."
                )
            lines.append("")
    lines += ["## Interpretation limits", ""] + [
        f"- **{key}:** {value}" for key, value in methodology.items()
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if figure and not sequence_frame.empty:
        draw_figure(sequence_frame, output)
    return result


def draw_figure(frame: pd.DataFrame, output: Path) -> None:
    """Regenerate a static paired sequence RTE plot from report CSV values only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, cohort in zip(axes, ("historical46", "extended_new"), strict=True):
        selected = frame.loc[(frame.cohort == cohort) & frame.complete]
        table = selected.pivot(index="sequence", columns="model", values="RTE_pct")
        if OWN in table and GARL in table:
            table = table[[OWN, GARL]].dropna()
            axis.scatter(table[GARL], table[OWN])
            bound = max(float(table.max().max()), 1.0) if len(table) else 1.0
            axis.plot([0, bound], [0, bound], "k--", linewidth=1)
        axis.set(title=cohort, xlabel="Garl RTE (%)", ylabel="Final TRAIN40 RTE (%)")
        axis.grid(alpha=0.25)
    fig.suptitle("Each point is a complete physical sequence; below diagonal favors TRAIN40")
    fig.savefig(output / "PER_SEQUENCE_RTE.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    base = ROOT / "artifacts/final_train40_transfer_20261027"
    parser.add_argument("--manifest", type=Path, default=base / "QUERY_MANIFEST.json")
    parser.add_argument("--registration", type=Path, default=base / "COHORT_REGISTRATION.json")
    parser.add_argument("--own", type=Path, default=base / "final_train40")
    parser.add_argument("--garl", type=Path, default=base / "garl")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "docs/final_train40_transfer_20261027"
    )
    parser.add_argument("--copy-to", type=Path)
    args = parser.parse_args()
    result = build_report(
        args.manifest, args.registration, args.own, args.garl, args.candidate, args.output
    )
    if args.copy_to:
        if (
            args.copy_to.resolve() == args.candidate.resolve()
            or args.copy_to.resolve().is_relative_to(args.candidate.resolve())
        ):
            raise ValueError("New transfer evidence must not mutate preserved candidate closure")
        args.copy_to.mkdir(parents=True, exist_ok=True)
        for path in sorted(args.output.iterdir()):
            if path.is_file():
                shutil.copy2(path, args.copy_to / path.name)
        write(
            args.copy_to / "REPORT_COPY_RECEIPT.json",
            {
                "files": {
                    path.name: sha(path) for path in sorted(args.output.iterdir()) if path.is_file()
                }
            },
        )
    print(json.dumps({"status": result["status"], "sensor_queries": result["sensor_queries"]}))


if __name__ == "__main__":
    main()
