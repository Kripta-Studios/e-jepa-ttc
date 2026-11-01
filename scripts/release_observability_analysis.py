"""Preregistered group-held-out gain probes after RAW feature sealing; no R0 updates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from operational.release_research.normal_quality import PHYSICAL_FIELDS, SUPPORT_FIELDS
from operational.release_research.protocol import ResearchRun, sha256, write_json
from scripts.release_research import read, sources, verify_complete

PLAN = Path("docs/release_research_implementation_20261031/OBSERVABILITY_PREREGISTRATION.md")


def balanced_weights(groups: np.ndarray) -> np.ndarray:
    """Give acquisitions equal total mass while retaining an average row weight of one."""
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return len(groups) / (len(counts) * counts[inverse])


def design(train: np.ndarray, validation: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit imputation and standardization exclusively on the current training acquisitions."""
    if train.ndim != 2 or validation.ndim != 2 or train.shape[1] != 6 or validation.shape[1] != 6:
        raise ValueError("Expected six feature fields")
    available = np.isfinite(train)
    medians = np.array(
        [np.median(train[available[:, i], i]) if available[:, i].any() else 0.0 for i in range(6)]
    )
    filled = np.where(available, train, medians)
    mean, scale = filled.mean(0), np.maximum(filled.std(0), 1e-6)

    def convert(x: np.ndarray) -> np.ndarray:
        valid = np.isfinite(x)
        values = (np.where(valid, x, medians) - mean) / scale
        return np.column_stack((np.ones(len(x)), values, valid.astype(float)))

    return (
        convert(train),
        convert(validation),
        {
            "medians": medians.tolist(),
            "mean": mean.tolist(),
            "scale": scale.tolist(),
        },
    )


def grouped_probe(
    features: np.ndarray, gain: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, list]:
    """One fixed ridge solve per held-out acquisition, with no endpoint or hyperparameter search."""
    if gain.shape != (len(features),) or groups.shape != gain.shape or len(set(groups)) < 3:
        raise ValueError("Require aligned queries from at least three acquisitions")
    prediction = np.full(len(features), np.nan)
    fits = []
    for group in sorted(set(groups)):
        test = groups == group
        training = ~test
        train_x, val_x, preprocessing = design(features[training], features[test])
        weights = balanced_weights(groups[training])
        penalty = np.eye(train_x.shape[1])
        penalty[0, 0] = 0.0
        coefficient = np.linalg.solve(
            train_x.T @ (weights[:, None] * train_x) + penalty,
            train_x.T @ (weights * gain[training]),
        )
        prediction[test] = val_x @ coefficient
        fits.append(
            {
                "held_out_group": str(group),
                "training_groups": sorted(set(groups[training].tolist())),
                "training_rows": int(training.sum()),
                "held_out_rows": int(test.sum()),
                "ridge_lambda": 1.0,
                "coefficient": coefficient.tolist(),
                "preprocessing": preprocessing,
            }
        )
    return prediction, fits


def permute_physical(
    physical: np.ndarray, support: np.ndarray, groups: np.ndarray, seed: int = 7
) -> tuple[np.ndarray, dict]:
    """Label-free support/domain strata preserving the complete physical missingness pattern."""
    count = support[:, 0]
    valid = np.isfinite(count)
    limits = np.quantile(count[valid], [1 / 3, 2 / 3]) if valid.any() else np.array([0.0, 0.0])
    bucket = np.where(valid, np.searchsorted(limits, count), -1)
    strata: dict[tuple, list[int]] = {}
    for index in range(len(physical)):
        key = (
            "slider" if groups[index] == "Slider-500" else "road",
            int(bucket[index]),
            tuple(np.isfinite(physical[index]).tolist()),
        )
        strata.setdefault(key, []).append(index)
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(physical))
    for indices in strata.values():
        permutation[indices] = rng.permutation(indices)
    shuffled = physical[permutation].copy()
    np.testing.assert_array_equal(np.isfinite(physical), np.isfinite(shuffled))
    changed = np.any(
        ~((physical == shuffled) | (~np.isfinite(physical) & ~np.isfinite(shuffled))), axis=1
    )
    return shuffled, {
        "seed": seed,
        "strata": len(strata),
        "tercile_boundaries": limits.tolist(),
        "permutation": permutation.tolist(),
        "changed_rows": int(changed.sum()),
        "changed_fraction": float(changed.mean()),
        "deployment_candidate": False,
    }


def within_group_correlations(
    features: np.ndarray, gain: np.ndarray, groups: np.ndarray, names: tuple[str, ...]
) -> dict:
    """Report each field's association after removing acquisition means on available rows."""
    result = {}
    for column, name in enumerate(names):
        valid = np.isfinite(features[:, column])
        x, y = [], []
        group_count = 0
        for group in sorted(set(groups)):
            rows = valid & (groups == group)
            if rows.sum() < 2:
                continue
            xv, yv = features[rows, column], gain[rows]
            x.extend((xv - xv.mean()).tolist())
            y.extend((yv - yv.mean()).tolist())
            group_count += 1
        correlation = None
        if len(x) >= 3 and np.std(x) > 1e-12 and np.std(y) > 1e-12:
            correlation = float(np.corrcoef(x, y)[0, 1])
        result[name] = {
            "available": int(valid.sum()),
            "within_group_rows": len(x),
            "groups": group_count,
            "pearson_after_group_centering": correlation,
        }
    return result


def summarize_probes(
    gain: np.ndarray, predictions: dict[str, np.ndarray], groups: np.ndarray, permutation: dict
) -> dict:
    """Apply the fixed diagnostic admission threshold with acquisition bootstrap."""
    unique = sorted(set(groups))
    rng = np.random.default_rng(7)
    indices = rng.integers(0, len(unique), (2000, len(unique)))
    squared = {
        name: np.asarray([np.mean((value[groups == g] - gain[groups == g]) ** 2) for g in unique])
        for name, value in predictions.items()
    }
    arms = {
        name: {
            "macro_mse": float(errors.mean()),
            "by_acquisition": {str(g): float(e) for g, e in zip(unique, errors, strict=True)},
        }
        for name, errors in squared.items()
    }
    deltas = {}
    for control in ("constant", "support", "permuted"):
        delta = squared["physical"] - squared[control]
        deltas[control] = {
            "mean": float(delta.mean()),
            "ci95": np.quantile(delta[indices].mean(1), [0.025, 0.975]).tolist(),
        }
    improvements = all(
        arms["physical"]["macro_mse"] <= 0.98 * arms[name]["macro_mse"]
        for name in ("constant", "support", "permuted")
    )
    admitted = (
        improvements
        and deltas["support"]["ci95"][1] < 0
        and permutation["changed_fraction"] >= 0.25
    )
    return {
        "arms": arms,
        "physical_minus_control_mse": deltas,
        "bootstrap": {"unit": "acquisition", "resamples": 2000, "seed": 7},
        "gate_pilot_admitted": bool(admitted),
        "independent_confirmation": False,
    }


def analyze(features_path: Path, evaluated_path: Path, output: Path) -> None:
    """Join only complete RAW features and the already authorized development evaluation."""
    verify_complete(features_path)
    verify_complete(evaluated_path)
    frozen = read(features_path / "FREEZE.json")
    if frozen["config"]["queries"] != 149 or frozen["config"]["profiling_only"]:
        raise ValueError("Require all 149 queries, not a profile-selected subset")
    if frozen["inputs"].get("preregistration") != sha256(PLAN):
        raise ValueError("Preregistered criteria changed after extraction")
    rows = read(features_path / "FEATURES.json")["records"]
    scored = read(evaluated_path / "DIAGNOSTICS.json")["records"]
    if [(r["token"], r["group"], r["query_us"]) for r in rows] != [
        (r["token"], r["group"], r["query_us"]) for r in scored
    ]:
        raise ValueError("Feature/target query identity mismatch")
    inputs = {
        **sources(),
        "source:" + str(Path(__file__)): Path(__file__),
        "features": features_path / "FEATURES.json",
        "features_complete": features_path / "COMPLETE.json",
        "evaluation": evaluated_path / "DIAGNOSTICS.json",
        "evaluation_complete": evaluated_path / "COMPLETE.json",
        "preregistration": PLAN,
    }
    run = ResearchRun(
        output,
        inputs,
        {
            "experiment_id": "O0-incremental-gain-probe-v1",
            "seed": 7,
            "exposure": "development with R0-exposed parents",
            "ridge_lambda": 1.0,
            "validation": "leave one acquisition out; not full-chain OOF",
            "target": "gain_seconds",
            "R0_optimizer_updates": 0,
        },
    )
    try:
        supported = np.array([r.get("gain_seconds") is not None for r in scored])
        physical = np.asarray([r["physical"] for r in rows], float)[supported]
        support = np.asarray([r["support"] for r in rows], float)[supported]
        gain = np.asarray([r["gain_seconds"] for r in scored if r.get("gain_seconds") is not None])
        groups = np.asarray([r["group"] for r, valid in zip(rows, supported, strict=True) if valid])
        shuffled, shuffle_report = permute_physical(physical, support, groups)
        predictions, fits = {}, {}
        for name, values in (
            ("constant", np.zeros_like(physical)),
            ("support", support),
            ("physical", physical),
            ("permuted", shuffled),
        ):
            predictions[name], fits[name] = grouped_probe(values, gain, groups)
        summary = summarize_probes(gain, predictions, groups, shuffle_report)
        summary.update(
            required=len(rows),
            labeled_pairs=int(supported.sum()),
            unavailable_original=int((~supported).sum()),
            correlations_physical=within_group_correlations(
                physical, gain, groups, PHYSICAL_FIELDS
            ),
            correlations_support=within_group_correlations(support, gain, groups, SUPPORT_FIELDS),
            permutation=shuffle_report,
            correction_helpful=int((gain > 0).sum()),
            correction_harmful=int((gain < 0).sum()),
            zero_new_R0_updates=True,
        )
        for i, index in enumerate(np.flatnonzero(supported)):
            rows[index]["gain_seconds"] = float(gain[i])
            rows[index]["held_out_gain_predictions"] = {
                name: float(value[i]) for name, value in predictions.items()
            }
        write_json(output / "DIAGNOSTICS.json", {"records": rows})
        write_json(output / "FITS.json", fits)
        write_json(output / "METRICS.json", summary)
        write_json(
            output / "DECISION.json",
            {
                "family": "gate",
                "decision": "admit" if summary["gate_pilot_admitted"] else "diagnose",
                "conditions": ["incremental_observability"]
                if summary["gate_pilot_admitted"]
                else [],
                "scope": "pilot admission only; no candidate or release claim",
                "release_modified": False,
            },
        )
        lines = [
            "# Observabilidad RAW en las ROI del release",
            "",
            "Desarrollo expuesto; sondas con adquisición excluida, padres R0 ya expuestos.",
            "",
            "| Sonda de gain | MSE macro por adquisición |",
            "|---|---:|",
        ]
        for name, result in summary["arms"].items():
            lines.append(f"| {name} | {result['macro_mse']:.9f} |")
        lines.extend(
            [
                "",
                f"Consultas: {len(rows)}; pares evaluables: {supported.sum()}.",
                f"Piloto de gate admitido por regla prefijada: {summary['gate_pilot_admitted']}.",
                "MSE se refiere a predecir la utilidad de la corrección, no a TTC final.",
                "No se ha entrenado ni sustituido ningún componente de R0.",
            ]
        )
        (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def main() -> None:
    """Evaluate the fixed diagnostic, retaining all fits, queries and denominators."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument(
        "--evaluation", type=Path, default=Path("output/release_research_20261031/evaluation149_v2")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.features, args.evaluation, args.output)


if __name__ == "__main__":
    main()
