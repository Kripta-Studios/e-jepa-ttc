"""Recover expansion from the available-data photometric proxy before using TTC labels."""

from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.evttc_transfer.inputs import _calibration, _calibration_sha256
from operational.release_research.evaluation import compare
from operational.release_research.physics import q_to_raw_phase, transport
from operational.release_research.protocol import ResearchRun, sha256, write_json
from scripts.release_observability import check_stat
from scripts.release_research import read, sources, verify_complete
from scripts.release_sensor_proxy import project_event_points, sample_bilinear

PLAN = Path("docs/release_research_implementation_20261031/PHOTOMETRIC_Q_PREREGISTRATION.md")
ARMS = ("radial", "translation", "affine", "spatial_permutation")


def photometric_design(
    frame: np.ndarray, calibration: dict, xedges: np.ndarray, yedges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Cell means of affine brightness derivatives in one isotropic sensor coordinate unit."""
    gray = frame.astype(float).mean(axis=2) if frame.ndim == 3 else frame.astype(float)
    patches = []
    for y0, y1 in zip(yedges[:-1], yedges[1:], strict=True):
        for x0, x1 in zip(xedges[:-1], xedges[1:], strict=True):
            xx, yy = np.meshgrid(np.linspace(x0, x1 - 1, 5), np.linspace(y0, y1 - 1, 5))
            patches.append(np.column_stack((xx.ravel(), yy.ravel())))
    points = np.concatenate(patches)
    offsets = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]])
    mapped, rays = project_event_points(
        (points[None] + offsets[:, None]).reshape(-1, 2), calibration
    )
    sampled, inside = sample_bilinear(gray, mapped)
    sampled = sampled.reshape(5, -1)
    valid = (rays & inside).reshape(5, -1).all(axis=0)
    valid &= ((sampled >= 5) & (sampled <= 250)).all(axis=0)
    logs = np.log(np.maximum(sampled, 1) / 255)
    gx, gy = (logs[1] - logs[2]) / 2, (logs[3] - logs[4]) / 2
    center = [(xedges[0] + xedges[-1] - 1) / 2, (yedges[0] + yedges[-1] - 1) / 2]
    dx, dy = (points - center).T
    scale = float(xedges[-1] - xedges[0])
    matrix = np.column_stack(
        (-gx * dx, -gx * dy, -gy * dx, -gy * dy, -gx * scale, -gy * scale, np.ones(len(points)))
    )
    matrix = matrix.reshape(-1, 25, 7)
    mask = valid.reshape(-1, 25)
    average = (
        np.where(mask[:, :, None], matrix, 0).sum(axis=1) / np.maximum(mask.sum(axis=1), 1)[:, None]
    )
    return average, mask.mean(axis=1)


def fit_functional(design: np.ndarray, target: np.ndarray, functional: np.ndarray) -> dict:
    """Robustly estimate an observable functional, without inventing unobserved directions."""
    if (
        design.ndim != 2
        or target.shape != (len(design),)
        or functional.shape != (design.shape[1],)
        or not all(np.isfinite(a).all() for a in (design, target, functional))
        or not np.linalg.norm(functional) > 0
    ):
        raise ValueError("Invalid finite photometric regression arrays")
    if len(design) < 32:
        return {"q": None, "reason": "fewer_than_32_cells", "cells": len(design)}
    weights = np.ones(len(design))
    for _ in range(5):
        root = np.sqrt(weights)
        u, singular, vh = np.linalg.svd(design * root[:, None], full_matrices=False)
        admitted = singular > singular[0] * 1e-8
        basis = vh[admitted]
        coefficient = basis.T @ ((u[:, admitted].T @ (target * root)) / singular[admitted])
        residual = target - design @ coefficient
        scale = max(1e-6, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
        weights = np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(residual), 1e-15))
    projection = basis @ functional
    leakage = float(np.linalg.norm(functional - basis.T @ projection) / np.linalg.norm(functional))
    estimable = leakage <= 1e-8
    return {
        "q": float(functional @ coefficient) if estimable else None,
        "reason": "ok" if estimable else "q_not_observable",
        "cells": len(design),
        "rank": int(admitted.sum()),
        "residual_dof": len(design) - int(admitted.sum()),
        "null_leakage": leakage,
        "sensitivity": float(np.linalg.norm(projection / singular[admitted]))
        if estimable
        else None,
        "median_absolute_residual": float(np.median(np.abs(residual))),
        "coefficient": coefficient.tolist(),
    }


def source_inputs() -> dict[str, Path]:
    result = {**sources(), "preregistration": PLAN}
    for name in (
        "scripts/release_photometric_q.py",
        "scripts/release_sensor_proxy.py",
        "scripts/release_observability.py",
        "operational/evttc_transfer/inputs.py",
        "src/e_jepa_ttc/data/evttc_object_cache.py",
    ):
        result["source:" + name] = Path(name)
    return result


def extract(capture: Path, calibration_run: Path, expanded: Path, output: Path) -> None:
    """Fit local nuisance motion using held-out sensor scale, without reading TTC targets."""
    for folder in (capture, calibration_run, expanded):
        verify_complete(folder)
    calibration_freeze = read(calibration_run / "FREEZE.json")
    if (
        calibration_freeze["inputs"]["capture"] != sha256(capture / "COMPLETE.json")
        or calibration_freeze["config"]["extra_control"] != "rgb_observed"
    ):
        raise ValueError("Calibration must use this exact reference-control cache")
    records = read(capture / "FEATURES.json")["records"]
    if len(records) != 149:
        raise ValueError("Expected the entire 149-query inventory")
    fits = {
        f["held_out_group"]: f
        for f in read(calibration_run / "FITS.json")
        if f["arm"] == "rgb_observed"
    }
    inputs = {
        **source_inputs(),
        "capture": capture / "COMPLETE.json",
        "features": capture / "FEATURES.json",
        "calibration": calibration_run / "COMPLETE.json",
        "fits": calibration_run / "FITS.json",
        "expanded": expanded / "COMPLETE.json",
    }
    for row in records:
        if row["status"] == "measured":
            identity = hashlib.sha256(row["token"].encode()).hexdigest()[:24]
            inputs["array:" + identity] = capture / row["arrays"]
            inputs["source_query:" + identity] = expanded / "queries" / identity / "QUERY.json"
    run = ResearchRun(
        output,
        inputs,
        {
            "experiment_id": "photometric-q-extract-v1",
            "targets_read": False,
            "optimizer_updates": 0,
            "arms": list(ARMS),
            "state_time": "event window midpoint",
            "scope": "development proxy, no learned-family admission",
        },
    )
    try:
        results = []
        for number, row in enumerate(records):
            result = {k: row[k] for k in ("token", "group", "query_us")}
            result["readouts"] = {
                name: {"q_at_query": None, "ttc": None, "reason": row["status"]} for name in ARMS
            }
            if row["status"] == "measured":
                fit = fits[row["group"]]
                if row["group"] in fit["training_groups"]:
                    raise ValueError("Scale calibration includes evaluated acquisition")
                if fit["slope"] <= 0:
                    for value in result["readouts"].values():
                        value["reason"] = "nonpositive_scale"
                else:
                    identity = hashlib.sha256(row["token"].encode()).hexdigest()[:24]
                    source = read(expanded / "queries" / identity / "QUERY.json")
                    path = Path(source["raw_path"])
                    check_stat(path, source["raw_stat"])
                    with h5py.File(path, "r") as handle:
                        calibration = _calibration(handle)
                        if _calibration_sha256(calibration) != source["calibration_sha256"]:
                            raise ValueError("Geometric calibration changed")
                        times = np.asarray(handle["blackflys/left/ts"])
                        index = int(np.searchsorted(times, row["reference_us"]))
                        if index >= len(times) or int(times[index]) != row["reference_us"]:
                            raise ValueError("Reference timestamp changed")
                        data = handle["blackflys/left/data"]
                        if not isinstance(data, h5py.Dataset):
                            raise ValueError("RGB frames must be a dataset")
                        frame = np.asarray(data[index])
                    if (
                        hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()
                        != row["reference_sha256"]
                    ):
                        raise ValueError("Reference bytes changed")
                    roi = np.asarray(row["roi"])
                    left, top = max(0, int(np.ceil(roi[0]))), max(0, int(np.ceil(roi[1])))
                    right, bottom = min(1280, int(np.ceil(roi[2]))), min(720, int(np.ceil(roi[3])))
                    design, coverage = photometric_design(
                        frame,
                        calibration,
                        np.linspace(left, right, 17).astype(int),
                        np.linspace(top, bottom, 17).astype(int),
                    )
                    with np.load(capture / row["arrays"], allow_pickle=False) as archive:
                        np.testing.assert_array_equal(coverage, archive["coverage"])
                        np.testing.assert_allclose(
                            design[:, 0] + design[:, 3],
                            archive["texture_motion"],
                            rtol=1e-12,
                            atol=1e-12,
                        )
                        valid = archive["valid"]
                        target = (archive["target_rate"][valid] - fit["intercept"]) / fit["slope"]
                    design = design[valid]
                    radial = design[:, 0] + design[:, 3]
                    systems = {
                        "radial": (
                            np.column_stack((radial, np.ones(len(radial)))),
                            np.array([1.0, 0.0]),
                        ),
                        "translation": (
                            np.column_stack((radial, design[:, 4:6], np.ones(len(radial)))),
                            np.array([1.0, 0.0, 0.0, 0.0]),
                        ),
                        "affine": (design, np.array([0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0])),
                    }
                    seed = int(hashlib.sha256(("7:" + row["token"]).encode()).hexdigest()[:16], 16)
                    permutation = np.random.default_rng(seed).permutation(len(design))
                    systems["spatial_permutation"] = (design[permutation], systems["affine"][1])
                    midpoint = sum(row["windows_us"][-1]) / 2
                    for arm, (matrix, functional) in systems.items():
                        value = fit_functional(matrix, target, functional)
                        value.update(q_at_query=None, ttc=None)
                        if value["q"] is not None:
                            try:
                                q = transport(value["q"], (row["query_us"] - midpoint) * 1e-6)[0]
                                value["q_at_query"] = q
                                phase = q_to_raw_phase(q)
                                value["ttc"] = float(
                                    phase_to_ttc(torch.tensor([phase], dtype=torch.float32))[0]
                                )
                            except ValueError as error:
                                value["reason"] = str(error)
                        result["readouts"][arm] = value
                    result["scale_fit"] = fit
                    check_stat(path, source["raw_stat"])
                print(f"{number + 1}/149 {row['group']}", flush=True)
            results.append(result)
        write_json(output / "PREDICTIONS.json", {"records": results, "targets_read": False})
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def evaluate(predictions: Path, evaluation: Path, output: Path) -> None:
    """Join exposed TTC only after sealing all four measurements and rejections."""
    for folder in (predictions, evaluation):
        verify_complete(folder)
    freeze = read(predictions / "FREEZE.json")
    if freeze["config"]["targets_read"] or freeze["inputs"]["preregistration"] != sha256(PLAN):
        raise ValueError("Require unchanged target-blind photometric-q protocol")
    rows = read(predictions / "PREDICTIONS.json")["records"]
    scored = read(evaluation / "DIAGNOSTICS.json")["records"]
    if [(r["token"], r["group"], r["query_us"]) for r in rows] != [
        (r["token"], r["group"], r["query_us"]) for r in scored
    ]:
        raise ValueError("Target query identity mismatch")
    run = ResearchRun(
        output,
        {
            **source_inputs(),
            "predictions": predictions / "PREDICTIONS.json",
            "parent": predictions / "COMPLETE.json",
            "evaluation": evaluation / "DIAGNOSTICS.json",
            "evaluation_complete": evaluation / "COMPLETE.json",
        },
        {
            "experiment_id": "photometric-q-evaluation-v1",
            "targets_read": True,
            "optimizer_updates": 0,
            "learned_family_admitted": False,
        },
    )
    try:
        target = np.array([r["target"] for r in scored], float)
        baseline = np.array([r["R0"] for r in scored], float)
        event = np.array([r["EVENT_CONTEXT"] for r in scored], float)
        groups = np.array([r["group"] for r in rows])
        result = {}
        for arm in ARMS:
            values = np.array([r["readouts"][arm]["ttc"] for r in rows], float)
            q = np.array([r["readouts"][arm]["q_at_query"] for r in rows], float)
            valid = np.isfinite(q)
            candidate = np.where(np.isfinite(values), values, baseline)
            candidate[~np.isfinite(baseline)] = np.nan
            road = groups != "Slider-500"
            result[arm] = {
                "q_available": int(valid.sum()),
                "ttc_available": int(np.isfinite(values).sum()),
                "median_q_error": float(np.median(np.abs(q[valid] - 1 / target[valid])))
                if valid.any()
                else None,
                "q_sign_errors": int((np.sign(q[valid]) != np.sign(target[valid])).sum()),
                "with_R0_fallback": compare(target, baseline, candidate, groups),
                "road_measurement_vs_event_context": compare(
                    target[road], event[road], values[road], groups[road]
                ),
                "reasons": dict(Counter(r["readouts"][arm]["reason"] for r in rows)),
            }
        write_json(output / "METRICS.json", {"readouts": result, "learned_family_admitted": False})
        lines = [
            "# Recuperación de q desde el proxy fotométrico",
            "",
            "Desarrollo expuesto. Escala ajustada excluyendo la adquisición "
            "y lecturas selladas antes de unir TTC.",
            "",
            "| Lectura | q disponible | TTC disponible | Mediana error q | "
            "Signos erróneos | MAE con fallback R0 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for arm, value in result.items():
            error = value["median_q_error"]
            q_text = "—" if error is None else f"{error:.6f}"
            mae = value["with_R0_fallback"]["candidate"]["macro_mae"]
            lines.append(
                f"| {arm} | {value['q_available']} | {value['ttc_available']} | "
                f"{q_text} | {value['q_sign_errors']} | {mae:.9f} |"
            )
        lines += [
            "",
            "La lectura afín permite traslación, deformación y luz global; "
            "no una textura libre por píxel.",
            "La escala RGB/eventos sigue siendo un proxy: "
            "desfase, ISP, paralaje y respuesta pueden sesgar q.",
            "Se conservan todos los fallos, la ausencia original y el control espacial permutado.",
            "No se modifica R0 ni se abre automáticamente entrenamiento de productores o JEPA.",
        ]
        (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def main() -> None:
    """Keep target-free measurement and exposed TTC evaluation in separate runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("output/release_research_20261031")
    parser.add_argument("mode", choices=("extract", "evaluate"))
    parser.add_argument("--capture", type=Path, default=root / "sensor_reference149_v1")
    parser.add_argument(
        "--calibration", type=Path, default=root / "sensor_reference_analysis149_v1"
    )
    parser.add_argument(
        "--expanded", type=Path, default=Path("output/phase_observation_cache/expanded149")
    )
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--evaluation", type=Path, default=root / "evaluation149_v2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "extract":
        extract(args.capture, args.calibration, args.expanded, args.output)
    elif args.predictions is None:
        parser.error("evaluate requires --predictions")
    else:
        evaluate(args.predictions, args.evaluation, args.output)


if __name__ == "__main__":
    main()
