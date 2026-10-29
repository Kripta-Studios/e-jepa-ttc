"""Frozen V queue, unchanged V14 gates and prospective seed12345 group bootstrap."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from operational.range_transfer.analyze import gate, metrics, predict

from .common import HISTORY, OWNER, RUN, freeze, read, sha, utc, write

SCREEN = ("S-M0-L0-S7", "S-M1-L0-S7", "R-C05-CTX-S7", "R-C20-CTX-S7")


def freeze_queue(names: tuple[str, ...], stage: str) -> dict[str, Any]:
    """No prediction is permitted until the whole admitted queue is complete."""
    if stage not in ("screen", "replication"):
        raise ValueError("Unknown evaluation stage")
    if stage == "screen" and names != SCREEN:
        raise ValueError("All four preregistered heads must freeze together")
    path = RUN / ("SCREEN_ENDPOINTS.json" if stage == "screen" else "REPLICATION_ENDPOINTS.json")
    if path.exists():
        return read(path)
    heads = {}
    parents = {}
    for name in names:
        receipt = read(RUN / "fits" / name / "CHECKPOINT_RECEIPT.json")
        if receipt["completed_updates"] != 2500 or receipt["status"] != "COMPLETE":
            raise ValueError("Queue endpoint incomplete: " + name)
        if sha(Path(receipt["checkpoint_path"])) != receipt["checkpoint_sha256"]:
            raise ValueError("Checkpoint hash differs")
        heads[name] = receipt
        if name.startswith("R-"):
            parent = name.replace("-CTX", "")
            parents[parent] = read(RUN / "fits" / parent / "CHECKPOINT_RECEIPT.json")
            if (
                parents[parent]["completed_updates"] != 6000
                or parents[parent]["status"] != "COMPLETE"
            ):
                raise ValueError("Readout parent not complete")
            if (
                sha(Path(parents[parent]["checkpoint_path"]))
                != parents[parent]["checkpoint_sha256"]
            ):
                raise ValueError("Readout parent hash changed")
    value = {
        "heads": heads,
        "parents": parents,
        "original_parents": read(RUN / "ADMISSION.json")["parents"],
        "frozen_utc": utc(),
        "stage": stage,
        "V_scores_opened_before_freeze": False,
    }
    freeze(path, value)
    return value


def bootstrap(reference: pd.DataFrame, candidate: pd.DataFrame, output: Path) -> None:
    columns = ["sample_token", "group_id", "target_ttc"]
    if not reference[columns].equals(candidate[columns]):
        raise ValueError("Bootstrap rows/targets/groups differ")
    differences = []
    for group in sorted(set(reference.group_id)):
        mask = (
            (reference.group_id == group)
            & (reference.target_ttc > 0.1)
            & (reference.target_ttc <= 3)
        )
        truth = reference.loc[mask, "target_ttc"].to_numpy()
        if not len(truth):
            continue
        errors = []
        for frame in (reference, candidate):
            pred = frame.loc[mask, "prediction_ttc"].to_numpy()
            if not np.isfinite(pred).all() or not ((pred < 0) | (pred > 0.1)).all():
                write(output / "SUMMARY.json", {"status": "INVALID_COVERAGE"})
                return
            errors.append(10000 * np.abs(np.log1p(-0.1 / pred) - np.log1p(-0.1 / truth)).mean())
        differences.append(errors[1] - errors[0])
    rng = np.random.default_rng(12345)
    diff = np.asarray(differences)
    draws = diff[rng.integers(len(diff), size=(1000, len(diff)))].mean(1)
    freeze(output / "DRAWS.json", {"seed": 12345, "unit": "group", "draws": draws.tolist()})
    write(
        output / "SUMMARY.json",
        {
            "delta_MiDc": float(diff.mean()),
            "CI95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "groups": len(diff),
            "replicates": 1000,
            "seed": 12345,
            "selection_gate": False,
        },
    )


def evaluate_queue(stage: str) -> dict[str, Any]:
    frozen = read(
        RUN / ("SCREEN_ENDPOINTS.json" if stage == "screen" else "REPLICATION_ENDPOINTS.json")
    )
    reference = pd.read_csv(HISTORY / "evaluation/screen/RT-M0-L0-S7/PREDICTIONS.csv")
    results = {}
    for name, receipt in frozen["heads"].items():
        endpoint = Path(receipt["checkpoint_path"])
        if sha(endpoint) != receipt["checkpoint_sha256"]:
            raise ValueError("Frozen endpoint changed")
        if name.startswith("S-"):
            cache_path = OWNER / "features/V/E_HISTORY.npz"
            cache_receipt = read(cache_path.with_suffix(".json"))
            if sha(cache_path) != cache_receipt["output_sha256"]:
                raise ValueError("V cache changed")
        else:
            family = name.replace("-CTX", "")
            cache_path = RUN / "families" / family / "V/HISTORY.npz"
            if sha(cache_path) != read(cache_path.parent / "REBUILD.json")["history_sha256"]:
                raise ValueError("Family V cache changed")
        directory = RUN / "evaluation" / stage / name
        identity = {
            "endpoint": receipt["checkpoint_sha256"],
            "cache": sha(cache_path),
            "queue_freeze_sha256": sha(
                RUN
                / ("SCREEN_ENDPOINTS.json" if stage == "screen" else "REPLICATION_ENDPOINTS.json")
            ),
        }
        freeze(directory / "BINDING.json", identity)
        if (directory / "METRICS.json").exists():
            if (
                sha(directory / "PREDICTIONS.csv")
                != read(directory / "RECEIPT.json")["predictions_sha256"]
            ):
                raise ValueError("Predictions changed")
            results[name] = read(directory / "METRICS.json")
            continue
        with np.load(cache_path, allow_pickle=False) as archive:
            values = {k: archive[k] for k in archive.files}
        target = -0.1 / np.expm1(-values["target_phase"].astype(np.float64))
        tokens = values["sample_token"].astype(str)
        groups = values["group_id"].astype(str)
        if not (
            np.array_equal(tokens, reference.sample_token.to_numpy())
            and np.array_equal(groups, reference.group_id.to_numpy())
            and np.allclose(target, reference.target_ttc.to_numpy(), atol=1e-12, rtol=1e-12)
        ):
            raise ValueError("V native cohort changed")
        pred, elapsed = predict(
            endpoint, {k: values[k] for k in ("features", "timing", "valid", "expert_phase")}
        )
        frame = pd.DataFrame(
            {
                "sample_token": tokens,
                "group_id": groups,
                "target_ttc": target,
                "prediction_ttc": pred,
            }
        )
        frame.to_csv(directory / "PREDICTIONS.csv", index=False)
        score = metrics(target, pred, groups)
        write(directory / "METRICS.json", score)
        write(
            directory / "GROUP_METRICS.json",
            {
                g: metrics(target[groups == g], pred[groups == g], groups[groups == g])
                for g in sorted(set(groups))
            },
        )
        write(
            directory / "COST.json",
            {
                "scope": "CPU_FP32_cached_head_forward_batch128_all_queries",
                "seconds": elapsed,
                "queries": len(target),
                "end_to_end": False,
            },
        )
        write(
            directory / "RECEIPT.json",
            {"predictions_sha256": sha(directory / "PREDICTIONS.csv"), **identity},
        )
        results[name] = score
    return results


def select(results: dict[str, Any]) -> dict[str, Any]:
    historical = read(HISTORY / "evaluation/screen/RT-M0-L0-S7/METRICS.json")
    candidates = {"S": ("S-M0-L0-S7", "S-M1-L0-S7"), "R": ("R-C05-CTX-S7", "R-C20-CTX-S7")}
    gates = {}
    for family, (control, candidate) in candidates.items():
        paired = gate(results[control], results[candidate])
        old = gate(historical, results[candidate])
        gates[family] = {
            "paired": paired,
            "historical_V14_M0": old,
            "eligible": paired["eligible"] and old["eligible"],
        }
    eligible = [family for family in candidates if gates[family]["eligible"]]
    winner = (
        min(
            eligible,
            key=lambda f: (
                results[candidates[f][1]]["bins"]["crucial"]["group_macro_mid"],
                results[candidates[f][1]]["overall_MiD"],
                read(RUN / "evaluation/screen" / candidates[f][1] / "COST.json")["seconds"],
            ),
        )
        if eligible
        else None
    )
    decision = {
        "status": "SELECTED" if winner else "NO_CANDIDATE",
        "family": winner,
        "gates": gates,
        "candidate": candidates[winner][1] if winner else None,
        "selection_source": "exposed_internal_V",
        "CI_gate": False,
        "transfer_used_for_selection": False,
    }
    freeze(RUN / "SELECTION_FREEZE.json", decision)
    comparisons = [
        ("S-M0-L0-S7", "RT-M0-L0-S7"),
        ("S-M1-L0-S7", "RT-M1-L0-S7"),
        ("S-M1-L0-S7", "S-M0-L0-S7"),
        ("S-M1-L0-S7", "RT-M0-L0-S7"),
        ("R-C20-CTX-S7", "R-C05-CTX-S7"),
        ("R-C20-CTX-S7", "RT-M0-L0-S7"),
        ("R-C05-CTX-S7", "RT-M0-L0-S7"),
    ]
    for candidate, control in comparisons:
        root = HISTORY if control.startswith("RT-") else RUN
        a = pd.read_csv(root / "evaluation/screen" / control / "PREDICTIONS.csv")
        b = pd.read_csv(RUN / "evaluation/screen" / candidate / "PREDICTIONS.csv")
        # Parse both targets through the exact same frozen native phase values.
        if not np.allclose(a.target_ttc, b.target_ttc, atol=1e-12, rtol=1e-12):
            raise ValueError("Truth differs")
        b["target_ttc"] = a.target_ttc
        bootstrap(a, b, RUN / "bootstrap" / f"{candidate}_vs_{control}")
    return decision
