"""Regenerable TRAIN40 report, physical accounting and SHA-256 essential bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from operational.efficient_context.common import ROOT, digest
from operational.train40_system.comparison_metrics import evaluate
from operational.train40_system.contracts import read, verified_endpoint, verify_sources
from operational.train40_system.data_audit import OUTPUT
from operational.train40_system.durable_io import atomic_json, replace

ENDPOINTS = {
    "a5_seed7": 49932,
    "c2f_seed7": 49932,
    "pair_seed7": 6840,
    "h8_seed7": 2500,
    "h8_seed13": 2500,
    "h8_seed23": 2500,
}


def accounting(output: Path) -> dict:
    """Derive updates from journals; never rely on the initial technical receipt's stale zero."""
    authorization = read(output / "AUTHORIZATION.json")
    technical = read(output / "TECHNICAL_ACCOUNTING.json")
    fits, committed, pending, recovery = {}, 0, 0, 0
    for name, limit in ENDPOINTS.items():
        path = output / "fits" / name / "UPDATE_JOURNAL.json"
        if path.exists():
            journal = read(path)
            current = journal["committed_updates"]
            fits[name] = {
                key: journal[key]
                for key in (
                    "committed_updates",
                    "durable_updates",
                    "pending_update_upper",
                    "recovery_upper",
                )
            }
            if not 0 <= current <= limit:
                raise ValueError("Unregistered scientific update endpoint")
            committed += current
            pending += journal["pending_update_upper"]
            recovery += journal["recovery_upper"]
        else:
            fits[name] = {"status": "NOT_STARTED", "updates_limit": limit}
    physical = (
        authorization["previous_physical_execution_upper"]
        + technical["synthetic_optimizer_updates"]
        + committed
        + pending
        + recovery
    )
    if physical > 240000 or recovery > 2000:
        raise ValueError("Joint physical or recovery accounting exceeded")
    return {
        "previous_campaign_physical_upper": authorization["previous_physical_execution_upper"],
        "new_scientific_committed": committed,
        "new_pending_upper": pending,
        "new_recovery_upper": recovery,
        "synthetic_optimizer_updates": technical["synthetic_optimizer_updates"],
        "physical_upper": physical,
        "physical_cap": 240000,
        "new_fixed_scientific_endpoint": sum(ENDPOINTS.values()),
        "fits": fits,
        "sampled_utc": datetime.now(UTC).isoformat(),
    }


def prediction_metrics(output: Path) -> dict:
    """Compute all rows from saved predictions, explicitly as descriptive TRAIN diagnostics."""
    with np.load(output / "TRAIN40_INDEX.npz", allow_pickle=False) as stored:
        truth = stored["ttc_s"]
    results = {}
    if (output / "TRAIN_FIT_DIAGNOSTICS.json").exists():
        arrays = {seed: np.empty(88744, np.float32) for seed in (7, 13, 23)}
        for start in range(0, 88744, 128):
            stop = min(start + 128, 88744)
            path = output / "train_predictions" / f"batch_{start:06d}.npz"
            if digest(path) != read(path.with_suffix(".json"))["sha256"]:
                raise ValueError("Prediction fragment bytes changed")
            with np.load(path, allow_pickle=False) as stored:
                if not np.array_equal(stored["ordinals"], np.arange(start, stop)):
                    raise ValueError("Prediction fragment query identity changed")
                for seed in arrays:
                    arrays[seed][start:stop] = stored[f"ttc_seed{seed}"]
        results.update({f"H8_seed{seed}": evaluate(value, truth) for seed, value in arrays.items()})
    if (output / "PUBLIC_GARL_PREDICTION_MANIFEST.json").exists():
        manifest = read(output / "PUBLIC_GARL_PREDICTION_MANIFEST.json")
        if manifest["status"] != "COMPLETE_VERIFIED" or manifest["row_count"] != 88744:
            raise ValueError("Complete published Garl population required")
        item = manifest["files"][0]
        path = output / item["path"]
        if digest(path) != item["sha256"]:
            raise ValueError("Published Garl predictions changed")
        with np.load(path, allow_pickle=False) as stored:
            if not np.array_equal(stored["ordinals"], np.arange(88744)):
                raise ValueError("Published Garl comparison query identity changed")
            results["public_Garl_event_lhr"] = evaluate(stored["ttc"], truth)
    return results


def cost_and_curves(output: Path) -> dict:
    """Regenerate learning curves and observed compute/I/O cost from fragmented receipts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    result = {}
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    for name in ENDPOINTS:
        curves = [
            update
            for path in sorted((output / "fits" / name).glob("curve_*.json"))
            for update in read(path)["updates"]
        ]
        if not curves:
            continue
        positions = [row["update"] for row in curves]
        losses = [row["losses"]["total"] if "losses" in row else row["loss"] for row in curves]
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(positions, losses, linewidth=0.5)
        ax.set(xlabel="Optimizer update", ylabel="TRAIN loss", title=name)
        fig.tight_layout()
        fig.savefig(directory / f"{name}_training_loss.png", dpi=150)
        plt.close(fig)
        result[name] = {
            "recorded_updates": len(curves),
            "last_recorded_update": positions[-1],
            "compute_seconds": sum(row.get("compute_ms", 0) / 1000 for row in curves),
            "unhidden_input_wait_seconds": sum(row.get("data_wait_ms", 0) / 1000 for row in curves),
            "head_update_seconds": sum(row.get("seconds", 0) for row in curves),
        }
    return result


def write_bundle(files: dict[str, Path], destination: Path) -> dict:
    """Publish an archive only after independently hashing every streamed member."""
    entries = [
        {"path": name, "sha256": digest(path), "bytes": path.stat().st_size}
        for name, path in sorted(files.items())
    ]
    manifest = {"schema": "train40_essential_SHA256_v1", "files": entries}
    temporary = destination.with_suffix(".pending.zip")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        archive.writestr("MANIFEST.json", json.dumps(manifest, indent=2))
        for name, path in sorted(files.items()):
            archive.write(path, name)
    with zipfile.ZipFile(temporary, "r") as archive:
        if len(archive.namelist()) != len(entries) + 1:
            raise ValueError("Duplicate or missing essential bundle members")
        for item in entries:
            sha = hashlib.sha256()
            with archive.open(item["path"]) as member:
                for block in iter(lambda: member.read(1024**2), b""):
                    sha.update(block)
            if sha.hexdigest() != item["sha256"]:
                raise ValueError("Independent archived SHA-256 verification failed")
    replace(temporary, destination)
    return {
        "status": "PASSED",
        "sha256": digest(destination),
        "bytes": destination.stat().st_size,
        "verified_members": len(entries),
        "path": str(destination),
        "optimizer_updates": 0,
    }


def run(output: Path, *, bundle: bool = False) -> None:
    """Report a truthful partial state or deliver complete trained endpoints and predictions."""
    ledger = accounting(output)
    atomic_json(output / "ACCOUNTING_SNAPSHOT.json", ledger)
    results = prediction_metrics(output)
    costs = cost_and_curves(output)
    complete = ledger["new_scientific_committed"] == sum(ENDPOINTS.values())
    checkpoint_receipts = {}
    if complete:
        for name, limit in ENDPOINTS.items():
            _, checkpoint_receipts[name] = verified_endpoint(output, name, limit)
    comparison = {
        "role": "TRAIN_FIT_DIAGNOSTIC_NOT_GENERALIZATION",
        "queries": 88744,
        "TRAIN_sequences": 40,
        "public_checkpoint_training_manifest_verified": False,
        "public_checkpoint_SHA256": read(output / "PUBLIC_GARL_REAL_INPUT_ADMISSION.json")[
            "checkpoint_sha256"
        ],
        "checkpoint_strict_load_and_real_sensor_input": "PASSED",
        "no_published_accuracy_claim_or_paper_score_reused": True,
        "independent_evaluation_dependency": "Common authorized holdout with accessible signed-TTC "
        "labels, exact native evaluation adapter and verified checkpoint training "
        "sequence history.",
        "does_not_block_local_training": True,
    }
    atomic_json(output / "COMPARATOR_CONTRACT.json", comparison)
    status = "TRAINING_COMPLETE" if complete else "IN_PROGRESS"
    report = {
        "status": status,
        "accounting": ledger,
        "TRAIN_fit_metrics": results,
        "observed_cost": costs,
        "comparator_contract": comparison,
        "checkpoint_receipts": checkpoint_receipts,
        "encoder_system_seeds": 1,
        "temporal_head_seeds": [7, 13, 23],
    }
    atomic_json(output / "REPORT.json", report)
    decision = (
        "Evaluate the frozen trained system and published comparator only on an authorized "
        "common labeled holdout once checkpoint provenance is verified."
        if complete
        else "Continue the recoverable local TRAIN40 queue without opening protected holdouts."
    )
    atomic_json(
        output / "NEXT_DECISION.json",
        {
            "status": status,
            "decision": decision,
            "unresolved_dependency_is_not_a_negative_result": True,
            "scientific_updates_additionally_authorized_by_this_report": 0,
        },
    )
    lines = [
        "# TRAIN40 local campaign",
        "",
        f"Status: {status}.",
        "",
        "All reported prediction metrics are TRAIN fitting diagnostics. They do not establish "
        "holdout generalization or reproduce the published Garl paper evaluation.",
        "",
        f"New scientific updates: {ledger['new_scientific_committed']}/114204. "
        f"Joint physical upper: {ledger['physical_upper']}/240000.",
        "",
        "One independently trained encoder/system seed; three temporal-head seeds.",
        "",
        "| Model | TRAIN queries | Finite coverage | TRAIN MAE (s) | Failure rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, value in results.items():
        lines.append(
            f"| {name} | {value['population']} | {value['finite_coverage']:.6f} | "
            f"{value['MAE_seconds_finite']} | {value['failure_rate']:.6f} |"
        )
    lines += [
        "",
        "NEXT_DECISION: " + decision,
        "",
        "Independent comparison dependency: " + comparison["independent_evaluation_dependency"],
        "",
        "Full resumable checkpoints, prediction fragments and cost/loss receipts are retained. "
        "Raw media and large input caches remain external SHA-pinned dependencies.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if bundle:
        if not complete or len(results) != 4:
            raise ValueError(
                "Essential final bundle requires complete own and public TRAIN predictions"
            )
        freeze = read(output / "DELIVERY_FREEZE.json")
        verify_sources(freeze)
        files = {}
        for pattern in (
            "*.json",
            "*.txt",
            "*.md",
            "TRAIN40*.npz",
            "TRAIN40*.parquet",
            "H8_FEATURES.npz",
            "PAIR_FEATURES.npz",
            "PUBLIC_GARL_TRAIN_PREDICTIONS.npz",
        ):
            for path in output.glob(pattern):
                if path.name != "BUNDLE_VERIFICATION.json":
                    files["outputs/" + path.name] = path
        for folder in ("fits", "train_predictions", "figures", "admission_sources", "public_garl"):
            for path in (output / folder).rglob("*"):
                if path.is_file() and path.suffix in {
                    ".json",
                    ".pt",
                    ".pth",
                    ".npz",
                    ".png",
                    ".py",
                    ".yaml",
                    ".txt",
                }:
                    files["outputs/" + path.relative_to(output).as_posix()] = path
        for item in freeze["files"]:
            files["sources/" + Path(item["path"]).as_posix()] = ROOT / item["path"]
        native_root = Path(freeze["native_code_root"])
        for item in freeze["native_source_files"]:
            path = native_root / item["path"]
            if digest(path) != item["sha256"]:
                raise ValueError("Pinned native comparator source changed before packaging")
            files["native_sources/" + item["path"]] = path
        if sum(path.stat().st_size for path in files.values()) > 120_000_000_000:
            raise ValueError("Essential bundle exceeds authorized artifact cap")
        atomic_json(
            output / "BUNDLE_VERIFICATION.json",
            write_bundle(files, output / "essential_bundle.zip"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--bundle", action="store_true")
    args = parser.parse_args()
    run(args.output.resolve(), bundle=args.bundle)
