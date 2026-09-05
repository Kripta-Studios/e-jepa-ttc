"""Build the essential, checkpoint-free Stage 63–65 verification bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import torch

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash  # noqa: E402
from e_jepa_ttc.artifacts.stage63_65 import sign_stage63_65_artifact  # noqa: E402
from e_jepa_ttc.artifacts.training_authorization import (  # noqa: E402
    read_signed,
    verify_output_bindings,
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_index(output_root: Path, *, allow_invalid: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    indexed: set[Path] = set()
    for path in sorted((output_root / "stage64").glob("seed*/outer*/S64-*/frozen_manifest.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        checkpoint = Path(value["checkpoint_path"])
        valid = not (
            not verify_artifact_hash(value)
            or not checkpoint.is_file()
            or checkpoint.stat().st_size != value["checkpoint_bytes"]
            or _sha(checkpoint) != value["checkpoint_sha256"]
        )
        if not valid and not allow_invalid:
            raise ValueError(f"packaging endpoint checkpoint identity mismatch: {checkpoint}")
        indexed.add(checkpoint.resolve())
        records.append(
            {
                "seed": value["seed"],
                "outer_fold": value["outer_fold"],
                "arm": value["arm"],
                "updates": value["completed_updates"],
                "path": value["checkpoint_path"],
                "bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
                "sha256": _sha(checkpoint) if checkpoint.is_file() else None,
                "status": "frozen_bytes_verified" if valid else "invalid_endpoint_preserved",
                "declared_sha256": value["checkpoint_sha256"],
            }
        )
    for pattern in (
        "seed*/outer*/S64-*/checkpoint_last.pt",
        "seed*/outer*/S64-*/checkpoint_last.pt.previous",
    ):
        for checkpoint in sorted((output_root / "stage64").glob(pattern)):
            if checkpoint.resolve() in indexed:
                continue
            receipt = checkpoint.with_suffix(checkpoint.suffix + ".sha256")
            actual = _sha(checkpoint)
            declared = receipt.read_text(encoding="ascii").strip() if receipt.is_file() else None
            updates = None
            try:
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                updates = int(payload["completed_updates"])
            except (
                OSError,
                RuntimeError,
                ValueError,
                KeyError,
                TypeError,
                EOFError,
                pickle.UnpicklingError,
            ):
                pass  # Byte identity is still reported; no valid endpoint is inferred.
            records.append(
                {
                    "seed": checkpoint.parents[2].name.removeprefix("seed"),
                    "outer_fold": checkpoint.parents[1].name.removeprefix("outer"),
                    "arm": checkpoint.parent.name,
                    "updates": updates,
                    "path": str(checkpoint.resolve()),
                    "bytes": checkpoint.stat().st_size,
                    "sha256": actual,
                    "declared_sha256": declared,
                    "status": "partial_bytes_verified_not_final"
                    if actual == declared
                    else "partial_unverified_preserved",
                }
            )
    return records


def _verify_bundle(path: Path, required: set[str]) -> None:
    """Reopen the closed ZIP and verify inventory, CRC, sizes and SHA-256 bytes."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or not required <= set(names):
            raise ValueError("essential ZIP duplicates or lacks contractual members")
        if archive.testzip() is not None:
            raise ValueError("essential ZIP CRC verification failed")
        manifest = json.loads(archive.read("ESSENTIAL_MANIFEST.json"))
        listed = {record["path"] for record in manifest["files"]}
        if listed | {"ESSENTIAL_MANIFEST.json"} != set(names):
            raise ValueError("essential ZIP manifest inventory mismatch")
        for record in manifest["files"]:
            data = archive.read(record["path"])
            if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError(f"essential ZIP member identity mismatch: {record['path']}")


def _report(campaign: dict[str, Any], next_decision: dict[str, Any], output_root: Path) -> str:
    stage63 = campaign.get("stage63", {})
    stage64 = campaign.get("stage64", {})
    stage65 = campaign.get("stage65")
    qa_path = output_root / "qa/QA_MANIFEST.json"
    qa = read_signed(qa_path) if qa_path.is_file() else {}
    smoke_path = output_root / "qa/STAGE64_REAL_TRAIN_ONLY_SMOKE.json"
    smoke = read_signed(smoke_path) if smoke_path.is_file() else {}
    lines = [
        "# E-JEPA-TTC — Stage 63 / Stage 64 / Stage 65 final report",
        "",
        f"Final protocol status: `{campaign['status']}`.",
        "",
        "## What actually ran",
        "",
        f"- Stage 63 physical raw audit: `{stage63.get('decision', 'not_completed')}`.",
        f"- Physical raw ready: `{stage63.get('physical_data_ready', False)}`.",
        f"- Stage 64 training ready: `{stage63.get('training_ready', False)}`.",
        f"- Stage 64 seeds with completed results: `{sorted(stage64)}`.",
        f"- Stage 65 status: `{stage65.get('status') if stage65 else 'not_authorized_branch'}`.",
        f"- Non-selectable real train-only smoke microbatches: `{smoke.get('batches', 0)}`.",
        f"- Final-code QA acceptance: `{qa.get('status', 'not_accepted')}`; historical failures "
        f"retained: `{len(qa.get('historical_failures_classified_not_waived', []))}`.",
        "",
        "Reading or hashing raw events is not reported as training. Stage 64 is listed above only "
        "when fixed update-3000 endpoints and OOF evaluation artifacts exist.",
        "Partial attempts are indexed separately in CHECKPOINT_INDEX.csv and must not be "
        "interpreted as unexecuted work or as final training endpoints.",
        f"Operational failure: `{campaign.get('failure_phase')}` / `{campaign.get('error')}`.",
        "",
        "## Raw data and provenance",
        "",
        f"- Binding rows: `{stage63.get('binding_manifest', {}).get('rows')}`.",
        f"- Raw cache tokens: `{stage63.get('raw_cache_manifest', {}).get('tokens')}`.",
        "- Supported fraction: "
        f"`{stage63.get('raw_cache_manifest', {}).get('supported_fraction')}`.",
        f"- Forbidden paths opened: `{stage63.get('forbidden_paths_opened')}`.",
        f"- A5 replay passed: `{stage63.get('a5_replay', {}).get('passed')}`.",
        "",
        "The raw representation remains detection-assisted: the same audited common ROI derived "
        "from the two train bounding boxes is used for both windows and every temporal bin.",
        "The preserved handoff decision X3_BLOCKED describes its historical pin, not the later "
        "pending X3 export (X3_DATA_READY). Neither substitutes for the physical Stage 63 audit.",
        "",
        "## Scores and gates",
        "",
    ]
    if stage64:
        for seed, value in sorted(stage64.items()):
            lines.append(f"### Stage 64 seed {seed}")
            lines.append("")
            lines.append(f"Status: `{value.get('status')}`.")
            scores = value.get("gates", {}).get("scores", {})
            for name, item in scores.items():
                lines.append(f"- {name}: `{item.get('score')}` MiD.")
            for name, gate in value.get("gates", {}).get("comparisons", {}).items():
                stats = gate.get("statistics", {})
                lines.append(
                    f"- RAW vs {name}: delta `{stats.get('point_delta')}`, "
                    f"CI95 `[{stats.get('ci95_low')}, {stats.get('ci95_high')}]`, "
                    f"fraction_negative `{stats.get('fraction_negative')}`, "
                    f"passed `{gate.get('passed')}`."
                )
            lines.append("")
    if stage65:
        lines.extend(["### Stage 65", "", f"Status: `{stage65.get('status')}`.", ""])
        for name, item in stage65.get("scores_and_diagnostics", {}).items():
            lines.append(f"- {name}: `{item.get('score', {}).get('score')}` MiD.")
        for name, gate in stage65.get("comparisons", {}).items():
            stats = gate.get("statistics", {})
            lines.append(
                f"- RISK17 vs {name}: delta `{stats.get('point_delta')}`, "
                f"CI95 high `{stats.get('ci95_high')}`, passed `{gate.get('passed')}`."
            )
        lines.append("")
    lines.extend(
        [
            "## Integrity and interpretation",
            "",
            f"Training commit: `{campaign.get('training_commit')}`.",
            f"Automatic next action: `{next_decision['automatic_next_action']}`.",
            "",
            "These nine groups are adaptively reused development data. No public validation, "
            "private test, EvTTC test or CodaBench data were authorized. This protocol cannot "
            "establish SOTA, including when a local score crosses 144.353 MiD.",
            "",
            "Checkpoint bytes are not bundled. Their absolute paths, sizes and SHA-256 values are "
            "listed in `CHECKPOINT_INDEX.csv` for later local audit.",
        ]
    )
    return "\n".join(lines) + "\n"


def package(args: argparse.Namespace) -> tuple[Path, Path]:
    repo = Path(__file__).resolve().parents[1]
    args.output_root = args.output_root.resolve()
    args.handoff_root = args.handoff_root.resolve()
    remediation_root = args.remediation_root.resolve() if args.remediation_root else None
    prior_root = args.prior_attempt_root.resolve() if args.prior_attempt_root else None
    campaign = json.loads((args.output_root / "CAMPAIGN_RESULT.json").read_text(encoding="utf-8"))
    # Failure closure can occur after earlier stages completed. Recover only
    # signed, byte-verified completed results, preserving the terminal failure.
    feasibility_path = args.output_root / "stage63/X3_FEASIBILITY_V2.json"
    if "stage63" not in campaign and feasibility_path.is_file():
        campaign["stage63"] = read_signed(feasibility_path)
    campaign.setdefault("stage64", {})
    for result_path in (args.output_root / "stage64").glob("seed*/STAGE64_RESULT.json"):
        result = read_signed(result_path)
        verify_output_bindings(result_path.parent, result)
        campaign["stage64"][str(result["seed"])] = result
    risk_path = args.output_root / "stage65/STAGE65_RESULT.json"
    if risk_path.is_file():
        risk = read_signed(risk_path)
        verify_output_bindings(risk_path.parent, risk)
        campaign["stage65"] = risk
    lock_path = args.output_root / "TRAINING_LOCK.json"
    if lock_path.is_file():
        campaign.setdefault("training_commit", read_signed(lock_path)["training_commit"])
    analysis_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    stage63 = campaign.get("stage63", {})
    next_decision = {
        "artifact_type": "scientific_recovery_v9_next_decision_v2",
        "historical_decision_preserved": "STAGE61_AND_X2_NEGATIVE_X3_BLOCKED",
        "raw_physical_status": stage63.get("decision", "not_completed"),
        "training_readiness": stage63.get("training_ready", False),
        "status_by_stage": {
            "stage63": stage63.get("decision", "not_completed"),
            "stage64": {
                seed: value.get("status") for seed, value in campaign.get("stage64", {}).items()
            },
            "stage65": campaign.get("stage65", {}).get("status")
            if campaign.get("stage65")
            else "not_run",
        },
        "authorized_branches_executed": {
            "stage64_seeds": sorted(campaign.get("stage64", {})),
            "stage65": campaign.get("stage65") is not None,
        },
        "training_commit": campaign.get("training_commit"),
        "analysis_commit": analysis_commit,
        "source_hashes": {
            "original_bundle": "5ec3e5ebe1f0c24a04bda42bd2c7699e976fae019602b9e2a0b0c1aafd8224ce",
            "hdf5": stage63.get("binding_manifest", {}).get("hdf5_files", {}),
        },
        "seeds_and_replication_scope": {
            "a5_producer_seed": 7,
            "new_module_seeds_executed": sorted(campaign.get("stage64", {})),
            "claim": "conditional_new_encoder_adapter_only_not_full_system_multiseed",
        },
        "gate_details": {
            seed: value.get("gates") for seed, value in campaign.get("stage64", {}).items()
        }
        | (
            {"stage65": campaign["stage65"].get("comparisons", {})}
            if campaign.get("stage65")
            else {}
        ),
        "sealed_access_status": {
            "forbidden_paths_opened": stage63.get("forbidden_paths_opened"),
            "derived_from_access_ledger": False,
            "recorded_access_assessment": stage63.get("access_audit"),
            "exhaustive_access_claim": False,
        },
        "abort_or_stop_reason": campaign["status"],
        "automatic_next_action": "none",
        "recommended_next_protocol": (
            "independent_confirmation_protocol_required"
            if campaign["status"]
            in {"RAW_CONDITIONAL_REPLICATION_COMPLETE", "RISK_ROUTER_DEV_CANDIDATE"}
            else "preserve_negative_or_blocked_result_and_redesign_before_new_data_or_training"
        ),
    }
    next_decision = sign_stage63_65_artifact(
        next_decision, evidence_type="campaign_next_decision", repository_root=repo
    )
    report_path = repo / "CODEX_STAGE63_STAGE64_STAGE65_FINAL_REPORT.md"
    next_path = repo / "NEXT_DECISION_V2.json"
    report_path.write_text(_report(campaign, next_decision, args.output_root), encoding="utf-8")
    next_path.write_text(
        json.dumps(next_decision, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    checkpoint_records = _checkpoint_index(
        args.output_root,
        allow_invalid=campaign["status"]
        in {
            "INTEGRITY_BLOCKED",
            "RESOURCE_BLOCKED",
            "TECHNICAL_FAILURE",
            "NUMERICAL_FAILURE",
            "INPUT_MISSING",
            "INTERRUPTED",
        },
    )
    for record in checkpoint_records:
        record["attempt"] = "corrected_campaign"
    if prior_root is not None:
        previous_records = _checkpoint_index(prior_root, allow_invalid=True)
        for record in previous_records:
            record["attempt"] = "pre_review_invalidated_not_scientifically_accepted"
        checkpoint_records.extend(previous_records)
    checkpoint_path = args.output_root / "CHECKPOINT_INDEX.csv"
    with checkpoint_path.open("w", encoding="utf-8", newline="") as stream:
        fields = [
            "seed",
            "outer_fold",
            "arm",
            "updates",
            "path",
            "bytes",
            "sha256",
            "status",
            "declared_sha256",
            "attempt",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(checkpoint_records)
    short = analysis_commit[:12]
    zip_path = repo / f"E_JEPA_TTC_STAGE63_STAGE64_STAGE65_ESSENTIAL_RESULTS_{short}.zip"
    if zip_path.exists():
        raise FileExistsError("essential ZIP already exists; preserve the prior delivery")
    diff_path = args.output_root / "IMPLEMENTATION.patch"
    diff_path.write_text(
        subprocess.check_output(
            ["git", "diff", "e69bc5195ed4cccf88835be79797c5ef6cf6be2c", "--"],
            cwd=repo,
            text=True,
            encoding="utf-8",
        ),
        encoding="utf-8",
    )
    candidates = [
        report_path,
        next_path,
        args.output_root / "CAMPAIGN_RESULT.json",
        checkpoint_path,
        diff_path,
    ]
    candidates.extend(
        path
        for path in args.output_root.rglob("*")
        if path.is_file()
        and not {"targeted_tmp", "historical_tmp"}.intersection(
            path.relative_to(args.output_root).parts
        )
        and path.suffix.lower()
        in {".json", ".jsonl", ".csv", ".log", ".xml", ".md", ".txt", ".npz"}
        and path.stat().st_size < 50 * 1024 * 1024
    )
    for evidence_root in (remediation_root, prior_root):
        if evidence_root is None:
            continue
        candidates.extend(
            path
            for path in evidence_root.rglob("*")
            if path.is_file()
            and "source" not in path.relative_to(evidence_root).parts
            and path.suffix.lower() in {".json", ".jsonl", ".csv", ".log", ".xml", ".md", ".txt"}
            and path.stat().st_size < 50 * 1024 * 1024
        )
    # Small fitted routers are required objects, not optional text attachments.
    if (campaign.get("stage65") or {}).get("status") in {
        "RISK_ROUTER_DEV_CANDIDATE",
        "RISK_ROUTER_NEGATIVE",
    }:
        for outer in range(3):
            for arm in ("S65-RISK8", "S65-RISK17"):
                fit = args.output_root / "stage65" / f"outer{outer}" / f"{arm}.npz"
                if not fit.is_file():
                    raise FileNotFoundError(f"contractual router fit missing: {fit}")
                candidates.append(fit)
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "e69bc5195ed4cccf88835be79797c5ef6cf6be2c", "--"],
        cwd=repo,
        text=True,
    ).splitlines()
    candidates.extend(
        repo / relative
        for relative in changed
        if Path(relative).suffix in {".py", ".ps1", ".json", ".md", ".toml"}
        and (repo / relative).is_file()
    )
    candidates.extend(
        [
            args.handoff_root / "SOURCE_PINS.json",
            args.handoff_root / "PROTOCOL.json",
            repo / "configs" / "protocol" / "scientific_recovery_v9_stage63_65.json",
        ]
    )
    unique = sorted(set(candidates), key=lambda path: str(path))
    manifest: list[dict[str, Any]] = []
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in unique:
            if not path.is_file():
                raise FileNotFoundError(f"essential bundle input disappeared: {path}")
            if path.is_relative_to(args.output_root):
                name = Path("run") / path.relative_to(args.output_root)
            elif remediation_root is not None and path.is_relative_to(remediation_root):
                name = Path("remediation") / path.relative_to(remediation_root)
            elif prior_root is not None and path.is_relative_to(prior_root):
                name = Path("prior_invalidated_attempt") / path.relative_to(prior_root)
            elif path.is_relative_to(repo):
                name = Path("repository") / path.relative_to(repo)
            else:
                name = Path("handoff") / path.name
            archive.write(path, name.as_posix())
            manifest.append(
                {"path": name.as_posix(), "bytes": path.stat().st_size, "sha256": _sha(path)}
            )
        payload = json.dumps(
            {"artifact_type": "stage63_65_essential_manifest_v1", "files": manifest},
            indent=2,
            sort_keys=True,
        ).encode()
        archive.writestr("ESSENTIAL_MANIFEST.json", payload)
    checksum_path = zip_path.with_suffix(zip_path.suffix + ".sha256")
    _verify_bundle(zip_path, {record["path"] for record in manifest} | {"ESSENTIAL_MANIFEST.json"})
    checksum_path.write_text(f"{_sha(zip_path)}  {zip_path.name}\n", encoding="ascii")
    return zip_path, checksum_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--remediation-root", type=Path)
    parser.add_argument("--prior-attempt-root", type=Path)
    zip_path, checksum_path = package(parser.parse_args())
    print(json.dumps({"zip": str(zip_path), "sha256_file": str(checksum_path)}))


if __name__ == "__main__":
    main()
