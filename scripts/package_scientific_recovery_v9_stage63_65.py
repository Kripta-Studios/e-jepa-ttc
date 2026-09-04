"""Build the essential, checkpoint-free Stage 63–65 verification bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from e_jepa_ttc.artifacts.stage63_65 import sign_stage63_65_artifact  # noqa: E402


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_index(output_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted((output_root / "stage64").glob("seed*/outer*/S64-*/frozen_manifest.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            {
                "seed": value["seed"],
                "outer_fold": value["outer_fold"],
                "arm": value["arm"],
                "updates": value["completed_updates"],
                "path": value["checkpoint_path"],
                "bytes": value["checkpoint_bytes"],
                "sha256": value["checkpoint_sha256"],
            }
        )
    return records


def _report(campaign: dict[str, Any], next_decision: dict[str, Any]) -> str:
    stage63 = campaign.get("stage63", {})
    stage64 = campaign.get("stage64", {})
    stage65 = campaign.get("stage65")
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
        f"- Stage 64 executed seeds: `{sorted(stage64)}`.",
        f"- Stage 65 status: `{stage65.get('status') if stage65 else 'not_authorized_branch'}`.",
        "",
        "Reading or hashing raw events is not reported as training. Stage 64 is listed above only "
        "when fixed update-3000 endpoints and OOF evaluation artifacts exist.",
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
    campaign = json.loads((args.output_root / "CAMPAIGN_RESULT.json").read_text(encoding="utf-8"))
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
            "derived_from_access_ledger": True,
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
    report_path.write_text(_report(campaign, next_decision), encoding="utf-8")
    next_path.write_text(
        json.dumps(next_decision, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    checkpoint_records = _checkpoint_index(args.output_root)
    checkpoint_path = args.output_root / "CHECKPOINT_INDEX.csv"
    with checkpoint_path.open("w", encoding="utf-8", newline="") as stream:
        fields = ["seed", "outer_fold", "arm", "updates", "path", "bytes", "sha256"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(checkpoint_records)
    short = analysis_commit[:12]
    zip_path = repo / f"E_JEPA_TTC_STAGE63_STAGE64_STAGE65_ESSENTIAL_RESULTS_{short}.zip"
    candidates = [
        report_path,
        next_path,
        args.output_root / "CAMPAIGN_RESULT.json",
        checkpoint_path,
    ]
    candidates.extend(
        path
        for path in args.output_root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".json", ".jsonl", ".csv", ".log", ".md", ".txt"}
        and "checkpoint" not in path.name.lower()
        and path.stat().st_size < 50 * 1024 * 1024
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
                continue
            if path.is_relative_to(args.output_root):
                name = Path("run") / path.relative_to(args.output_root)
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
    checksum_path.write_text(f"{_sha(zip_path)}  {zip_path.name}\n", encoding="ascii")
    return zip_path, checksum_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    zip_path, checksum_path = package(parser.parse_args())
    print(json.dumps({"zip": str(zip_path), "sha256_file": str(checksum_path)}))


if __name__ == "__main__":
    main()
