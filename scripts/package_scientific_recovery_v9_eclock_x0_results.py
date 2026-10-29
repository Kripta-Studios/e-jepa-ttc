#!/usr/bin/env python
"""Create and self-verify the essential E-Clock X0 result bundle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any

from e_jepa_ttc.artifacts.hashing import verify_artifact_hash

ARMS = {"X0-A5-REPLAY", "X0-BASE-U", "X0-DYN-U", "X0-PAIR-U"}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    campaign = args.campaign_root.resolve()
    repo = Path(__file__).resolve().parents[1]
    observed = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed != args.git_commit:
        raise ValueError("packager commit differs from authorized training commit")
    report = campaign / "CODEX_X0_FINAL_CAMPAIGN_REPORT.md"
    if not report.is_file():
        raise ValueError("final report is missing")
    short = args.git_commit[:12]
    output = args.output_root / f"E_JEPA_TTC_X0_SEED7_ESSENTIAL_RESULTS_{short}.zip"
    if output.exists():
        raise FileExistsError(f"result bundle already exists: {output}")
    members: dict[str, bytes] = {}
    allowed_suffixes = {".json", ".csv", ".md", ".log", ".jsonl", ".yaml"}
    for path in sorted(campaign.rglob("*")):
        if not path.is_file() or path == output or path.suffix == ".pt":
            continue
        if path.suffix not in allowed_suffixes:
            continue
        relative_path = path.relative_to(campaign)
        relative = relative_path.as_posix()
        data = path.read_bytes()
        if path.name.endswith(".pt.manifest.json"):
            destination = Path("checkpoint_manifests", relative_path).as_posix()
            members[destination] = data
            continue
        if relative_path.parts[0] in ARMS:
            destination = Path("runs", relative_path).as_posix()
        elif relative_path.parts[0] == "master_logs":
            destination = Path("qa", "logs", *relative_path.parts[1:]).as_posix()
        elif relative_path.parts[0] == "state":
            destination = Path("qa", "state", *relative_path.parts[1:]).as_posix()
        elif relative_path.parts[0] == "preflight":
            destination = Path("cache_engineering", *relative_path.parts[1:]).as_posix()
        else:
            destination = relative
        if destination.endswith("progress.jsonl"):
            members[f"{destination}.gz"] = gzip.compress(data, mtime=0)
        else:
            members[destination] = data
    for path in sorted(
        (repo / "configs/experiment/scientific_recovery_v9_eclock").glob("x0_*.yaml")
    ):
        members[f"configs/{path.name}"] = path.read_bytes()
    for name in ("scientific_recovery_v9_eclock_x0.json",):
        path = repo / "configs/protocol" / name
        members[f"protocol/{name}"] = path.read_bytes()
    reference_name = "scientific_recovery_v9_eclock_x0_reference.json"
    reference_path = repo / "configs/protocol" / reference_name
    members[f"reference/{reference_name}"] = reference_path.read_bytes()
    members["final_report.md"] = report.read_bytes()
    provenance_path = campaign / "provenance_exception.json"
    provenance = None
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if not isinstance(provenance, dict) or not verify_artifact_hash(provenance):
            raise ValueError("cross-commit provenance signature mismatch")
    provenance_text = (
        f"Finalization/PAIR commit: `{args.git_commit}`. BASE/DYN training commit: "
        f"`{provenance['source_training_commit']}`. See `provenance_exception.json`."
        if provenance is not None
        else f"Training commit: `{args.git_commit}`."
    )
    members["README.md"] = (
        "# E-Clock X0 seed-7 essential results\n\n"
        f"{provenance_text} Verify every payload member with "
        "`MANIFEST.json` and verify the ZIP with the adjacent `.sha256` file. "
        "Checkpoints are intentionally excluded; "
        "their physical paths, byte counts and hashes are recorded in checkpoint manifests.\n"
    ).encode()
    arm_training_commits = (
        {
            "X0-A5-REPLAY": "external_official_a5_checkpoints",
            "X0-BASE-U": provenance["source_training_commit"],
            "X0-DYN-U": provenance["source_training_commit"],
            "X0-PAIR-U": args.git_commit,
        }
        if provenance is not None
        else {arm: args.git_commit for arm in sorted(ARMS)}
    )
    git_identity = {
        "branch": subprocess.check_output(
            ["git", "-C", str(repo), "branch", "--show-current"], text=True
        ).strip(),
        "starting_head": "af66f2c8ca2017059d7765b5f171e1cda866ab07",
        "training_commit": args.git_commit,
        "training_commit_scope": ["X0-PAIR-U"] if provenance is not None else sorted(ARMS),
        "arm_training_commits": arm_training_commits,
        "cross_commit_reuse_declared": provenance is not None,
        "push_performed": False,
        "tracked_worktree_clean": not bool(
            subprocess.check_output(
                ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=no"],
                text=True,
            ).strip()
        ),
    }
    members["git_identity.json"] = (
        json.dumps(git_identity, indent=2, sort_keys=True) + "\n"
    ).encode()
    members["git_diff_starting_head.txt"] = subprocess.check_output(
        [
            "git",
            "-C",
            str(repo),
            "diff",
            "--stat",
            "af66f2c8ca2017059d7765b5f171e1cda866ab07..HEAD",
        ]
    )
    checkpoint_lines: list[str] = []
    for path in sorted(campaign.glob("X0-*/fold-*/fold_summary.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        checkpoint_hash = value["checkpoint_file_sha256"]
        checkpoint_lines.append(
            f"{checkpoint_hash}  {value['checkpoint_bytes']}  {value['checkpoint_path']}"
        )
    members["CHECKPOINT_SHA256SUMS.txt"] = ("\n".join(checkpoint_lines) + "\n").encode()
    analysis = json.loads((campaign / "analysis.json").read_text(encoding="utf-8"))
    if analysis.get("decision") != "X0_CAMPAIGN_FATAL_INCOMPLETE":
        required = {
            "README.md",
            "final_report.md",
            "environment.json",
            "git_identity.json",
            "cache_engineering/cache_engineering_decision.json",
            "telemetry/summaries/summary.json",
            "smoke/smoke_summary.json",
            "comparisons/x0_dyn_vs_base.json",
            "comparisons/x0_dyn_vs_base_gate.json",
            "CHECKPOINT_SHA256SUMS.txt",
        }
        if provenance is not None:
            required.add("provenance_exception.json")
        for arm in sorted(ARMS):
            required.add(f"runs/{arm}/aggregate.json")
            for fold in (0, 1, 2):
                root = f"runs/{arm}/fold-{fold}"
                required.add(f"{root}/fold_summary.json")
                required.add(f"{root}/oof_predictions.csv")
                if arm != "X0-A5-REPLAY":
                    required.add(f"{root}/progress.jsonl.gz")
        missing = sorted(required - set(members))
        if missing:
            raise ValueError(f"essential bundle is incomplete: {missing}")
    manifest: dict[str, Any] = {
        "schema": "eclock_x0_essential_bundle_manifest_v1",
        "git_commit": args.git_commit,
        "arm_training_commits": arm_training_commits,
        "provenance_exception_sha256": (
            provenance.get("artifact_sha256") if provenance is not None else None
        ),
        "manifest_self_hash_policy": "MANIFEST.json excluded to avoid impossible recursive digest",
        "members": [
            {"path": name, "bytes": len(data), "sha256": _sha(data)}
            for name, data in sorted(members.items())
        ],
    }
    members["MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    args.output_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, data in sorted(members.items()):
            archive.writestr(name, data)
    with zipfile.ZipFile(output, "r") as archive:
        if set(archive.namelist()) != set(members):
            raise ValueError("ZIP member universe changed during packaging")
        for record in manifest["members"]:
            data = archive.read(record["path"])
            if len(data) != record["bytes"] or _sha(data) != record["sha256"]:
                raise ValueError(f"ZIP member verification failed: {record['path']}")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    checksum = output.with_suffix(output.suffix + ".sha256")
    checksum.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    print(json.dumps({"zip": str(output), "sha256": digest, "checksum": str(checksum)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
