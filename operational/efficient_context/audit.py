"""Bounded recovery of comparator metadata and local resource/role interfaces."""

import subprocess
import sys
from pathlib import Path

from .common import ROOT, Campaign, atomic_bytes, atomic_json, digest, read


def audit(c: Campaign) -> None:
    """Inspect only manifest-addressed sources; keep protected evaluations unopened."""
    import psutil

    processes = []
    for process in psutil.process_iter(["pid", "ppid", "name", "cmdline", "memory_info"]):
        info = process.info
        if "python" in (info["name"] or "").lower():
            processes.append(
                {
                    "pid": info["pid"],
                    "ppid": info["ppid"],
                    "command": info["cmdline"],
                    "rss": info["memory_info"].rss if info["memory_info"] else None,
                }
            )
    ack_path = Path(c.local["shared_coordination"]) / "SIMPLEX_T_STAGE70_ACK.json"
    ack = read(ack_path)
    bindings = []
    for key in ("role_manifest", "time_charter"):
        row = ack["interfaces"][key]
        path = Path(row["path"])
        ok = digest(path) == row["sha256"]
        if not ok:
            raise ValueError("Stage70 interface hash changed: " + str(path))
        bindings.append({"kind": key, "path": str(path), "sha256": row["sha256"], "verified": ok})
    atomic_json(
        c.out / "LOCAL_STATE.json",
        {
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "processes": processes,
            "Stage70_interfaces": bindings,
            "executable": sys.executable,
            "protected_payloads_opened": False,
        },
    )
    atomic_bytes(
        c.out / "STAGE70_INTERFACE_STATUS.md",
        (
            b"Stage70 role/time interfaces verified against the existing ACK.\n"
            b"No Stage76 payload or confirmation score opened. "
            b"Active processes are recorded in LOCAL_STATE.json.\n"
            b"Absence of a PID does not establish completion of the owner's campaign.\n"
        ),
    )
    review_path = (
        c.historical
        / "artifacts/simplex_t/post_campaign_20261003/COMPARATOR_REVIEW_RECONCILED.json"
    )
    review = read(review_path)
    for row in review["comparisons"]:
        verified = []
        for value, sha in zip(row["evidence_paths"], row["evidence_sha256"], strict=True):
            path = Path(value)
            verified.append(
                {
                    "path": value,
                    "expected_sha256": sha,
                    "exists": path.exists(),
                    "verified": path.exists() and digest(path) == sha,
                }
            )
        row["current_verification"] = verified
    atomic_json(c.out / "COMPARATOR_CONTRACTS.json", review)
    code = Path(c.local["garl_code_candidate"])
    upstream = {"path": str(code), "exists": code.exists()}
    if code.exists():
        upstream["commit"] = subprocess.check_output(
            ["git", "-C", str(code), "rev-parse", "HEAD"], text=True
        ).strip()
        upstream["config_blob"] = subprocess.check_output(
            ["git", "-C", str(code), "hash-object", "configs/ablation/event_lhr.yaml"], text=True
        ).strip()
        upstream["verified"] = (
            upstream["commit"] == c.config["garl_upstream_commit"]
            and upstream["config_blob"] == c.config["garl_config_blob"]
        )
    # This audit is a dependency record; the producer branch must resolve native
    # sample coverage, permitted labels and exact50-epoch budgets before training.
    atomic_json(
        c.out / "GARL_SOURCE_AUDIT.json",
        {
            "upstream": upstream,
            "train_annotations": str(c.local["garl_annotations_candidate"]),
            "raw_train_root": str(c.raw),
            "status": "NATIVE_SAMPLE_AND_PRODUCER_ADMISSION_PENDING",
            "optimizer_updates": 0,
        },
    )
    assets = read(ROOT / "docs/efficient_context_handoff_20261004/RELEASE_ASSETS.json")
    candidates = [
        c.historical / "artifacts/simplex_t/deliveries",
        c.historical / "artifacts/simplex_t/github_release_complete_20261004",
        c.historical / "artifacts/simplex_t/github_release_20261004",
    ]
    for row in assets["archives"]:
        paths = [p / row["filename"] for p in candidates if (p / row["filename"]).exists()]
        row["local_paths"] = [str(v) for v in paths]
        row["local_verification"] = [
            {"path": str(v), "sha256": digest(v), "matches": digest(v) == row["sha256"]}
            for v in paths
        ]
    atomic_json(c.out / "RELEASE_ASSET_BINDINGS.json", assets)
    print("E0_LOCAL_AND_COMPARATOR_CONTRACTS_AUDITED_ZERO_UPDATES", flush=True)
