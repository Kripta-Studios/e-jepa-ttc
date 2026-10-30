"""Build a local, hash-bound release-research transfer kit; never deploy or train."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any


def digest(path: Path) -> str:
    """Stream a file digest with bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def relative(path: Path, roots: dict[str, str]) -> str:
    """Map only explicitly registered roots, rejecting traversal or unknown roots."""
    source = PureWindowsPath(str(path))
    if ".." in source.parts:
        raise ValueError("Unresolved path traversal")
    for root, target in sorted(roots.items(), key=lambda item: len(item[0]), reverse=True):
        try:
            suffix = source.relative_to(PureWindowsPath(root))
        except ValueError:
            continue
        return "/".join((target, *suffix.parts))
    raise ValueError(f"Path outside transfer roots: {path}")


def build(repo: Path, output: Path, pause: Path) -> None:
    """Snapshot sources and private inputs, inventory all forty public train sequences."""
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    roots = {
        str(repo.parent): "workspace",
        "E:\\": "drives/E",
        "D:\\PHASE_TTC_DETECTION_CAMPAIGN_20261029": "campaign",
        str(Path.home() / ".cache/huggingface/hub"): "teacher_weights",
    }
    files: dict[str, dict[str, Any]] = {}
    pending: list[Path] = []
    unresolved: set[str] = set()
    excluded_test: set[str] = set()

    def add(path: Path, purpose: str, expected: str | None = None) -> None:
        path = path.resolve(strict=True)
        target = relative(path, roots)
        lowered = target.lower()
        if any(
            part in lowered for part in ("/data/test", "/annotations/test", "ejepa_garlttc_test12")
        ):
            excluded_test.add(str(path))
            return
        if target in files:
            if expected and files[target]["sha256"] != expected:
                raise ValueError(f"Conflicting expected digest: {path}")
            return
        if path.suffix.lower() in {".py", ".json", ".yaml", ".yml", ".toml", ".sh"}:
            if path.stat().st_size < 10_000_000:
                text = path.read_text(encoding="utf-8-sig", errors="replace")
                if re.search(r"(?:hf_[A-Za-z0-9]{25,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)", text):
                    raise ValueError(f"Secret-like content requires inspection: {path}")
        actual = digest(path)
        if expected and actual != expected:
            raise ValueError(f"Pinned input changed: {path}")
        files[target] = {
            "source": str(path),
            "target": target,
            "bytes": path.stat().st_size,
            "sha256": actual,
            "kind": "private",
            "purpose": purpose,
        }
        if path.suffix == ".json" and path.stat().st_size < 20_000_000:
            pending.append(path)

    def tree(path: Path, purpose: str) -> None:
        for item in sorted(path.rglob("*")):
            if item.is_file() and not any(
                p in {"__pycache__", ".git", ".venv"} for p in item.parts
            ):
                add(item, purpose)

    # Public metadata and sensor files only from the explicitly separated train roles.
    public = read(repo / "data/manifests/eap_detection_public_files_v1.json")
    split = read(repo / "data/splits/eap_detection_candidate_v1.json")
    roles = {seq: role for role, sequences in split["assignments"].items() for seq in sequences}
    if len(roles) != 40 or public["revision"] != split["dataset_revision"]:
        raise ValueError("Expected the pinned forty-sequence role partition")
    for name, entry in public["files"].items():
        if name != "data/train.parquet" and not name.startswith("data/train/"):
            continue
        sequence = name.split("/")[2] if name.startswith("data/train/") else None
        if sequence is not None and sequence not in roles:
            raise ValueError(f"Unassigned sequence: {sequence}")
        path = Path("E:/eAP_dataset") / name
        if not path.is_file() or path.stat().st_size != entry["size_bytes"]:
            raise FileNotFoundError(f"Missing or wrong-size public asset: {path}")
        target = relative(path, roots)
        modality = (
            "events" if name.endswith(".h5") else "rgb" if name.endswith(".tar") else "metadata"
        )
        if not entry["sha256"]:
            raise ValueError(f"Public payload lacks SHA-256: {name}")
        files[target] = {
            "source": str(path),
            "target": target,
            "bytes": entry["size_bytes"],
            "sha256": entry["sha256"],
            "kind": "public",
            "modality": modality,
            "role": roles.get(sequence, "role_index"),
            "sequence_id": sequence,
            "repository": public["repository"],
            "revision": public["revision"],
            "repository_path": name,
            "url": f"https://huggingface.co/datasets/{public['repository']}/resolve/{public['revision']}/{name}",
            "local_verification": "EXISTS_AND_SIZE_MATCH; POD_MUST_VERIFY_SHA256",
        }
    for folder in ("src", "operational", "configs", "data/manifests", "data/splits"):
        tree(repo / folder, "source_or_protocol")
    for path in (repo / "scripts").glob("*.py"):
        if any(
            word in path.name for word in ("detection", "monodgp", "independent", "runpod_release")
        ):
            add(path, "source")
    for name in ("pyproject.toml", "uv.lock", "README.md", "AGENTS.md", "LICENSE"):
        add(repo / name, "source")
    tree(repo / "docs/detection_sota_20261029", "plan_and_evidence")
    tree(repo / "output/campaign_data", "data_admission_and_population")
    for name in (
        "train24_population_v2",
        "raw_history_index_v4",
        "raw_history_h5_inventory_v2",
        "raw_history_h5_metadata_preflight_v4",
        "raw_history_h5_admission_v1",
        "teacher_crosswalk_v3",
        "train24_teacher_cache_v2",
        "foundation_smoke_v2",
        "e_a5_rotation0_source_admission_v3",
        "e_a5_rotation0_cpu_engineering_smoke_v2",
        "r_a5_rotation0_rgb_source_v2",
        "producer_er_source_snapshot_v2",
    ):
        tree(repo / "output/independent_phase" / name, "independent_phase")
    producer = repo / "output/detection_campaign/rgb_screen_v5_seed7_v1/producer"
    tree(producer, "frozen_detector_producer")
    backbone = (
        repo.parent / "phase-ttc-detection-local/output/external_weights/resnet50-0676ba61.pth"
    )
    add(
        backbone,
        "external_initialization",
        "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a",
    )
    contract = read(
        repo / "output/independent_phase/e_a5_rotation0_source_admission_v3/RUN_CONTRACT.json"
    )
    for item in contract["parents"].values():
        add(Path(item["path"]), "independent_parent", item["sha256"])
    input_manifest = read(Path(contract["parents"]["input_manifest"]["path"]))
    cache = Path(input_manifest["directory"])
    for item in input_manifest["ordered_shards"]:
        add(cache / item["name"], "row_local_input_cache", item["sha256"])
    for path in cache.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".parquet"}:
            add(path, "input_cache_metadata")
    teacher = read(repo / "output/independent_phase/train24_teacher_cache_v2/TEACHER_PARENT.json")
    add(Path(teacher["weights_path"]), "external_frozen_teacher", teacher["weights_file_sha256"])
    state = read(pause / "status_before_pause.json")
    for name, record in state["completed"].items():
        run = Path(record["summary_path"]).parent
        add(run / "summary.json", "completed_result", record["summary_sha256"])
        for cp in ("last.pt", "selected.pt"):
            add(run / "checkpoints" / cp, f"completed_{name}_checkpoint")
        for path in run.glob("*.json"):
            add(path, "completed_provenance")
        add(run / "metrics.jsonl", "completed_curve")
    tree(pause, "pause_boundary")
    for item in read(pause / "CHECKPOINTS.json").values():
        add(Path(item["path"]), "paused_checkpoint", item["sha256"])
    qroot = pause.parent
    for q in (qroot, Path(read(qroot / "plan.json")["base_plan"]["path"]).parent):
        for path in q.glob("*.json"):
            add(path, "queue_provenance")
        if (q / "configs").is_dir():
            tree(q / "configs", "queue_configs")

    # Close explicit metadata file references, without following arbitrary directories.
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        data = read(path)
        stack = [data]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, list):
                stack.extend(value)
            elif isinstance(value, str) and len(value) < 1000:
                if PureWindowsPath(value).is_absolute():
                    candidate = Path(value)
                    if candidate.is_file():
                        try:
                            relative(candidate.resolve(), roots)
                        except ValueError:
                            unresolved.add(value)
                            continue
                        add(candidate, "referenced_metadata_or_parent")
                    elif candidate.suffix and not candidate.is_dir():
                        unresolved.add(value)
                elif value.endswith((".json", ".parquet")):
                    candidate = path.parent / value
                    if candidate.is_file():
                        add(candidate, "relative_metadata_reference")
    manifest = {
        "schema": "phase_ttc_release_transfer_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "roots": roots,
        "workspace": relative(repo, roots),
        "files": list(files.values()),
        "roles": split["assignments"],
        "test_data_included": False,
        "unresolved_historical_references": sorted(unresolved),
        "excluded_test_references": sorted(excluded_test),
        "release_model_selected": False,
        "cloud_gpu_execution_validated": False,
    }
    write(output / "MANIFEST.json", manifest)
    print(
        json.dumps(
            {
                "phase": "INVENTORY_COMPLETE",
                "files": len(files),
                "bytes_by_kind": dict(
                    Counter(
                        {
                            kind: sum(x["bytes"] for x in files.values() if x["kind"] == kind)
                            for kind in ("private", "public")
                        }
                    )
                ),
            }
        ),
        flush=True,
    )
    private = [row for row in files.values() if row["kind"] == "private"]
    if shutil.disk_usage(output).free < sum(row["bytes"] for row in private) + 2 * 2**30:
        raise RuntimeError("Not enough free disk to archive private payloads")
    parts = []
    batches: list[list[dict[str, Any]]] = [[]]
    size = 0
    for row in sorted(private, key=lambda row: row["target"]):
        if batches[-1] and size + row["bytes"] > 4 * 2**30:
            batches.append([])
            size = 0
        batches[-1].append(row)
        size += row["bytes"]
    for i, batch in enumerate(batches):
        dest = output / f"private_{i:03d}.tar"
        with tarfile.open(dest, "w") as archive:
            for row in batch:
                archive.add(row["source"], arcname=row["target"], recursive=False)
        parts.append({"name": dest.name, "bytes": dest.stat().st_size, "sha256": digest(dest)})
        write(output / "PARTS.json", parts)
        print(
            json.dumps({"phase": "ARCHIVED", "part": dest.name, "bytes": dest.stat().st_size}),
            flush=True,
        )
    write(
        output / "BUILD_RECEIPT.json",
        {
            "status": "TRANSFER_KIT_BUILT_NOT_CLOUD_EXECUTED",
            "manifest_sha256": digest(output / "MANIFEST.json"),
            "parts": parts,
            "files": len(files),
            "private_bytes": sum(x["bytes"] for x in private),
            "public_bytes": sum(x["bytes"] for x in files.values() if x["kind"] == "public"),
            "public_hashes": "PINNED_RELEASE;VERIFY_AT_DESTINATION",
            "optimizer_updates": 0,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pause", type=Path, required=True)
    args = parser.parse_args()
    build(args.repo.resolve(), args.output.resolve(), args.pause.resolve())


if __name__ == "__main__":
    main()
