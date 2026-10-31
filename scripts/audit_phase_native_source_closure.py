"""Supplement a running native cache with frozen numerical dependency byte proofs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
import tokenize
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Numerical producers, model construction, pair/context/head features and sensor
# preparation dependencies supplement the running builder's direct-source manifest.
DEPENDENCIES = (
    "src/e_jepa_ttc/models/causal_scale_ttc.py",
    "src/e_jepa_ttc/models/local_transport.py",
    "src/e_jepa_ttc/models/collision_clock_math.py",
    "src/e_jepa_ttc/rgb_port/features.py",
    "src/e_jepa_ttc/rgb_port/normalization.py",
    "src/e_jepa_ttc/rgb_port/data.py",
    "src/e_jepa_ttc/simplex_t/model.py",
    "src/e_jepa_ttc/simplex_t/phase.py",
    "src/e_jepa_ttc/training/stage61_pair_head.py",
    "src/e_jepa_ttc/data/stage61_pair_feature_cache.py",
    "src/e_jepa_ttc/data/event_v4_geometry.py",
    "operational/final_refit/test12.py",
    "operational/final_refit/training.py",
    "operational/final_refit/features.py",
    "operational/final_refit/recipe.py",
    "operational/rgb_port/train_producers.py",
    "operational/rgb_port/train_heads.py",
    "operational/rgb_port/transfer.py",
    "operational/rgb_port/infer_experts.py",
    "operational/rgb_port_revision/transfer_inputs.py",
    "operational/rgb_port_streaming/contracts.py",
    "operational/evttc_transfer/inputs.py",
    "operational/evttc_rgb_transfer/inputs.py",
)


def sha(path: Path) -> str:
    """Hash bytes with bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict[str, Any]:
    """Read an explicitly selected evidence receipt."""
    return json.loads(path.read_text(encoding="utf8"))


def write_new(path: Path, value: dict[str, Any]) -> None:
    """Create supplementary evidence without modifying any original extraction file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def archive_source_hashes(archive: Path, expected_sha: str, names: list[str]) -> dict[str, str]:
    """Bind unique regular source members to a fully verified archive, without extraction."""
    if sha(archive) != expected_sha:
        raise ValueError("Frozen archive hash differs")
    hashes = {}
    with tarfile.open(archive) as source:
        members = source.getmembers()
        for name in names:
            matches = [m for m in members if m.isfile() and m.name.endswith("/" + name)]
            if len(matches) != 1:
                raise ValueError(f"Expected unique archived source: {name}")
            stream = source.extractfile(matches[0])
            if stream is None:
                raise ValueError("Archived source has no readable bytes")
            digest = hashlib.sha256()
            while block := stream.read(1 << 20):
                digest.update(block)
            hashes[name] = digest.hexdigest()
    return hashes


def formatting_equivalent(original: str, current: str) -> bool:
    """Accept line wrapping only when both AST and non-layout lexical tokens agree."""
    if ast.dump(ast.parse(original), include_attributes=False) != ast.dump(
        ast.parse(current), include_attributes=False
    ):
        return False
    excluded = {tokenize.NL, tokenize.NEWLINE, tokenize.ENDMARKER, tokenize.ENCODING}

    def tokens(text: str) -> list[tuple[int, str]]:
        return [
            (token.type, token.string)
            for token in tokenize.generate_tokens(io.StringIO(text).readline)
            if token.type not in excluded
        ]

    return tokens(original) == tokens(current)


def archive_equivalence(archive: Path, original: dict[str, str], current: dict[str, str]) -> dict:
    """Retain both byte identities and explicitly prove any formatting exception."""
    results = {}
    with tarfile.open(archive) as source:
        members = source.getmembers()
        for name in DEPENDENCIES:
            kind = "byte_identical"
            if original[name] != current[name]:
                match = [m for m in members if m.isfile() and m.name.endswith("/" + name)]
                if len(match) != 1:
                    raise ValueError("Missing or ambiguous equivalence source")
                stream = source.extractfile(match[0])
                if stream is None or not formatting_equivalent(
                    stream.read().decode("utf8"), Path(name).read_text(encoding="utf8")
                ):
                    raise ValueError(f"Numerical source differs beyond formatting: {name}")
                kind = "ast_and_lexical_tokens_identical_line_wrapping_only"
            results[name] = {
                "current_sha256": current[name],
                "frozen_sha256": original[name],
                "equivalence": kind,
            }
    return results


def verify_snapshot(root: Path, expected: dict[str, str]) -> None:
    """Require the exact declared inventory and current/snapshotted byte equality."""
    if set(expected) != set(DEPENDENCIES):
        raise ValueError("Supplemental dependency inventory differs")
    for name, digest in expected.items():
        if sha(Path(name)) != digest or sha(root / "executed_source" / name) != digest:
            raise ValueError(f"Dependency changed since snapshot: {name}")


def snapshot(extraction: Path, preservation: Path, output: Path) -> None:
    """Snapshot during extraction and prove exact equality with preserved cloud sources."""
    if output.exists():
        raise ValueError("Use a fresh supplemental output directory")
    freeze = read(extraction / "FREEZE.json")
    entry = read(preservation)["groups"]["frozen_code"]
    archive = Path(entry["archive"])
    current = {name: sha(Path(name)) for name in DEPENDENCIES}
    # Capture current bytes before potentially slower archive reads.
    for name, digest in current.items():
        destination = output / "executed_source" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, destination)
        if sha(destination) != digest:
            raise ValueError("Source changed during snapshot")
    archived = archive_source_hashes(archive, entry["sha256"], list(DEPENDENCIES))
    mismatches = {
        name: {"current": current[name], "archived": archived[name]}
        for name in DEPENDENCIES
        if current[name] != archived[name]
    }
    git_status = subprocess.run(
        ["git", "status", "--porcelain", "--", *DEPENDENCIES],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    receipt = {
        "schema": "phase_native_source_closure_snapshot_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "extraction_root": str(extraction.resolve()),
        "extraction_freeze_sha256": sha(extraction / "FREEZE.json"),
        "extraction_complete_at_snapshot": (extraction / "COMPLETE.json").exists(),
        "completed_query_count_at_snapshot": len(
            list((extraction / "queries").glob("*/COMPLETE.json"))
        ),
        "preservation_receipt": str(preservation.resolve()),
        "preservation_sha256": sha(preservation),
        "frozen_archive": str(archive.resolve()),
        "frozen_archive_sha256": entry["sha256"],
        "dependency_sha256": current,
        "archived_dependency_sha256": archived,
        "frozen_sources_match": not mismatches,
        "mismatches": mismatches,
        "git_status_declared_dependencies": git_status,
        "builder_source_sha256": freeze["source_hashes"],
        "scope": "Declared numerical execution dependencies plus original builder inventory; "
        "not all transitively imported training/CLI/analysis modules or third-party packages",
        "chronology": "Supplemental observation during extraction, not a retroactive pre-run seal. "
        "Historical frozen-archive equality and final unchanged bytes support provenance; no claim "
        "of continuous filesystem monitoring or process-memory attestation.",
        "labels_opened": False,
        "producer_calls": 0,
        "audit_source_sha256": sha(Path(__file__)),
    }
    write_new(output / "SNAPSHOT.json", receipt)
    for path in (Path(__file__), Path("tests/unit/test_phase_native_source_closure.py")):
        destination = output / "audit_source" / path.resolve().relative_to(Path.cwd())
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    if mismatches:
        raise ValueError(f"Frozen numerical dependencies differ: {list(mismatches)}")
    verify_snapshot(output, current)
    print(f"SNAPSHOT PASS {len(current)} frozen dependencies; no producer execution")


def finalize(extraction: Path, output: Path) -> None:
    """Bind unchanged source evidence to the completed149/447 extraction and all hashes."""
    snapshot_path = output / "SNAPSHOT.json"
    initial = read(snapshot_path)
    if (
        initial["extraction_root"] != str(extraction.resolve())
        or initial["extraction_freeze_sha256"] != sha(extraction / "FREEZE.json")
        or initial["audit_source_sha256"]
        != sha(output / "audit_source/scripts/audit_phase_native_source_closure.py")
    ):
        raise ValueError("Supplemental snapshot identity/code differs")
    verify_snapshot(output, initial["dependency_sha256"])
    if sha(Path(initial["preservation_receipt"])) != initial["preservation_sha256"]:
        raise ValueError("Preservation receipt changed")
    archived = archive_source_hashes(
        Path(initial["frozen_archive"]), initial["frozen_archive_sha256"], list(DEPENDENCIES)
    )
    equivalence = archive_equivalence(
        Path(initial["frozen_archive"]), archived, initial["dependency_sha256"]
    )
    complete = read(extraction / "COMPLETE.json")
    if (
        complete.get("status") != "COMPLETE"
        or complete.get("queries") != 149
        or complete.get("augmentations") != 447
        or complete.get("clean_exact") is not True
        or complete.get("disk_roundtrip_exact") is not True
        or complete.get("targets_read") is not False
        or complete.get("freeze_sha256") != sha(extraction / "FREEZE.json")
    ):
        raise ValueError("Expanded extraction is not the complete clean target-blind149/447 run")
    for relative, digest in complete["artifacts"].items():
        path = (extraction / relative).resolve(strict=True)
        if not path.is_relative_to(extraction.resolve()) or sha(path) != digest:
            raise ValueError("Expanded extraction artifact changed or escapes root")
    for name, digest in initial["builder_source_sha256"].items():
        if sha(Path(name)) != digest or sha(extraction / "executed_source" / name) != digest:
            raise ValueError("Original builder source binding differs")
    receipt = {
        "schema": "phase_native_source_closure_v1",
        "status": "PASS",
        "created_utc": datetime.now(UTC).isoformat(),
        "snapshot_sha256": sha(snapshot_path),
        "extraction_root": str(extraction.resolve()),
        "expanded_complete_sha256": sha(extraction / "COMPLETE.json"),
        "expanded_freeze_sha256": sha(extraction / "FREEZE.json"),
        "dependency_sha256": initial["dependency_sha256"],
        "frozen_archive_sha256": initial["frozen_archive_sha256"],
        "preservation_sha256": initial["preservation_sha256"],
        "audit_source_sha256": sha(Path(__file__)),
        "final_audit_source_sha256": {
            str(path): sha(path)
            for path in (Path(__file__), Path("tests/unit/test_phase_native_source_closure.py"))
        },
        "snapshot_to_completion_unchanged": True,
        "frozen_numerical_sources_match": True,
        "all_frozen_bytes_match": archived == initial["dependency_sha256"],
        "dependency_equivalence": equivalence,
        "initial_strict_byte_check_pass": initial["frozen_sources_match"],
        "original_artifacts_verified": len(complete["artifacts"]),
        "labels_opened": False,
        "producer_calls": 0,
        "chronology": initial["chronology"],
    }
    write_new(output / "SOURCE_CLOSURE.json", receipt)
    for path in (Path(__file__), Path("tests/unit/test_phase_native_source_closure.py")):
        destination = output / "final_audit_source" / path.resolve().relative_to(Path.cwd())
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    print("SOURCE CLOSURE PASS; original extraction untouched; trainer may bind this receipt")


def main() -> None:
    """Snapshot while extraction runs, then finalize only after its immutable completion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--preservation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("snapshot", "finalize"), required=True)
    args = parser.parse_args()
    if args.phase == "snapshot":
        if args.preservation is None:
            raise ValueError("Snapshot requires preservation receipt")
        snapshot(args.extraction, args.preservation, args.output)
    else:
        finalize(args.extraction, args.output)


if __name__ == "__main__":
    main()
