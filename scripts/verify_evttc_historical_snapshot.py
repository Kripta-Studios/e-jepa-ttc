"""Verify original receipts against exact preserved source bytes in a separate snapshot."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import ast
import hashlib
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from operational.evttc_transfer_audit.common import file_sha, read_document, write_receipt
from scripts.verify_evttc_competition_run import audit_run


def main() -> None:
    """Recover only hash-matching historical files and run the existing verifier."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "data-root", "frozen-archive", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    freeze = read_document(args.run / "RUN_FREEZE.json")
    archive_hash = file_sha(args.frozen_archive)
    if archive_hash != freeze["frozen_runtime"]["archive_sha256"]:
        raise ValueError("Preserved archive identity mismatch")
    expected = dict(freeze["executed_source_sha256"])
    for name, checksum in freeze["frozen_runtime"]["source_sha256"].items():
        if name in expected and expected[name] != checksum:
            raise ValueError("Conflicting source identities")
        expected[name] = checksum
    snapshot = (args.output / "source").resolve()
    changes = []
    with tarfile.open(args.frozen_archive) as archive:
        for relative, checksum in expected.items():
            destination = (snapshot / relative).resolve()
            source = (ROOT / relative).resolve()
            if not destination.is_relative_to(snapshot) or not source.is_relative_to(ROOT):
                raise ValueError("Nonlocal source path")
            current = source.read_bytes()
            data = current
            if hashlib.sha256(data).hexdigest() != checksum:
                members = [
                    m
                    for m in archive.getmembers()
                    if m.isfile() and m.name.endswith("/" + relative)
                ]
                if len(members) != 1:
                    raise ValueError(f"Ambiguous or missing historical source: {relative}")
                stream = archive.extractfile(members[0])
                if stream is None:
                    raise ValueError("Missing archive member bytes")
                data = stream.read()
                changes.append(
                    {
                        "path": relative,
                        "current_sha256": hashlib.sha256(current).hexdigest(),
                        "historical_sha256": checksum,
                        "ast_equal": ast.dump(ast.parse(current)) == ast.dump(ast.parse(data)),
                    }
                )
            if hashlib.sha256(data).hexdigest() != checksum:
                raise ValueError(f"Historical source differs from freeze: {relative}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and destination.read_bytes() != data:
                raise ValueError("Refusing to replace a different historical snapshot")
            destination.write_bytes(data)
    result = audit_run(args.run, args.data_root, project_root=snapshot)
    result["source_reconstruction"] = {
        "archive_sha256": archive_hash,
        "differences_from_current": changes,
        "original_worktree_modified": False,
    }
    write_receipt(args.output / "report.json", result)
    print(result["audit_status"], result["rows_verified"])


if __name__ == "__main__":
    main()
