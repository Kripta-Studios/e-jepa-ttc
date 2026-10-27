"""Verify downloaded cloud results in isolation; never admit them to the active queue."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

from .inventory import digest, read
from .paths import mapped_path_class
from .runtime import ProjectFinder


class ReturnPathMap:
    """Translate only declared cloud output/input prefixes back to local locations."""

    def __init__(self, migration: dict, downloaded: Path) -> None:
        self.outputs = str(migration["run_root"])
        self.bundle = str(migration["bundle_root"])
        self.downloaded = downloaded.resolve(strict=True)
        self.roots = {relative: original for original, relative in migration["roots"].items()}

    def translate(self, value: str | os.PathLike[str]) -> str:
        raw = os.fspath(value)
        if PureWindowsPath(raw).drive or not raw.startswith("/"):
            return raw
        path = PurePosixPath(raw)
        if ".." in path.parts:
            raise ValueError("Cloud return path has unresolved traversal")
        for prefix, destination in [
            (self.outputs, self.downloaded),
            *(
                (str(PurePosixPath(self.bundle) / relative), Path(original))
                for relative, original in self.roots.items()
            ),
        ]:
            try:
                suffix = path.relative_to(PurePosixPath(prefix))
            except ValueError:
                continue
            return str(destination.joinpath(*suffix.parts))
        raise ValueError(f"Unregistered cloud return path: {raw}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloaded-run", type=Path, required=True)
    parser.add_argument("--rotation", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.downloaded_run.resolve(strict=True)
    live = Path(__file__).resolve().parents[2] / "artifacts/final_refit_20261019_scientific_oof"
    if root == live.resolve() or root.is_relative_to(live.resolve()):
        raise ValueError("Downloads must remain outside the active scientific run")
    migration = read(root / "MIGRATION_BINDING.json")
    if migration.get("schema") != "ejepa_linux_migration_binding_v1":
        raise ValueError("Unknown migration binding")
    if digest(root / "SCIENTIFIC_CONFIG.json") != migration["original_authority_sha256"]:
        raise ValueError("Returned scientific authority differs")
    mapper = ReturnPathMap(migration, root)
    sys.meta_path.insert(0, ProjectFinder(mapped_path_class(mapper)))
    from operational.final_refit.scientific_fold_heads import verify_scientific_admission

    rotation_root = root / "rotations" / str(args.rotation)
    reports = {}
    for path in sorted((rotation_root / "fits").glob("*/ENDPOINT_ADMISSION.json")):
        admission = verify_scientific_admission(path)
        if admission["rotation"] != args.rotation:
            raise ValueError("Mixed rotation outputs")
        reports[path.parent.name] = {
            "admission_sha256": digest(path),
            "checkpoint_sha256": admission["checkpoint_sha256"],
            "updates": admission["updates"],
        }
    if not reports:
        raise ValueError("No completed scientific endpoint was downloaded")
    report = {
        "schema": "ejepa_cloud_return_verification_v1",
        "status": "PASS",
        "downloaded_run": str(root),
        "rotation": args.rotation,
        "migration_binding_sha256": digest(root / "MIGRATION_BINDING.json"),
        "endpoints": reports,
        "active_queue_modified": False,
        "approved_for_automatic_local_consumption": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
