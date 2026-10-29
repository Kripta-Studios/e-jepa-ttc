"""Regenerate and validate the public PHASE-TTC manuscript in CI."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
import sys
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, cast

REPOSITORY = Path(__file__).resolve().parents[1]
PAPER = REPOSITORY / "paper"
PUBLIC_MANIFEST = PAPER / "data" / "public_manifest.json"
SOURCE_MANIFEST_NAME = "SOURCE_MANIFEST.json"
SOURCE_ZIP_NAME = "PHASE_TTC_Final_Complete_Source.zip"
SOURCE_ZIP_PREFIX = "PHASE_TTC_Final_Source"
EXPECTED_PUBLIC_INPUTS = 13


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return cast(dict[str, Any], value)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_manifest_path(raw: object) -> PurePosixPath:
    if not isinstance(raw, str) or not raw:
        raise TypeError("Source-manifest paths must be nonempty strings")
    relative = PurePosixPath(raw)
    if (
        relative.is_absolute()
        or raw != relative.as_posix()
        or any(part in {"", ".", ".."} for part in relative.parts)
        or any(":" in part or "\x00" in part for part in relative.parts)
        or "\\" in raw
    ):
        raise ValueError(f"Unsafe or noncanonical source-manifest path: {raw!r}")
    return relative


def verify_source_package(paper: Path = PAPER) -> dict[str, int]:
    """Verify that the committed source tree, manifest, and ZIP have identical bytes."""
    manifest_path = paper / SOURCE_MANIFEST_NAME
    manifest_bytes = manifest_path.read_bytes()
    manifest = _read_object(manifest_path)
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("The source manifest must contain at least one file")

    declared: dict[str, tuple[int, str, bytes]] = {}
    casefolded: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise TypeError("Each source-manifest entry must be an object")
        relative = _safe_manifest_path(entry.get("path"))
        name = relative.as_posix()
        size = entry.get("bytes")
        expected_sha256 = entry.get("sha256")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise TypeError(f"Invalid byte count for source-manifest entry: {name}")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise TypeError(f"Invalid SHA-256 for source-manifest entry: {name}")
        if name in declared or name.casefold() in casefolded:
            raise ValueError(f"Duplicate source-manifest path: {name}")
        if name in {SOURCE_MANIFEST_NAME, SOURCE_ZIP_NAME}:
            raise ValueError(f"Self-referential source-manifest entry: {name}")

        path = paper.joinpath(*relative.parts)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or non-regular source file: {name}")
        data = path.read_bytes()
        if len(data) != size or hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ValueError(f"Source file differs from its manifest: {name}")
        declared[name] = (size, expected_sha256, data)
        casefolded.add(name.casefold())

    zip_path = paper / SOURCE_ZIP_NAME
    if zip_path.is_symlink() or not zip_path.is_file():
        raise ValueError(f"Missing source archive: {SOURCE_ZIP_NAME}")
    expected_members = {f"{SOURCE_ZIP_PREFIX}/{name}" for name in declared} | {
        f"{SOURCE_ZIP_PREFIX}/{SOURCE_MANIFEST_NAME}"
    }

    with zipfile.ZipFile(zip_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("The source archive contains duplicate members")
        if set(names) != expected_members:
            missing = sorted(expected_members - set(names))
            unexpected = sorted(set(names) - expected_members)
            raise ValueError(
                f"Source archive membership differs from the manifest; "
                f"missing={missing}, unexpected={unexpected}"
            )

        for info in infos:
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Non-regular source archive member: {info.filename}")
            relative_name = info.filename.removeprefix(f"{SOURCE_ZIP_PREFIX}/")
            data = archive.read(info)
            if relative_name == SOURCE_MANIFEST_NAME:
                if data != manifest_bytes:
                    raise ValueError(
                        "The archived source manifest differs from the committed bytes"
                    )
                continue
            size, expected_sha256, local_data = declared[relative_name]
            if (
                info.file_size != size
                or data != local_data
                or hashlib.sha256(data).hexdigest() != expected_sha256
            ):
                raise ValueError(
                    f"Archived source differs from the committed bytes: {relative_name}"
                )

    return {"manifest_files": len(declared), "zip_members": len(expected_members)}


def _public_input_hashes() -> dict[str, str]:
    manifest = _read_object(PUBLIC_MANIFEST)
    entries = manifest.get("files")
    if not isinstance(entries, list) or len(entries) != EXPECTED_PUBLIC_INPUTS:
        raise ValueError(
            f"Expected {EXPECTED_PUBLIC_INPUTS} immutable public inputs, got "
            f"{len(entries) if isinstance(entries, list) else 'an invalid manifest'}"
        )

    observed: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise TypeError("Each public-manifest entry must be an object")
        relative = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise TypeError("Public-manifest path and sha256 fields must be strings")
        path = PAPER / relative
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"Public evidence hash mismatch: {relative}")
        observed[relative] = actual
    return observed


def _all_zero(checks: object) -> bool:
    return isinstance(checks, Mapping) and all(value == 0 for value in checks.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-package-only",
        action="store_true",
        help="Verify the source manifest and ZIP without regenerating the manuscript.",
    )
    args = parser.parse_args()

    package = verify_source_package()
    if args.verify_package_only:
        print(json.dumps(package, indent=2))
        return

    before = _public_input_hashes()
    subprocess.run(
        [sys.executable, "scripts/build_final.py"],
        cwd=PAPER,
        check=True,
    )
    after = _public_input_hashes()
    if after != before:
        raise RuntimeError("The manuscript build modified immutable public evidence")

    # Run the numerical validator again so the post-generation state is checked too.
    subprocess.run(
        [sys.executable, "scripts/verify_public_data.py"],
        cwd=PAPER,
        check=True,
    )

    receipt = _read_object(PAPER / "revision" / "build_receipt.json")
    source_validation = receipt.get("source_validation")
    if not isinstance(source_validation, Mapping):
        raise ValueError("The build receipt has no source-validation record")
    if source_validation.get("public_hashes_verified") != EXPECTED_PUBLIC_INPUTS:
        raise ValueError("The build receipt did not verify all public inputs")
    if not _all_zero(receipt.get("modular_log_checks")):
        raise ValueError("The modular manuscript has unresolved TeX warnings")
    if not _all_zero(receipt.get("delivered_log_checks")):
        raise ValueError("The consolidated manuscript has unresolved TeX warnings")

    competition = _read_object(PAPER / "revision" / "competition_validation.json")
    for field in ("prediction_rows", "query_identities", "sequence_variant_cells", "stats"):
        if field not in competition:
            raise ValueError(f"Competition validation is missing {field}")

    pdf = PAPER / "PHASE_TTC_Final.pdf"
    if not pdf.is_file() or pdf.stat().st_size == 0:
        raise ValueError("The consolidated manuscript PDF was not produced")

    print(
        json.dumps(
            {
                "source_package": package,
                "immutable_public_inputs": len(after),
                "pdf_bytes": pdf.stat().st_size,
                "pages": receipt.get("pages"),
                "numbered_figures": receipt.get("numbered_figures"),
                "numbered_tables": receipt.get("numbered_tables"),
                "tex_warning_checks": {
                    "modular": receipt["modular_log_checks"],
                    "consolidated": receipt["delivered_log_checks"],
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
