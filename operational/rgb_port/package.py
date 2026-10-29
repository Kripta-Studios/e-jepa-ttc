"""Build and independently verify the local RGB-PORT essential bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from .accounting import (
    atomic_write_bytes,
    atomic_write_json,
    durable_replace_file,
    read_bytes_shared,
    read_json_shared,
    sha256_file,
)

FORBIDDEN_PARTS = {"private", "sealed", "stage76", "test_labels", "raw_data"}
FIXED_TIME = (1980, 1, 1, 0, 0, 0)
MODEL_IDS = (
    "E_A5_MATCHED",
    "E_C2F_MATCHED",
    "R_A5",
    "R_C2F",
    "PAIR_E_MATCHED",
    "PAIR_R",
    "E_H1_MATCHED",
    "E_CTX_MATCHED",
    "R_H1",
    "R_CTX",
    "F_TRUE",
    "F_ZERO",
)


def _safe_member(relative: str) -> PurePosixPath:
    member = PurePosixPath(relative.replace("\\", "/"))
    if (
        member.is_absolute()
        or ".." in member.parts
        or any(part.lower() in FORBIDDEN_PARTS for part in member.parts)
    ):
        raise ValueError(f"Unsafe or prohibited bundle member: {relative}")
    return member


def campaign_status(run_root: Path) -> str:
    state_path = run_root / "RGB_PORT_STATE.json"
    if not state_path.exists():
        return "INCOMPLETE"
    state = read_json_shared(state_path)
    tasks = state.get("tasks", {})
    return (
        "COMPLETE"
        if tasks and all(item.get("status") == "COMPLETE" for item in tasks.values())
        else "INCOMPLETE"
    )


def build_bundle(
    run_root: Path,
    output: Path,
    members: list[str],
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    selected: list[tuple[PurePosixPath, Path]] = []
    missing_optional: list[str] = []
    for name in members:
        optional = name.startswith("optional:")
        if optional:
            name = name.removeprefix("optional:")
        if name.startswith("repo-tree:"):
            if repository_root is None:
                raise ValueError("Repository tree member requires repository_root")
            relative = _safe_member(name.removeprefix("repo-tree:"))
            directory = repository_root / Path(*relative.parts)
            if not directory.is_dir():
                raise FileNotFoundError(f"Required repository tree is absent: {directory}")
            for source in sorted(path for path in directory.rglob("*") if path.is_file()):
                if "__pycache__" in source.parts or source.suffix.lower() == ".pyc":
                    continue
                child = source.relative_to(repository_root).as_posix()
                selected.append((_safe_member(f"source/{child}"), source))
            continue
        if name.startswith("repo:"):
            if repository_root is None:
                raise ValueError("Repository bundle member requires repository_root")
            relative = _safe_member(name.removeprefix("repo:"))
            member = _safe_member(f"source/{relative}")
            source = repository_root / Path(*relative.parts)
        else:
            member = _safe_member(name)
            source = run_root / Path(*member.parts)
        if not source.is_file():
            if optional:
                missing_optional.append(str(member))
                continue
            raise FileNotFoundError(f"Required bundle member is absent: {source}")
        selected.append((member, source))
    selected_names = [str(member) for member, _ in selected]
    if len(selected_names) != len(set(selected_names)):
        raise ValueError("Bundle member specification expands to duplicate archive names")
    hashes = {str(member): sha256_file(source) for member, source in selected}
    status = campaign_status(run_root)
    manifest = {
        "schema": "rgb_port_essential_bundle_v1",
        "campaign_status": status,
        "payload_members": hashes,
        "missing_optional_members": sorted(missing_optional),
        "contains_full_model_states": all(
            any(name.endswith(f"fits/{fit_id}/checkpoint_last.pt") for name in hashes)
            for fit_id in MODEL_IDS
        ),
    }
    if status == "COMPLETE" and not manifest["contains_full_model_states"]:
        raise ValueError("A complete bundle must embed all twelve full model states")
    if status == "COMPLETE" and missing_optional:
        raise ValueError(
            "A complete campaign bundle cannot omit generated payload members: "
            f"{sorted(missing_optional)}"
        )
    checksum_text = "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items()))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for member, source in sorted(selected, key=lambda pair: str(pair[0])):
            info = zipfile.ZipInfo(str(member), FIXED_TIME)
            info.compress_type = (
                zipfile.ZIP_STORED
                if source.suffix.lower() in {".pt", ".npz", ".zip"}
                else zipfile.ZIP_DEFLATED
            )
            info.external_attr = 0o100644 << 16
            # Stream full model states; they can exceed the bounded live-JSON reader.
            with source.open("rb") as input_stream, archive.open(info, "w") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
        for name, payload in (
            (
                "BUNDLE_MANIFEST.json",
                (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
            ),
            ("SHA256SUMS", checksum_text.encode()),
        ):
            info = zipfile.ZipInfo(name, FIXED_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    durable_replace_file(temporary, output)
    verification = verify_bundle(output)
    atomic_write_bytes(
        output.with_suffix(output.suffix + ".sha256"),
        f"{sha256_file(output)}  {output.name}\n".encode(),
    )
    atomic_write_json(output.with_suffix(output.suffix + ".verification.json"), verification)
    return verification


def verify_bundle(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if (
            len(names) != len(set(names))
            or "SHA256SUMS" not in names
            or "BUNDLE_MANIFEST.json" not in names
        ):
            raise ValueError("Bundle has duplicate names or lacks its manifests")
        expected: dict[str, str] = {}
        for line in archive.read("SHA256SUMS").decode("utf-8").splitlines():
            digest, name = line.split("  ", 1)
            _safe_member(name)
            expected[name] = digest
        actual: dict[str, str] = {}
        for name in expected:
            digest = hashlib.sha256()
            with archive.open(name, "r") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual[name] = digest.hexdigest()
        if actual != expected:
            raise ValueError("Bundle payload verification failed")
        manifest = json.loads(archive.read("BUNDLE_MANIFEST.json"))
        if manifest.get("payload_members") != expected:
            raise ValueError("Bundle manifest and SHA256SUMS disagree")
    return {
        "schema": "rgb_port_bundle_verification_v1",
        "status": "VERIFIED",
        "bundle_sha256": sha256_file(path),
        "payload_count": len(expected),
        "campaign_status": manifest["campaign_status"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or verify a local RGB-PORT essential bundle."
    )
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build")
    build.add_argument("--run", type=Path, required=True)
    build.add_argument(
        "--manifest", type=Path, required=True, help="JSON list of run-relative payload members"
    )
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--repository", type=Path)
    verify = sub.add_parser("verify")
    verify.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "verify":
        print(json.dumps(verify_bundle(args.bundle), sort_keys=True))
        return 0
    manifest = json.loads(read_bytes_shared(args.manifest).decode("utf-8"))
    if not isinstance(manifest, list) or not all(isinstance(item, str) for item in manifest):
        raise ValueError("Package manifest must be a JSON string list")
    print(
        json.dumps(
            build_bundle(args.run, args.output, manifest, repository_root=args.repository),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
