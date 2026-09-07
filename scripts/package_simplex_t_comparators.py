"""Package completed comparator artifacts without copying the live temporal cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.simplex_t.bundle_integrity import verify_bundle
from e_jepa_ttc.simplex_t.coordination import shared_write_admission
from e_jepa_ttc.simplex_t.lifecycle import ExclusiveLease, admitted


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument(
        "--evidence",
        type=Path,
        action="append",
        default=[],
        help="additional immutable T0 evidence file; may be repeated",
    )
    args = parser.parse_args()
    root = Path.cwd().resolve()
    if not args.output.resolve().is_relative_to(root) or args.other_reserved_bytes < 0:
        raise ValueError("companion output and explicit reservations required")
    names = ("CODEX_SIMPLEX_T_FINAL_REPORT.md", "NEXT_DECISION_SIMPLEX_T.json")
    files = {args.output / name for name in names}
    evidence_root = root / "artifacts/simplex_t/T0"
    for evidence in args.evidence:
        path = evidence.resolve(strict=True)
        if not path.is_relative_to(evidence_root) or not path.is_file():
            raise ValueError("additional evidence must be a file inside companion T0")
        files.add(path)
    for pattern in (
        "src/e_jepa_ttc/simplex_t/*.py",
        "scripts/*simplex_t*.py",
        "tests/unit/test_simplex_t*.py",
        "configs/experiment/simplex_t*.json",
        "artifacts/simplex_t/T1/risk17_frozen_replay/*",
        "artifacts/simplex_t/T1/simplex17_frozen_replay/*",
        "artifacts/simplex_t/T1/fixed_baselines_outer0_fp64_emission/*",
        "artifacts/simplex_t/T0/*REVERIFICATION.json",
        "artifacts/simplex_t/T0/*RISK17*.json",
        "artifacts/simplex_t/T0/*SIMPLEX17*.json",
        "artifacts/simplex_t/T0/*VALIDATOR_QA.xml",
        "artifacts/simplex_t/T0/HISTORICAL_SELECTOR_INPUT_QA.xml",
        "artifacts/simplex_t/T0/real_context_cpu_resume/*.json",
        "artifacts/simplex_t/T0/real_context_cpu_resume/*/checkpoint_last.pt",
    ):
        files.update(p for p in root.glob(pattern) if p.is_file())
    files.add(root / "artifacts/simplex_t/TECHNICAL_BUDGET.json")
    reservation = 2 * sum(p.stat().st_size for p in files) + 67_108_864
    snapshot = admitted([root])
    if not snapshot["has_headroom"] or not shared_write_admission(
        snapshot["written_volume_free_bytes"][0], args.other_reserved_bytes + reservation
    ):
        raise RuntimeError("RESOURCE_PAUSE before immutable comparator package")
    with ExclusiveLease(args.output / "PACKAGE.lock"):
        inventory = {p.resolve().relative_to(root).as_posix(): digest(p) for p in sorted(files)}
        analysis = hashlib.sha256(json.dumps(inventory, sort_keys=True).encode()).hexdigest()[:12]
        manifest = args.output / "CONTENT_MANIFEST.json"
        write_new_json(
            manifest,
            {
                "files": inventory,
                "live_cache_included": False,
                "scientific_complete": False,
                "analysis_shortsha": analysis,
            },
        )
        archive = args.output / f"E_JEPA_TTC_SIMPLEX_T_ESSENTIAL_RESULTS_{analysis}.zip"
        with zipfile.ZipFile(
            archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1
        ) as bundle:
            for relative, expected in inventory.items():
                path = root / relative
                if digest(path) != expected:
                    raise ValueError("completed input changed during packaging")
                bundle.write(path, path.name if path.name in names else relative)
            bundle.write(manifest, manifest.name)
        archived_inventory = {
            Path(relative).name if Path(relative).name in names else relative: expected
            for relative, expected in inventory.items()
        }
        if len(archived_inventory) != len(inventory):
            raise ValueError("bundle member name collision")
        archived_inventory[manifest.name] = digest(manifest)

        def resource_ok() -> bool:
            state = admitted([root])
            return state["has_headroom"] and shared_write_admission(
                state["written_volume_free_bytes"][0], args.other_reserved_bytes + reservation
            )

        verification = verify_bundle(archive, archived_inventory, resource_ok=resource_ok)
        for relative, expected in inventory.items():
            if digest(root / relative) != expected:
                raise ValueError("input changed before package validation completed")
        checksum = digest(archive)
        write_new_json(args.output / "BUNDLE_VERIFICATION.json", verification)
        with archive.with_suffix(".zip.sha256").open("x", encoding="ascii") as stream:
            stream.write(f"{checksum}  {archive.name}\n")
        write_new_json(
            args.output / "ZIP_SHA256.json", {"archive": archive.name, "sha256": checksum}
        )
        print(
            json.dumps({"archive": str(archive), "sha256": checksum, "scientific_complete": False})
        )


if __name__ == "__main__":
    main()
