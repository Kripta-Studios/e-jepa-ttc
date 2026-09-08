"""Build the real freeze launch after source preparation and current QA exist."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.coordination import shared_write_admission, verified_ack
from e_jepa_ttc.simplex_t.expansion_authority import verify_expansion_authority
from e_jepa_ttc.simplex_t.freeze_integrity import FrozenFile, verify_code_commit, verify_files
from e_jepa_ttc.simplex_t.lifecycle import admitted
from e_jepa_ttc.simplex_t.scientific_admission import validate_scientific_admission


def consumer_references(work: Path, historical: Path) -> list[tuple[Path, str]]:
    """Enumerate the exact OLD/control/history files consumed after freeze."""
    result = [
        (historical / "FROZEN_EXPERT_TABLE_INDEX.json", "producers"),
        (
            historical / "frozen_audit/extracted_input/run/stage65/ALL_RIDGE_FITS_FROZEN.json",
            "producers",
        ),
        (work / "artifacts/simplex_t/T1/risk17_frozen_replay/REPLAY.json", "producers"),
    ]
    for fold in range(3):
        for suffix in ("csv", "npz"):
            result.append((historical / "tables" / f"outer{fold}_outer_dev.{suffix}", "producers"))
    for prefix in ("", "expansion_", "dense_"):
        for folder, filename in (
            ("query_context_index", "INDEX_MANIFEST.json"),
            ("query_context_dedup", "DEDUP_MANIFEST.json"),
        ):
            result.append((work / "artifacts/simplex_t/T1" / (prefix + folder) / filename, "time"))
    return result


def main() -> int:
    """Materialize metadata, not a freeze; the publisher independently validates it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--local-paths-sha256", required=True)
    parser.add_argument("--source-configuration", type=Path, required=True)
    parser.add_argument("--evidence-profile", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-output", type=Path, required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    work = Path(__file__).resolve().parents[1]
    if len(args.code_commit) != 40 or set(args.code_commit) - set("0123456789abcdef"):
        parser.error("full lowercase source commit required")
    if args.output.resolve() == args.freeze_output.resolve():
        parser.error("launch and scientific freeze require distinct paths")
    if args.other_reserved_bytes < 0:
        parser.error("nonnegative outstanding reservation required")
    for path in (
        args.output,
        args.freeze_output,
        args.source_configuration,
        args.evidence_profile,
        args.preparation,
    ):
        if not path.resolve().is_relative_to(work / "artifacts"):
            raise ValueError("generated metadata must remain inside companion artifacts")
    if sha256(args.local_paths) != args.local_paths_sha256:
        raise ValueError("local configuration changed")
    missing = [
        str(path)
        for path in (args.source_configuration, args.evidence_profile, args.preparation)
        if not path.is_file()
    ]
    if missing:
        print(
            json.dumps(
                dict(
                    status="WAITING_EXACT_FREEZE_INPUTS",
                    missing=missing,
                    scientific_completion=False,
                )
            )
        )
        return 10 if args.verify_only else 2

    def read(path: Path) -> dict:
        if path.stat().st_size > 8_388_608:
            raise ValueError("freeze metadata exceeds 8 MiB")
        return json.loads(path.read_text("utf-8"))

    paths = read(args.local_paths)
    if Path(paths["worktree"]).resolve(strict=True) != work:
        raise ValueError("local configuration belongs to another worktree")

    def resource_ok() -> bool:
        snapshot = admitted([work])
        return snapshot["has_headroom"] and shared_write_admission(
            snapshot["written_volume_free_bytes"][0], args.other_reserved_bytes + 8_388_608
        )

    if not resource_ok():
        return 3
    ack_path = Path(paths["shared_coordination"]) / "SIMPLEX_T_STAGE70_ACK.json"
    ack_hash = "3e55ab3c6e9a57eecd862ad05e999627ea90957e58e329b2eb3652120e953318"
    ack = verified_ack(ack_path, ack_hash)
    ancestry = ack["producers"]["authoritative_historical_manifest"]
    roots = dict(
        work=work,
        historical=Path(ancestry["path"]).parent.resolve(strict=True),
        stage70=Path(paths["stage70_worktree_read_only"]).resolve(strict=True),
        garl=Path(paths["garl_annotations_candidate"]).parent.parent.resolve(strict=True),
        garl_code=Path(paths["garl_code_candidate"]).resolve(strict=True),
        eap=Path(paths["eap_root"]).resolve(strict=True),
        coordination=Path(paths["shared_coordination"]).resolve(strict=True),
        local=args.local_paths.resolve().parent,
        handoff=Path(paths["handoff_root"]).resolve(strict=True),
    )
    files: dict[Path, FrozenFile] = {}

    def add(path: Path, category: str, expected: str | None = None) -> FrozenFile:
        path = path.resolve(strict=True)
        digest = sha256(path)
        if expected is not None and digest != expected:
            raise ValueError("authoritative input changed: " + path.name)
        if path in files:
            if files[path].sha256 != digest:
                raise ValueError("freeze input changed during inventory")
            return files[path]
        choices = [(name, root) for name, root in roots.items() if path.is_relative_to(root)]
        if not choices:
            raise ValueError("input outside explicitly acknowledged roots: " + str(path))
        name, root = max(choices, key=lambda pair: len(pair[1].parts))
        pin = FrozenFile(category, name, path.relative_to(root).as_posix(), digest)
        files[path] = pin
        return pin

    for folder in ("src", "scripts"):
        for path in sorted((work / folder).rglob("*")):
            if path.is_file() and path.suffix.lower() in {".py", ".ps1", ".psm1"}:
                add(path, "code")
    verify_code_commit(list(files.values()), roots, args.code_commit)
    add(args.local_paths, "config", args.local_paths_sha256)
    source_pin = add(args.source_configuration, "config")
    evidence_pin = add(args.evidence_profile, "qa")
    preparation = add(args.preparation, "normalizers")
    ledger = add(work / "artifacts/simplex_t/TECHNICAL_BUDGET.json", "qa")
    add(ack_path, "roles", ack_hash)
    for field, category in (("role_manifest", "roles"), ("time_charter", "time")):
        entry = ack["interfaces"][field]
        add(Path(entry["path"]), category, entry["sha256"])
    add(Path(ancestry["path"]), "producers", ancestry["sha256"])
    for path, category in consumer_references(work, roots["historical"]):
        add(path, category)
    add(
        work / "artifacts/simplex_t/T1/query_context_index/INDEX_MANIFEST.json",
        "time",
        "93a4f62e5025c5046fc82fcb1428a428f8a8df869b34486b92d5c753b0f68a3b",
    )
    for name in (
        "simplex_t_throughput_amendment.json",
        "simplex_t_resource_amendment.json",
        "simplex_t_technical_qa_amendment.json",
        "simplex_t_consumer_binding_qa_amendment.json",
    ):
        add(work / "configs/experiment" / name, "config")
    add(roots["handoff"] / "configs/RESOLVED_FIT_MANIFEST.schema.json", "schemas")
    add(roots["handoff"] / "configs/CAMPAIGN.json", "config")
    config = read(args.source_configuration)
    try:
        if any(key in config for key in ("expansion", "dense", "matched")):
            temporal = verify_expansion_authority(args.local_paths, resource_ok=resource_ok)
            add(Path(temporal["path"]), "time", temporal["sha256"])
            for entry in temporal["evidence"]:
                add(Path(entry["path"]), "time", entry["sha256"])
        value = dict(
            schema="simplex_t_freeze_launch_v1",
            local_paths=str(args.local_paths.resolve()),
            source_configuration=str(args.source_configuration.resolve()),
            source_configuration_sha256=source_pin.sha256,
            evidence_profile=str(args.evidence_profile.resolve()),
            evidence_profile_sha256=evidence_pin.sha256,
            roots={key: str(root) for key, root in roots.items()},
            files=[asdict(pin) for pin in files.values()],
            preparation=asdict(preparation),
            technical_ledger=asdict(ledger),
            code_commit=args.code_commit,
            output=str(args.freeze_output.resolve()),
        )
        validate_scientific_admission(
            dict(files=value["files"], source_contract=read(args.preparation)["contract"]),
            roots=roots,
            local_paths=args.local_paths,
            source_configuration=args.source_configuration,
            source_configuration_sha256=source_pin.sha256,
            evidence_profile=args.evidence_profile,
            evidence_profile_sha256=evidence_pin.sha256,
            resource_ok=resource_ok,
        )
        verify_files(list(files.values()), roots)
        if not resource_ok():
            return 3
        if args.output.exists():
            if read(args.output) != value:
                raise ValueError("existing freeze launch differs; no overwrite")
        elif args.verify_only:
            return 10
        else:
            write_new_json(args.output, value)
    except InterruptedError:
        return 3
    print(
        json.dumps(
            dict(
                status="FREEZE_LAUNCH_PREPARED_NOT_FROZEN",
                path=str(args.output),
                sha256=sha256(args.output),
                optimizer_updates=0,
            )
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
