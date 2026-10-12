"""Build an immutable, byte-verified research snapshot without replacing historical bundles."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from datetime import UTC, datetime

from operational.multimodal_validation.common import CODE, ROOT, RUN, V12, read, sha, write
from operational.multimodal_validation.common import OUT as PREVIOUS

from .common import OUT, accounting


def main() -> None:
    ledger = accounting()
    assert ledger["pending_upper"] == 0
    for row in ledger["fits"]:
        assert row["completed_updates"] == 2500, "Snapshot only at fixed complete fit endpoints"
    local_complete = all(
        (OUT / name).exists()
        for name in (
            "test12_phase_only/PACKAGE_VERIFICATION.json",
            "test12_phase_only/OFFICIAL_PARSER_VERIFICATION.json",
            "test12_phase_only/CACHE_RESUME_PARITY.json",
            "FCWD_GPU_ROUTE_COST.json",
            "MULTIMODAL_REVIEW_FINAL.md",
        )
    )
    status = (
        "LOCAL_RESULTS_COMPLETE_OFFICIAL_EVALUATION_PENDING" if local_complete else "IN_PROGRESS"
    )
    kind = "RESULT" if local_complete else "PROGRESS"
    if local_complete:
        assert read(OUT / "test12_phase_only/OFFICIAL_PARSER_VERIFICATION.json")["status"] == "PASS"
        assert read(OUT / "test12_phase_only/CACHE_RESUME_PARITY.json")["status"] == "PASS"
        assert read(OUT / "FCWD_GPU_ROUTE_COST.json")[
            "all_registration_states_and_predictions_exact"
        ]
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = ROOT / "artifacts" / f"E_JEPA_TTC_RESEARCH_{kind}_{stamp}.zip"
    assert not destination.exists()
    paths = set()
    for directory in (
        ROOT / "src",
        ROOT / "operational",
        ROOT / "configs",
        ROOT / "tests",
        ROOT / "docs/multimodal_next_handoff_20261012",
        ROOT / "docs/dev32_transfer_20261012",
        OUT,
        PREVIOUS,
    ):
        for path in directory.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if ".pending" in path.name or path.suffix in {".lock", ".tmp"}:
                continue
            paths.add(path)
    for name in ("README.md", "AGENTS.md", "pyproject.toml", "uv.lock", ".gitattributes"):
        path = ROOT / name
        if path.exists():
            paths.add(path)
    parent_paths = set((RUN / "fits").glob("*/checkpoint_last.pt"))
    for name in ("CHECKPOINT_RECEIPT.json", "RUN_PROVENANCE.json", "UPDATE_JOURNAL.json"):
        parent_paths.update((RUN / "fits").glob("*/" + name))
    parent_paths.update((RUN / "features/H").glob("*_NORMALIZER.json"))
    parent_paths.update(
        RUN / name
        for name in (
            "P_MANIFEST.json",
            "H_MANIFEST.json",
            "V_MANIFEST.json",
            "rounds/PRODUCERS_PAIR_ENDPOINT_FREEZE.json",
        )
    )
    dependency_index = []
    official_prefix = "DEPENDENCIES/OFFICIAL_TEST12/"
    official_inputs = {
        V12 / "artifacts/sota_evidence_20261010/sources/test_inputs.parquet": (
            official_prefix + "test_inputs.parquet"
        ),
        V12 / "artifacts/garlttc_submission_20261010/INPUT_FREEZE.json": (
            official_prefix + "H8_CURRENT_SENSOR_INPUT_FREEZE.json"
        ),
    }
    garl = read(PREVIOUS / "garl_full/BINDING.json")
    for relative, expected in garl["native_source_sha256"].items():
        path = CODE / relative
        assert sha(path) == expected
        official_inputs[path] = "DEPENDENCIES/GARL_RELEASE/" + relative
    config_path = (
        V12 / "artifacts/evttc_rgb_transfer_20261008/public_garl/configs/ablation/ours_full.yaml"
    )
    assert sha(config_path) == garl["config_sha256"]
    official_inputs[config_path] = "DEPENDENCIES/GARL_RELEASE/configs/ablation/ours_full.yaml"
    for relative in (
        "garl_ttc_benchmark/codabench/build_codabench_bundle.py",
        "garl_ttc_benchmark/codabench/scoring_program/score.py",
    ):
        official_inputs[CODE / relative] = "DEPENDENCIES/GARL_RELEASE/" + relative
    metadata = dict(
        utc=stamp,
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        git_status=subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        status=status,
        official_SOTA=False,
        accounting=ledger,
        next_decision=read(OUT / "NEXT_DECISION.json"),
        dependencies=(
            "New readout checkpoints, H/V state caches, predictions and sources included. "
            "All twelve original V13 endpoints, receipts and H normalizers are included as "
            "byte-preserved reference dependencies. Their historical absolute path bindings "
            "are preserved, not advertised as a standalone relocated runtime. Original sensor "
            "HDF5/RGB archives, historical H8 current-input caches and the public Garl "
            "full/event checkpoints remain external; the downloaded visual checkpoint is included. "
            "Partial sensor "
            "replay directories are recoverable per query and are complete only with SEALED.json."
        ),
        scientific_scope=(
            "Adaptive development over exposed V/Dev32/FCWD. No private test labels, "
            "submissions or claim of official SOTA. Old V16 result archive preserved."
        ),
    )
    verifier = """from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'MANIFEST.json').read_text(encoding='utf-8'))
for entry in manifest['files']:
    path=(root/entry['path']).resolve()
    assert path.is_relative_to(root)
    data=path.read_bytes()
    assert len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256'], path
print('Verified',len(manifest['files']),'files')
"""
    entries = []
    with zipfile.ZipFile(
        destination, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        for path in sorted(paths):
            data = path.read_bytes()
            name = path.relative_to(ROOT).as_posix()
            archive.writestr(name, data)
            entries.append(
                dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            )
        for path in sorted(parent_paths):
            data = path.read_bytes()
            name = "DEPENDENCIES/V13/" + path.relative_to(RUN).as_posix()
            digest = hashlib.sha256(data).hexdigest()
            archive.writestr(name, data)
            entries.append(dict(path=name, bytes=len(data), sha256=digest))
            dependency_index.append(
                dict(original_path=str(path), archived_path=name, sha256=digest)
            )
        for path, name in official_inputs.items():
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            archive.writestr(name, data)
            entries.append(dict(path=name, bytes=len(data), sha256=digest))
            dependency_index.append(
                dict(original_path=str(path), archived_path=name, sha256=digest)
            )
        extras = {
            "SNAPSHOT.json": json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8"),
            "VERIFY.py": verifier.encode("utf-8"),
            "DEPENDENCY_INDEX.json": json.dumps(
                dependency_index, ensure_ascii=False, indent=2
            ).encode("utf-8"),
        }
        for name, data in extras.items():
            archive.writestr(name, data)
            entries.append(
                dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            )
        archive.writestr(
            "MANIFEST.json", json.dumps(dict(files=entries), ensure_ascii=False, indent=2)
        )
    with zipfile.ZipFile(destination) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("MANIFEST.json"))
        assert len(archive.namelist()) == len(entries) + 1
        for entry in manifest["files"]:
            data = archive.read(entry["path"])
            assert (
                len(data) == entry["bytes"] and hashlib.sha256(data).hexdigest() == entry["sha256"]
            )
    receipt = dict(
        path=str(destination),
        bytes=destination.stat().st_size,
        sha256=sha(destination),
        files=len(entries),
        crc_pass=True,
        every_payload_sha256_verified=True,
        status=status,
        source_sha256=sha(__file__),
        official_SOTA=False,
    )
    write(destination.with_suffix(".verification.json"), receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
