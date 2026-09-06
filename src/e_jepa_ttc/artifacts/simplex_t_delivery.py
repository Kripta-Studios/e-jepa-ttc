"""Package honest, untrained T0 prerequisite evidence without mutable telemetry."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json

SOURCE_FILES = (
    "src/e_jepa_ttc/artifacts/simplex_t_preflight.py",
    "src/e_jepa_ttc/artifacts/simplex_t_delivery.py",
    "scripts/run_simplex_t_companion.py",
    "scripts/run_simplex_t_companion.ps1",
    "scripts/audit_simplex_t_interfaces.py",
    "scripts/audit_simplex_t_timeline.py",
    "tests/unit/test_simplex_t_preflight.py",
    "configs/experiment/simplex_t_resource_amendment.json",
    "docs/SIMPLEX_T_T0_RESUME.md",
    "docs/SIMPLEX_T_STAGE70_OWNER_MESSAGE.md",
)


def qa_summary(path: Path) -> dict[str, Any]:
    """Read exact executed test identities, preserving failures and skips."""
    root = ElementTree.parse(path).getroot()
    cases = list(root.iter("testcase"))
    failures = [
        f"{case.attrib.get('classname')}::{case.attrib['name']}"
        for case in cases
        if case.find("failure") is not None or case.find("error") is not None
    ]
    return {
        "tests": len(cases),
        "failure_ids": failures,
        "skipped": sum(case.find("skipped") is not None for case in cases),
        "summed_case_seconds": sum(float(case.attrib.get("time", "0")) for case in cases),
        "evidence_sha256": sha256(path),
    }


def package_t0(local_paths: Path, output: Path) -> dict[str, Any]:
    """Snapshot only bounded T0 evidence; never report absent fits as complete."""
    paths = json.loads(local_paths.read_text(encoding="utf-8"))
    worktree = Path(paths["worktree"])
    evidence = worktree / "artifacts/simplex_t/T0"
    audit_path = evidence / "LOCAL_DATA_AUDIT_FINAL.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["scientific_fits"] != 0:
        raise ValueError("T0 packager cannot package scientific fits")
    qa = {
        name: qa_summary(evidence / f"qa_{name}.xml")
        for name in ("baseline", "companion", "reference", "integrated_final")
    }
    if any(item["failure_ids"] for item in qa.values()):
        raise ValueError("failed QA must be investigated before this T0 handoff")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip()
    decision = {
        "artifact_type": "simplex_t_next_decision_v1",
        "execution_status": "WAITING_SHARED_ROLE_MANIFEST",
        "source_status": "OWNER_INTERFACES_DISCOVERED_NOT_ACKNOWLEDGED",
        "numerical_status": "INTEGRATED_CPU_ENGINE_SYNTHETIC_RESUME_PASS_REAL_REPLAY_PENDING",
        "mechanism_status": "NOT_EVALUATED",
        "replication_scope": "NONE",
        "claim_ceiling": "T0_METADATA_AND_SYNTHETIC_ENGINEERING_ONLY",
        "analysis_commit": commit,
        "scientific_fits": 0,
        "scientific_optimizer_updates": 0,
        "technical_optimizer_updates": 65,
        "total_optimizer_updates": 65,
        "technical_accounting": [
            {"test": "test_resume_exact", "updates": 20, "scope": "SYNTHETIC_REFERENCE"},
            {
                "test": "test_resume_rejects_contract_change",
                "updates": 5,
                "scope": "SYNTHETIC_REFERENCE",
            },
        ],
        "integrated_technical_accounting": {
            "test": "test_production_cpu_10_versus_5_plus_5",
            "invocations": 2,
            "updates": 40,
            "scope": "INTEGRATED_ENGINE_SYNTHETIC",
        },
        "implementation_status": (
            "KERNELS_CACHE_ENGINE_METRICS_LEDGER_IMPLEMENTED_INTEGRATION_PENDING"
        ),
        "scientific_freeze": False,
        "T1_T6": "NOT_RUN_PREREQUISITE_BLOCK",
        "canonical_D": "UNRESOLVED_BEFORE_SCORES",
        "D1_density_latent_availability": "UNRESOLVED",
        "history_index": "NOT_BUILT_LABEL_INDEPENDENT_TIMELINE_UNVERIFIED",
        "production_replay_parity": "NOT_RUN",
        "production_device_resume": "CPU_ENGINE_PASS_ON_SYNTHETIC_DATA",
        "qa": qa,
        "qa_scope": "11 historical baseline; 19 combined T0; 65 reference; 86 integrated tests",
        "full_repository_qa": "NOT_RUN",
        "initial_failure_ids": [
            "RUFF_FORMAT_AND_REFERENCE_STYLE",
            "test_simplex_t_training::test_schedule_exact_endpoints",
            "PYRIGHT_FACTORIAL_TUPLE_AND_CONTEXT_MANAGER_RETURN",
        ],
        "resolved_initial_failures": [
            "RUFF_FORMAT_AND_REFERENCE_STYLE",
            "test_simplex_t_training::test_schedule_exact_endpoints",
            "PYRIGHT_FACTORIAL_TUPLE_AND_CONTEXT_MANAGER_RETURN",
        ],
        "historical_failure_reclassification": False,
        "required_next_evidence": [
            "SIMPLEX_T_STAGE70_ACK.json acknowledging exact role/time byte hashes and quotas",
            "Adopt verified 36-producer ancestry and bind new history cache preprocessing",
            "Full permitted input-only OBJECT timeline and clock/ROI mapping",
            "Integrated production loader parity, all-arm implementation and pre-fit QA/freeze",
            "Exclusive inference scheduling boundary followed by CACHE_READY evidence",
        ],
        "interfaces": audit["interfaces"],
        "producer_audit": json.loads(
            (evidence / "DISCOVERED_INTERFACES.json").read_text(encoding="utf-8")
        ),
        "timeline_feasibility": json.loads(
            (evidence / "LOCAL_TIMELINE_FEASIBILITY.json").read_text(encoding="utf-8")
        ),
        "new_weights": [],
        "per_query_predictions": [],
        "holdout_opened": False,
        "scientific_negative": False,
        "background_completion_promised": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    decision_path = output / "NEXT_DECISION_SIMPLEX_T.json"
    write_new_json(decision_path, decision)
    report_path = output / "CODEX_SIMPLEX_T_FINAL_REPORT.md"
    report = f"""# SIMPLEX-T — resumable T0 prerequisite block

Status: **WAITING_SHARED_ROLE_MANIFEST** (owner acknowledgement pending).
Stage70 role/time files exist and their byte hashes are recorded. The local
configuration has not adopted them; shared ownership/resource acknowledgement
and the full-history source remain unresolved. All 36 historical checkpoints and
their bounded metadata bindings match the historical ancestry audit. This is not a
scientific negative or completed T0–T6 campaign.

Starting commit: {audit["base_commit"]}. Analysis/implementation commit: {commit}.
All 35 handoff payload hashes passed. No Stage70 scientific files were changed;
no architecture scores or confirmation/test payloads were read.

## Actual work and tests

Bounded TRAIN footer/sequence metadata audit: 118247 eAP FRAME rows and 88744
Garl TTC-PAIR input rows, with separate 88744-row supervision footer. FRAME,
OBJECT and TTC-PAIR identities remain distinct. The whole OBJECT timeline and
its label-independent history membership are not yet proven.

Selected historical baseline: {qa["baseline"]["tests"]} tests, zero failures.
Combined historical plus T0 suite: {qa["companion"]["tests"]} tests, zero failures.
Reference suite: {qa["reference"]["tests"]} tests, zero failures. This includes
CPU 10 versus 5+save+5 equality for weights, optimizer and losses. Its recipe is
synthetic reference only. The integrated suite additionally passes
{qa["integrated_final"]["tests"]} tests including CPU 10 versus 5+5 on the actual
new head engine at batch128 with the registered schedule. That proves synthetic
engine resume, not real TRAIN replay parity or local loader integration.
Initial Ruff format failure was corrected and retained as a QA ID. Full repository
QA, real TRAIN replay, production loader integration and scientific freeze are
pending. No historical failing result has been relabelled.

Executed scientific fits: **0**. Scientific optimizer updates: **0**. Technical
updates: **65** (25 reference + two 20-update integrated resume invocations).
Raw expert forwards: **0**. No model endpoint, scientific prediction, factor
interaction, bootstrap interval, hull gain/harm or lag result exists to report.
No partial checkpoint was evaluated as a scientific endpoint.

Implemented independent components: registered GRU/Transformer/latent kernels,
four temporal controls, input-only dependency-checked histories, deduplicated
feature gathering and train-only normalizers, FP32 head engine, full 84-fit graph,
absolute-resource admission, exclusive leases/update ledger, per-query output
diagnostics and D/H/C interactions. These components still require real source
adapter, coordinated replay, production lifecycle wiring and pre-fit freeze.

The temporal source search checked all nine original eAP OBJECT schema footers:
none provides 2D bbox fields. Filename discovery found no original annotation
pickle or alternate full timeline in the three local roots (excluded environment,
download-cache and RGB-shard directories are documented). The inspected Garl
builder filters by TTC and 3D height, so its full pair table is not a safe primary
history substitute. H1 is not intrinsically blocked by this missing timeline;
it remains pending coordinated real replay/production pre-fit prerequisites.

## Resources and owner activity

Minimum available RAM: 8 GiB; companion process-tree RSS maximum: 4 GiB; minimum
free space per written volume: 60 GiB. Limits use absolute bytes/GiB only. The user
amendment changes operational resource limits, not model/statistical constants.
T0 metadata audit elapsed: {audit["elapsed_seconds"]:.3f} seconds.
Saved audit includes process commands, host RAM, disks and per-source read timings.

Read-only observation found active Stage71 count-cache construction, growing from
7616 to 7680 rows with 120 physical count shards. State declared zero fits/updates.
The process CPU counter advanced; the declared state is not a claim that every
artifact is independently validated. GPU/heavy I/O remains unacknowledged. No
owner process was stopped, resumed, wrapped retroactively or otherwise modified.

## Resume boundary

The shared SIMPLEX_T_STAGE70_REQUEST.json requests immutable role/time paths and
hashes, complete producer ancestry, permitted timeline and cooperative resource
quotas. The discovered owner manifest references are in NEXT_DECISION_SIMPLEX_T.json.
No competing holdout assignment was made. Obtain owner acknowledgement, complete
timeline/producer integration and all registered pre-fit QA, freeze, then run every
available registered fit. D*, density controls and latent availability remain
unresolved before scores. No canonical candidate has been replaced by a control.

See docs/SIMPLEX_T_T0_RESUME.md for the exact local evidence and production work
remaining. `run --resume` currently returns exit 3; it does not train automatically.
The bundle includes new source, metadata audit, QA XML, resource amendment and
this decision. Missing scientific weights/predictions are intentional absences
caused by the prerequisite block, not successful deliverables.

Regenerate after QA into a new directory:
`python scripts/run_simplex_t_companion.py package --local-paths PATH --output NEW_DIR`
Use the existing Python 3.11 environment with this worktree's src on PYTHONPATH.
No future background completion is promised.
"""
    with report_path.open("x", encoding="utf-8") as stream:
        stream.write(report)
    payload = {
        "CODEX_SIMPLEX_T_FINAL_REPORT.md": report_path.read_bytes(),
        "NEXT_DECISION_SIMPLEX_T.json": decision_path.read_bytes(),
    }
    for relative in SOURCE_FILES:
        payload[relative] = (worktree / relative).read_bytes()
    for path in sorted((worktree / "src/e_jepa_ttc/simplex_t").glob("*.py")):
        payload[path.relative_to(worktree).as_posix()] = path.read_bytes()
    for path in sorted((worktree / "tests/unit").glob("test_simplex_t*.py")):
        payload[path.relative_to(worktree).as_posix()] = path.read_bytes()
    for name in (
        "LOCAL_DATA_AUDIT.json",
        "LOCAL_DATA_AUDIT_FINAL.json",
        "qa_baseline.xml",
        "qa_companion.xml",
        "qa_reference.xml",
        "QA_CHECKS.json",
        "DISCOVERED_INTERFACES.json",
        "LOCAL_TIMELINE_FEASIBILITY.json",
        "qa_integrated_final.xml",
        "qa_training.xml",
        "qa_training_schedule_fixed.xml",
        "qa_kernels.xml",
        "qa_cache.xml",
        "qa_evaluation.xml",
        "qa_post_types.xml",
    ):
        payload[f"evidence/{name}"] = (evidence / name).read_bytes()
    request = Path(paths["shared_coordination"]) / "SIMPLEX_T_STAGE70_REQUEST.json"
    payload["coordination/SIMPLEX_T_STAGE70_REQUEST.json"] = request.read_bytes()
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()}
    payload["PAYLOAD_SHA256.json"] = json.dumps(manifest, indent=2).encode()
    bundle = output / f"E_JEPA_TTC_SIMPLEX_T_ESSENTIAL_RESULTS_{commit[:12]}.zip"
    with zipfile.ZipFile(bundle, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(payload.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 9, 6, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, data)
    digest = sha256(bundle)
    with bundle.with_suffix(".zip.sha256").open("x", encoding="ascii") as stream:
        stream.write(f"{digest}  {bundle.name}\n")
    with zipfile.ZipFile(bundle) as archive:
        if archive.testzip() is not None:
            raise ValueError("bundle CRC verification failed")
        for name, expected in manifest.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"bundle payload changed: {name}")
    return {"bundle": str(bundle), "sha256": digest, "decision": str(decision_path)}
