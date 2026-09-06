"""Package honest, untrained T0 prerequisite evidence without mutable telemetry."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from e_jepa_ttc.artifacts.simplex_t_preflight import interface_status, sha256, write_new_json

SOURCE_FILES = (
    "src/e_jepa_ttc/artifacts/simplex_t_preflight.py",
    "src/e_jepa_ttc/artifacts/simplex_t_delivery.py",
    "scripts/run_simplex_t_companion.py",
    "scripts/run_simplex_t_companion.ps1",
    "scripts/audit_simplex_t_interfaces.py",
    "scripts/audit_simplex_t_timeline.py",
    "scripts/profile_simplex_t_cpu.py",
    "scripts/audit_simplex_t_projection.py",
    "scripts/fetch_simplex_t_original_annotations.py",
    "scripts/audit_simplex_t_original_release.py",
    "scripts/audit_simplex_t_current_tables.py",
    "scripts/probe_simplex_t_current_resume.py",
    "scripts/audit_simplex_t_current_timing.py",
    "scripts/audit_simplex_t_release_identity.py",
    "configs/experiment/simplex_t_coordination.json",
    "docs/SIMPLEX_T_ACK_AND_PROJECTION_AUDIT.md",
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
    current_interfaces = interface_status(paths)
    if audit["scientific_fits"] != 0:
        raise ValueError("T0 packager cannot package scientific fits")
    qa = {
        name: qa_summary(evidence / f"qa_{name}.xml")
        for name in (
            "baseline",
            "companion",
            "reference",
            "integrated_final",
            "selector_export_fix",
            "pools",
            "checkpoint_resume",
            "technical_budget",
            "ack_40gb",
        )
    }
    if any(item["failure_ids"] for item in qa.values()):
        raise ValueError("failed QA must be investigated before this T0 handoff")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip()
    decision = {
        "artifact_type": "simplex_t_next_decision_v1",
        "execution_status": current_interfaces["execution_status"],
        "source_status": "OWNER_INTERFACES_ACKNOWLEDGED_GEOMETRY_PARITY_UNRESOLVED",
        "numerical_status": "INTEGRATED_CPU_ENGINE_SYNTHETIC_RESUME_PASS_REAL_REPLAY_PENDING",
        "mechanism_status": "NOT_EVALUATED",
        "replication_scope": "NONE",
        "claim_ceiling": "T0_METADATA_AND_SYNTHETIC_ENGINEERING_ONLY",
        "analysis_commit": commit,
        "scientific_fits": 0,
        "scientific_optimizer_updates": 0,
        "technical_optimizer_updates": 605,
        "total_optimizer_updates": 605,
        "current_array_resume_accounting": {
            "updates": 20,
            "scope": "REAL_TRAIN_ARRAYS_ZERO_TIMING_TECHNICAL_FIXTURE",
            "complete_state_exact_resume": True,
            "raw_replay_parity": False,
        },
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
            "invocations": 3,
            "updates": 60,
            "scope": "INTEGRATED_ENGINE_SYNTHETIC",
        },
        "synthetic_profile_accounting": {
            "updates": 500,
            "scope": "SYNTHETIC_CPU_PROFILE_ONLY",
            "evidence": "cpu_profile_500/RESOURCE_PROFILE.json",
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
        "qa_scope": (
            "11 historical baseline; 19 combined T0; 65 reference; 86 integrated tests; "
            "5 export tests, 6 pool tests, 9 checkpoint/resume tests after continuation fixes"
        ),
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
            "Explicit exclusive inference slot after live architecture owner releases resources",
            "Adopt verified 36-producer ancestry and bind new history cache preprocessing",
            "Authoritative Garl-to-eAP persistent-object crosswalk or unfiltered Garl source tracks",
            "Historical producer clock/ROI dependency mapping, including nonzero exposure age",
            "Integrated production loader parity, all-arm implementation and pre-fit QA/freeze",
            "Exclusive inference scheduling boundary followed by CACHE_READY evidence",
        ],
        "interfaces": current_interfaces,
        "original_release_identity_audit": json.loads(
            (evidence / "ORIGINAL_RELEASE_IDENTITY_FULL.json").read_text(encoding="utf-8")
        ),
        "minimum_written_volume_remaining_bytes": 40_000_000_000,
        "producer_audit": json.loads(
            (evidence / "DISCOVERED_INTERFACES.json").read_text(encoding="utf-8")
        ),
        "timeline_feasibility": json.loads(
            (evidence / "LOCAL_TIMELINE_FEASIBILITY.json").read_text(encoding="utf-8")
        ),
        "cpu_profile": json.loads(
            (evidence / "cpu_profile_500/RESOURCE_PROFILE.json").read_text(encoding="utf-8")
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

Status: **{current_interfaces["execution_status"]}**.
Stage70 role/time files and the user-pinned ACK have been verified and adopted
read-only. Exclusive inference scheduling and production history parity remain
unresolved. All 36 historical checkpoints and
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
Continuation QA adds {qa["selector_export_fix"]["tests"]} export tests and
{qa["pools"]["tests"]} query-pool tests, all passing without optimizer updates.
The selector export now preserves the original expert TTC exactly; the median
diagnostic scores its finite emitted output. D1/density query selection is
implemented from input identity only, but real availability remains unresolved.
Checkpoint continuation adds {qa["checkpoint_resume"]["tests"]} passing tests:
complete-state digest validation, corruption refusal, interrupted-publication
preservation, false-endpoint refusal and another CPU exact-resume comparison.
Budget continuation adds {qa["technical_budget"]["tests"]} passing resource/ledger
tests without optimizer updates. The persistent technical ledger reconciles 585
executed updates and prevents repeating the profile in another output directory.
Future reservations are upper bounds, not claims of executed updates.
Initial Ruff format failure was corrected and retained as a QA ID. Full repository
QA, real TRAIN replay, production loader integration and scientific freeze are
pending. No historical failing result has been relabelled.

Executed scientific fits: **0**. Scientific optimizer updates: **0**. Technical
updates: **605** (25 reference + three 20-update synthetic resume invocations
+ one 500-update synthetic CPU profile +20 real-array resume updates with explicit
zero timing fixture). The latter gives exact complete-state CPU resume, not raw
expert replay or production timing validation. No technical result selects a model.
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

## Original annotation acquisition

After explicit user download authorization, selective HTTP ranges retrieved
annotations.pkl and frames.pkl from original-role sequence ZIPs linked by the
official eAP release catalog. Per-sequence receipts bind archive ETag, member
CRC verification, extracted SHA256 and transferred bytes. No media was downloaded.
Acquisition does not establish complete-history eligibility or Garl identity/ROI
parity. These sources require a separate input-only schema and lineage audit.
Earlier filename-search findings describe the state before this acquisition.

## Resources and owner activity

Minimum available RAM: 8 GiB; companion process-tree RSS maximum: 4 GiB; minimum
free space after reservations per written volume: 40,000,000,000 bytes (40 GB).
The latest explicit user instruction supersedes prior60 GiB and ACK120 GB floors.
Limits use absolute bytes only. The user
amendment changes operational resource limits, not model/statistical constants.
T0 metadata audit elapsed: {audit["elapsed_seconds"]:.3f} seconds.
Saved audit includes process commands, host RAM, disks and per-source read timings.
The 145-input C160/H8 GRU CPU profile completed 500 synthetic updates in 16.982s,
with 639238144 bytes peak process-tree RSS and 17568727040 bytes minimum host
available RAM, four threads/two interop. This measures the head on generated
inputs, not real expert replay, full-system latency or useful TTC performance.

The initial saved read-only observation found Stage71 count-cache construction, growing from
7616 to 7680 rows with 120 physical count shards. State declared zero fits/updates.
The process CPU counter advanced; this is a historical snapshot, not current job status.
A later continuation observed Stage71 QA processes; the final bounded process-command
check found no matching Stage70/architecture Python or PowerShell command. Absence
of a matching command is not an owner acknowledgement or proof of GPU availability.
The declared state is not a claim that every artifact is independently validated.
GPU/heavy I/O remains unacknowledged. No
owner process was stopped, resumed, wrapped retroactively or otherwise modified.

## Resume boundary

The complete original-release identity audit covers182086 eAP object keys,
exactly shared by ZIP and HF, and22716 unique Garl observations across21471
original-role pairs. Only938 Garl frame/instance keys match directly; none has
an exact same-ID box under the two audited conventions. No correspondence was
invented. CURRENT_EXPOSURE_TIMING.json also records nonzero selected exposure
age for all8192 D0 queries (1003–19992us); producer cutoff parity remains open.

The shared SIMPLEX_T_STAGE70_ACK.json acknowledges the immutable interfaces and
conditional light-I/O CPU overlap. It does not grant exclusive inference.
References and verified ACK identity are in NEXT_DECISION_SIMPLEX_T.json.
No competing holdout assignment was made. Coordinate the inference slot, complete
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
        "qa_selector_export_fix.xml",
        "qa_pools.xml",
        "QA_CONTINUATION_01.json",
        "qa_checkpoint_resume.xml",
        "QA_CONTINUATION_02.json",
    ):
        payload[f"evidence/{name}"] = (evidence / name).read_bytes()
    for name in ("RESERVATION.json", "RESOURCE_PROFILE.json", "checkpoint_last.pt"):
        payload[f"technical_cpu_profile/{name}"] = (
            evidence / "cpu_profile_500" / name
        ).read_bytes()
    payload["evidence/qa_technical_budget.xml"] = (
        evidence / "qa_technical_budget.xml"
    ).read_bytes()
    payload["evidence/TECHNICAL_BUDGET.json"] = (
        worktree / "artifacts/simplex_t/TECHNICAL_BUDGET.json"
    ).read_bytes()
    for name in (
        "qa_ack_40gb.xml",
        "PROJECTION_FEASIBILITY.json",
        "PROJECTION_FEASIBILITY_INSTANCE_ID.json",
        "PROJECTION_FRAME_CANDIDATES.json",
    ):
        payload[f"evidence/{name}"] = (evidence / name).read_bytes()
    payload["coordination/SIMPLEX_T_STAGE70_ACK.json"] = (
        Path(paths["shared_coordination"]) / "SIMPLEX_T_STAGE70_ACK.json"
    ).read_bytes()
    request = Path(paths["shared_coordination"]) / "SIMPLEX_T_STAGE70_REQUEST.json"
    for receipt in sorted(
        (worktree / "artifacts/simplex_t/original_annotations").glob("*/DOWNLOAD_RECEIPT.json")
    ):
        payload[f"acquisition/{receipt.parent.name}/DOWNLOAD_RECEIPT.json"] = receipt.read_bytes()
    for original_audit in sorted(evidence.glob("ORIGINAL_RELEASE*.json")):
        payload[f"evidence/{original_audit.name}"] = original_audit.read_bytes()
    for name in (
        "qa_original_release.xml",
        "CURRENT_TABLE_AUDIT.json",
        "CURRENT_INPUT_INTEGRATION.json",
        "CURRENT_EXPOSURE_TIMING.json",
        "qa_current_inputs.xml",
        "qa_release_identity_dependencies.xml",
    ):
        payload[f"evidence/{name}"] = (evidence / name).read_bytes()
    payload["evidence/current_array_resume/RESUME_QA.json"] = (
        evidence / "current_array_resume/RESUME_QA.json"
    ).read_bytes()
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
