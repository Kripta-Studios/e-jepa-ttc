"""Read actual same-layout replay arrays, not a claimed success flag."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .coordination import verified_ack
from .h16_qa_plan import bind_h16_reference_receipts, require_exact_h16_arrays


def verify_h16_execution_identity(work: Path, local_paths: Path, record: dict) -> None:
    """Bind the executable's full declared identity to current audited inputs."""
    paths = json.loads(local_paths.read_text(encoding="utf-8"))
    work = work.resolve(strict=True)
    if Path(paths["worktree"]).resolve(strict=True) != work:
        raise ValueError("H16 execution worktree differs")
    base = work / "artifacts/simplex_t"
    cache = json.loads((base / "T1/context_features_fp32/IDENTITY.json").read_text("utf-8"))
    qa = json.loads((base / "T0/coherent_fp32_extractor_receipt/QA.json").read_text("utf-8"))
    # bind_h16_reference_receipts independently verifies these fixed input hashes.
    code = {
        "src/e_jepa_ttc/simplex_t/" + filename: cache[field]
        for filename, field in (
            ("expert_features.py", "extractor_sha256"),
            ("expert_phase.py", "expert_phase_sha256"),
            ("query_context_voxel.py", "voxel_sha256"),
            ("context_raw_union.py", "union_reader_sha256"),
        )
    }
    for relative in (
        "src/e_jepa_ttc/simplex_t/expanded_inference.py",
        "src/e_jepa_ttc/simplex_t/h16_qa_execution.py",
        "src/e_jepa_ttc/simplex_t/h16_qa_plan.py",
        "scripts/run_simplex_t_h16_replay_qa.py",
    ):
        code[relative] = sha256(work / relative)
    for relative, digest in code.items():
        if sha256(work / relative) != digest:
            raise ValueError("H16 numerical implementation differs from cache")
    # JSON emitted on Windows uses native path separators for this inventory.
    code = {str(Path(relative)): digest for relative, digest in code.items()}
    ack_hash = "3e55ab3c6e9a57eecd862ad05e999627ea90957e58e329b2eb3652120e953318"
    verified_ack(Path(paths["shared_coordination"]) / "SIMPLEX_T_STAGE70_ACK.json", ack_hash)
    expected = {
        "code": code,
        "local_paths_sha256": sha256(local_paths),
        "ack_sha256": ack_hash,
        "preprocessing_sha256": cache["preprocessing_sha256"],
        "device": qa["runtime"]["device"],
        "torch": cache["torch"],
    }
    if record != expected:
        raise ValueError("real H16 executable identity differs")


def verify_h16_replay(
    work: Path,
    root: Path,
    expected_report_sha256: str,
    *,
    validate_execution_identity: Callable[[dict], None],
    resource_ok: Callable[[], bool],
) -> dict:
    """Recheck every bound reference and independent output; no inference or writes.

    The mandatory identity validator must bind the real executable, source code,
    producer authority, preprocessing and runtime. This reader cannot establish
    those facts from a caller-written success flag or an arbitrary code inventory.
    """
    work, root = work.resolve(strict=True), root.resolve(strict=True)
    if not root.is_relative_to(work / "artifacts/simplex_t/T0"):
        raise ValueError("H16 QA evidence outside companion T0")

    def read(path: Path, digest: str | None = None) -> dict:
        if path.stat().st_size > 8_388_608 or (digest is not None and sha256(path) != digest):
            raise ValueError("H16 evidence size or hash mismatch")
        return json.loads(path.read_text(encoding="utf-8"))

    def boundary() -> None:
        if not resource_ok():
            raise InterruptedError("RESOURCE_PAUSE: H16 evidence verification")

    boundary()
    report = read(root / "QA.json", expected_report_sha256)
    if (
        report.get("status") != "SAME_LAYOUT_H16_EXACT_PASS"
        or type(report.get("completed")) is not int
        or report["completed"] != 64
        or type(report.get("optimizer_updates")) is not int
        or report["optimizer_updates"] != 0
        or report.get("scientific_admission") is not False
        or len(report["results"]) != 64
    ):
        raise ValueError("complete real signed64 H16 report required")
    contract = read(root / "CONTRACT.json", report["contract_sha256"])
    validate_execution_identity(contract["execution_identity"])
    bound = bind_h16_reference_receipts(work)
    if bound["missing"] or contract["references"] != bound:
        raise ValueError("H16 QA references differ from the complete real cohort")
    expected = bound["references"]
    if len(expected) != 64 or [r["reference"] for r in report["results"]] != expected:
        raise ValueError("H16 QA cohort reordered, repeated or changed")
    checked = []
    for item, result in zip(expected, report["results"], strict=True):
        boundary()
        stem = root / item["reference_stem"]
        receipt_hash = sha256(stem.with_suffix(".json"))
        receipt = read(stem.with_suffix(".json"), receipt_hash)
        if receipt != result or result["status"] != "EXACT_PASS" or result["error"] is not None:
            raise ValueError("H16 QA verdict or receipt changed")
        if sha256(stem.with_suffix(".npz")) != result["sha256"]:
            raise ValueError("H16 independent output changed")
        reference = work / "artifacts/simplex_t/T1/context_features_fp32" / item["reference_stem"]
        with (
            np.load(reference.with_suffix(".npz"), allow_pickle=False) as left,
            np.load(stem.with_suffix(".npz"), allow_pickle=False) as right,
        ):
            require_exact_h16_arrays(dict(left), dict(right))
        checked.append((stem.with_suffix(".npz"), result["sha256"]))
        checked.append((stem.with_suffix(".json"), receipt_hash))
    boundary()
    validate_execution_identity(contract["execution_identity"])
    if bind_h16_reference_receipts(work) != bound:
        raise ValueError("H16 reference cohort changed during verification")
    for path, digest in checked:
        if sha256(path) != digest:
            raise ValueError("H16 independent evidence changed during verification")
    read(root / "QA.json", expected_report_sha256)
    read(root / "CONTRACT.json", report["contract_sha256"])
    return {
        "status": "REAL_H16_ARRAY_PARITY_VERIFIED_NOT_SCIENTIFIC_ADMISSION",
        "report_sha256": expected_report_sha256,
        "contract_sha256": report["contract_sha256"],
        "queries": 64,
        "optimizer_updates_executed": 0,
        "scientific_admission": False,
    }
