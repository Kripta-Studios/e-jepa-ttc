"""Resumable independent H16 recomputation under the production replay lease."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

import numpy as np

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json

from .h16_qa_plan import bind_h16_reference_receipts, require_exact_h16_arrays
from .lifecycle import ExclusiveLease


def execute_h16_qa(
    work: Path,
    output: Path,
    *,
    family_factory: Callable[[int], AbstractContextManager[Callable[[int], dict[str, np.ndarray]]]],
    validate_prerequisites: Callable[[], None],
    resource_ok: Callable[[], bool],
    execution_identity: dict,
) -> dict:
    """Keep all outputs, including mismatches; no tolerance or retry of failed rows.

    Models are acquired only after all 64 references are verified and the shared
    lease is held. Completed rows are independently rechecked on resume. A crash
    between payload and receipt leaves an explicit orphan, never overwritten.
    """
    work = work.resolve(strict=True)
    output = output.resolve()
    if not output.is_relative_to(work / "artifacts/simplex_t/T0"):
        raise ValueError("independent QA output must remain under companion T0")
    validate_prerequisites()
    bound = bind_h16_reference_receipts(work)
    if bound["missing"]:
        raise ValueError("WAITING_H16_REFERENCE_COHORT: no GPU model was loaded")
    contract = {"references": bound, "execution_identity": execution_identity}
    with ExclusiveLease(work / "artifacts/simplex_t/T1/CURRENT_REPLAY.lock"):
        if not resource_ok():
            return {"status": "RESOURCE_PAUSE", "completed": 0, "optimizer_updates": 0}
        output.mkdir(parents=True, exist_ok=True)
        identity = output / "CONTRACT.json"
        if identity.exists():
            if json.loads(identity.read_text(encoding="utf-8")) != contract:
                raise ValueError("H16 QA contract changed; resume refused")
        else:
            write_new_json(identity, contract)
        completed, pending = [], {}

        def reference(item: dict) -> dict[str, np.ndarray]:
            stem = work / "artifacts/simplex_t/T1/context_features_fp32" / item["reference_stem"]
            if (
                sha256(stem.with_suffix(".json")) != item["receipt_sha256"]
                or sha256(stem.with_suffix(".npz")) != item["payload_sha256"]
            ):
                raise ValueError("bound H16 reference changed")
            with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
                return {name: archive[name] for name in archive.files}

        for item in bound["references"]:
            stem = output / item["reference_stem"]
            receipt = stem.with_suffix(".json")
            if receipt.exists():
                saved = json.loads(receipt.read_text(encoding="utf-8"))
                if (
                    saved["reference"] != item
                    or sha256(stem.with_suffix(".npz")) != saved["sha256"]
                ):
                    raise ValueError("saved H16 QA output changed")
                with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
                    require_exact_h16_arrays(reference(item), dict(archive))
                if saved["status"] != "EXACT_PASS":
                    raise ValueError("prior H16 QA failure preserved; no automatic retry")
                completed.append(saved)
            else:
                if stem.with_suffix(".npz").exists() or stem.with_suffix(".partial").exists():
                    raise ValueError("orphan H16 QA payload preserved; inspect before resuming")
                pending.setdefault(item["family"], []).append(item)
        for family, queries in sorted(pending.items()):
            validate_prerequisites()
            if not resource_ok():
                return {
                    "status": "RESOURCE_PAUSE",
                    "completed": len(completed),
                    "optimizer_updates": 0,
                }
            with family_factory(family) as infer:
                for item in queries:
                    validate_prerequisites()
                    if not resource_ok():
                        return {
                            "status": "RESOURCE_PAUSE",
                            "completed": len(completed),
                            "optimizer_updates": 0,
                        }
                    expected = reference(item)
                    started = time.perf_counter()
                    observed = infer(item["query"])
                    elapsed = time.perf_counter() - started
                    stem = output / item["reference_stem"]
                    with stem.with_suffix(".partial").open("xb") as stream:
                        np.savez_compressed(stream, allow_pickle=False, **observed)
                    stem.with_suffix(".partial").rename(stem.with_suffix(".npz"))
                    error = None
                    try:
                        require_exact_h16_arrays(expected, observed)
                    except ValueError as mismatch:
                        error = str(mismatch)
                    saved = {
                        "reference": item,
                        "sha256": sha256(stem.with_suffix(".npz")),
                        "status": "EXACT_PASS" if error is None else "EXACT_FAIL",
                        "error": error,
                        "inference_seconds": elapsed,
                    }
                    write_new_json(stem.with_suffix(".json"), saved)
                    if error is not None:
                        raise ValueError(error)
                    completed.append(saved)
        validate_prerequisites()
        for item in bound["references"]:
            reference(item)
        result = {
            "status": "SAME_LAYOUT_H16_EXACT_PASS",
            "completed": len(completed),
            "contract_sha256": sha256(identity),
            "results": completed,
            "optimizer_updates": 0,
            "scientific_admission": False,
        }
        report = output / "QA.json"
        if report.exists():
            if json.loads(report.read_text(encoding="utf-8")) != result:
                raise ValueError("completed H16 QA report changed")
        else:
            write_new_json(report, result)
        return result
