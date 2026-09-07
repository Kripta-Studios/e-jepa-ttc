"""Predetermine same-layout replay QA from the existing signed64 cohort."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .compiled_context import validate_block


def plan_h16_replay_qa(work: Path) -> dict:
    """Bind all 64 original QA queries, never select by observed discrepancies.

    This metadata-only plan does not acquire an inference slot or enable a fit.
    Execution must pin all completed reference receipts before reserving the
    same CURRENT_REPLAY.lock as production and loading any GPU model.
    """
    base = work / "artifacts/simplex_t"
    relative_pins = {
        "T0/coherent_fp32_extractor_receipt/QA.json": (
            "5801fee3e03f967b471344a72bdbaa9830679a1e0d71539478fc578c5a1694de"
        ),
        "T1/query_context_index/INDEX_MANIFEST.json": (
            "93a4f62e5025c5046fc82fcb1428a428f8a8df869b34486b92d5c753b0f68a3b"
        ),
        "T1/context_features_fp32/IDENTITY.json": (
            "00097c6e78bff173f9fe1497aae49da93f6ad7e86e678fe714da4269219955b1"
        ),
        "T1/query_context_dedup/DEDUP_MANIFEST.json": (
            "8f2bbfbacbb762a39d6d329dbe3f135fd58f6ac454e7117962c3e84edb60c1a3"
        ),
    }
    for relative, digest in relative_pins.items():
        if sha256(base / relative) != digest:
            raise ValueError("H16 QA input authority changed")
    qa = json.loads(
        (base / "T0/coherent_fp32_extractor_receipt/QA.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (base / "T1/query_context_index/INDEX_MANIFEST.json").read_text(encoding="utf-8")
    )
    index_path = base / "T1/query_context_index/query_context_index.npz"
    if sha256(index_path) != manifest["index_sha256"]:
        raise ValueError("H16 QA query index changed")
    with np.load(index_path, allow_pickle=False) as archive:
        tokens, assignments, valid = archive["tokens"], archive["producer_family"], archive["valid"]
    positions = {str(token): row for row, token in enumerate(tokens)}
    queries = []
    families = set()
    for record in qa["results"]:
        outer, role = record["family"].split("/")
        outer_id = int(outer[-1])
        family = 4 * outer_id + (3 if role == "outer_dev" else int(role[-1]))
        if (
            family in families
            or record["checkpoint_sha256"] != manifest["families"][family]["experts"]
        ):
            raise ValueError("duplicate or changed QA producer family")
        families.add(family)
        path = (
            base
            / "T0/coherent_fp32_extractor_receipt"
            / (record["family"].replace("/", "_") + ".npz")
        )
        if sha256(path) != record["output_sha256"]:
            raise ValueError("signed64 query source changed")
        with np.load(path, allow_pickle=False) as archive:
            selected = archive["tokens"]
        for token in selected:
            query = positions[str(token)]
            if assignments[outer_id, query] != family or not valid[query, -1]:
                raise ValueError("QA query producer/current availability differs")
            queries.append(
                {
                    "token": str(token),
                    "query": query,
                    "family": family,
                    "valid_slots": int(valid[query].sum()),
                    "reference_stem": f"family{family:02d}_query{query:05d}",
                }
            )
    if families != set(range(12)) or len(queries) != 64 or len({q["token"] for q in queries}) != 64:
        raise ValueError("entire existing signed64 cohort required")
    return {
        "schema": "simplex_t_same_layout_h16_qa_plan_v1",
        "status": "METADATA_PLAN_ONLY_NOT_EXECUTED",
        "selection": "Entire pre-existing signed64 cohort, independent of diagnostic differences",
        "input_pins_relative_to_simplex_artifacts": relative_pins,
        "queries": queries,
        "reference_cache": "T1/context_features_fp32",
        "shared_lease": "T1/CURRENT_REPLAY.lock",
        "expected_gpu_name": qa["runtime"]["device"],
        "comparison": "Exact shape, dtype and element equality for every valid H16 slot",
        "arrays": [
            "features145",
            "expert_ttc",
            "pair_features",
            "known",
            "observation_ids",
            "anchor_us",
            "available_us",
        ],
        "batch_size": 16,
        "precision": "FP32",
        "tf32": False,
        "all_reference_receipts_required_before_gpu": True,
        "optimizer_updates_authorized": 0,
        "scientific_fit_authorized": False,
        "numerical_amendment_authorized": False,
    }


def require_exact_h16_arrays(
    reference: dict[str, np.ndarray], observed: dict[str, np.ndarray]
) -> None:
    """No tolerance or reference replacement is permitted for same-layout replay."""
    expected = {
        "features145",
        "expert_ttc",
        "pair_features",
        "known",
        "observation_ids",
        "anchor_us",
        "available_us",
    }
    if set(reference) != expected or set(observed) != expected:
        raise ValueError("H16 QA array set differs")
    for name in sorted(expected):
        left, right = reference[name], observed[name]
        if (
            left.dtype != right.dtype
            or left.shape != right.shape
            or not np.array_equal(left, right)
        ):
            raise ValueError(f"same-layout H16 replay differs: {name}")


def bind_h16_reference_receipts(work: Path) -> dict:
    """Validate available immutable references; incomplete cohorts never admit QA."""
    plan = plan_h16_replay_qa(work)
    base = work / "artifacts/simplex_t"
    dedup = base / "T1/query_context_dedup"
    manifest = json.loads((dedup / "DEDUP_MANIFEST.json").read_text(encoding="utf-8"))
    histories = {}
    for outer, record in enumerate(manifest["outputs"]):
        path = dedup / f"outer{outer}.npz"
        if record["path"] != path.name or sha256(path) != record["sha256"]:
            raise ValueError("H16 reference observation index changed")
        with np.load(path, allow_pickle=False) as archive:
            histories[outer] = archive["history"]
    with np.load(
        base / "T1/query_context_index/query_context_index.npz", allow_pickle=False
    ) as archive:
        index = {name: archive[name] for name in archive.files}
    references, missing = [], []
    for item in plan["queries"]:
        stem = base / plan["reference_cache"] / item["reference_stem"]
        receipt, payload = stem.with_suffix(".json"), stem.with_suffix(".npz")
        if not receipt.exists():
            missing.append(item)
            continue
        receipt_hash = sha256(receipt)
        record = json.loads(receipt.read_text(encoding="utf-8"))
        query, family = item["query"], item["family"]
        if (
            record["query"] != query
            or record["family"] != family
            or sha256(payload) != record["sha256"]
        ):
            raise ValueError("H16 reference receipt or payload changed")
        with np.load(payload, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        mask = index["valid"][query]
        validate_block(
            arrays,
            histories[family // 4][query, mask],
            index["anchor_us"][query] - index["lag_us"][mask],
            int(index["roi_available_us"][query]),
        )
        if sha256(receipt) != receipt_hash or sha256(payload) != record["sha256"]:
            raise ValueError("H16 reference changed during validation")
        references.append(
            dict(item, receipt_sha256=receipt_hash, payload_sha256=record["sha256"])
        )
    return {
        "status": "REFERENCES_BOUND_NOT_REPLAYED" if not missing else "REFERENCES_INCOMPLETE",
        "plan": plan,
        "references": references,
        "missing": missing,
        "gpu_replay_performed": False,
        "optimizer_updates": 0,
        "scientific_admission": False,
    }
