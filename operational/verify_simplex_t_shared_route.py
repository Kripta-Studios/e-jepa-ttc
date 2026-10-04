"""Verify hashes, cost quantiles and frozen CPU head predictions from a route bundle."""

# JSON model constructors and NPZ state dictionaries are checked dynamic payloads.
# ruff: noqa: ANN401
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np


def record(path: Path) -> Any:
    """Read a declared receipt."""
    return json.loads(path.read_bytes())


def sha(path: Path) -> str:
    """Bounded file hashing."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def summaries(rows: list[dict], *, require_full: bool = True) -> list[dict]:
    """Recompute linear quantiles from the durable measured requests."""
    output = []
    blocks = {(r["model"], r["mode"]) for r in rows}
    query_blocks = {
        token: {(r["model"], r["mode"]) for r in rows if r["sample_token"] == token}
        for token in {r["sample_token"] for r in rows}
    }
    complete_queries = {token for token, seen in query_blocks.items() if seen == blocks}
    if len(blocks) != 27 or not complete_queries or (require_full and len(complete_queries) != 64):
        raise ValueError("matched complete query blocks required")
    rows = [r for r in rows if r["sample_token"] in complete_queries]
    for model, mode in sorted({(r["model"], r["mode"]) for r in rows}):
        values = [r for r in rows if (r["model"], r["mode"]) == (model, mode)]
        if len(values) != len(complete_queries):
            raise ValueError("duplicate or missing matched query request")
        row: dict[str, Any] = dict(
            model=model,
            mode=mode,
            measurements=len(complete_queries),
            planned_queries=64,
            inventory_complete=require_full,
            shared_gpu=True,
            scope="supplied_roi_resident_estimator_raw_to_ttc",
        )
        for field in (
            "total_ms",
            "raw_read_ms",
            "roi_voxel_ms",
            "experts_transfers_ms",
            "normalize_ms",
            "head_emission_ms",
        ):
            raw = np.asarray([v[field] for v in values], np.float64)
            if not np.isfinite(raw).all() or (raw < 0).any():
                raise ValueError("invalid measured stage time")
            row[field + "_p50"] = float(np.quantile(raw, 0.5))
            row[field + "_p95"] = float(np.quantile(raw, 0.95))
        row["own_gpu_allocated_bytes_max"] = max(v["gpu_allocated_bytes"] for v in values)
        row["own_gpu_reserved_bytes_max"] = max(v["gpu_reserved_bytes"] for v in values)
        output.append(row)
    return output


def verify(root: Path, heads: bool) -> dict:
    """Check included evidence; raw events and elapsed wall time are not regenerated."""
    manifest = record(root / "CONTENT_MANIFEST.json")
    for name, row in manifest["members"].items():
        path = (root / name).resolve(strict=True)
        if not path.is_relative_to(root.resolve()) or path.stat().st_size != row["bytes"]:
            raise ValueError("unsafe or changed member")
        if sha(path) != row["sha256"]:
            raise ValueError("member SHA changed")
    rows = [record(p) for p in sorted((root / "results/fragments").glob("*.json"))]
    accounting = record(root / "ACCOUNTING.json")
    if len(rows) != accounting["measured_fragments"] or not 0 < len(rows) <= 1728:
        raise ValueError("measured fragment inventory changed")
    is_complete = len(rows) == 1728
    protocol = record(root / "PROTOCOL.json")
    protocol_hash = sha(root / "PROTOCOL.json")
    planned = {
        (q["sample_token"], label, mode)
        for q in protocol["queries"]
        for label in protocol["models"]
        for mode in protocol["modes"]
    }
    observed = {(r["sample_token"], r["model"], r["mode"]) for r in rows}
    if len(planned) != 1728 or len(observed) != len(rows) or not observed <= planned or (
        is_complete and observed != planned
    ):
        raise ValueError("observed IDs do not equal the prospectively fixed queue")
    query_map = {q["sample_token"]: q for q in protocol["queries"]}
    checked_inputs = set()
    for row in rows:
        if row["protocol_sha256"] != protocol_hash:
            raise ValueError("timing fragment has a different protocol binding")
        query = query_map[row["sample_token"]]
        if row["family"] != query["family_id"] or row["sequence_id"] != query["sequence_id"]:
            raise ValueError("query producer family changed")
        label = row["model"]
        allowed = (
            {"A5"} if label == "A5_ONLY_C0" else
            {"C2F"} if label == "C2F_ONLY_C0" else
            {"A5", "PAIR"} if label == "A5_PAIR_C0" else
            {"A5", "C2F", "PAIR"}
        )
        calls = {name: int(name in allowed) for name in ("A5", "C2F", "PAIR")}
        if row["actual_producer_calls"] != calls or set(row["allowed_producers"]) != allowed:
            raise ValueError("excluded producer was called or required producer missing")
        key = row["inputs_relative_path"]
        member = manifest["members"].get(key)
        if member is None or member["sha256"] != row["inputs_sha256"]:
            raise ValueError("captured head input binding changed")
        checked_inputs.add(key)
        parity = row["parity"]
        bounds = {"features": 1e-4, "times": 1e-7, "valid": 0, "experts": 1e-5}
        if any(not 0 <= parity["input_errors"][k] <= v for k, v in bounds.items()):
            raise ValueError("captured input fails the fixed numerical admission")
        if not 0 <= parity["point_phase_error"] <= 1e-5 or not (
            0 <= parity["ttc_error_seconds"] <= .01
        ):
            raise ValueError("TTC emission fails numerical admission")
        partition = sum(row[k] for k in (
            "raw_read_ms", "roi_voxel_ms", "experts_transfers_ms", "normalize_ms",
            "head_emission_ms",
        ))
        if abs(partition - row["total_ms"]) > 1e-6:
            raise ValueError("stage timers do not partition the direct total")
    expected = list(csv.DictReader((root / "ROUTE_COST.csv").open(encoding="utf-8")))
    actual = summaries(rows, require_full=is_complete)
    if len(actual) != 27 or len(expected) != 27:
        raise ValueError("nine routes across three modes required")
    errors = [
        abs(float(a[k]) - float(e[k]))
        for a, e in zip(actual, expected, strict=True)
        for k in a
        if k.endswith(("_p50", "_p95"))
    ]
    if any(
        (a["model"], a["mode"]) != (e["model"], e["mode"])
        for a, e in zip(actual, expected, strict=True)
    ):
        raise ValueError("cost table labels changed")
    shares = list(csv.DictReader((root / "STAGE_SHARES.csv").open(encoding="utf-8")))
    if len(shares) != 27:
        raise ValueError("stage contribution inventory changed")
    query_counts = {
        token: sum(r["sample_token"] == token for r in rows)
        for token in {r["sample_token"] for r in rows}
    }
    matched = [r for r in rows if query_counts[r["sample_token"]] == 27]
    contribution_errors = []
    for share in shares:
        group = [r for r in matched if (r["model"], r["mode"]) == (
            share["model"], share["mode"]
        )]
        if len(group) != int(share["measurements"]):
            raise ValueError("stage share query count differs")
        total = sum(r["total_ms"] for r in group)
        for field in (
            "raw_read_ms", "roi_voxel_ms", "experts_transfers_ms",
            "normalize_ms", "head_emission_ms",
        ):
            contribution_errors.append(abs(
                sum(r[field] for r in group) / total
                - float(share[field + "_fraction_of_measured_sum"])
            ))
    if max(contribution_errors) > 1e-12:
        raise ValueError("stage contribution differs from physical fragments")
    accuracy = {
        r["model"]: r
        for r in csv.DictReader((root / "audit/ACCURACY_COST_FRONTIER.csv").open(
            encoding="utf-8"
        ))
    }
    frontier = list(csv.DictReader((root / "COST_ACCURACY_SHARED.csv").open(encoding="utf-8")))
    if len(frontier) != 9 or {r["model"] for r in frontier} != set(protocol["models"]):
        raise ValueError("accuracy/cost model inventory changed")
    for row in frontier:
        if float(row["OLD_DEV_MiD"]) != float(accuracy[row["OLD_DEV_evidence_model"]]["MiD"]):
            raise ValueError("referenced full-precision OLD_DEV score differs")
    parity_receipts = [
        record(path)
        for path in sorted((root / "results/parity_historical_runtime").glob("*.json"))
    ]
    if len(parity_receipts) != 64 or any(
        r["protocol_sha256"] != protocol_hash for r in parity_receipts
    ):
        raise ValueError("64 admitted raw-context parity receipts required")
    prediction_rows = rows + [r["observation"] for r in parity_receipts]
    for row in prediction_rows:
        key = row["inputs_relative_path"]
        member = manifest["members"].get(key)
        if member is None or member["sha256"] != row["inputs_sha256"]:
            raise ValueError("admitted parity input binding changed")
        checked_inputs.add(key)
    prediction_error = None
    replayed = 0
    if heads:
        guard = runpy.run_path(str(root / "resource_guard.py"))["resource_guard"]
        guard(root)
        sys.path[:0] = [str(root / "provenance"), str(root / "provenance/src")]
        import torch

        from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
        from e_jepa_ttc.simplex_t.phase import phase_to_ttc
        from operational.simplex_t_cost_context.model import build_model

        torch.set_num_threads(4)
        torch.set_num_interop_threads(2)
        torch.use_deterministic_algorithms(True)
        differences = []
        index = record(root / "HEAD_INDEX.json")
        with torch.inference_mode():
            for label, row in index["models"].items():
                guard(root)
                if sha(root / row["weights_path"]) != row["weights_sha256"]:
                    raise ValueError("head endpoint binding changed")
                c = row["constructor"]
                model = (
                    TemporalRefiner(TemporalConfig(**c["config"]))
                    if c["kind"] == "historical_temporal"
                    else build_model(c["arm"])
                )
                with np.load(root / row["weights_path"], allow_pickle=False) as archive:
                    model.load_state_dict(
                        {k: torch.from_numpy(archive[k].copy()) for k in archive.files}, strict=True
                    )
                model.cpu().float().eval()
                cache = {}
                for measured in [r for r in prediction_rows if r["model"] == label]:
                    path = measured["inputs_relative_path"]
                    if path not in cache:
                        with np.load(root / path, allow_pickle=False) as archive:
                            xs = tuple(
                                torch.from_numpy(archive[k].copy())
                                for k in ("features", "times", "valid", "experts")
                            )
                        output = model(*xs)
                        cache[path] = (
                            float(output["point_phase"][0]),
                            float(phase_to_ttc(output["point_phase"].to(torch.float64))[0]),
                        )
                        replayed += 1
                    phase, ttc = cache[path]
                    differences.extend(
                        [abs(phase - measured["point_phase"]), abs(ttc - measured["ttc_seconds"])]
                    )
        prediction_error = max(differences)
        if prediction_error > 1e-6:
            raise ValueError("included CPU head point/ttc differs")
    if max(errors) > 1e-9:
        raise ValueError("cost table differs from physical timing fragments")
    return dict(
        status="VERIFIED",
        inventory_complete=is_complete,
        planned_fragments=1728,
        pending_fragments=1728 - len(rows),
        members=len(manifest["members"]),
        fragments=len(rows),
        cost_rows=27,
        captured_input_files=len(checked_inputs),
        raw_parity_predictions_checked=64,
        prediction_scope=(
            "measured requests and admitted raw parity; failed diagnostic evidence hashed"
        ),
        prospective_inventory_verified=True,
        excluded_producer_calls_verified=True,
        maximum_quantile_difference=max(errors),
        maximum_stage_contribution_difference=max(contribution_errors),
        referenced_accuracy_scores_verified=9,
        heads_replayed=heads,
        unique_head_inputs_replayed=replayed,
        maximum_prediction_difference=prediction_error,
        optimizer_updates=0,
        raw_reconstruction=False,
        timings_recollected=False,
        gpu_shared=True,
    )


def main() -> None:
    """CLI for independent extracted-bundle verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", action="store_true")
    args = parser.parse_args()
    result = verify(args.root.resolve(strict=True), args.heads)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
