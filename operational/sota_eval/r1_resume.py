"""Plan and resume R1 without re-ingesting completed independent stream groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol

from operational.efficient_context import r1_gib_measure, r1_profile
from operational.efficient_context.common import ROOT, Campaign, Lease, atomic_json, digest, read
from operational.efficient_context.r1_reader import ChunkSource, ResidentReplay

CAPACITY_BYTES = 1024**3
OUTPUT = ROOT / "artifacts/efficient_context_20261004/r1_20261008/resident_1g"
ARTIFACTS = ROOT / "artifacts/sota_campaign_20261008/r1_resume"


class NamedSource(ChunkSource, Protocol):
    """A sensor reader whose stream identity is available without opening it."""

    path: Path


def validate_receipt(receipt: dict, *, block: str, arm: str, query: dict, qi: int) -> None:
    """Reject mismatched or failed saved pairs before treating a group as complete."""
    import math

    rows = receipt["requests"]
    if len(rows) != 2 or {row["route"] for row in rows} != {"reference", "mapped"}:
        raise ValueError("saved receipt must contain the two admitted routes")
    for row in rows:
        if (row["block"], row["label"], row["query"], row["original_query"]) != (
            block,
            arm,
            query["sample_token"],
            qi,
        ) or row["regime"] != "R1_REPLAY":
            raise ValueError("saved R1 receipt identity changed")
        if not math.isfinite(row["milliseconds"]) or row["milliseconds"] < 0:
            raise ValueError("invalid saved request timing")
    evidence = receipt["parity"]
    if evidence["status"] not in ("EXACT", "NUMERICALLY_VALIDATED"):
        raise ValueError("failed parity cannot be resumed as completed")
    limits = dict(
        preprocessing=0, features=1e-4, times=1e-7, mask=0, experts=1e-5, phase=1e-5, ttc=0.01
    )
    for key, limit in limits.items():
        value = evidence["max_absolute_errors"][key]
        if not math.isfinite(value) or not 0 <= value <= limit:
            raise ValueError(f"saved parity exceeds frozen tolerance: {key}")


def receipt_snapshot(output: Path) -> dict:
    """Fingerprint all durable paired receipts without opening raw event payloads."""
    files = [
        dict(path=str(path.relative_to(output)), sha256=digest(path))
        for path in sorted((output / "measurement/fragments").glob("*.json"))
    ]
    aggregate = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode())
    return dict(count=len(files), files=files, aggregate_sha256=aggregate.hexdigest())


def verify_snapshot(output: Path, snapshot: dict) -> None:
    """Require every previously saved receipt to remain byte-identical."""
    for row in snapshot["files"]:
        if digest(output / row["path"]) != row["sha256"]:
            raise ValueError("pre-existing R1 receipt changed: " + row["path"])


def plan(output: Path) -> dict:
    """Build the exact construction/advance schedule of the immutable R1 runner."""
    import numpy as np

    from e_jepa_ttc.efficient_context.sparse_history import WIDE_SLOTS

    freeze = read(output / "SUPERVISOR_GIB_FREEZE.json")
    for path, expected in freeze["sources"].items():
        if digest(ROOT / path) != expected:
            raise ValueError("frozen original R1 source changed: " + path)
    if freeze["capacity_mib_per_route"] != 1024:
        raise ValueError("only the admitted one-GiB replay is supported")
    snapshot = receipt_snapshot(output)
    base = Campaign(ROOT / "configs/campaign/efficient_context_v1.json")
    base.freeze()
    protocol, indexes, _ = r1_profile.load_inputs(base)
    ordered = r1_profile.schedule(protocol, indexes)
    groups: list[dict] = []
    unseen = False
    first_missing = None
    saved_count = 0
    for block in r1_profile.BLOCKS:
        for arm, length in r1_profile.ARMS.items():
            group: dict | None = None
            for qi, query in ordered:
                if group is None or group["sequence"] != query["sequence_id"]:
                    group = dict(
                        block=block,
                        arm=arm,
                        sequence=query["sequence_id"],
                        source_path=str(base.raw / query["sequence_id"] / "events.h5"),
                        queries=[],
                    )
                    groups.append(group)
                index, i = indexes[query["pool"]], query["index_row"]
                if str(index["tokens"][i]) != query["sample_token"]:
                    raise ValueError("frozen query identity changed")
                valid = index["valid"][i].copy()
                if arm == "WIDE":
                    valid &= np.isin(np.arange(16), WIDE_SLOTS)
                else:
                    valid[:-length] = False
                intervals = index["base_windows_us"][i][None] - index["lag_us"][:, None, None]
                path = output / "measurement/fragments" / f"{block}_{arm}_{qi:02d}.json"
                saved = path.exists()
                if saved:
                    if unseen:
                        raise ValueError("saved receipts must be a chronological prefix, not holes")
                    validate_receipt(read(path), block=block, arm=arm, query=query, qi=qi)
                    saved_count += 1
                else:
                    unseen = True
                    if first_missing is None:
                        first_missing = dict(
                            block=block,
                            arm=arm,
                            sequence=query["sequence_id"],
                            query=query["sample_token"],
                            original_query=qi,
                        )
                group["queries"].append(
                    dict(
                        original_query=qi,
                        token=query["sample_token"],
                        saved=saved,
                        start_us=int(intervals[valid].min()),
                        cutoff_us=int(intervals[valid].max()),
                    )
                )
    if saved_count != snapshot["count"]:
        raise ValueError("unexpected receipt files outside the admitted query set")
    for group in groups:
        group["saved_count"] = sum(q["saved"] for q in group["queries"])
        group["complete"] = group["saved_count"] == len(group["queries"])
        group["restore_prefix_count"] = 0 if group["complete"] else group["saved_count"]
    verify_snapshot(output, snapshot)
    return dict(
        schema="R1_SEQUENCE_BOUNDARY_RESUME_V1",
        sources_preserved=True,
        capacity_bytes_per_route=CAPACITY_BYTES,
        optimizer_updates=0,
        parent_freeze_sha256=digest(output / "SUPERVISOR_GIB_FREEZE.json"),
        wrapper_sha256=digest(Path(__file__)),
        receipt_snapshot=snapshot,
        completed_pairs=saved_count,
        remaining_pairs=768 - saved_count,
        complete_groups=sum(g["complete"] for g in groups),
        partial_groups=sum(0 < g["saved_count"] < len(g["queries"]) for g in groups),
        restore_prefix_advances_per_route=sum(g["restore_prefix_count"] for g in groups),
        first_missing=first_missing,
        groups=groups,
        cache_semantics="Each independent group owns two sensor readers; no state crosses groups.",
        measurement_semantics="Partial-group restoration occurs inside existing recovery_ms timer.",
        coldness_limit="OS/HDF5 storage coldness remains uncontrolled as in original R1 protocol.",
    )


class ResumeReader(ResidentReplay):
    """Skip complete groups; restore a partial group's exact advance sequence."""

    def __init__(self, source: ChunkSource, capacity_bytes: int, group: dict) -> None:
        super().__init__(source, capacity_bytes)
        self.group = group
        self.position = 0
        self.pending: list[tuple[int, int]] = []

    def advance(self, start_us: int, cutoff_us: int) -> None:
        """Do no replay restoration inside a new timed request."""
        if self.position >= len(self.group["queries"]):
            raise ValueError("more reader advances than the frozen group plan")
        query = self.group["queries"][self.position]
        if (start_us, cutoff_us) != (query["start_us"], query["cutoff_us"]):
            raise ValueError("reader advance does not match the frozen chronological plan")
        self.position += 1
        if query["saved"]:
            if self.group["complete"]:
                return
            self.pending.append((start_us, cutoff_us))
            if self.position == self.group["saved_count"]:
                for first, last in self.pending:
                    super().advance(first, last)
                self.pending.clear()
            return
        if self.pending:
            raise ValueError("recovery would contaminate a new measured request")
        super().advance(start_us, cutoff_us)


@contextmanager
def installed(plan_data: dict) -> Iterator[list[dict]]:
    """Patch only the constructor binding of an unchanged original runner."""
    contexts = iter(
        (group, route) for group in plan_data["groups"] for route in ("reference", "mapped")
    )
    audits: list[dict] = []
    original = r1_profile.ResidentReplay

    class PlannedReader(ResumeReader):
        def __init__(self, source: NamedSource, capacity_bytes: int) -> None:
            try:
                group, _route = next(contexts)
            except StopIteration as exc:
                raise ValueError("unexpected reader construction outside frozen plan") from exc
            if source.path != Path(group["source_path"]):
                raise ValueError("reader source or group construction order changed")
            if capacity_bytes != plan_data["capacity_bytes_per_route"]:
                raise ValueError("resident capacity differs from admitted plan")
            super().__init__(source, capacity_bytes, group)
            self.audit = dict(position=0, planned_queries=len(group["queries"]))
            audits.append(self.audit)

        def advance(self, start_us: int, cutoff_us: int) -> None:
            super().advance(start_us, cutoff_us)
            self.audit["position"] = self.position

    r1_profile.ResidentReplay = PlannedReader
    try:
        yield audits
    finally:
        r1_profile.ResidentReplay = original


def execute(output: Path, artifacts: Path, plan_data: dict) -> None:
    """Resume only after the root coordinator has released its stop and lease."""
    if (output / "STOP_REQUEST").exists():
        raise RuntimeError("R1 remains paused; coordinator must release its owned stop first")
    verify_snapshot(output, plan_data["receipt_snapshot"])
    previous_argv = sys.argv
    with Lease(output), installed(plan_data) as audits:
        sys.argv = [previous_argv[0], "--output", str(output)]
        try:
            code = r1_gib_measure.main()
            if code != 0:
                raise RuntimeError(f"original R1 runner returned {code}")
            if len(audits) != 2 * len(plan_data["groups"]) or any(
                audit["position"] != audit["planned_queries"] for audit in audits
            ):
                raise ValueError("original runner did not consume the planned groups exactly")
        finally:
            sys.argv = previous_argv
            verify_snapshot(output, plan_data["receipt_snapshot"])
            atomic_json(
                artifacts / "RECEIPTS_AFTER.json",
                dict(
                    status="PRESERVED",
                    original_count=plan_data["receipt_snapshot"]["count"],
                    original_aggregate_sha256=plan_data["receipt_snapshot"]["aggregate_sha256"],
                    current_snapshot=receipt_snapshot(output),
                    optimizer_updates=0,
                ),
            )


def main() -> int:
    """Default is CPU-only planning; --execute explicitly enables frozen inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--artifacts", type=Path, default=ARTIFACTS)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    output, artifacts = args.output.resolve(), args.artifacts.resolve()
    plan_data = plan(output)
    atomic_json(artifacts / "PLAN.json", plan_data)
    if args.execute:
        execute(output, artifacts, plan_data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
