"""Bind declared observed invocation receipts; never derive fits from attempt counts."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .bundle_creation import BundleMember
from .bundle_integrity import validate_bundle_inventory


@dataclass(frozen=True)
class ResourceAttempt:
    """One pinned launch and its corresponding terminal observation receipt."""

    launch: Path
    launch_sha256: str
    receipt: Path
    receipt_sha256: str


def resource_bundle_members(
    attempts: list[ResourceAttempt],
    *,
    work_root: Path,
    freeze_sha256: str,
    completed_stages: set[str],
    resource_ok: Callable[[], bool],
) -> dict[str, BundleMember]:
    """Verify metadata identity and counters for every explicitly supplied attempt.

    Include paused/failed attempts as observations, not completed fits. Coverage
    requires a declared receipt for every completed stage, but cannot prove that
    a hard-killed process left a receipt or that the supplied list is exhaustive.
    Preserve that limitation in reports: measured elapsed time is not a complete
    end-to-end campaign duration. No referenced data/model files are opened.
    """
    if not attempts or not completed_stages or not completed_stages <= {"T2", "T3", "T4", "T5"}:
        raise ValueError("observed attempts and registered completed stages required")
    work = work_root.resolve(strict=True)
    members: dict[str, BundleMember] = {}
    stages: set[str] = set()
    seen: set[Path] = set()

    def read(path: Path, digest: str, limit: int) -> dict:
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: observed resource transport")
        target = path.resolve(strict=True)
        if not target.is_relative_to(work) or target in seen:
            raise ValueError("duplicate or out-of-scope resource evidence")
        seen.add(target)
        with target.open("rb") as stream:
            payload = stream.read(limit + 1)
        if len(payload) > limit or hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("resource evidence bytes or size differ")
        name = "resources/" + target.relative_to(work).as_posix()
        members[name] = BundleMember(target, digest, len(payload))
        return json.loads(payload)

    for attempt in attempts:
        launch = read(attempt.launch, attempt.launch_sha256, 1_048_576)
        receipt = read(attempt.receipt, attempt.receipt_sha256, 65_536)
        if (
            launch.get("schema") != "simplex_t_frozen_phase_launch_v2"
            or receipt.get("schema") != "simplex_t_resource_observations_v1"
            or launch["freeze_sha256"] != freeze_sha256
            or receipt["freeze_sha256"] != freeze_sha256
            or receipt["launch_sha256"] != attempt.launch_sha256
            or launch["stage"] != receipt["stage"]
            or receipt["stage"] not in completed_stages
            or Path(launch["resource_receipt"]).resolve(strict=True)
            != attempt.receipt.resolve(strict=True)
            or receipt.get("continuous_peak_measurement") is not False
            or receipt.get("hard_termination_receipt_guaranteed") is not False
            or receipt.get("campaign_complete") is not False
            or receipt.get("observer_optimizer_updates") != 0
            or receipt.get("vram_measurement") is not None
            or receipt.get("disk_floor_after_reservations_bytes") != 40_000_000_000
            or receipt.get("timing_scope")
            != "INVOCATION_ADMISSION_LOADING_FITS_AND_PUBLICATION_NOT_OPTIMIZER_ONLY"
            or type(receipt.get("resume_requested")) is not bool
            or not isinstance(receipt.get("execution_result_status"), str)
            or not receipt["execution_result_status"]
        ):
            raise ValueError("resource receipt changes launch identity or measurement scope")
        stages.add(receipt["stage"])
        for field in (
            "admission_samples",
            "denied_admission_samples",
            "missing_counter_samples",
            "other_reserved_bytes",
            "own_reserved_bytes",
        ):
            if type(receipt[field]) is not int or receipt[field] < 0:
                raise ValueError("nonnegative integer resource counter required")
        samples = receipt["admission_samples"]
        if (
            receipt["denied_admission_samples"] > samples
            or receipt["missing_counter_samples"] > samples
            or receipt["own_reserved_bytes"] < 65536
        ):
            raise ValueError("resource sample or reservation counters differ")
        elapsed = receipt["elapsed_seconds"]
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (float, int))
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError("finite nonnegative observed elapsed time required")
        for field in (
            "sampled_process_tree_rss_max_bytes",
            "sampled_host_available_min_bytes",
            "sampled_written_volume_free_min_bytes",
        ):
            value = receipt[field]
            if samples == receipt["missing_counter_samples"]:
                if value is not None:
                    raise ValueError("missing measurements cannot become zero-valued samples")
            elif type(value) is not int or value < 0:
                raise ValueError("observed byte extrema required")
    if stages != completed_stages:
        raise ValueError("resource observations omit completed stages")
    validate_bundle_inventory({name: member.sha256 for name, member in members.items()})
    return members
