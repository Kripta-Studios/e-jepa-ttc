"""Research admission, immutable run receipts and conservative decision rules."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal


def sha256(path: Path) -> str:
    """Hash bytes without loading large media/checkpoints into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Atomically write strict JSON, rejecting NaN and infinity."""
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_artifacts(root: Path, artifacts: dict[str, str]) -> None:
    """Verify a receipt only within its declared root; never accept traversal paths."""
    if not artifacts:
        raise ValueError("Empty artifact binding")
    root = root.resolve()
    for name, digest in artifacts.items():
        path = (root / name.replace("\\", "/")).resolve()
        if not path.is_relative_to(root) or len(digest) != 64 or sha256(path) != digest:
            raise ValueError(f"Artifact identity differs: {name}")


@dataclass(frozen=True)
class Admission:
    """Conditional family admission; evidence hashes refer to reviewed diagnostic receipts."""

    family: Literal["gate", "clock", "sensor", "visibility", "jepa"]
    updates: int
    arms: int
    seeds: tuple[int, ...]
    exposure: Literal["development", "confirmation"]
    evidence: dict[str, str]

    def validate(self, root: Path) -> None:
        """Enforce screen budget and family-specific evidence before GPU allocation.

        Receipt verification authenticates bytes, not their scientific conclusion;
        explicit reviewed `decision=admit` and scope fields are required too.
        """
        requirements = {
            "gate": {"incremental_observability"},
            "clock": {"observer_beats_simple_controls"},
            "sensor": {"photometric_identification", "sensor_alignment"},
            "visibility": {"coordinate_audit", "visibility_incremental"},
            "jepa": {
                "observed_q_recoverable",
                "producer_dynamic_failure",
                "readout_controls_excluded",
            },
        }
        if self.family not in requirements or self.exposure not in ("development", "confirmation"):
            raise ValueError("Unknown family/exposure")
        if not 1 <= self.updates <= 1000 or not 1 <= self.arms <= 4:
            raise ValueError("Screen cap: 1000 updates and four matched arms")
        expected = 1 if self.exposure == "development" else 3
        if len(set(self.seeds)) != expected or len(self.seeds) != expected:
            raise ValueError("Use one screen seed or three distinct confirmation seeds")
        verify_artifacts(root, self.evidence)
        observed = set()
        for name in self.evidence:
            receipt = json.loads((root / name).read_text(encoding="utf-8"))
            if receipt.get("decision") != "admit" or receipt.get("family") != self.family:
                raise ValueError("Diagnostic evidence has not admitted this learned family")
            observed.update(receipt.get("conditions", []))
        if not requirements[self.family] <= observed:
            raise ValueError(
                f"Missing family conditions: {sorted(requirements[self.family] - observed)}"
            )


def validate_group_roles(train: set[str], calibration: set[str], evaluation: set[str]) -> None:
    """Keep acquisitions, not windows, disjoint; producer exposure is declared separately."""
    if not train or not calibration or not evaluation:
        raise ValueError("Three nonempty acquisition role sets are required")
    if train & calibration or train & evaluation or calibration & evaluation:
        raise ValueError("Acquisition leakage across fitting, calibration or evaluation")


def promotion_decision(
    baseline_mae: float,
    candidate_mae: float,
    baseline_relative: float,
    candidate_relative: float,
    newly_missing: int,
    sign_urgency_regression: bool,
    *,
    confirmation: bool = False,
    delta_ci_upper: float | None = None,
    native_simultaneous_ci_upper: dict[str, float] | None = None,
    latency_ratio: float | None = None,
    independent_acquisitions: bool = False,
) -> dict[str, object]:
    """Conservative v1; aggregate native scorer receipts cannot fabricate bootstrap CIs."""
    values = (baseline_mae, candidate_mae, baseline_relative, candidate_relative)
    if (
        not all(math.isfinite(x) and x >= 0 for x in values)
        or baseline_mae == 0
        or newly_missing < 0
    ):
        raise ValueError("Invalid comparison metrics")
    reasons = []
    if candidate_mae > 0.98 * baseline_mae:
        reasons.append("less_than_2_percent_macro_mae_improvement")
    if candidate_relative > baseline_relative:
        reasons.append("relative_error_regression")
    if newly_missing:
        reasons.append("lost_coverage")
    if sign_urgency_regression:
        reasons.append("sign_or_urgency_regression")
    if confirmation:
        if not independent_acquisitions:
            reasons.append("no_independent_confirmation")
        if delta_ci_upper is None or not math.isfinite(delta_ci_upper) or delta_ci_upper >= 0:
            reasons.append("paired_improvement_not_demonstrated")
        native = native_simultaneous_ci_upper or {}
        if set(native) != {"overall", "crucial", "negative"} or any(
            not math.isfinite(v) or v > 0 for v in native.values()
        ):
            reasons.append("native_noninferiority_not_demonstrated")
        if (
            latency_ratio is None
            or not math.isfinite(latency_ratio)
            or not 0 < latency_ratio <= 1.1
        ):
            reasons.append("total_latency_budget_not_demonstrated")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "status": "confirmation_pass"
        if confirmation and not reasons
        else "development_signal"
        if not reasons
        else "not_admitted",
        "release_modified": False,
    }


class ResearchRun:
    """Fresh, source-bound output with COMPLETE/FAILED and post-execution hash checking."""

    def __init__(self, output: Path, inputs: dict[str, Path], config: dict[str, object]) -> None:
        if output.exists():
            raise ValueError("Use a fresh run directory; never overwrite scientific evidence")
        self.output, self.inputs = output, inputs
        self.hashes = {name: sha256(path) for name, path in inputs.items()}
        output.mkdir(parents=True)
        self.freeze = {
            "schema": "release_research_v1",
            "inputs": self.hashes,
            "input_paths": {name: str(path) for name, path in inputs.items()},
            "config": config,
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "python_version": platform.python_version(),
            "host": platform.node(),
            "start_time": datetime.now(UTC).isoformat(),
            "scope": "development or mechanism only; never official promotion",
        }
        write_json(output / "FREEZE.json", self.freeze)
        # A git commit cannot reproduce new/uncommitted sources. Preserve the
        # exact executed bytes as well as their hashes, without copying datasets.
        source_root = Path.cwd().resolve()
        for name, path in inputs.items():
            if name.startswith("source:"):
                relative = path.resolve().relative_to(source_root)
                destination = output / "executed_source" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)

    def finish(self) -> None:
        """Publish completion only after all sources and inputs still match."""
        if (self.output / "COMPLETE.json").exists() or (self.output / "FAILED.json").exists():
            raise ValueError("A terminal scientific receipt cannot be overwritten")
        for name, path in self.inputs.items():
            if sha256(path) != self.hashes[name]:
                raise ValueError(f"Run input/source changed: {name}")
        artifacts = {
            str(p.relative_to(self.output)): sha256(p)
            for p in self.output.rglob("*")
            if p.is_file()
        }
        write_json(
            self.output / "COMPLETE.json",
            {
                "status": "COMPLETE",
                "end_time": datetime.now(UTC).isoformat(),
                "artifacts": artifacts,
                "release_modified": False,
            },
        )

    def fail(self, error: Exception) -> None:
        """Retain failed work and actionable reason instead of a false completion."""
        if (self.output / "COMPLETE.json").exists() or (self.output / "FAILED.json").exists():
            raise ValueError("A terminal scientific receipt cannot be overwritten")
        write_json(
            self.output / "FAILED.json",
            {
                "status": "FAILED",
                "error": repr(error),
                "end_time": datetime.now(UTC).isoformat(),
            },
        )
