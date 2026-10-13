"""Replay retained optimized-resume state using only an extracted bundle and CPU."""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch

STATE_KEYS = (
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "sampler_generator_state",
    "torch_rng_state",
    "cuda_rng_state_all",
    "numpy_random_state",
    "python_random_state",
    "cursor",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _exact(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and torch.equal(
                left.contiguous().reshape(-1).view(torch.uint8),
                right.contiguous().reshape(-1).view(torch.uint8),
            )
        )
    if isinstance(left, np.ndarray):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and left.tobytes() == right.tobytes()
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_exact(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _exact(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, float):
        return struct.pack("!d", left) == struct.pack("!d", right)
    return bool(left == right)


def _checkpoint(directory: Path, count: int, offset: int, registration: str) -> tuple[dict, str]:
    pointer = _read(directory / "CHECKPOINT_POINTER.json")
    version = pointer["version"]
    if Path(version).name != version or ":" in version or "\\" in version:
        raise ValueError("Unsafe runtime checkpoint version")
    path = directory / "checkpoint_versions" / version
    receipt = _read(directory / "CHECKPOINT_RECEIPT.json")
    journal = _read(directory / "UPDATE_JOURNAL.json")
    digest = _sha(path)
    if digest != pointer["checkpoint_sha256"] or digest != receipt["checkpoint_sha256"]:
        raise ValueError("Runtime checkpoint differs from pointer or receipt")
    if (
        receipt["status"] != "COMPLETE"
        or not receipt["complete_state"]
        or receipt["accumulation_index"] != 0
        or any(item["completed_updates"] != count for item in (receipt, journal, pointer))
        or journal["durable_updates"] != count
        or journal["pending_update_upper"]
        or journal["recovery_upper"]
    ):
        raise ValueError("Runtime witness has incomplete or unexpected physical accounting")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    identity = _read(directory / "IDENTITY.json")
    if (
        payload["identity"] != identity
        or payload["completed_updates"] != count
        or identity["prefix_offset"] != offset
        or identity["updates"] != count
        or identity["registration"] != registration
        or identity["origin_updates"] != 6131
    ):
        raise ValueError("Runtime witness identity or logical update offset differs")
    return payload, digest


def replay_runtime(root: Path) -> dict:
    """Recompute nine exact states; distinguish physical work from the shared prefix."""
    campaign = root / "campaign"
    runtime = campaign / "runtime/optimization_20261013"
    binding = _read(runtime / "RUNTIME_BINDING.json")
    admission = _read(runtime / "FINAL_ADMISSION.json")
    if (
        binding["status"] != "ADMITTED"
        or not binding["graph"]
        or admission["status"] != "PASS"
        or admission["eager_bitwise_equivalence"]
        or binding["eager_bitwise_equivalence"]
        or _sha(runtime / "FINAL_ADMISSION.json") != binding["admission_sha256"]
        or _sha(runtime / "NUMERICAL_GRAPH_POLICY.json") != binding["numerical_graph_policy_sha256"]
        or _sha(runtime / "ADMISSION.json") != admission["retained_initial_admission_sha256"]
    ):
        raise ValueError("Optimized runtime admission lineage differs")
    # Source paths are checked against the extracted committed snapshot, not this host's tree.
    for name, expected in binding["runtime_sources"].items():
        path = (root / "source" / name).resolve()
        if not path.is_relative_to((root / "source").resolve()) or _sha(path) != expected:
            raise ValueError("Bound runtime source differs in extracted snapshot")
    for name, expected in admission["runtime_sources"].items():
        path = (root / "source/operational/joint_measurement" / name).resolve()
        if not path.is_relative_to((root / "source").resolve()) or _sha(path) != expected:
            raise ValueError("Bound admission source differs in extracted snapshot")
    registration = _sha(runtime / "RESUME_PROOF_REGISTRATION.json")
    if registration != admission["resume_registration_sha256"]:
        raise ValueError("Resume registration differs")
    diagnostics = campaign / "diagnostics/fast_runtime_20261013"
    reference, reference_hash = _checkpoint(diagnostics / "resume_continuous", 6, 0, registration)
    resumed, resumed_hash = _checkpoint(diagnostics / "resume_restored", 2, 4, registration)
    witness = _read(runtime / "INDEPENDENT_RESUME_CHECK.json")
    if reference_hash != witness["reference_sha256"] or resumed_hash != witness["restored_sha256"]:
        raise ValueError("Independent witness refers to different checkpoint bytes")
    checks = {key: _exact(reference[key], resumed[key]) for key in STATE_KEYS}
    if not all(checks.values()):
        raise ValueError(f"Optimized resume state differs: {checks}")
    curves = reference["cursor"]["curve"]
    if len(curves) != 6 or not all(len(row["ids"]) == 32 for row in curves):
        raise ValueError("Shared-prefix witness must contain six logical batch updates")
    steps = {int(value["step"]) for value in reference["optimizer_state_dict"]["state"].values()}
    if steps != {6137}:
        raise ValueError("Witness optimizer did not continue the admitted origin")
    if not all(torch.isfinite(value).all() for value in reference["model_state_dict"].values()):
        raise ValueError("Runtime witness contains nonfinite model weights")
    return {
        "status": "PASS",
        "checks": checks,
        "logical_updates": 6,
        "retained_physical_updates": 8,
        "shared_prefix_updates": 4,
        "reference_sha256": reference_hash,
        "restored_sha256": resumed_hash,
        "optimized_resume_bit_exact": True,
        "eager_bitwise_equivalence": False,
        "optimizer_updates": 0,
        "cuda_initialized": torch.cuda.is_initialized(),
        "scope": "CPU comparison of retained graph6 and shared4+restored2 states; no retraining",
    }
