"""Verified original-bundle geometry; no unverified HF conversion is provided."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

import numpy as np


def original_bundle_corners(box_xyzlhwyaw: Sequence[float]) -> np.ndarray:
    """Return original eAP bundle corners, checked against the official visualizer.

    This function accepts the ORIGINAL bundle contract only: ``x,y,z,l,h,w,yaw``;
    local X carries width, local Y carries length, and positive yaw is clockwise.
    It does not establish equivalence with ``bbox_3d_ego`` in Hugging Face.
    The mean of the corners is the supplied geometric center.
    """

    box = np.asarray(box_xyzlhwyaw, dtype=np.float64)
    if box.shape != (7,) or not np.isfinite(box).all() or (box[3:6] <= 0).any():
        raise ValueError("Original bundle box requires seven finite values and positive l,h,w.")
    signs = np.asarray(
        [
            (-1, -1, 1),
            (1, -1, 1),
            (1, 1, 1),
            (-1, 1, 1),
            (-1, -1, -1),
            (1, -1, -1),
            (1, 1, -1),
            (-1, 1, -1),
        ],
        dtype=np.float64,
    )
    local = signs * box[[5, 3, 4]] / 2.0
    cosine, sine = np.cos(box[6]), np.sin(box[6])
    rotation = np.asarray([[cosine, sine, 0], [-sine, cosine, 0], [0, 0, 1]])
    return local @ rotation.T + box[:3]


def validate_calibration(intrinsic: object, event_from_ego: object) -> None:
    """Check finite pinhole/rigid-transform structure, not sensor alignment."""

    intrinsic_array = np.asarray(intrinsic, dtype=np.float64)
    transform = np.asarray(event_from_ego, dtype=np.float64)
    if intrinsic_array.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError("K_event must be 3x3 and T_event_ego must be 4x4.")
    if not np.isfinite(intrinsic_array).all() or not np.isfinite(transform).all():
        raise ValueError("Calibration contains non-finite values.")
    if min(intrinsic_array[0, 0], intrinsic_array[1, 1]) <= 0:
        raise ValueError("Focal lengths must be positive.")
    if not np.allclose(intrinsic_array[2], [0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError("Invalid homogeneous row in K_event.")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError("Invalid homogeneous row in T_event_ego.")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0):
        raise ValueError("T_event_ego rotation is not orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1, atol=1e-5, rtol=0):
        raise ValueError("T_event_ego contains a reflection or non-rigid rotation.")


def check_causal_availability(
    *,
    exposure_start_us: int,
    exposure_end_us: int,
    max_input_us: int,
    input_available_us: int,
    decision_us: int,
) -> None:
    """Reject a physical sample or arrival unavailable at the declared decision."""

    values = (exposure_start_us, exposure_end_us, max_input_us, input_available_us, decision_us)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise ValueError("Causal clock fields must be integer microseconds.")
    if exposure_start_us > exposure_end_us or exposure_end_us > max_input_us:
        raise ValueError("The RGB exposure must be included in max_input_us.")
    if max_input_us > decision_us or input_available_us > decision_us:
        raise ValueError("Future or late input is unavailable at decision_us.")
    if input_available_us < max_input_us:
        raise ValueError("Input cannot arrive before its physical measurement completes.")


def geometry_blockers(
    contract: Mapping[str, object],
    *,
    evidence_root: Path | None = None,
) -> list[str]:
    """Require claim-specific, byte-bound evidence, not just an approval boolean.

    Evidence integrity is a prerequisite, not a proof of physical correctness.
    New evidence still needs independent scientific review before approval.
    """

    required = (
        "hf_to_bundle_equivalence",
        "hf_to_submission",
        "rgb_event_rectification",
        "velocity_semantics",
        "causal_sensor_audit",
    )
    claims = contract.get("claims", {})
    if not isinstance(claims, Mapping):
        return ["geometry claims missing or malformed"]
    blockers = []
    revision = contract.get("dataset_revision")
    for key in required:
        claim = claims.get(key)
        if not isinstance(claim, Mapping) or claim.get("status") != "VERIFIED":
            blockers.append(key)
            continue
        digest, relative_path = claim.get("evidence_sha256"), claim.get("evidence_path")
        if (
            evidence_root is None
            or not isinstance(relative_path, str)
            or not isinstance(digest, str)
        ):
            blockers.append(f"{key}: unbound evidence")
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            blockers.append(f"{key}: malformed evidence reference")
            continue
        portable_path = PurePosixPath(relative_path)
        if (
            not relative_path
            or relative_path == "."
            or portable_path.is_absolute()
            or ".." in portable_path.parts
            or "\\" in relative_path
            or ":" in relative_path
            or str(portable_path) != relative_path
        ):
            blockers.append(f"{key}: non-canonical evidence path")
            continue
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            blockers.append(f"{key}: unpinned dataset revision")
            continue
        path = evidence_root / relative_path
        try:
            if not path.resolve(strict=False).is_relative_to(evidence_root.resolve(strict=False)):
                raise ValueError("Evidence path resolves outside its declared root.")
            if path.stat().st_size > 5_000_000:
                raise ValueError("Evidence must be a bounded JSON receipt, not raw media.")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("Evidence SHA-256 mismatch.")
            evidence = json.loads(raw)
            if not isinstance(evidence, dict) or any(
                (
                    evidence.get("artifact_type") != "eap_geometry_evidence_v1",
                    evidence.get("claim_id") != key,
                    evidence.get("dataset_revision") != contract.get("dataset_revision"),
                    evidence.get("assertions_passed") is not True,
                    not isinstance(evidence.get("method"), str) or not evidence.get("method"),
                    not isinstance(evidence.get("source_refs"), list)
                    or not evidence.get("source_refs"),
                    not isinstance(evidence.get("limitations"), list),
                )
            ):
                raise ValueError(
                    "Evidence has the wrong claim/schema/revision or incomplete result."
                )
        except (OSError, ValueError, TypeError) as error:
            blockers.append(f"{key}: invalid evidence ({error})")
    return blockers
