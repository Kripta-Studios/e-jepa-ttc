"""Prove exact frozen primary geometry for one fold-local A6, A8, or V6.1 run."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact, verify_artifact_hash  # noqa: E402
from e_jepa_ttc.models.causal_scale_ttc import (  # noqa: E402
    CausalScaleTTC,
    CausalScaleTTCConfig,
)
from e_jepa_ttc.training.causal_scale_eap import _module_tensor_sha256  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], CausalScaleTTC]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("artifact_type") != "causal_scale_eap_grouped_dev_checkpoint_v1":
        raise ValueError(f"grouped checkpoint type is incompatible: {path}")
    config = payload.get("model_config")
    state = payload.get("model_state_dict")
    if not isinstance(config, dict) or not isinstance(state, dict):
        raise ValueError(f"grouped checkpoint is malformed: {path}")
    model = CausalScaleTTC(CausalScaleTTCConfig(**config))
    model.load_state_dict(state, strict=True)
    model.eval()
    return payload, model


def _tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


@torch.inference_mode()
def build_audit(
    *,
    parent_checkpoint: Path,
    child_checkpoint: Path,
    child_summary: Path,
    arm: Literal["a6", "a8_0", "v6_1"],
    fold: int,
) -> dict[str, Any]:
    """Build fail-closed state, optimizer, and fixed-probe geometry evidence."""

    _, parent = _load_checkpoint(parent_checkpoint)
    _, child = _load_checkpoint(child_checkpoint)
    summary = json.loads(child_summary.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or not verify_artifact_hash(summary):
        raise ValueError("child summary signature is invalid")
    development = summary.get("development_protocol", {})
    if development.get("fold_identity", {}).get("fold") != fold:
        raise ValueError("child summary fold differs from requested audit fold")
    initialization = summary.get("initialization")
    if not isinstance(initialization, dict):
        raise ValueError("child summary initialization evidence is missing")
    parent_sha = _sha256(parent_checkpoint)
    if initialization.get("checkpoint_sha256") != parent_sha:
        raise ValueError("child initialization does not bind the requested parent")

    parent_encoder = parent.encoder.state_dict()
    child_encoder = child.encoder.state_dict()
    if parent_encoder.keys() != child_encoder.keys():
        raise ValueError("parent and child primary encoder schemas differ")
    unequal = [
        name
        for name in parent_encoder
        if not torch.equal(parent_encoder[name], child_encoder[name])
    ]
    if unequal:
        raise ValueError(f"primary geometry tensors changed: {unequal}")
    parent_encoder_hash = _module_tensor_sha256(parent.encoder)
    child_encoder_hash = _module_tensor_sha256(child.encoder)
    expected_hashes = (
        initialization.get("initial_primary_encoder_sha256"),
        initialization.get("final_primary_encoder_sha256"),
    )
    if expected_hashes != (parent_encoder_hash, parent_encoder_hash):
        raise ValueError("summary primary encoder hashes do not equal the parent")
    if initialization.get("primary_encoder_exact_initial") is not True:
        raise ValueError("summary does not attest exact primary encoder preservation")

    encoder_parameter_names = {f"encoder.{name}" for name, _ in child.encoder.named_parameters()}
    frozen_names = set(initialization.get("frozen_parameter_names", []))
    optimizer_names = set(initialization.get("optimizer_parameter_names", []))
    if not encoder_parameter_names <= frozen_names:
        raise ValueError("not every primary encoder parameter was frozen")
    if encoder_parameter_names & optimizer_names:
        raise ValueError("primary encoder parameters entered the optimizer")
    if initialization.get("frozen_optimizer_overlap") != []:
        raise ValueError("summary reports frozen/optimizer parameter overlap")

    generator = torch.Generator().manual_seed(20260813 + fold)
    events = torch.randn(
        2,
        3,
        parent.config.in_channels,
        32,
        32,
        generator=generator,
    )
    delta = torch.full((2, 2), 0.1)
    parent_output = parent(events, delta)
    child_output = child(events, delta)
    probe_names = (
        "foreground_logits",
        "visible_height_normalized",
        "visible_width_normalized",
        "analytic_log_height_ratio",
    )
    probe_hashes: dict[str, str] = {}
    for name in probe_names:
        parent_value = getattr(parent_output, name)
        child_value = getattr(child_output, name)
        if not torch.equal(parent_value, child_value):
            raise ValueError(f"fixed-probe primary geometry changed: {name}")
        probe_hashes[name] = _tensor_hash(parent_value)

    transport: dict[str, Any] | None = None
    if arm in {"a8_0", "v6_1"}:
        if child.transport_encoder is None:
            raise ValueError(f"{arm} checkpoint lacks its transport encoder")
        transport = {
            "initial_equal_parent": (
                initialization.get("initial_transport_encoder_sha256") == parent_encoder_hash
            ),
            "changed_after_training": initialization.get("transport_encoder_changed_from_initial"),
            "final_sha256": _module_tensor_sha256(child.transport_encoder),
        }
        if transport["initial_equal_parent"] is not True:
            raise ValueError(f"{arm} transport encoder did not initialize from the parent")
        if transport["changed_after_training"] is not True:
            raise ValueError(f"{arm} transport encoder did not receive an observable update")

    report: dict[str, Any] = {
        "artifact_type": "scientific_recovery_v5_fold_geometry_audit_v1",
        "status": "completed_exact_primary_geometry",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "arm": arm,
        "fold": fold,
        "sources": {
            "parent_checkpoint": {
                "path": str(parent_checkpoint.resolve()),
                "sha256": parent_sha,
            },
            "child_checkpoint": {
                "path": str(child_checkpoint.resolve()),
                "sha256": _sha256(child_checkpoint),
            },
            "child_summary": {
                "path": str(child_summary.resolve()),
                "sha256": _sha256(child_summary),
                "artifact_sha256": summary["artifact_sha256"],
            },
        },
        "primary_geometry": {
            "tensor_names": sorted(parent_encoder),
            "tensor_count": len(parent_encoder),
            "parent_sha256": parent_encoder_hash,
            "child_sha256": child_encoder_hash,
            "exact_tensor_equality": True,
            "requires_grad_false": True,
            "absent_from_optimizer": True,
            "fixed_probe_exact": True,
            "fixed_probe_hashes": probe_hashes,
        },
        "transport_encoder": transport,
        "contracts": {
            "geometry_before_equals_parent": True,
            "geometry_after_equals_parent": True,
            "private_test_opened": False,
        },
    }
    sign_artifact(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--child-checkpoint", type=Path, required=True)
    parser.add_argument("--child-summary", type=Path, required=True)
    parser.add_argument("--arm", choices=("a6", "a8_0", "v6_1"), required=True)
    parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_audit(
            parent_checkpoint=args.parent_checkpoint.resolve(strict=True),
            child_checkpoint=args.child_checkpoint.resolve(strict=True),
            child_summary=args.child_summary.resolve(strict=True),
            arm=args.arm,
            fold=args.fold,
        )
    except Exception as error:
        parser.exit(2, f"fold geometry audit failed: {type(error).__name__}: {error}\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"output": str(args.output), "status": report["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
