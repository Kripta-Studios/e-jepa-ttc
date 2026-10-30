"""Explicit provenance-only migration to the admitted evaluation scheduling revision."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

OLD_RUNNER = "d1eef04a4f4d5bdf2a7d77d88d8fb44efc6e3dded0919d3e9ce001b8f2628352"
NEW_RUNNER = "e8c7291de32f39e7326857c0722bb04068f81f605f0ab22ebdb5240f317519db"
RUNNER = "scripts/train_detection_campaign.py"
ADDED = {
    "src/e_jepa_ttc/detection/evaluation_prefetch.py": "ce56072e26cbfc7b75f292a82ea20232c0d874dd073e26c5047d8ac7423d99bd",
    "src/e_jepa_ttc/detection/bounded_prefetch.py": "5854b9e2b45b1d0d42683b0cf279a8523e5d3fa27ef2860cd62b05cf5f4e40b3",
}


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def fingerprint(value: Any) -> object:
    """Bind every tensor byte, scalar, container type and mapping key."""
    if isinstance(value, torch.Tensor):
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return {
            "tensor": hashlib.sha256(raw).hexdigest(),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "stride": list(value.stride()),
            "requires_grad": value.requires_grad,
        }
    if isinstance(value, Mapping):
        return {
            "mapping": [
                [fingerprint(k), fingerprint(v)]
                for k, v in sorted(value.items(), key=lambda item: repr(item[0]))
            ]
        }
    if isinstance(value, (list, tuple)):
        return {type(value).__name__: [fingerprint(item) for item in value]}
    if value is None or isinstance(value, (str, int, float, bool)):
        return {type(value).__name__: value}
    raise TypeError(f"Unsupported state: {type(value)}")


def state_digest(payload: Mapping[str, Any]) -> str:
    return canonical(
        fingerprint(
            {
                k: v
                for k, v in payload.items()
                if k not in {"runtime_provenance", "runtime_provenance_sha256"}
            }
        )
    )


def verify_transition(old: dict[str, Any], new: dict[str, Any]) -> None:
    """Permit only the admitted runner and two CPU-prefetch dependencies."""
    excluded = {"code_sha256", "code_sha256_manifest"}
    if {k: v for k, v in old.items() if k not in excluded} != {
        k: v for k, v in new.items() if k not in excluded
    }:
        raise ValueError("Scientific runtime fields changed")
    a, b = old["code_sha256"], new["code_sha256"]
    if a[RUNNER] != OLD_RUNNER or b[RUNNER] != NEW_RUNNER:
        raise ValueError("Unadmitted producer pair")
    if set(b) - set(a) != set(ADDED) or set(a) - set(b):
        raise ValueError("Runtime dependency population changed")
    if any(a[k] != b[k] for k in a if k != RUNNER) or any(b[k] != v for k, v in ADDED.items()):
        raise ValueError("Runtime source outside the allowlist changed")
    for runtime in (old, new):
        if canonical(runtime["code_sha256"]) != runtime["code_sha256_manifest"]:
            raise ValueError("Invalid code manifest hash")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    old_run = root / "cloud_runs_input_v1/r0_seed7"
    admission = json.loads(
        (root / "evaluation_optimization_v1/deployment_admission.json").read_text()
    )
    if admission["status"] != "PASS_EXACT_PAYLOAD_AND_TRAINING_SMOKE":
        raise ValueError("Evaluation optimization is not admitted")
    if admission["runner_sha256"] != NEW_RUNNER or admission["modules"] != ADDED:
        raise ValueError("Admission producer differs")
    for relative, expected in admission["evidence_sha256"].items():
        if sha(root / relative) != expected:
            raise ValueError(f"Admission evidence changed: {relative}")
    source = old_run / "checkpoints/selected.pt"
    source_digest = sha(source)
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if (
        payload["update"] != 4096
        or payload.get("producer") != RUNNER
        or payload.get("schema") != "eap_detection_campaign_checkpoint_v1"
        or payload.get("inference_contract") != "campaign_v1_strict_loader_required"
    ):
        raise ValueError("This migration requires the retained update-4096 selected state")
    old = json.loads((old_run / "runtime_preregistration.json").read_text())
    new = json.loads(
        (
            root / "evaluation_optimization_v1/hardware_smoke/runtime_preregistration.json"
        ).read_text()
    )
    new["config"], new["config_sha256"] = old["config"], old["config_sha256"]
    verify_transition(old, new)
    for runtime, code in [
        (old, old_run / "code"),
        (new, root / "evaluation_optimization_v1/hardware_smoke/code"),
    ]:
        for relative, expected in runtime["code_sha256"].items():
            if sha(code / relative) != expected:
                raise ValueError(f"Snapshot differs: {relative}")
    old_provenance = payload["runtime_provenance"]
    expected_old = {
        "runtime_preregistration_sha256": sha(old_run / "runtime_preregistration.json"),
        **{key: old[key] for key in old_provenance if key != "runtime_preregistration_sha256"},
    }
    if (
        old_provenance["runtime_preregistration_sha256"]
        != sha(old_run / "runtime_preregistration.json")
        or old_provenance != expected_old
        or canonical(old_provenance) != payload["runtime_provenance_sha256"]
        or payload["config_sha256"] != old["config_sha256"]
        or payload["protocol_sha256"] != old["protocol_sha256"]
    ):
        raise ValueError("Source checkpoint provenance differs")
    destination = root / "evaluation_optimization_v1/r0_migration"
    destination.mkdir()
    preregistration = destination / "runtime_preregistration.json"
    preregistration.write_text(json.dumps(new, indent=2) + "\n")
    provenance = {
        "runtime_preregistration_sha256": sha(preregistration),
        **{key: new[key] for key in old_provenance if key != "runtime_preregistration_sha256"},
    }
    before = state_digest(payload)
    payload["runtime_provenance"] = provenance
    payload["runtime_provenance_sha256"] = canonical(provenance)
    target = destination / "checkpoints"
    target.mkdir()
    torch.save(payload, target / "last.pt")
    restored = torch.load(target / "last.pt", map_location="cpu", weights_only=True)
    if state_digest(restored) != before:
        raise ValueError("Checkpoint state changed during migration")
    # The update-4096 checkpoint is also the best selected checkpoint.
    shutil.copy2(target / "last.pt", target / "selected.pt")
    if sha(source) != source_digest:
        raise ValueError("Source checkpoint changed during migration")
    receipt = {
        "status": "PASS_EXACT_STATE_PROVENANCE_ONLY",
        "source": str(source),
        "source_sha256": source_digest,
        "checkpoint": str(target / "last.pt"),
        "checkpoint_sha256": sha(target / "last.pt"),
        "update": 4096,
        "state_sha256": before,
        "new_runtime_provenance": provenance,
        "pending_full_evaluation": 4096,
        "original_training_state_preserved": True,
    }
    (destination / "MIGRATION.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
