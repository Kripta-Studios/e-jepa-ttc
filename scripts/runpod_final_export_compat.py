"""Additive TorchScript compatibility for the sealed PHASE_ONLY final head.

The scientific architecture and checkpoint are unchanged.  This module only
re-expresses the two phase helpers with literal, TorchScript-compatible bounds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from operational.final_refit import training
from operational.final_refit.training import PhaseOnlyFinal as _FrozenPhaseOnlyFinal

MIN_PHASE_LITERAL = -0.6931461805599453
MAX_PHASE_LITERAL = 8.0
_ORIGINAL_PHASE_ONLY_FINAL = _FrozenPhaseOnlyFinal


def _validate_bound_literals() -> None:
    from e_jepa_ttc.simplex_t import phase

    if phase.MIN_PHASE != MIN_PHASE_LITERAL or phase.MAX_PHASE != MAX_PHASE_LITERAL:
        raise ValueError("Frozen phase bounds differ from export compatibility literals")


class ScriptablePhaseOnlyFinal(_FrozenPhaseOnlyFinal):
    """State-identical PHASE_ONLY head with a scriptable forward method."""

    def forward(self, features: Tensor, rgb_width_logits: Tensor) -> dict[str, Tensor]:
        if features.dim() != 2 or features.size(1) != 340:
            raise ValueError("PHASE_ONLY expects frozen [B,340] states")
        if rgb_width_logits.dim() != 2 or rgb_width_logits.size(0) != features.size(0):
            raise ValueError("PHASE_ONLY expects frozen R_CTX width logits [B,2]")
        if rgb_width_logits.size(1) != 2:
            raise ValueError("PHASE_ONLY expects frozen R_CTX width logits [B,2]")
        selected = torch.cat((features[:, 160:170], features[:, 330:340]), dim=-1).clone()
        selected[:, :5] = selected[:, :5] / 0.03
        selected[:, 10:15] = selected[:, 10:15] / 0.03
        correction = self.output(self.mlp(selected))
        location = features[:, 160] + 0.03 * correction[:, 0]
        if not torch.isfinite(location).all():
            raise ValueError("nonfinite phase")
        widths = 0.03 * torch.nn.functional.softplus(rgb_width_logits + correction[:, 1:])
        phi = location.clamp(-0.6931461805599453, 8.0)
        q = -torch.expm1(-phi) / 0.1
        sign = torch.where(q < 0, -torch.ones_like(q), torch.ones_like(q))
        ttc = sign / q.abs().clamp_min(1.0 / 60.0)
        if not torch.isfinite(ttc).all() or not ((ttc < 0) | (ttc > 0.1)).all():
            raise ValueError("TTC must be finite with T<0 or T>0.1 seconds")
        point_phase = -torch.log1p(-0.1 / ttc)
        return {
            "raw_location": location,
            "point_phase": point_phase,
            "ttc": ttc,
            "q10": (location - widths[:, 0]).clamp(-0.6931461805599453, 8.0),
            "q90": (location + widths[:, 1]).clamp(-0.6931461805599453, 8.0),
            "correction": correction,
        }


def install() -> type[ScriptablePhaseOnlyFinal]:
    """Install the export-only class for callers importing from training."""
    _validate_bound_literals()
    current = training.PhaseOnlyFinal
    if current not in {_ORIGINAL_PHASE_ONLY_FINAL, ScriptablePhaseOnlyFinal}:
        raise RuntimeError("PHASE_ONLY class was already replaced by an unknown implementation")
    training.PhaseOnlyFinal = ScriptablePhaseOnlyFinal
    return ScriptablePhaseOnlyFinal


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    pending = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.pending")
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def _parity_case(
    eager: nn.Module, scripted: torch.jit.ScriptModule, features: Tensor, widths: Tensor
) -> float:
    with torch.inference_mode():
        expected = eager(features, widths)
        observed = scripted(features, widths)
    if set(expected) != set(observed):
        raise ValueError("TorchScript output fields differ from the sealed head")
    maximum = 0.0
    for name in expected:
        if expected[name].shape != observed[name].shape:
            raise ValueError(f"TorchScript output shape differs: {name}")
        if not torch.isfinite(expected[name]).all() or not torch.isfinite(observed[name]).all():
            raise ValueError(f"TorchScript output is nonfinite: {name}")
        maximum = max(maximum, float((expected[name] - observed[name]).abs().max()))
    return maximum


def diagnostic_export(run: Path, output: Path) -> dict[str, Any]:
    """Write and validate a noncanonical diagnostic export with zero updates."""
    _validate_bound_literals()
    run = run.resolve(strict=True)
    output = output.resolve()
    receipt_path = output.with_suffix(".json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError("Diagnostic export destination already exists")
    seal_path = (run / "final/MODEL_SEAL.json").resolve(strict=True)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (
        seal.get("schema") != "scientific_oof_final_model_seal_v1"
        or seal.get("status") != "SEALED_NOT_EVALUATED"
    ):
        raise ValueError("Scientific final model is not sealed")
    component = seal["trained_endpoints"]["PHASE_ONLY_EVENT_S7_TRAIN40"]
    checkpoint = Path(component["checkpoint_path"]).resolve(strict=True)
    if _sha256(checkpoint) != component["checkpoint_sha256"]:
        raise ValueError("Sealed PHASE_ONLY checkpoint changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    eager = _ORIGINAL_PHASE_ONLY_FINAL().eval()
    eager.load_state_dict(payload["model_state_dict"])
    compatible = ScriptablePhaseOnlyFinal().eval()
    compatible.load_state_dict(payload["model_state_dict"])
    eager_state = eager.state_dict()
    compatible_state = compatible.state_dict()
    if set(eager_state) != set(compatible_state) or any(
        not torch.equal(eager_state[name], compatible_state[name]) for name in eager_state
    ):
        raise ValueError("Compatibility adapter changed sealed model state")

    scripted = torch.jit.script(compatible)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp.ts")
    torch.jit.save(scripted, temporary)
    restored = torch.jit.load(str(temporary), map_location="cpu").eval()
    restored_state = restored.state_dict()
    if set(restored_state) != set(eager_state) or any(
        not torch.equal(restored_state[name], eager_state[name]) for name in eager_state
    ):
        temporary.unlink(missing_ok=True)
        raise ValueError("Diagnostic TorchScript weights differ from sealed state")

    maximum = 0.0
    for batch in (1, 7):
        features = torch.linspace(-0.25, 0.25, batch * 340).reshape(batch, 340)
        widths = torch.linspace(-0.1, 0.2, batch * 2).reshape(batch, 2)
        maximum = max(maximum, _parity_case(eager, restored, features, widths))
    edges = torch.tensor([-100.0, MIN_PHASE_LITERAL, -1e-8, 0.0, 1e-8, MAX_PHASE_LITERAL, 100.0])
    edge_features = torch.zeros((len(edges), 340), dtype=torch.float32)
    edge_features[:, 160] = edges
    edge_widths = torch.zeros((len(edges), 2), dtype=torch.float32)
    maximum = max(maximum, _parity_case(eager, restored, edge_features, edge_widths))
    if maximum > 1e-6:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"TorchScript parity differs by {maximum}")
    os.replace(temporary, output)
    receipt: dict[str, Any] = {
        "schema": "ejepa_final_phase_only_export_compat_diagnostic_v1",
        "status": "DIAGNOSTIC_PARITY_PASS",
        "model_seal_sha256": _sha256(seal_path),
        "checkpoint_sha256": _sha256(checkpoint),
        "output": str(output),
        "output_sha256": _sha256(output),
        "state_dict_exact": True,
        "parity_batches": [1, 7],
        "edge_phases": edges.tolist(),
        "parity_max_abs": maximum,
        "parity_tolerance": 1e-6,
        "optimizer_updates": 0,
        "canonical_export_replaced": False,
    }
    receipt["receipt_sha256"] = _canonical(receipt)
    _atomic_json(receipt_path, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(diagnostic_export(args.run, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
