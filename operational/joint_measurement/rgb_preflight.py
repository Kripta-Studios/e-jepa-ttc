"""Zero-update CPU RGB integration check on label-blind selected real P pairs."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.joint_measurement.loss import phase_components
from e_jepa_ttc.simplex_t.phase import ttc_to_phase
from operational.range_transfer.common import freeze, sha, utc

from .common import ROOT, RUN, require_resources
from .rgb_data import RGBPairDataset
from .rgb_model import load_parent, pair_from_parent


def run() -> dict:
    """Check mixed T2/T3, own RGB clocks, gradients and untouched parent; no optimizer."""
    require_resources(starting=True)
    torch.set_num_threads(1)
    dataset = RGBPairDataset("P")
    try:
        rows = dataset.rows
        indices = []
        for count in sorted(set(dataset.source.frame_counts.tolist())):
            candidates = np.flatnonzero(dataset.source.frame_counts == count).tolist()
            candidates.sort(
                key=lambda i: hashlib.sha256(str(rows.iloc[i].sample_token).encode()).hexdigest()
            )
            indices.extend(candidates[:2])
        batch = dataset.batch(indices)
        parent, identity = load_parent()
        original = {k: v.clone() for k, v in parent.state_dict().items()}
        outputs = {}
        heads = None
        for correlation, frozen in ((False, False), (False, True), (True, False), (True, True)):
            model = pair_from_parent(parent, correlation=correlation, frozen=frozen, seed=7)
            current_heads = {
                k: v.clone() for k, v in model.state_dict().items() if not k.startswith("encoder.")
            }
            if heads is None:
                heads = current_heads
            assert all(torch.equal(v, current_heads[k]) for k, v in heads.items())
            parent_parameters = dict(parent.encoder.features.named_parameters())
            assert all(
                p.data_ptr() != parent_parameters[k].data_ptr()
                for k, p in model.encoder.named_parameters()
            )
            model.train()
            assert model.encoder.training != frozen
            initial = {k: v.clone() for k, v in model.state_dict().items()}
            prediction = model(batch.events, batch.delta_t_s)
            terms = phase_components(prediction["raw_phase"], ttc_to_phase(batch.target_ttc_s))
            loss = (terms["emitted"] + terms["raw"]).mean()
            assert torch.isfinite(loss)
            loss.backward()
            gradients = [p.grad for p in model.encoder.parameters() if p.grad is not None]
            assert all(torch.isfinite(g).all() for g in gradients)
            gradient_l2 = sum(float(g.square().sum()) for g in gradients) ** 0.5
            assert (gradient_l2 > 0) != frozen
            assert any(
                p.grad is not None and bool((p.grad != 0).any()) for p in model.readout.parameters()
            )
            assert all(torch.equal(v, model.state_dict()[k]) for k, v in initial.items())
            outputs[f"{'CORR' if correlation else 'DIRECT'}-{'FROZEN' if frozen else 'FT'}"] = {
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "encoder_gradient_l2": gradient_l2,
                "raw_phase_finite": bool(torch.isfinite(prediction["raw_phase"]).all()),
                "optimizer_updates": 0,
            }
        assert all(torch.equal(v, parent.state_dict()[k]) for k, v in original.items())
        result = {
            "status": "PASS",
            "scope": (
                "CPU integration only; no accuracy result, recipe selection or optimizer updates"
            ),
            "role": "P",
            "indices": indices,
            "sample_tokens": batch.sample_tokens,
            "source_frame_counts": dataset.source.frame_counts[indices].tolist(),
            "input_shape": list(batch.events.shape),
            "sensor_delta_t_s": batch.delta_t_s.tolist(),
            "parent": identity,
            "checks": outputs,
            "parent_unchanged": True,
            "new_modules_identical": True,
            "optimizer_updates": 0,
            "sources": {
                p.relative_to(ROOT).as_posix(): sha(p)
                for p in (
                    Path(__file__),
                    ROOT / "operational/joint_measurement/rgb_data.py",
                    ROOT / "operational/joint_measurement/rgb_model.py",
                    ROOT / "src/e_jepa_ttc/joint_measurement/model.py",
                )
            },
            "utc": utc(),
        }
        freeze(RUN / "verification/RGB_CPU_PREFLIGHT.json", result)
        return result
    finally:
        dataset.close()


if __name__ == "__main__":
    print(run())
