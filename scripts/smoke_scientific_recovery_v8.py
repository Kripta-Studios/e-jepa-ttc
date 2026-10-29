#!/usr/bin/env python
"""Run a bounded CPU V8 smoke through temporal frontends, models, JEPA and router."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact  # noqa: E402
from e_jepa_ttc.data.scientific_recovery_v8 import (  # noqa: E402
    CausalExponentialStateRepresentation,
    GarlTimeVolumeRepresentation,
)
from e_jepa_ttc.data.types import EventBatch  # noqa: E402
from e_jepa_ttc.models.causal_expert_router import (  # noqa: E402
    ROUTER_FEATURES,
    CausalExpertRouter,
)
from e_jepa_ttc.models.causal_scale_jepa_v8 import (  # noqa: E402
    CausalScaleJEPAV8,
    CausalScaleJEPAV8Config,
)
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig  # noqa: E402
from e_jepa_ttc.training.scientific_recovery_v8_jepa import (  # noqa: E402
    ScientificRecoveryV8JEPATrainer,
    ScientificRecoveryV8JEPATrainerConfig,
)


def _events() -> EventBatch:
    return EventBatch(
        x=np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int32),
        y=np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int32),
        t_us=np.asarray([0, 200, 400, 600, 800, 1_000], dtype=np.int64),
        polarity=np.asarray([1, -1, 1, 1, -1, 1], dtype=np.int8),
        width=8,
        height=8,
        sequence_id="v8-smoke",
        t_start_us=0,
        t_end_us=1_000,
    )


def run(device: str) -> dict[str, object]:
    if device != "cpu":
        raise ValueError("the bounded V8 smoke is intentionally CPU-only")
    torch.manual_seed(7)
    events = _events()
    roi = torch.tensor([0, 0, 8, 8])
    timevol = GarlTimeVolumeRepresentation(target_size=(16, 16)).encode(events, 1_000, roi)
    exp6 = CausalExponentialStateRepresentation(target_size=(16, 16)).encode(events, 1_000, roi)
    time_model = CausalScaleTTC(
        CausalScaleTTCConfig(
            in_channels=20,
            hidden_dim=8,
            geometry_dim=16,
            residual_depth=1,
            foreground_temporal_smoothing_mode="causal_left",
        )
    )
    exp_model = CausalScaleTTC(
        CausalScaleTTCConfig(
            in_channels=6,
            hidden_dim=8,
            geometry_dim=16,
            residual_depth=1,
            foreground_temporal_smoothing_mode="causal_left",
        )
    )
    time_input = timevol.tensor.unsqueeze(0).unsqueeze(0).repeat(2, 3, 1, 1, 1)
    exp_input = exp6.tensor.unsqueeze(0).unsqueeze(0).repeat(2, 3, 1, 1, 1)
    delta = torch.full((2, 2), 0.1)
    time_output = time_model(time_input, delta, return_dense_features=True)
    exp_output = exp_model(exp_input, delta, return_dense_features=True)
    jepa = CausalScaleJEPAV8(
        exp_model,
        CausalScaleJEPAV8Config(ema_total_updates=1, predictor_hidden_dim=8, collapse_patience=10),
    )
    trainer = ScientificRecoveryV8JEPATrainer(
        jepa,
        ScientificRecoveryV8JEPATrainerConfig(total_updates=1, seed=7),
    )
    jepa_step = trainer.step(exp_input[:, 0], exp_input[:, 1], exp_input[:, 2])
    features = pd.DataFrame(
        np.asarray(
            [
                [1.0, 1.0, 0.1, 0.1, -3.0, 0.2, 0.2, -3.0],
                [2.0, 2.0, 0.2, 0.2, -2.0, 0.1, 0.1, -2.0],
                [3.0, 3.0, 0.3, 0.3, -1.0, 0.4, 0.4, -1.0],
                [4.0, 4.0, 0.4, 0.4, -0.5, 0.3, 0.3, -0.5],
            ]
        ),
        columns=ROUTER_FEATURES,
    )
    router = CausalExpertRouter(seed=7).fit(
        features,
        np.asarray([0, 1, 0, 1]),
        sample_tokens=("a", "b", "c", "d"),
        effective_sample_weights=np.ones(4),
    )
    return {
        "status": "smoke_passed",
        "timevol_shape": list(timevol.tensor.shape),
        "exp6_shape": list(exp6.tensor.shape),
        "timevol_finite": bool(torch.isfinite(time_output.ttc_mean_seconds).all()),
        "exp6_finite": bool(torch.isfinite(exp_output.ttc_mean_seconds).all()),
        "jepa_loss_finite": bool(np.isfinite(jepa_step["loss"])),
        "router_signature": router.signature.artifact_sha256,
        "sealed_data_opened": False,
    }


def _write_manifest(output_dir: Path, result: dict[str, object]) -> None:
    """Publish smoke completion only after all CPU model checks have succeeded."""

    payload: dict[str, object] = {
        "artifact_type": "scientific_recovery_v8_cpu_smoke_v1",
        "status": "completed",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "closed_evaluation": {
            "public_validation_used_for_selection": False,
            "private_test_opened": False,
            "evttc_test_opened": False,
            "codabench_opened": False,
        },
        "checks": result,
    }
    sign_artifact(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "manifest.json"
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/scientific_recovery_v8/smoke",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps({"status": "planned", "command": "bounded CPU V8 smoke"}))
        return 0
    try:
        result = run(args.device)
        _write_manifest(args.output_dir, result)
        print(json.dumps(result, sort_keys=True))
    except (RuntimeError, ValueError) as error:
        parser.exit(2, f"V8 smoke failed: {type(error).__name__}: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
