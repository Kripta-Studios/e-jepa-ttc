"""Regenerate included numerical evidence from an independently extracted bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def replay(root: Path) -> dict:
    """Recompute scores and intervals; no external raw, holdouts or checkpoints needed."""
    frame = pd.read_parquet(root / "analysis/EWMA_TRANSPORT_CV.parquet")
    with np.load(root / "analytical/PAIRED_LOSSES.npz", allow_pickle=False) as z:
        losses, mass = z["losses"].copy(), z["mass"].copy()
    truth = -np.log1p(-0.1 / frame.target_ttc.to_numpy(np.float64))
    prediction = -np.log1p(-0.1 / frame.prediction_ttc_s.to_numpy(np.float64))
    if not np.array_equal(losses[:, 0], 10000 * abs(prediction - truth)):
        raise ValueError("included point losses do not regenerate")
    if len(frame) != 8192 or frame.sample_token.duplicated().any() or not np.isclose(mass.sum(), 1):
        raise ValueError("included population or mass changed")
    blocks = [
        np.load(p, allow_pickle=False)
        for p in sorted((root / "analytical/bootstrap/.resume").glob("*/LOSSES.npy"))
    ]
    draws = np.concatenate(blocks)
    if draws.shape != (8192, 3):
        raise ValueError("included hierarchical fragments incomplete")
    result = {
        "candidate_score": float(mass @ losses[:, 0]),
        "reference_score": float(mass @ losses[:, 1]),
        "point_delta": float(mass @ (losses[:, 0] - losses[:, 1])),
        "hierarchical_ci95": np.percentile(draws[:, 0] - draws[:, 1], [2.5, 97.5]).tolist(),
        "sequence_deltas": {},
    }
    for sequence in sorted(frame.sequence_id.unique()):
        use = frame.sequence_id.to_numpy() == sequence
        result["sequence_deltas"][sequence] = float(
            (mass[use] / mass[use].sum()) @ (losses[use, 0] - losses[use, 1])
        )
    expected = json.loads((root / "EWMA_TRANSPORT_CV_RESULTS.json").read_text())
    for key in ("candidate_score", "reference_score", "point_delta", "hierarchical_ci95"):
        if not np.allclose(result[key], expected[key], rtol=0, atol=1e-10):
            raise ValueError("independent regeneration disagrees: " + key)
    for sequence, value in result["sequence_deltas"].items():
        if not np.isclose(value, expected["sequence_deltas"][sequence], rtol=0, atol=1e-10):
            raise ValueError("independent sequence regeneration disagrees")
    profile = pd.read_csv(root / "prepared_heads/MEASUREMENTS.csv")
    if len(profile) != 768:
        raise ValueError("prepared TRAIN head measurements incomplete")
    head_outputs = replay_heads(root, profile)
    result.update(
        status="PASS",
        prepared_head_requests=len(profile),
        head_output_replay=head_outputs,
        scope="included analytical predictions/losses/bootstrap and TRAIN head outputs",
        independent_raw_training_replay=False,
    )
    return result


def replay_heads(root: Path, measurements: pd.DataFrame) -> dict:
    """Regenerate all256 distinct TRAIN outputs and verify their three recorded repeats."""
    import torch

    sys.path.insert(0, str(root / "source/src"))
    from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
    from e_jepa_ttc.simplex_t.phase import phase_to_ttc

    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    directory = root / "prepared_heads/replay"
    manifest = json.loads((directory / "MODELS.json").read_text())
    differences = []
    for label, record in manifest["models"].items():
        head = TemporalRefiner(TemporalConfig(**record["config"])).float().eval()
        with np.load(directory / f"{label}_weights.npz", allow_pickle=False) as z:
            head.load_state_dict({k: torch.from_numpy(z[k].copy()) for k in z.files})
        with np.load(directory / f"{label}_inputs.npz", allow_pickle=False) as z:
            for i, token in enumerate(manifest["query_tokens"]):
                xs = [
                    torch.from_numpy(z[k][i : i + 1].copy())
                    for k in ("features", "times", "valid", "experts")
                ]
                with torch.inference_mode():
                    phase = head(*xs)["point_phase"]
                    ttc = phase_to_ttc(phase.double())
                records = measurements.loc[
                    (measurements.label == label) & (measurements["query"] == token)
                ]
                if len(records) != 3:
                    raise ValueError("prepared replay query has missing blocks")
                differences.extend(abs(records.point_phase.to_numpy() - float(phase[0])))
                differences.extend(abs(records.ttc_s.to_numpy() - float(ttc[0])))
    maximum = float(np.max(differences))
    if maximum > 1e-12:
        raise ValueError("included prepared head output replay differs: " + str(maximum))
    return {
        "unique_outputs": 256,
        "recorded_outputs_verified": 768,
        "max_difference": maximum,
        "extra_optimizer_updates": 0,
        "latency_remeasured": False,
    }


def main() -> int:
    """Verify the delivered included analysis without the historical350MB archive."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.root.resolve()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
