"""Compare frozen A5 FP32/BF16 routes on identical canonical inputs, without fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import compute_file_hash  # noqa: E402
from e_jepa_ttc.data.collision_clock_cache import (  # noqa: E402
    CollisionClockTrain8192Cache,
    load_canonical_supervision,
)
from e_jepa_ttc.data.stage61_pair_feature_cache import load_feature_cache  # noqa: E402
from e_jepa_ttc.evaluation.scientific_recovery_v8 import (  # noqa: E402
    load_causal_scale_replay_checkpoint,
)
from e_jepa_ttc.models.collision_clock_math import ttc_to_benchmark_phase  # noqa: E402


def main() -> None:
    """Run a paired precision diagnostic with one frozen producer on GPU at a time."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--stage61-worktree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batch-layout",
        choices=("hash_selected", "original_dense", "original_historical"),
        default="hash_selected",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("diagnostic evidence already exists")
    protocol = json.loads(
        (ROOT / "configs/protocol/scientific_recovery_v9_eclock_x0.json").read_text()
    )
    reference = json.loads(
        (ROOT / "configs/protocol/scientific_recovery_v9_eclock_x0_reference.json").read_text()
    )
    adapter = CollisionClockTrain8192Cache(
        args.reference_root / "artifacts/cache/garl_object_event_common_roi_train8192_v1",
        protocol,
        cache_mode="shard_lru",
        lru_capacity=2,
        canonical_supervision=load_canonical_supervision(reference, args.reference_root),
    )
    locators = adapter.verify_and_index()
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = []
    for outer in range(3):
        producer = args.reference_root / (
            f"artifacts/scientific_recovery_v8/results/router/outer_fold{outer}_seed7/a5/outer_dev"
        )
        checkpoint = producer / "train/model_best.pt"
        historical_all = pd.read_csv(producer / "expert_oof.csv").set_index("token_id")
        selected = sorted(
            [item for item in locators if item.outer_fold == outer],
            key=lambda item: hashlib.sha256(item.sample_token.encode()).hexdigest(),
        )[:32]
        if args.batch_layout == "original_dense":
            anchor = next(
                index
                for index, item in enumerate(locators)
                if item.sample_token == selected[0].sample_token
            )
            selected = locators[(anchor // 32) * 32 : (anchor // 32 + 1) * 32]
        elif args.batch_layout == "original_historical":
            original_tokens = historical_all.index.tolist()
            anchor = original_tokens.index(selected[0].sample_token)
            locator_map = {item.sample_token: item for item in locators}
            selected = [
                locator_map[token]
                for token in original_tokens[(anchor // 32) * 32 : (anchor // 32 + 1) * 32]
            ]
        tokens = [item.sample_token for item in selected]
        inputs, delta, _ = adapter._materialize(selected)
        input_sha = hashlib.sha256(inputs.numpy().tobytes()).hexdigest()
        inputs, delta = inputs.to(device), delta.to(device)
        arrays, metadata, _ = load_feature_cache(
            args.stage61_worktree
            / "artifacts/scientific_recovery_v9_stage61_stage62"
            / "feature_cache"
            / f"outer{outer}_final.npz"
        )
        indices = metadata.reset_index().set_index("sample_token").loc[tokens, "index"].to_numpy()
        dense = arrays["patch_features"][indices, 0, -3:]
        historical_mask = np.array([token in historical_all.index for token in tokens])
        historical = historical_all.loc[
            [token for token in tokens if token in historical_all.index]
        ]
        model = load_causal_scale_replay_checkpoint(checkpoint, device=device)
        outputs = {}
        with torch.inference_mode():
            for precision in ("fp32", "fp32_cudnn_tf32", "bf16", "bf16_cudnn_tf32"):
                torch.backends.cudnn.allow_tf32 = precision.endswith("cudnn_tf32")
                with torch.autocast(
                    "cuda", dtype=torch.bfloat16, enabled=precision.startswith("bf16")
                ):
                    output = model(inputs, delta, return_dense_features=True)
                phase, valid = ttc_to_benchmark_phase(output.ttc_mean_seconds, metric_delta_t_s=0.1)
                if not bool(valid.all()):
                    raise ValueError("diagnostic phase invalid")
                state = torch.stack(
                    (phase, output.ttc_log_variance, output.sensor_support[:, -1]), -1
                )
                values = state.float().cpu().numpy()
                lv = historical.prediction_log_variance.to_numpy()
                outputs[precision] = {
                    "dense_max_abs_by_component": np.abs(values - dense).max(axis=0).tolist(),
                    "historical_compared_rows": len(lv),
                    "historical_log_variance_max_abs": float(
                        np.abs(values[historical_mask, 1] - lv).max()
                    ),
                    "historical_log_variance_mean_abs": float(
                        np.abs(values[historical_mask, 1] - lv).mean()
                    ),
                    "state": values.tolist(),
                }
        results.append(
            {
                "outer_fold": outer,
                "checkpoint_sha256": compute_file_hash(str(checkpoint)),
                "input_sha256": input_sha,
                "tokens": tokens,
                "model_training": model.training,
                "outputs": outputs,
            }
        )
        print(
            json.dumps(
                {
                    "outer_fold": outer,
                    "comparisons": {
                        key: {name: value for name, value in record.items() if name != "state"}
                        for key, record in outputs.items()
                    },
                }
            ),
            flush=True,
        )
        del model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "diagnostic_completed",
                "training_executed": False,
                "selection": args.batch_layout,
                "torch": torch.__version__,
                "results": results,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
