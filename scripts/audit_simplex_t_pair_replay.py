"""Isolate PAIR checkpoint replay from its historical A5 feature extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.data.stage61_pair_feature_cache import PairFeatureBatch, load_feature_cache
from e_jepa_ttc.reproducibility import seed_everything
from e_jepa_ttc.training.stage61_pair_head import load_pair_head


def main() -> None:
    """Read only signed historical TRAIN caches and the already executed64-row replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--ancestry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-gpu-layout", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve existing PAIR diagnostic")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    if args.historical_gpu_layout:
        seed_everything(7, deterministic=True)
        torch.backends.cuda.matmul.allow_tf32 = False
    ancestry = json.loads(args.ancestry.read_text(encoding="utf-8"))
    bindings = {item["sha256"]: Path(item["path"]) for item in ancestry["input_bindings"].values()}
    replay = json.loads((args.replay / "REPLAY_DIAGNOSTIC.json").read_text(encoding="utf-8"))
    results = []
    for record in replay["results"]:
        outer, role = record["family"].split("/")
        name = f"{outer}_{'final' if role == 'outer_dev' else role}"
        arrays, metadata, manifest = load_feature_cache(args.feature_cache / f"{name}.npz")
        if manifest["identity"]["a5_checkpoint_sha256"] != record["checkpoint_sha256"]["A5"]:
            raise ValueError("PAIR input teacher identity mismatch")
        with np.load(args.replay / f"{outer}_{role}.npz", allow_pickle=False) as saved:
            tokens = saved["tokens"].tolist()
            rows = metadata.reset_index().set_index("sample_token").loc[tokens, "index"].to_numpy()
            features = arrays["pair_features"][rows].copy()
            regenerated = saved["pair_features"].copy()
            expected = saved["expected_ttc"][:, 2].copy()
        digest = record["checkpoint_sha256"]["PAIR"]
        checkpoint = bindings[digest]
        if compute_file_hash(str(checkpoint)) != digest:
            raise ValueError("PAIR checkpoint bytes changed")
        device = torch.device("cuda" if args.historical_gpu_layout else "cpu")
        model = load_pair_head(checkpoint, device=device)
        with torch.inference_mode():
            if args.historical_gpu_layout:
                producer = next(
                    p
                    for p in ancestry["producers"]
                    if p["outer_fold"] == int(outer.removeprefix("outer"))
                    and p["role"] == role
                    and p["expert"] == "A5"
                )
                dev_sequences = producer["split_validation"]["dev_sequence_ids"]
                original_tokens = metadata.loc[
                    metadata.sequence_id.isin(dev_sequences), "sample_token"
                ].tolist()
                values = []
                for index, token in enumerate(tokens):
                    position = original_tokens.index(token)
                    count = min(1024, len(original_tokens) - (position // 1024) * 1024)
                    batch = (
                        torch.from_numpy(features[index : index + 1])
                        .expand(count, -1)
                        .contiguous()
                        .to(device)
                    )
                    values.append(
                        float(model.predict_ttc(PairFeatureBatch(batch))[position % 1024])
                    )
                actual = np.asarray(values)
            else:
                actual = model.predict_ttc(PairFeatureBatch(torch.from_numpy(features))).numpy()
        results.append(
            {
                "family": record["family"],
                "tokens": tokens,
                "signed_cache_sha256": manifest["cache"]["sha256"],
                "a5_pair_token_max_abs_difference": float(
                    np.abs(features[:, :128] - regenerated[:, :128]).max()
                ),
                "input_timing_support_max_abs_difference": float(
                    np.abs(features[:, 128:] - regenerated[:, 128:]).max()
                ),
                "pair_from_signed_features_ttc": actual.tolist(),
                "expected_ttc": expected.tolist(),
                "pair_from_signed_features_max_abs_difference": float(
                    np.abs(actual - expected).max()
                ),
            }
        )
        print(json.dumps(results[-1]), flush=True)
    write_new_json(
        args.output,
        {
            "results": results,
            "optimizer_updates": 0,
            "historical_gpu_layout": args.historical_gpu_layout,
            "purpose": "Separate feature drift from PAIR-head replay; not scientific scoring",
        },
    )


if __name__ == "__main__":
    main()
