"""Test historical A5 producer batch shapes against signed feature values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.data.stage61_pair_feature_cache import load_feature_cache
from e_jepa_ttc.evaluation.scientific_recovery_v8 import load_causal_scale_replay_checkpoint
from e_jepa_ttc.models.collision_clock_math import ttc_to_benchmark_phase
from e_jepa_ttc.simplex_t.lifecycle import ExclusiveLease, admitted


def main() -> None:
    """Replicated padding tests shape only, not historical batch-content identity."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--ancestry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-position", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous diagnostic")
    receipt = json.loads((args.replay / "INPUTS_READY.json").read_text(encoding="utf-8"))
    if compute_file_hash(str(args.replay / "inputs.pt")) != receipt["sha256"]:
        raise ValueError("input hash changed")
    saved = torch.load(args.replay / "inputs.pt", weights_only=True, map_location="cpu")
    ancestry = json.loads(args.ancestry.read_text(encoding="utf-8"))
    bindings = {r["sha256"]: Path(r["path"]) for r in ancestry["input_bindings"].values()}
    records = json.loads((args.replay / "REPLAY_DIAGNOSTIC.json").read_text(encoding="utf-8"))[
        "results"
    ]
    # Verify the physical global-order metadata against its signed cache manifest.
    global_manifest = json.loads((args.feature_cache / "outer0_final.manifest.json").read_text())
    global_path = args.feature_cache / "outer0_final.metadata.csv"
    if compute_file_hash(str(global_path)) != global_manifest["metadata"]["sha256"]:
        raise ValueError("global input order changed")
    global_order = pd.read_csv(global_path)
    global_indices = {token: i for i, token in enumerate(global_order.sample_token)}
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    offset = 0
    with ExclusiveLease(args.replay.parent / "CURRENT_REPLAY.lock"), torch.inference_mode():
        for record in records:
            outer, role = record["family"].split("/")
            base = f"{outer}_{'final' if role == 'outer_dev' else role}"
            arrays, metadata, manifest = load_feature_cache(args.feature_cache / f"{base}.npz")
            digest = record["checkpoint_sha256"]["A5"]
            if manifest["identity"]["a5_checkpoint_sha256"] != digest:
                raise ValueError("wrong feature teacher")
            path = bindings[digest]
            if compute_file_hash(str(path)) != digest:
                raise ValueError("checkpoint changed")
            model = load_causal_scale_replay_checkpoint(path, device=torch.device("cuda"))
            indexed = metadata.reset_index().set_index("sample_token")
            sequences = set(manifest["identity"]["sequence_ids"])
            for _i in range(record["rows"]):
                if not admitted([args.replay])["has_headroom"]:
                    raise RuntimeError("RESOURCE_PAUSE")
                token = saved["tokens"][offset]
                global_index = global_indices[token]
                block = global_order.iloc[(global_index // 32) * 32 : (global_index // 32 + 1) * 32]
                count = int(block.sequence_id.isin(sequences).sum())
                selected_block = block.loc[block.sequence_id.isin(sequences)]
                position = (
                    selected_block.sample_token.tolist().index(token)
                    if args.historical_position
                    else 0
                )
                x = (
                    saved["inputs"][offset : offset + 1]
                    .expand(count, -1, -1, -1, -1)
                    .contiguous()
                    .cuda()
                )
                dt = saved["delta"][offset : offset + 1].expand(count, -1).contiguous().cuda()
                expected = arrays["pair_features"][int(indexed.loc[token, "index"]), :128]
                modes = {}
                phase_errors = {}
                for tf32 in (False, True):
                    torch.backends.cudnn.allow_tf32 = tf32
                    output = model(x, dt, return_dense_features=True)
                    actual = output.pair_tokens[position, -1].float().cpu().numpy()
                    modes[str(tf32)] = float(np.abs(actual - expected).max())
                    phase, valid = ttc_to_benchmark_phase(
                        output.ttc_mean_seconds[position : position + 1], metric_delta_t_s=0.1
                    )
                    if not bool(valid.all()):
                        raise ValueError("invalid replay phase")
                    expected_phase = arrays["a5_phase"][int(indexed.loc[token, "index"])]
                    phase_errors[str(tf32)] = abs(float(phase[0]) - float(expected_phase))
                    del output
                results.append(
                    {
                        "family": record["family"],
                        "token": token,
                        "historical_selected_batch_size": count,
                        "output_position": position,
                        "max_abs_pair_token_by_cudnn_tf32": modes,
                        "abs_phase_error_by_cudnn_tf32": phase_errors,
                    }
                )
                offset += 1
                del x, dt
            del model
            print(json.dumps({"family": record["family"], "completed": True}), flush=True)
        write_new_json(
            args.output,
            {
                "results": results,
                "optimizer_updates": 0,
                "scope": "Historical batch-shape diagnostic using repeated identical padding",
            },
        )


if __name__ == "__main__":
    main()
