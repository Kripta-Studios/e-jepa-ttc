"""Validate coherent extraction on the signed64-query real TRAIN replay inputs."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.evaluation.scientific_recovery_v8 import load_causal_scale_replay_checkpoint
from e_jepa_ttc.simplex_t.expert_features import extract_family
from e_jepa_ttc.simplex_t.lifecycle import ExclusiveLease, admitted
from e_jepa_ttc.training.stage61_pair_head import load_pair_head


def main() -> None:
    """No fits; compare against the existing same-runtime FP32 diagnostic."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--ancestry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve prior extraction evidence")
    if compute_file_hash(str(args.ancestry)) != (
        "62b185b32b5b26db939a526606a6948b10ae9be11b1f4741746861055d3db34e"
    ):
        raise ValueError("ancestry hash mismatch")
    if compute_file_hash(str(args.replay / "inputs.pt")) != (
        "5b509f3f2eb5549e2dcc4fa82ad4a8d809d80dce92c328faaf3d04d39467fb47"
    ):
        raise ValueError("TRAIN replay inputs changed")
    ancestry = json.loads(args.ancestry.read_text(encoding="utf-8"))
    paths = {r["sha256"]: Path(r["path"]) for r in ancestry["input_bindings"].values()}
    records = json.loads((args.replay / "REPLAY_DIAGNOSTIC.json").read_text(encoding="utf-8"))[
        "results"
    ]
    saved = torch.load(args.replay / "inputs.pt", weights_only=True, map_location="cpu")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    results = []
    offset = 0
    args.output.mkdir()
    with ExclusiveLease(args.replay.parent / "CURRENT_REPLAY.lock"):
        for record in records:
            resources = admitted([args.output])
            if not resources["has_headroom"]:
                raise RuntimeError("RESOURCE_PAUSE")
            hashes = record["checkpoint_sha256"]
            for digest in hashes.values():
                if compute_file_hash(str(paths[digest])) != digest:
                    raise ValueError("producer bytes changed")
            a5 = load_causal_scale_replay_checkpoint(paths[hashes["A5"]], device=device)
            c2f = load_causal_scale_replay_checkpoint(paths[hashes["C2F"]], device=device)
            pair = load_pair_head(paths[hashes["PAIR"]], device=device)
            count = record["rows"]
            x = saved["inputs"][offset : offset + count].to(device)
            dt = saved["delta"][offset : offset + count].to(device)
            arrays = extract_family(a5, c2f, pair, x, dt)
            stem = record["family"].replace("/", "_")
            with np.load(args.replay / f"{stem}.npz", allow_pickle=False) as old:
                exact = np.array_equal(arrays["expert_ttc"], old["actual_ttc"])
                pair_exact = np.array_equal(arrays["pair_features"], old["pair_features"])
            np.savez_compressed(
                args.output / f"{stem}.npz",
                allow_pickle=False,
                **arrays,
                tokens=np.asarray(saved["tokens"][offset : offset + count]),
            )
            results.append(
                {
                    "family": record["family"],
                    "rows": count,
                    "ttc_exact": exact,
                    "pair_features_exact": pair_exact,
                    "checkpoint_sha256": hashes,
                    "output_sha256": compute_file_hash(str(args.output / f"{stem}.npz")),
                    "resources": admitted([args.output]),
                }
            )
            print(json.dumps(results[-1]), flush=True)
            offset += count
            del a5, c2f, pair, x, dt, arrays
            gc.collect()
    write_new_json(
        args.output / "QA.json",
        {
            "status": "COHERENT_FP32_CURRENT_EXTRACTOR_CHECKED",
            "rows": offset,
            "results": results,
            "optimizer_updates": 0,
            "all_exact": all(r["ttc_exact"] and r["pair_features_exact"] for r in results),
            "scope": "current64 real inputs; temporal production cache still pending",
            "runtime": {
                "torch": torch.__version__,
                "device": torch.cuda.get_device_name(),
                "precision": "FP32",
                "cudnn_tf32": False,
                "matmul_tf32": False,
                "batch_layout": "signed64 cohort family order,6 or5 rows",
            },
        },
    )


if __name__ == "__main__":
    main()
