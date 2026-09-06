"""Compare documented producer precision routes, without changing scientific recipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.evaluation.scientific_recovery_v8 import load_causal_scale_replay_checkpoint
from e_jepa_ttc.simplex_t.lifecycle import ExclusiveLease, admitted


def main() -> None:
    """Reuse already extracted inputs; never read raw shards or labels again."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--ancestry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous precision evidence")
    receipt = json.loads((args.replay / "INPUTS_READY.json").read_text(encoding="utf-8"))
    if compute_file_hash(str(args.replay / "inputs.pt")) != receipt["sha256"]:
        raise ValueError("replay input changed")
    ancestry = json.loads(args.ancestry.read_text(encoding="utf-8"))
    paths = {r["sha256"]: Path(r["path"]) for r in ancestry["input_bindings"].values()}
    replay = json.loads((args.replay / "REPLAY_DIAGNOSTIC.json").read_text(encoding="utf-8"))
    saved = torch.load(args.replay / "inputs.pt", map_location="cpu", weights_only=True)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    results = []
    offset = 0
    with ExclusiveLease(args.replay.parent / "CURRENT_REPLAY.lock"), torch.inference_mode():
        for record in replay["results"]:
            if not admitted([args.replay])["has_headroom"]:
                raise RuntimeError("RESOURCE_PAUSE")
            outer_name, role = record["family"].split("/")
            count = record["rows"]
            x = saved["inputs"][offset : offset + count].cuda()
            dt = saved["delta"][offset : offset + count].cuda()
            offset += count
            with np.load(args.replay / f"{outer_name}_{role}.npz", allow_pickle=False) as data:
                expected = data["expected_ttc"]
            for index, expert in enumerate(("A5", "C2F")):
                digest = record["checkpoint_sha256"][expert]
                if compute_file_hash(str(paths[digest])) != digest:
                    raise ValueError("producer checkpoint changed")
                model = load_causal_scale_replay_checkpoint(
                    paths[digest], device=torch.device("cuda")
                )
                modes = {}
                for mode in ("fp32_cudnn_tf32", "bf16", "bf16_cudnn_tf32"):
                    torch.backends.cudnn.allow_tf32 = mode.endswith("cudnn_tf32")
                    with torch.autocast(
                        "cuda", dtype=torch.bfloat16, enabled=mode.startswith("bf16")
                    ):
                        output = model(x, dt, return_dense_features=False)
                    prediction = output.ttc_mean_seconds.float().cpu().numpy()
                    modes[mode] = {
                        "ttc": prediction.tolist(),
                        "max_abs_ttc_difference": float(
                            np.abs(prediction - expected[:, index]).max()
                        ),
                    }
                    del output
                results.append({"family": record["family"], "expert": expert, "modes": modes})
                del model
            print(
                json.dumps({"family": record["family"], "precision_diagnostic": "completed"}),
                flush=True,
            )
        write_new_json(
            args.output,
            {
                "status": "TECHNICAL_PRECISION_DIAGNOSTIC_NOT_RECIPE_SELECTION",
                "results": results,
                "optimizer_updates": 0,
                "input_sha256": receipt["sha256"],
            },
        )


if __name__ == "__main__":
    main()
