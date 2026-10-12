"""Observe the existing fusion's own hidden state, without fitting or reading TTC."""

from __future__ import annotations

import argparse
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.rgb_port.fusion import DualClockFusion
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.multimodal_validation.ancestry import historical_source_binding
from operational.multimodal_validation.common import RUN, admission, read, sha, write
from operational.rgb_port.train_heads import load_head_endpoint
from operational.rgb_port_fusion_fix.fusion import deterministic_forward

from .common import OUT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=("H", "V"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    directory = OUT / "fusion_states" / args.role
    inputs, sources = {}, {}
    for modality in ("E", "R"):
        path = RUN / f"features/{args.role}/{modality}_HISTORY.npz"
        assert sha(path) == read(path.with_suffix(".json"))["output_sha256"]
        sources[modality] = sha(path)
        with np.load(path, allow_pickle=False) as z:
            inputs[modality] = {
                key: z[key]
                for key in (
                    "features",
                    "timing",
                    "valid",
                    "expert_phase",
                    "sample_token",
                )
            }
    assert np.array_equal(inputs["E"]["sample_token"], inputs["R"]["sample_token"])
    binding = dict(
        sources=sources,
        role=args.role,
        source_sha256=sha(__file__),
        checkpoint_sha256=sha(RUN / "fits/F_TRUE/checkpoint_last.pt"),
        precision="CPU FP32 one thread batch128",
        targets_read=False,
        schema="F_TRUE combine hidden160, raw_location, emitted_phase",
        supervised_parent="F_TRUE trained on H; these states are not OOF",
        optimizer_updates=0,
    )
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "BINDING.json").exists():
        assert read(directory / "BINDING.json") == binding
    else:
        write(directory / "BINDING.json", binding)
    if (directory / "SEALED.json").exists():
        assert sha(directory / "states.npz") == read(directory / "SEALED.json")["states_sha256"]
        return
    with (
        historical_source_binding(),
        patch.object(DualClockFusion, "forward", deterministic_forward),
    ):
        model = load_head_endpoint(RUN / "fits/F_TRUE", "F_TRUE", torch.device("cpu"))
        model.eval().requires_grad_(False)
        tokens = inputs["E"]["sample_token"]
        parts = []
        for start in range(0, len(tokens), 128):
            admission()
            e, r = inputs["E"], inputs["R"]
            values = [
                torch.from_numpy(data[key][start : start + 128])
                for data, key in (
                    (e, "features"),
                    (e, "timing"),
                    (e, "valid"),
                    (e, "expert_phase"),
                    (r, "features"),
                    (r, "timing"),
                    (r, "valid"),
                )
            ]
            with torch.inference_mode():
                output = model(*values)
                parts.append(
                    torch.cat(
                        [
                            output["hidden"],
                            output["raw_location"][:, None],
                            output["point_phase"][:, None],
                        ],
                        -1,
                    ).numpy()
                )
        features = np.concatenate(parts)
    path = directory / "states.npz"
    np.savez_compressed(path, features=features, sample_token=tokens)
    write(
        directory / "SEALED.json",
        dict(
            states_sha256=sha(path),
            binding_sha256=sha(directory / "BINDING.json"),
            population=len(tokens),
            targets_read=False,
            optimizer_updates=0,
        ),
    )
    if args.role == "V":
        historical_path = RUN / "predictions/V/F_TRUE.csv"
        old = pd.read_csv(historical_path, usecols=["sample_token", "prediction_ttc"])
        old = old.set_index("sample_token")
        assert old.index.is_unique and set(old.index) == set(tokens)
        predictions = phase_to_ttc(torch.from_numpy(features[:, -1])).numpy()
        delta = np.abs(predictions - old.loc[tokens].prediction_ttc.to_numpy())
        write(
            directory / "PREDICTION_PARITY.json",
            dict(
                historical_sha256=sha(historical_path),
                max_delta_s=float(delta.max()),
                mean_delta_s=float(delta.mean()),
                targets_read=False,
            ),
        )
    print(args.role, features.shape, flush=True)


if __name__ == "__main__":
    main()
