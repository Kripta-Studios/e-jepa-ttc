"""Label-free physical-identity replay of complete frozen parent populations."""

from __future__ import annotations

import argparse
import hashlib

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.multimodal_validation.common import REG, RUN, read, sha, write

from .common import OUT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=("DEV32", "FCWD"))
    parser.add_argument("--route", choices=("cpu", "native_gpu"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    source = (
        REG.parent / "predictions/PREDICTIONS.csv"
        if args.cohort == "FCWD"
        else RUN / "transfer/dev32/PREDICTIONS.csv"
    )
    native_seal = read(source.parent / "PREDICTIONS_SEALED.json")
    assert sha(source) == native_seal["predictions_sha256"]
    historical = pd.read_csv(source)
    assert not historical.duplicated(["sequence_id", "anchor_us"]).any()
    historical = historical.set_index(["sequence_id", "anchor_us"])
    root = "external_states" if args.route == "cpu" else "external_states_gpu_native"
    directory = OUT / root / args.cohort
    seal = read(directory / "SEALED.json")
    assert sha(directory / "BINDING.json") == seal["binding_sha256"]
    assert len(historical) == seal["population"]
    records, keys = [], set()
    for qid, receipt in seal["queries"].items():
        key = receipt["sequence_id"], receipt["anchor_us"]
        assert key not in keys
        keys.add(key)
        old = historical.loc[key]
        assert old.query_id == qid and receipt["query_id"] == qid
        assert bool(old.rgb_available) == receipt["rgb_available"]
        assert receipt["binding_sha256"] == seal["binding_sha256"]
        path = directory / (hashlib.sha256(qid.encode()).hexdigest() + ".npz")
        assert sha(path) == receipt["sha256"]
        with np.load(path, allow_pickle=False) as z:
            for model, field in (
                ("E_CTX_MATCHED", "event_phase"),
                ("R_CTX", "rgb_phase"),
                ("F_TRUE", "fusion_phase"),
            ):
                if field not in z.files:
                    assert model == "R_CTX" and not receipt["rgb_available"]
                    assert pd.isna(old[model])
                    continue
                current = float(phase_to_ttc(torch.from_numpy(z[field]))[0])
                previous = float(old[model])
                records.append(
                    dict(
                        query_id=qid,
                        sequence_id=key[0],
                        anchor_us=key[1],
                        model=model,
                        replay=current,
                        historical=previous,
                        delta_s=abs(current - previous),
                        sign_changed=(current < 0) != (previous < 0),
                    )
                )
    assert keys == set(historical.index)
    frame = pd.DataFrame(records)
    target = OUT / "parent_replay" / args.route / args.cohort
    target.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target / "DELTAS.csv", index=False)
    summary = {
        name: dict(
            comparisons=len(part),
            max_delta_s=float(part.delta_s.max()),
            mean_delta_s=float(part.delta_s.mean()),
            sign_changes=int(part.sign_changed.sum()),
            worst_queries=part.nlargest(5, "delta_s").to_dict("records"),
        )
        for name, part in frame.groupby("model")
    }
    write(
        target / "REPLAY.json",
        dict(
            summary=summary,
            source_sha256=sha(__file__),
            historical_predictions_sha256=sha(source),
            historical_device=native_seal["device"],
            state_seal_sha256=sha(directory / "SEALED.json"),
            physical_identities_verified=len(keys),
            targets_read=False,
            optimizer_updates=0,
            route=args.route,
            cohort=args.cohort,
        ),
    )
    print(
        {
            name: {k: v for k, v in row.items() if k != "worst_queries"}
            for name, row in summary.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
