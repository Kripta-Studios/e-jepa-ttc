"""Verify completed endpoint lineage, paired draws, frozen tensors and exposure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from operational.range_transfer.admission import revalidate

from .common import OWNER, ROOT, RUN, read, sha, write


def main() -> None:
    """Read checkpoints only; never spend optimizer updates during the final audit."""
    torch.set_num_threads(1)
    binding = read(RUN / "ADMISSION.json")
    revalidate(binding)
    parent = torch.load(
        OWNER / "fits/E_A5_MATCHED/checkpoint_last.pt", map_location="cpu", weights_only=False
    )["model_state_dict"]
    checks = []
    clipping = []
    readout_draws = {}
    context_draws = {}
    for receipt_path in sorted((RUN / "fits").glob("*/CHECKPOINT_RECEIPT.json")):
        receipt = read(receipt_path)
        if receipt["status"] != "COMPLETE":
            raise ValueError("Endpoint incomplete")
        path = Path(receipt["checkpoint_path"])
        if sha(path) != receipt["checkpoint_sha256"]:
            raise ValueError("Endpoint hash changed")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        sources = payload["identity"].get("source_code", {})
        for relative, expected in sources.items():
            if sha(ROOT / relative) != expected:
                raise ValueError("Scientific source changed: " + relative)
        name = receipt_path.parent.name
        if "-CTX-" in name:
            context_draws[name] = read(receipt_path.parent / "RUN_PROVENANCE.json")[
                "draw_sequence_sha256"
            ]
        frozen_count = None
        if name.startswith("R-") and "-CTX-" not in name:
            frozen_count = 0
            for key, value in parent.items():
                if not key.startswith("residual.scorer."):
                    if not torch.equal(value, payload["model_state_dict"]["model." + key]):
                        raise ValueError("Frozen parent tensor changed: " + key)
                    frozen_count += 1
            readout_draws[name] = read(receipt_path.parent / "RUN_PROVENANCE.json")[
                "draw_sequence_sha256"
            ]
        checks.append(
            {
                "fit": name,
                "sha256": sha(path),
                "source_files_verified": len(sources),
                "unchanged_parent_tensors": frozen_count,
                "completed_updates": payload["completed_updates"],
            }
        )
        curve = pd.read_csv(receipt_path.parent / "CURVE.csv")
        norm = curve.gradient_l2_before_clip.to_numpy()
        clipping.append(
            {
                "fit": name,
                "updates": len(curve),
                "fraction_clipped": float((norm > 1).mean()),
                "last250_fraction_clipped": float((norm[-250:] > 1).mean()),
                "gradient_norm_p95": float(np.quantile(norm, 0.95)),
                "gradient_norm_max": float(norm.max()),
            }
        )
    for seed in (7, 13, 23):
        names = [f"R-C{cap}-S{seed}" for cap in ("05", "20")]
        present = [name in readout_draws for name in names]
        if any(present) and (
            not all(present) or readout_draws[names[0]] != readout_draws[names[1]]
        ):
            raise ValueError("Paired readout draw sequence differs")
        names = [f"R-C{cap}-CTX-S{seed}" for cap in ("05", "20")]
        present = [name in context_draws for name in names]
        if any(present) and (
            not all(present) or context_draws[names[0]] != context_draws[names[1]]
        ):
            raise ValueError("Paired context draw sequence differs")
    cohorts = []
    for path in sorted((RUN / "families").glob("*/*/HISTORY.npz")):
        role = path.parent.name
        with np.load(OWNER / "features" / role / "E_HISTORY.npz", allow_pickle=False) as old:
            with np.load(path, allow_pickle=False) as new:
                keys = ("sample_token", "group_id", "target_phase", "timing", "valid")
                for key in keys:
                    if not np.array_equal(old[key], new[key]):
                        raise ValueError(f"Family cohort/target/clock drift: {path}/{key}")
        cohorts.append({"family": path.parent.parent.name, "role": role, "unchanged": list(keys)})
    owner_checks = []
    for key, value in read(RUN / "OWNER_OBSERVATION.json").items():
        if isinstance(value, dict) and "path" in value and "sha256" in value:
            owner_checks.append(
                {"name": key, "unchanged": sha(Path(value["path"])) == value["sha256"]}
            )
    pd.DataFrame(clipping).to_csv(RUN / "CLIPPING_ALL_FITS.csv", index=False)
    write(
        RUN / "qa/FINAL_ENDPOINT_AUDIT.json",
        {
            "status": "PASS",
            "optimizer_updates": 0,
            "endpoints": checks,
            "readout_paired_draw_hashes": readout_draws,
            "context_paired_draw_hashes": context_draws,
            "cohort_label_clock_identity": cohorts,
            "owner_files": owner_checks,
            "admitted_V13_sources_revalidated": True,
        },
    )


if __name__ == "__main__":
    main()
