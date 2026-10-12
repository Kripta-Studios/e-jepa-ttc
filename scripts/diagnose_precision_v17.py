"""Audit frozen A5 precision weights on P/H, without any optimizer or V labels."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from e_jepa_ttc.range_readout.readout import VectorReadout
from operational.range_transfer.common import read, sha, write


def main() -> None:
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    old = root.parent / "e-jepa-ttc-v15-range-readout/artifacts/range_readout_20261012"
    owner = root.parent / "e-jepa-ttc-v13-rgb-port/artifacts/rgb_port_20261008"
    output = root / "artifacts/representation_audit_20261012/diagnostics"
    parent = owner / "fits/E_A5_MATCHED/checkpoint_last.pt"
    if sha(parent) != read(old / "ADMISSION.json")["parents"]["E_A5_MATCHED"]:
        raise ValueError("Parent changed")
    payload = torch.load(parent, map_location="cpu", weights_only=False)
    native = CausalScaleTTC(CausalScaleTTCConfig(**payload["identity"]["model_config"]))
    native.load_state_dict(payload["model_state_dict"], strict=True)
    model = VectorReadout(native.eval().requires_grad_(False))
    summaries = []
    for role in ("P", "H"):
        manifest = read(owner / f"{role}_MANIFEST.json")
        if sha(Path(manifest["rows_path"])) != manifest["rows_sha256"]:
            raise ValueError("Role changed")
        meta = pd.read_parquet(manifest["rows_path"], columns=["sample_token", "target_ttc"])
        meta = meta.set_index("sample_token", verify_integrity=True)
        folder = old / "vector_cache" / role
        n = read(folder / "COMPLETE.json")["rows"]
        batches = []
        with torch.inference_mode():
            for i in range((n + 127) // 128):
                path = folder / f"part_{i:06d}.npz"
                if sha(path) != read(path.with_suffix(".json"))["sha256"]:
                    raise ValueError("Shard changed")
                with np.load(path, allow_pickle=False) as arc:
                    tokens = arc["sample_token"].astype(str)
                    values = {
                        k: torch.from_numpy(arc[k].copy())
                        for k in arc.files
                        if k != "sample_token" and not k.startswith(("target_", "dino_"))
                    }
                out = model(values)
                target = meta.loc[tokens, "target_ttc"].to_numpy(np.float64)
                logvar = out.log_ratio_log_variance[:, -1].numpy().astype(np.float64)
                ratio = out.pair_log_height_ratio[:, -1].numpy().astype(np.float64)
                valid = (target > 0) | (target < -values["delta"][:, -1].numpy())
                truth_ratio = np.full(len(target), np.nan)
                truth_ratio[valid] = np.log1p(values["delta"][:, -1].numpy()[valid] / target[valid])
                batches.append(
                    pd.DataFrame(
                        {
                            "target": target,
                            "logvar": logvar,
                            "precision": np.exp(-logvar),
                            "squared_standardized_ratio_error": np.exp(-logvar)
                            * (ratio - truth_ratio) ** 2,
                        }
                    )
                )
        frame = pd.concat(batches, ignore_index=True)
        masks = {
            "all": np.ones(len(frame), bool),
            "0-1": (frame.target > 0) & (frame.target <= 1),
            "1-3": (frame.target > 1) & (frame.target <= 3),
            "3-6": (frame.target > 3) & (frame.target <= 6),
            "6+": frame.target > 6,
            "negative": frame.target < 0,
        }
        for band, mask in masks.items():
            part = frame.loc[mask]
            summaries.append(
                {
                    "role": role,
                    "band": band,
                    "n": len(part),
                    "median_precision": float(part.precision.median()),
                    "mean_precision": float(part.precision.mean()),
                    "p95_precision": float(part.precision.quantile(0.95)),
                    "share_precision_mass": float(part.precision.sum() / frame.precision.sum()),
                    "mean_squared_standardized_error": float(
                        part.squared_standardized_ratio_error.mean()
                    ),
                    "std_error_missing": int(part.squared_standardized_ratio_error.isna().sum()),
                }
            )
    table = pd.DataFrame(summaries)
    table.to_csv(output / "FROZEN_PRECISION_P_H.csv", index=False)
    write(
        output / "PRECISION_SCOPE.json",
        {
            "parent_sha256": sha(parent),
            "optimizer_updates": 0,
            "scope": "native current pair before temporal blend; frozen variance; descriptive P/H",
            "calibration_reference": (
                "Unit standardized mean-square is a Gaussian model diagnostic, "
                "not a fitted correction."
            ),
        },
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
