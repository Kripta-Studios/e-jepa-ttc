"""Strict read-only views of native histories; mass changes live in V14."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.range_transfer.mass import fit_mass_plans
from operational.rgb_port.train_heads import SealedCache


class HeadCache(SealedCache):
    """Native H input with stronger uniqueness, finiteness and domain checks."""

    def __init__(self, path: Path) -> None:
        super().__init__(path, "E_CTX_MATCHED")
        if len(set(map(str, self.values["sample_token"]))) != self.population:
            raise ValueError("Duplicate H sample tokens")
        for name in ("features", "timing", "expert_phase", "target_phase"):
            if self.values[name].dtype != np.float32 or not np.isfinite(self.values[name]).all():
                raise ValueError(f"Native FP32/finite contract violated: {name}")
        target = self.values["target_phase"].astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            self.target_ttc = -0.1 / np.expm1(-target)
        plans = fit_mass_plans(
            self.target_ttc, self.values["group_id"], self.values["mass"], fit_role="H"
        )
        self.plans = (
            plans.native,
            plans.benchmark.astype(np.float32),
            plans.subrange.astype(np.float32),
        )
        for value in self.values.values():
            value.setflags(write=False)

    def forward(self, model: torch.nn.Module, ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Keep all native observation slots and timing unchanged."""
        device = torch.device("cpu")
        return model(
            *(
                self.tensor(name, ids, device)
                for name in ("features", "timing", "valid", "expert_phase")
            )
        )


def verify_h_population(cache: HeadCache, owner: Path) -> None:
    """Bind tokens/groups and original masses to H metadata, without reading V GT."""
    import pandas as pd

    from .common import read, sha

    manifest = read(owner / "H_MANIFEST.json")
    path = Path(manifest["rows_path"])
    if sha(path) != manifest["rows_sha256"]:
        raise ValueError("H role metadata changed")
    rows = pd.read_parquet(path, columns=["sample_token", "group_id", "mass", "rgb_available"])
    rows = rows.loc[rows.rgb_available].set_index("sample_token", verify_integrity=True)
    tokens = cache.values["sample_token"].astype(str)
    if set(tokens) != set(rows.index):
        raise ValueError("Native H cache population differs; no silent target exclusion")
    ordered = rows.loc[tokens]
    if not np.array_equal(ordered.group_id.to_numpy().astype(str), cache.values["group_id"]):
        raise ValueError("H group identities changed")
    if not np.array_equal(ordered.mass.to_numpy(np.float32), cache.values["mass"]):
        raise ValueError("Native H masses differ from frozen role metadata")
