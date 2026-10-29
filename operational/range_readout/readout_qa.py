"""Synthetic scorer continuation and frozen-parameter verification, 20 updates."""

from pathlib import Path

import numpy as np
import torch

from operational.range_transfer.qa import same

from .common import RUN, read, sha, write
from .train_readout import fit


class SyntheticVectors:
    """Random synthetic vector fixture; no P/H/V values or labels reused."""

    def __init__(self) -> None:
        prototype = RUN / "qa/CPU_VECTOR_PILOT/part_000000.npz"
        rng = np.random.default_rng(88)
        with np.load(prototype, allow_pickle=False) as z:
            shapes = {k: z[k].shape[1:] for k in z.files if k != "sample_token"}
            kinds = {k: z[k].dtype for k in shapes}
        self.values = {}
        for key, shape in shapes.items():
            full = (64, *shape)
            if kinds[key].kind == "b":
                value = np.ones(full, bool)
                value[:, 0] = False
            elif kinds[key].kind in "iu":
                value = np.ones(full, np.int64)
            else:
                value = np.full(full, 0.5, np.float32)
                if key in ("geometry_tokens", "forward_transport", "reverse_transport"):
                    value = rng.normal(size=full).astype(np.float32) * 0.1
                elif key in ("analytic", "raw_foreground_ratio", "raw_foreground_width_ratio"):
                    value = rng.normal(size=full).astype(np.float32) * 0.01
                elif key == "delta":
                    value.fill(0.1)
                elif key == "log_height_correction":
                    value.fill(0)
                elif key == "target_ttc":
                    value = np.tile([0.5, 1.5, 5.0, -2.0], 16).astype(np.float32)
            self.values[key] = value
        self.population = 64
        self.binding = {
            "status": "SYNTHETIC_ONLY",
            "seed": 88,
            "population": 64,
            "scope": "shapes_only_no_data_values_from_P",
            "schema": "synthetic_vectors_v1",
        }

    def batch(self, ids: torch.Tensor) -> dict:
        return {k: torch.from_numpy(v[ids.numpy()].copy()) for k, v in self.values.items()}


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    target = RUN / "qa/READOUT_RESUME.json"
    if target.exists():
        for p, h in read(target)["checkpoints"].items():
            assert sha(Path(p)) == h
        print(read(target))
        return
    cache = SyntheticVectors()
    a = fit(cache, 0.20, technical="R_continuous10_guarded")
    fit(cache, 0.20, technical="R_split5_restore5_guarded", stop=5)
    b = fit(cache, 0.20, technical="R_split5_restore5_guarded", stop=5)
    x = torch.load(a["checkpoint_path"], weights_only=False, map_location="cpu")
    y = torch.load(b["checkpoint_path"], weights_only=False, map_location="cpu")
    keys = (
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "sampler_generator_state",
        "torch_rng_state",
        "numpy_random_state",
        "python_random_state",
        "completed_updates",
    )
    checks = {k: same(x[k], y[k]) for k in keys}
    checks["draws"] = same(x["cursor"]["draw_hashes"], y["cursor"]["draw_hashes"])
    checks["losses"] = same(x["cursor"]["losses"], y["cursor"]["losses"])
    assert all(checks.values()), checks
    result = {
        "status": "PASS",
        "checks": checks,
        "optimizer_updates": 20,
        "frozen_modules_verified_each_fit": True,
        "checkpoints": {r["checkpoint_path"]: r["checkpoint_sha256"] for r in (a, b)},
    }
    write(target, result)
    print(result)


if __name__ == "__main__":
    main()
