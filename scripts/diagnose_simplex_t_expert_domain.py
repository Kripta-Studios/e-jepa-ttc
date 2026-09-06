"""Observe failed original expert outputs; never modify or replace predictions."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.simplex_t import expert_features
from e_jepa_ttc.training import stage61_pair_head


def main() -> None:
    """Run the unchanged cache command with an exception-only diagnostic hook."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if args.evidence.exists():
        raise FileExistsError("preserve prior domain diagnostic")
    original = expert_features.build_router_features
    load_pair = stage61_pair_head.load_pair_head
    native_phase: list[str] = []

    def capture_phase(module: torch.nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        native_phase[:] = [repr(float(value)) for value in output.detach().cpu()]

    def observed_loader(
        path: Path, *, device: torch.device
    ) -> stage61_pair_head.CachedPairDirectPhase:
        model = load_pair(path, device=device)
        model.register_forward_hook(capture_phase)
        return model

    stage61_pair_head.load_pair_head = observed_loader

    def observe(
        a5: pd.DataFrame, c2f: pd.DataFrame, pair: np.ndarray
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            return original(a5, c2f, pair)
        except ValueError as error:
            records = {}
            for name, values in (
                ("A5", a5.prediction_ttc.to_numpy()),
                ("C2F", c2f.prediction_ttc.to_numpy()),
                ("PAIR", np.asarray(pair)),
            ):
                valid = np.isfinite(values) & ((values < 0) | (values.astype(np.float64) > 0.1))
                records[name] = {
                    "ttc_repr": [repr(float(v)) for v in values],
                    "dtype": str(values.dtype),
                    "bytes_hex": values.tobytes().hex(),
                    "invalid_slots": np.flatnonzero(~valid).tolist(),
                }
            write_new_json(
                args.evidence,
                {
                    "status": "ORIGINAL_EXPERT_DOMAIN_FAILURE_OBSERVED",
                    "pair_native_phase_repr": native_phase,
                    "error": str(error),
                    "experts": records,
                    "predictions_changed": False,
                    "targets_read": False,
                    "optimizer_updates": 0,
                },
            )
            raise

    expert_features.build_router_features = observe
    runner = Path("scripts/build_simplex_t_context_features.py")
    sys.argv = [str(runner), *remaining]
    runpy.run_path(str(runner), run_name="__main__")


if __name__ == "__main__":
    main()
