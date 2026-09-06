"""Verify and bind the completed component-wise historical64-row replay evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json


def main() -> None:
    """Require exact FP32 source values, not a tolerance chosen after inspection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--raw-parity", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = {}

    def read(path: Path) -> dict[str, Any]:
        sources[str(path)] = compute_file_hash(str(path))
        return json.loads(path.read_text(encoding="utf-8"))

    raw = read(args.raw_parity)
    original = read(args.replay / "REPLAY_DIAGNOSTIC.json")
    recovered = read(args.replay / "RECOVERED_INPUT_COMPARISON.json")
    a5 = read(args.replay / "A5_HISTORICAL_BATCH_POSITION.json")
    predictions = read(args.replay / "HISTORICAL_DETERMINISTIC_VALIDATION.json")
    pair = read(args.replay / "PAIR_HISTORICAL_GPU_LAYOUT.json")
    if len(raw["queries"]) != 64 or not raw["all_windows_bit_identical"]:
        raise ValueError("raw64 input parity incomplete")
    if len(recovered["results"]) != 64 or not recovered["all_input_tensors_bit_identical"]:
        raise ValueError("recovered64 input parity incomplete")
    if len(a5["results"]) != 64 or any(
        r["max_abs_pair_token_by_cudnn_tf32"]["True"] != 0
        or r["abs_phase_error_by_cudnn_tf32"]["True"] != 0
        for r in a5["results"]
    ):
        raise ValueError("A5 exact64 feature parity incomplete")
    families = {}
    all_tokens = set()
    for family in original["results"]:
        name = family["family"].replace("/", "_")
        file = args.replay / f"{name}.npz"
        sources[str(file)] = compute_file_hash(str(file))
        with np.load(file, allow_pickle=False) as arrays:
            expected = arrays["expected_ttc"].astype(np.float32)
            tokens = arrays["tokens"].tolist()
        for index, expert in enumerate(("A5", "C2F")):
            matches = [
                r
                for r in predictions["results"]
                if r["family"] == family["family"] and r["expert"] == expert
            ]
            if len(matches) != 1:
                raise ValueError("missing or duplicate expert evidence")
            actual = np.asarray(matches[0]["modes"]["bf16_cudnn_tf32"]["ttc"], dtype=np.float32)
            if not np.array_equal(actual, expected[:, index]):
                raise ValueError("historical TTC is not FP32-identical")
        matches = [r for r in pair["results"] if r["family"] == family["family"]]
        if len(matches) != 1 or matches[0]["tokens"] != tokens:
            raise ValueError("PAIR evidence token mismatch")
        if not np.array_equal(
            np.asarray(matches[0]["pair_from_signed_features_ttc"], dtype=np.float32),
            expected[:, 2],
        ):
            raise ValueError("PAIR TTC is not FP32-identical")
        families[family["family"]] = tokens
        all_tokens.update(tokens)
    if len(families) != 12 or len(all_tokens) != 64:
        raise ValueError("incomplete or duplicated cohort")
    if set(r["token"] for r in a5["results"]) != all_tokens:
        raise ValueError("A5 feature cohort differs")
    if set(r["token"] for r in raw["queries"]) != all_tokens:
        raise ValueError("raw input cohort differs")
    write_new_json(
        args.output,
        {
            "status": "HISTORICAL64_COMPONENT_REPLAY_PARITY_PASSED",
            "rows": 64,
            "producer_families": 12,
            "checkpoints": 36,
            "historical_predictions_verified": 192,
            "raw_windows_bit_identical": 192,
            "a5_phase_and_128d_tokens_bit_identical": 64,
            "comparison": "Exact source FP32 equality; CSV float64 parsing is not model drift",
            "sources": sources,
            "families": families,
            "optimizer_updates": 0,
            "production_cache_frozen": False,
            "scientific_fit_authorized_by_this_file_alone": False,
            "precision_boundary": (
                "Historical A5/C2F point TTC uses deterministic BF16; A5 feature cache uses "
                "its separate FP32 extraction runtime. Do not mix variance fields across "
                "these routes in a new coherent cache."
            ),
        },
    )


if __name__ == "__main__":
    main()
