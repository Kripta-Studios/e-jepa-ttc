"""Verify real cache contracts and report OOF versus in-sample states without training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jsonschema
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.cache_components import verify_components  # noqa: E402
from e_jepa_ttc.artifacts.stage63_65 import sign_stage63_65_artifact  # noqa: E402
from e_jepa_ttc.artifacts.training_authorization import read_signed  # noqa: E402
from e_jepa_ttc.data.crossfitted_a5_state import load_a5_state_split  # noqa: E402
from e_jepa_ttc.data.raw_temporal_cache import load_raw_cache  # noqa: E402


def run(root: Path, coherent_root: Path) -> dict[str, object]:
    """Read only complete caches; never replace OOF state by its in-sample counterpart."""
    feasibility = read_signed(root / "stage63/X3_FEASIBILITY_V2.json")
    if feasibility.get("integrity_passed") is not True:
        raise ValueError("input QA cannot approve a failed integrity audit")
    if feasibility.get("training_ready") is not True:
        from scripts.run_scientific_recovery_v9_stage65 import _validate_nested_sources

        proof = read_signed(root / "stage65/PREREQUISITES_MANIFEST.json")
        paths = [Path(item["path"]) for item in proof["input_bindings"].values()]
        router = next(
            path.parents[3]
            for path in paths
            if path.as_posix().endswith("/outer_fold0_seed7/a5/inner0/nested_protocol.json")
        )
        stage61 = next(
            path.parents[1] / "stage61"
            for path in paths
            if path.name == "outer0_final.metadata.csv"
        )
        teacher = next(path for path in paths if path.name == "FROZEN_TEACHER_PROVENANCE.json")
        current = _validate_nested_sources(router, stage61, teacher)
        if current["input_bindings"] != proof["input_bindings"]:
            raise ValueError("fallback prerequisites changed after audit")
        return {
            "status": "passed",
            "branch": "verified_cpu_fallback_sources",
            "training_executed": False,
            "producer_count": current["producer_count"],
        }
    for relative, schema in (
        ("raw_temporal_cache/manifest.json", "raw_cache_v1"),
        ("crossfitted_a5_state/manifest.json", "crossfitted_a5_state_v1"),
        ("stage63/X3_FEASIBILITY_V2.json", "stage63_feasibility_v2"),
    ):
        jsonschema.validate(
            read_signed(root / relative),
            json.loads(
                (ROOT / f"schemas/scientific_recovery_v9_{schema}.schema.json").read_text(
                    encoding="utf-8"
                )
            ),
        )
    cache = load_raw_cache(root / "raw_temporal_cache")
    summaries = {}
    for outer in range(3):
        train = load_a5_state_split(root / "crossfitted_a5_state", outer, "train")
        evaluation = load_a5_state_split(root / "crossfitted_a5_state", outer, "eval")
        if set(train.metadata.sample_token) & set(evaluation.metadata.sample_token):
            raise ValueError("train/evaluation state tokens overlap")
        if set(train.metadata.sample_token) | set(evaluation.metadata.sample_token) != set(
            cache.metadata.sample_token
        ):
            raise ValueError("A5 state folds do not cover the canonical raw universe")
        producer = coherent_root / f"outer{outer}_final"
        manifest = read_signed(producer / "manifest.json")
        verify_components(producer, manifest["components"], {"state.npy", "metadata.csv"})
        metadata = pd.read_csv(producer / "metadata.csv").reset_index(names="source_index")
        order = (
            metadata.set_index("sample_token")
            .loc[train.metadata.sample_token, "source_index"]
            .to_numpy(np.int64)
        )
        in_sample = np.load(producer / "state.npy", allow_pickle=False)[order]
        delta = np.abs(train.state.astype(np.float64) - in_sample.astype(np.float64))
        summaries[str(outer)] = {
            "train_rows": len(train.state),
            "eval_rows": len(evaluation.state),
            "absolute_difference_mean": delta.mean(axis=0).tolist(),
            "absolute_difference_max": delta.max(axis=0).tolist(),
            "source_producer_sha256": manifest["checkpoint_sha256"],
        }
    return sign_stage63_65_artifact(
        {
            "artifact_type": "stage63_65_real_input_contract_verification_v2",
            "status": "passed",
            "canonical_tokens": len(cache.metadata),
            "outer_folds": summaries,
            "state_order": ["benchmark_phase", "log_variance", "sensor_support"],
            "in_sample_comparison_scope": "diagnostic_only_never_used_for_training_or_selection",
            "training_executed": False,
        },
        evidence_type="cache_and_nested_state_qa",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--coherent-a5-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output_root, args.coherent_a5_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
