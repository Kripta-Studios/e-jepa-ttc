"""Recover coherent A5 state triplets from frozen producers, without fitting."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.cache_components import component_record  # noqa: E402
from e_jepa_ttc.artifacts.stage63_65 import sign_stage63_65_artifact  # noqa: E402
from e_jepa_ttc.data.collision_clock_cache import (  # noqa: E402
    CollisionClockTrain8192Cache,
    load_canonical_supervision,
)
from e_jepa_ttc.data.crossfitted_a5_state import _producer_state  # noqa: E402
from e_jepa_ttc.data.stage61_pair_feature_cache import load_feature_cache  # noqa: E402
from e_jepa_ttc.models.collision_clock_math import ttc_to_benchmark_phase  # noqa: E402
from e_jepa_ttc.training.campaign_budget import CampaignBudget  # noqa: E402
from scripts.build_scientific_recovery_v9_stage61_reference import _producer  # noqa: E402


@torch.no_grad()
def main() -> None:
    """Replay original FP32 batches and reject any stored phase/support mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--stage61-worktree", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if torch.__version__ != "2.11.0+cu128":
        raise RuntimeError("coherent A5 replay requires the diagnosed PyTorch 2.11.0+cu128 route")
    args.output_root.mkdir(parents=True, exist_ok=False)
    budget = CampaignBudget(args.output_root / "budget.json", hours=12)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    protocol = json.loads(
        (ROOT / "configs/protocol/scientific_recovery_v9_eclock_x0.json").read_text()
    )
    reference = json.loads(
        (ROOT / "configs/protocol/scientific_recovery_v9_eclock_x0_reference.json").read_text()
    )
    adapter = CollisionClockTrain8192Cache(
        args.reference_root / "artifacts/cache/garl_object_event_common_roi_train8192_v1",
        protocol,
        cache_mode="shard_lru",
        lru_capacity=2,
        canonical_supervision=load_canonical_supervision(reference, args.reference_root),
    )
    locators = adapter.verify_and_index()
    router = args.reference_root / "artifacts/scientific_recovery_v8/results/router"
    feature_root = args.stage61_worktree / (
        "artifacts/scientific_recovery_v9_stage61_stage62/feature_cache"
    )
    producers = [
        _producer(router, outer=outer, inner=inner, device=torch.device("cuda"))
        for outer in range(3)
        for inner in (0, 1, 2, None)
    ]
    provenance = {}
    states: dict[str, list[np.ndarray]] = {producer.name: [] for producer in producers}
    tokens: dict[str, list[str]] = {producer.name: [] for producer in producers}
    for producer in producers:
        role = "outer_dev" if producer.inner_fold is None else f"inner{producer.inner_fold}"
        _, _, evidence = _producer_state(
            feature_cache_base=feature_root / producer.name,
            expert_root=router / f"outer_fold{producer.outer_fold}_seed7/a5/{role}",
            expected_role="outer_dev" if producer.inner_fold is None else "inner_oof",
        )
        provenance[producer.name] = evidence
    for start in range(0, len(locators), 32):
        budget.check()
        batch_locators = locators[start : start + 32]
        inputs, delta, _ = adapter._materialize(batch_locators)
        inputs, delta = inputs.cuda(), delta.cuda()
        for producer in producers:
            budget.check()
            selected = [
                index
                for index, item in enumerate(batch_locators)
                if item.sequence_id in producer.sequences
            ]
            if not selected:
                continue
            index = torch.tensor(selected, device="cuda")
            output = producer.model(inputs[index], delta[index], return_dense_features=True)
            phase, valid = ttc_to_benchmark_phase(output.ttc_mean_seconds, metric_delta_t_s=0.1)
            if not bool(valid.all()):
                raise ValueError("coherent A5 replay has invalid phase")
            state = torch.stack((phase, output.ttc_log_variance, output.sensor_support[:, -1]), -1)
            values = state.float().cpu().numpy()
            if not np.isfinite(values).all():
                raise ValueError("coherent A5 replay has non-finite state")
            states[producer.name].append(values)
            tokens[producer.name].extend(batch_locators[i].sample_token for i in selected)
        if start % 512 == 0:
            print(f"coherent frozen A5 replay: {start}/{len(locators)}", flush=True)
    for producer in producers:
        budget.check()
        arrays, metadata, cache_manifest = load_feature_cache(feature_root / f"{producer.name}.npz")
        state = np.concatenate(states[producer.name])
        if metadata.sample_token.tolist() != tokens[producer.name]:
            raise ValueError("coherent A5 replay order differs from original feature builder")
        if not np.array_equal(state[:, 0], arrays["a5_phase"]):
            raise ValueError(f"coherent A5 replay phase differs: {producer.name}")
        if not np.array_equal(state[:, 2], arrays["pair_features"][:, -2]):
            raise ValueError(f"coherent A5 replay support differs: {producer.name}")
        if "patch_features" in arrays and not np.array_equal(
            state, arrays["patch_features"][:, 0, -3:]
        ):
            raise ValueError(f"coherent A5 replay dense state differs: {producer.name}")
        destination = args.output_root / producer.name
        destination.mkdir()
        np.save(destination / "state.npy", state)
        metadata.to_csv(destination / "metadata.csv", index=False, lineterminator="\n")
        manifest = sign_stage63_65_artifact(
            {
                "artifact_type": "coherent_frozen_a5_state_v2",
                "training_executed": False,
                "all_components_same_forward": True,
                "exact_feature_phase_support_replay": True,
                "torch": torch.__version__,
                "precision": "fp32",
                "cudnn_tf32": True,
                "matmul_tf32": False,
                "batch_layout": "original_global32",
                "checkpoint_sha256": cache_manifest["identity"]["a5_checkpoint_sha256"],
                "producer": producer.name,
                "prior_sources": provenance[producer.name],
                "components": {
                    name: component_record(destination / name)
                    for name in ("state.npy", "metadata.csv")
                },
            },
            evidence_type="frozen_producer_replay",
        )
        (destination / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print("coherent A5 replay complete; no corrector training performed", flush=True)


if __name__ == "__main__":
    main()
