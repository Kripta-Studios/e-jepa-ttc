"""Train-all, freeze-all, then evaluate-all for the locked Stage 64 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))

from e_jepa_ttc.artifacts.stage63_65 import sign_stage63_65_artifact  # noqa: E402
from e_jepa_ttc.data.crossfitted_a5_state import load_a5_state_split  # noqa: E402
from e_jepa_ttc.data.raw_temporal_cache import (  # noqa: E402
    RawCacheView,
    fold_rate_normalization,
    load_raw_cache,
)
from e_jepa_ttc.evaluation.stage63_65 import (  # noqa: E402
    benchmark_phase,
    evaluate_gate,
    paired_hierarchical_bootstrap,
    scientific_mid_per_row,
    scientific_ttc,
    strict_macro_mass,
    strict_score,
)
from e_jepa_ttc.models.raw_time_residual import (  # noqa: E402
    RAW_ARMS,
    RawArm,
    RawModelBatch,
    RawTimeResidual,
    normalize_arm_rates,
)
from e_jepa_ttc.training.raw_time_residual import (  # noqa: E402
    IndexedCounts,
    RawTrainingConfig,
    TrainSupervision,
    fixed_derangements,
    load_frozen_raw_endpoint,
    state_normalization,
    train_raw_time_residual,
)


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _supervision(stage61_root: Path) -> pd.DataFrame:
    path = stage61_root / "stage61" / "aggregate_seed7" / "R2_oof.csv"
    frame = pd.read_csv(path, dtype={"sample_token": str})
    required = {"sample_token", "sequence_id", "track_id", "target_ttc_s"}
    if not required <= set(frame) or len(frame) != 8192 or frame["sample_token"].duplicated().any():
        raise ValueError("canonical Stage 64 supervision source is invalid")
    benchmark_phase(frame["target_ttc_s"].to_numpy(np.float64))
    return frame


def _join_rows(
    cache_metadata: pd.DataFrame, state_metadata: pd.DataFrame, supervision: pd.DataFrame
) -> pd.DataFrame:
    cache = cache_metadata.reset_index(names="cache_index")
    result = state_metadata.reset_index(names="state_index").merge(
        cache,
        on=["sample_token", "sequence_id", "track_id", "outer_fold"],
        validate="one_to_one",
        suffixes=("", "_cache"),
    )
    result = result.merge(
        supervision[["sample_token", "sequence_id", "track_id", "target_ttc_s"]],
        on=["sample_token", "sequence_id", "track_id"],
        validate="one_to_one",
    )
    if len(result) != len(state_metadata):
        raise ValueError("raw/A5/supervision join lost rows")
    return result.sort_values("sample_token").reset_index(drop=True)


def _evaluate_fold_arm(
    *,
    endpoint: Path,
    arm: RawArm,
    seed: int,
    outer: int,
    cache: RawCacheView,
    state_root: Path,
    supervision: pd.DataFrame,
    device: torch.device,
    batch_size: int = 16,
) -> pd.DataFrame:
    model, checkpoint, manifest = load_frozen_raw_endpoint(endpoint, device=device)
    if manifest["arm"] != arm or manifest["seed"] != seed or manifest["outer_fold"] != outer:
        raise ValueError("frozen endpoint identity mismatch during evaluation")
    split = load_a5_state_split(state_root, outer, "eval")
    joined = _join_rows(cache.metadata, split.metadata, supervision)
    state = split.state[joined["state_index"].to_numpy(np.int64)]
    cache_index = joined["cache_index"].to_numpy(np.int64)
    tokens = joined["sample_token"].astype(str).tolist()
    permutations = fixed_derangements(tokens, seed)
    rate_mean = torch.as_tensor(checkpoint["rate_mean"], dtype=torch.float32, device=device)
    rate_std = torch.as_tensor(checkpoint["rate_std"], dtype=torch.float32, device=device)
    state_mean = torch.as_tensor(checkpoint["state_mean"], dtype=torch.float32, device=device)
    state_std = torch.as_tensor(checkpoint["state_std"], dtype=torch.float32, device=device)
    phase: list[np.ndarray] = []
    helper: list[np.ndarray] = []
    applied: list[np.ndarray] = []
    fallback: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(joined), batch_size):
            rows = slice(start, start + batch_size)
            indices = cache_index[rows]
            raw = torch.as_tensor(np.asarray(cache.counts[indices]), device=device)
            duration = torch.as_tensor(
                cache.durations_s[indices], dtype=torch.float32, device=device
            )
            normalized = normalize_arm_rates(
                raw,
                duration,
                rate_mean,
                rate_std,
                arm=arm,
                permutations=(
                    torch.as_tensor(permutations[rows], dtype=torch.long, device=device)
                    if arm == "S64-PERM-L1"
                    else None
                ),
            )
            state_value = torch.as_tensor(state[rows], dtype=torch.float32, device=device)
            output = model(
                RawModelBatch(
                    normalized,
                    state_value[:, 0],
                    (state_value - state_mean) / state_std,
                    torch.as_tensor(cache.times[indices], dtype=torch.float32, device=device),
                    torch.as_tensor(cache.valid_patches[indices], dtype=torch.bool, device=device),
                )
            )
            phase.append(output.benchmark_phase.cpu().numpy())
            helper.append(output.helper_input.cpu().numpy())
            applied.append(output.applied_phase_delta.cpu().numpy())
            fallback.append(output.fallback.cpu().numpy())
    phase_value = np.concatenate(phase).astype(np.float64)
    prediction = scientific_ttc(phase_value)
    target = joined["target_ttc_s"].to_numpy(np.float64)
    result = joined[["sample_token", "sequence_id", "track_id", "outer_fold"]].copy()
    result["seed"] = seed
    result["arm"] = arm
    result["target_ttc_s"] = target
    result["prediction_phase"] = phase_value
    result["prediction_ttc_s"] = prediction
    result["scientific_mid_per_row"] = scientific_mid_per_row(target, prediction)
    result["helper_input"] = np.concatenate(helper)
    result["applied_phase_delta"] = np.concatenate(applied)
    result["fallback"] = np.concatenate(fallback)
    result["failure"] = False
    result["checkpoint_sha256"] = manifest["checkpoint_sha256"]
    return cast(pd.DataFrame, result)


def _reference_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"sample_token": str})
    required = {"sample_token", "sequence_id", "track_id", "target_ttc_s", "prediction_ttc_s"}
    if not required <= set(frame) or len(frame) != 8192:
        raise ValueError(f"historical reference frame is invalid: {path}")
    return cast(pd.DataFrame, frame[list(required)].copy())


def _gates(
    predictions: dict[str, pd.DataFrame], stage61_root: Path, output: Path
) -> dict[str, Any]:
    candidate = predictions["S64-RAW-L1"]
    references: dict[str, pd.DataFrame] = {
        "S64-STATE-L1": predictions["S64-STATE-L1"],
        "S64-COUNT-L1": predictions["S64-COUNT-L1"],
        "S64-PERM-L1": predictions["S64-PERM-L1"],
        "S61-R2": _reference_frame(stage61_root / "stage61" / "aggregate_seed7" / "R2_oof.csv"),
        "RouterR": _reference_frame(
            stage61_root / "stage61" / "aggregate_seed7" / "RouterR_oof.csv"
        ),
    }
    thresholds = {
        "S64-STATE-L1": {"point_delta_lte": -1.0},
        "S64-COUNT-L1": {"point_delta_lte": -1.0},
        "S64-PERM-L1": {"point_delta_lt": 0.0},
        "S61-R2": {"point_delta_lte": -3.0},
        "RouterR": {},
    }
    gate_details: dict[str, Any] = {}
    draw_hashes: set[str] = set()
    for name, reference in references.items():
        bootstrap = paired_hierarchical_bootstrap(candidate, reference)
        draw_hashes.add(bootstrap.draws_sha256)
        gate_details[name] = evaluate_gate(bootstrap, **thresholds[name])
    # Draw construction depends only on paired identity, so every comparison must reuse it.
    if len(draw_hashes) != 1:
        raise AssertionError("Stage 64 comparisons did not reuse identical bootstrap draws")
    result = {
        "all_passed": all(item["passed"] for item in gate_details.values()),
        "candidate": "S64-RAW-L1",
        "comparisons": gate_details,
        "bootstrap_draws_sha256": next(iter(draw_hashes)),
        "scores": {name: strict_score(frame) for name, frame in predictions.items()},
    }
    _atomic_json(output, result)
    return result


def run_real_train_only_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Run two real outer-train microbatches without saving a selectable model."""

    output = args.output_root / "qa" / "STAGE64_REAL_TRAIN_ONLY_SMOKE.json"
    if output.is_file():
        return json.loads(output.read_text(encoding="utf-8"))
    stage61_root = args.stage61_worktree / "artifacts" / "scientific_recovery_v9_stage61_stage62"
    cache = load_raw_cache(args.output_root / "raw_temporal_cache")
    split = load_a5_state_split(args.output_root / "crossfitted_a5_state", 0, "train")
    joined = _join_rows(cache.metadata, split.metadata, _supervision(stage61_root))
    state = split.state[joined["state_index"].to_numpy(np.int64)]
    cache_index = joined["cache_index"].to_numpy(np.int64)
    target = benchmark_phase(joined["target_ttc_s"].to_numpy(np.float64)).astype(np.float32)
    mass = strict_macro_mass(
        joined["target_ttc_s"].to_numpy(np.float64),
        joined["sequence_id"].astype(str).to_numpy(),
    ).astype(np.float32)
    rate_mean, rate_std = fold_rate_normalization(cache, cache_index)
    state_mean, state_std = state_normalization(state)
    device = torch.device(args.device)
    torch.manual_seed(640064)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(640064)
        torch.cuda.reset_peak_memory_stats(device)
    model = RawTimeResidual().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    schedule = np.random.default_rng(640064).permutation(len(joined))[: 2 * args.microbatch]
    update0_exact = False
    losses: list[float] = []
    began = time.perf_counter()
    for batch_number in range(2):
        local = schedule[batch_number * args.microbatch : (batch_number + 1) * args.microbatch]
        index = cache_index[local]
        state_value = torch.as_tensor(state[local], dtype=torch.float32, device=device)
        normalized = normalize_arm_rates(
            torch.as_tensor(np.asarray(cache.counts[index]), device=device),
            torch.as_tensor(cache.durations_s[index], dtype=torch.float32, device=device),
            torch.as_tensor(rate_mean, device=device),
            torch.as_tensor(rate_std, device=device),
            arm="S64-RAW-L1",
        )
        batch = RawModelBatch(
            normalized,
            state_value[:, 0],
            (state_value - torch.as_tensor(state_mean, device=device))
            / torch.as_tensor(state_std, device=device),
            torch.as_tensor(cache.times[index], dtype=torch.float32, device=device),
            torch.as_tensor(cache.valid_patches[index], dtype=torch.bool, device=device),
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch).benchmark_phase
        if batch_number == 0:
            update0_exact = bool(torch.equal(prediction, batch.a5_phase))
        loss = (
            len(joined)
            * torch.as_tensor(mass[local], device=device)
            * torch.abs(prediction - torch.as_tensor(target[local], device=device))
        ).mean()
        loss.backward()
        if not bool(torch.isfinite(loss)) or any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise FloatingPointError("Stage 64 real smoke produced a non-finite loss/gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    result = {
        "artifact_type": "scientific_recovery_v9_stage64_real_train_only_smoke_v1",
        "status": "passed" if update0_exact else "failed",
        "outer_fold": 0,
        "role": "outer_train_inner_oof_only",
        "microbatch": args.microbatch,
        "batches": 2,
        "update0_exact_a5_phase": update0_exact,
        "losses": losses,
        "finite_gradients": True,
        "selectable_model_saved": False,
        "outer_dev_opened": False,
        "elapsed_seconds": time.perf_counter() - began,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
    }
    if result["status"] != "passed":
        raise RuntimeError("Stage 64 real train-only smoke failed update0 parity")
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(output, result)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    stage63 = json.loads(
        (args.output_root / "stage63" / "X3_FEASIBILITY_V2.json").read_text(encoding="utf-8")
    )
    if not stage63.get("training_ready"):
        raise RuntimeError("Stage 64 is forbidden because Stage 63 training_ready is false")
    stage61_root = args.stage61_worktree / "artifacts" / "scientific_recovery_v9_stage61_stage62"
    cache = load_raw_cache(args.output_root / "raw_temporal_cache")
    state_root = args.output_root / "crossfitted_a5_state"
    supervision = _supervision(stage61_root)
    device = torch.device(args.device)
    config = RawTrainingConfig(
        microbatch=args.microbatch,
        accumulation=64 // args.microbatch,
    )
    seed_root = args.output_root / "stage64" / f"seed{args.seed}"
    seed_root.mkdir(parents=True, exist_ok=args.resume)
    began = time.perf_counter()
    frozen: list[dict[str, Any]] = []
    # Train and freeze all 12 endpoints before constructing any outer prediction.
    for outer in range(3):
        split = load_a5_state_split(state_root, outer, "train")
        joined = _join_rows(cache.metadata, split.metadata, supervision)
        state = split.state[joined["state_index"].to_numpy(np.int64)]
        cache_index = joined["cache_index"].to_numpy(np.int64)
        target = joined["target_ttc_s"].to_numpy(np.float64)
        mass = strict_macro_mass(target, joined["sequence_id"].astype(str).to_numpy())
        rate_mean, rate_std = fold_rate_normalization(cache, cache_index)
        train_counts = IndexedCounts(cache.counts, cache_index)
        for arm in RAW_ARMS:
            endpoint = seed_root / f"outer{outer}" / arm
            manifest_path = endpoint / "frozen_manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            else:
                manifest = train_raw_time_residual(
                    counts=train_counts,
                    durations_s=cache.durations_s[cache_index],
                    times=cache.times[cache_index],
                    valid_patches=cache.valid_patches[cache_index],
                    a5_state=state,
                    sample_tokens=joined["sample_token"].astype(str).tolist(),
                    supervision=TrainSupervision(benchmark_phase(target), mass),
                    rate_mean=rate_mean,
                    rate_std=rate_std,
                    arm=arm,
                    seed=args.seed,
                    outer_fold=outer,
                    output_dir=endpoint,
                    device=device,
                    identity={
                        "training_lock_sha256": _sha256(args.output_root / "TRAINING_LOCK.json"),
                        "raw_cache_identity": stage63["raw_cache_manifest"]["identity_sha256"],
                        "a5_producer_seed": 7,
                    },
                    config=config,
                    resume=args.resume,
                )
            frozen.append(manifest)
            if time.perf_counter() - began > args.max_hours * 3600:
                raise TimeoutError("RAW_RESOURCE_BLOCKED: Stage 64 active-time cap reached")
    if len(frozen) != 12 or any(
        item.get("status") != "frozen" or item.get("completed_updates") != 3000 for item in frozen
    ):
        raise RuntimeError("not all Stage 64 endpoints reached fixed update3000")
    freeze_manifest = {
        "status": "all_endpoints_frozen_before_evaluation",
        "seed": args.seed,
        "endpoints": frozen,
        "config": asdict(config),
    }
    _atomic_json(seed_root / "ALL_ENDPOINTS_FROZEN.json", freeze_manifest)
    predictions: dict[str, pd.DataFrame] = {}
    for arm in RAW_ARMS:
        rows = [
            _evaluate_fold_arm(
                endpoint=seed_root / f"outer{outer}" / arm,
                arm=arm,
                seed=args.seed,
                outer=outer,
                cache=cache,
                state_root=state_root,
                supervision=supervision,
                device=device,
            )
            for outer in range(3)
        ]
        frame = (
            pd.concat(rows, ignore_index=True).sort_values("sample_token").reset_index(drop=True)
        )
        if len(frame) != 8192 or frame["sample_token"].nunique() != 8192:
            raise ValueError(f"Stage 64 {arm} OOF coverage is incomplete")
        path = seed_root / f"{arm}_oof.csv"
        frame.to_csv(path, index=False, lineterminator="\n")
        predictions[arm] = frame
    gates = _gates(predictions, stage61_root, seed_root / "STAGE64_GATES.json")
    result = {
        "artifact_type": "scientific_recovery_v9_stage64_result_v1",
        "seed": args.seed,
        "status": "RAW_ALL_GATES_PASSED" if gates["all_passed"] else "RAW_GATES_FAILED",
        "all_endpoints_frozen_before_evaluation": True,
        "gates": gates,
        "elapsed_active_seconds": time.perf_counter() - began,
    }
    result = sign_stage63_65_artifact(result, evidence_type="nested_outer_dev")
    _atomic_json(seed_root / "STAGE64_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage61-worktree", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, choices=(7, 13, 23), default=7)
    parser.add_argument("--microbatch", type=int, choices=(4, 8), default=8)
    parser.add_argument("--max-hours", type=float, default=24.0)
    parser.add_argument("--resume", action="store_true")
    result = run(parser.parse_args())
    print(json.dumps({"status": result["status"], "seed": result["seed"]}))


if __name__ == "__main__":
    main()
