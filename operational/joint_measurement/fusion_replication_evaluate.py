"""Three-seed fusion evaluation; average losses/metrics, never TTC or selected seeds."""

from __future__ import annotations

import argparse
import copy
import math

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.rgb_port.normalization import FrozenNormalizer
from operational.range_transfer.analyze import gate, metrics
from operational.range_transfer.common import freeze, read, sha, write
from operational.rgb_port.evaluate import BUCKETS, WEIGHTS, _bucket_mask

from .bootstrap import compare
from .common import ROOT, owner
from .data import role_rows
from .fusion_data import FusionTensorBank
from .fusion_export import export_head
from .fusion_features import FeatureIdentity, load_feature_cache
from .fusion_model import RGBAnchorComplement
from .fusion_replication_pipeline import DIRECTORY, event_block, freeze_all
from .fusion_replication_train import SEEDS, registered
from .fusion_train import DIRECTORY as ORIGINAL
from .fusion_train import inputs
from .train import committed_sources, setup
from .verify_predictions import verify


def mean_metrics(values: list[dict]) -> dict:
    """Apply native gates to seed-mean scalar metrics; retain each seed separately."""
    if not values or not all(value["complete_prediction_and_mid_coverage"] for value in values):
        raise ValueError("Seed aggregation requires complete signed-domain coverage for every seed")
    result = copy.deepcopy(values[0])
    scalar = ("overall_MiD", "p95_positive_crucial_overestimate_s")
    for name in scalar:
        result[name] = float(np.mean([value[name] for value in values]))
    for band in WEIGHTS:
        result["bins"][band]["group_macro_mid"] = float(
            np.mean([value["bins"][band]["group_macro_mid"] for value in values])
        )
    for name in ("mae_s", "sign_error_rate"):
        result["diagnostics"][name] = float(
            np.mean([value["diagnostics"][name] for value in values])
        )
    for name in ("late_warning", "false_alert"):
        if len({value[name]["denominator"] for value in values}) != 1:
            raise ValueError("Seed warning support changed")
        rates = [value[name]["rate"] for value in values]
        result[name]["rate"] = float(np.mean(rates)) if all(v is not None for v in rates) else None
    groups = set(result["group_crucial_mid"])
    if any(set(value["group_crucial_mid"]) != groups for value in values):
        raise ValueError("Seed crucial group support changed")
    result["group_crucial_mid"] = {
        group: float(np.mean([value["group_crucial_mid"][group] for value in values]))
        for group in groups
    }
    # Publish only explicitly averaged gate fields, never seed7's other diagnostics as means.
    result["bins"] = {
        band: {"group_macro_mid": result["bins"][band]["group_macro_mid"]} for band in WEIGHTS
    }
    result["diagnostics"] = {
        name: result["diagnostics"][name] for name in ("mae_s", "sign_error_rate")
    }
    return {
        name: result[name]
        for name in (
            "overall_MiD",
            "p95_positive_crucial_overestimate_s",
            "bins",
            "diagnostics",
            "late_warning",
            "false_alert",
            "group_crucial_mid",
            "complete_prediction_and_mid_coverage",
        )
    }


def loss_bootstrap(reference: list[pd.DataFrame], candidate: list[pd.DataFrame]) -> dict:
    """Resample acquisition groups after averaging per-group MiD over fixed seeds."""
    origin = candidate[0]
    matrices = []
    for collection in (reference, candidate):
        seed_arrays = []
        for frame in collection:
            for name in ("sample_token", "group_id", "target_ttc"):
                if not np.array_equal(frame[name], origin[name]):
                    raise ValueError("Seed-bootstrap cohorts differ")
            target, pred = frame.target_ttc.to_numpy(), frame.prediction_ttc.to_numpy()
            if not (np.isfinite(pred) & ((pred < 0) | (pred > 0.1))).all():
                raise ValueError("Seed-bootstrap cannot discard invalid predictions")
            mid = 10000 * np.abs(np.log1p(-0.1 / pred) - np.log1p(-0.1 / target))
            groups = []
            for group in sorted(frame.group_id.unique()):
                bands = {}
                for name, low, high in BUCKETS:
                    mask = (frame.group_id.to_numpy() == group) & _bucket_mask(target, low, high)
                    bands[name] = float(mid[mask].mean()) if mask.any() else np.nan
                groups.append([bands[name] for name in WEIGHTS])
            seed_arrays.append(groups)
        matrices.append(np.asarray(seed_arrays).mean(axis=0))
    draws = np.random.default_rng(12345).integers(len(matrices[0]), size=(2000, len(matrices[0])))
    means = []
    for matrix in matrices:
        selected = matrix[draws]
        count = np.isfinite(selected).sum(axis=1)
        means.append(
            np.divide(
                np.nansum(selected, axis=1),
                count,
                out=np.full_like(count, np.nan, dtype=np.float64),
                where=count > 0,
            )
        )
    delta = means[1] - means[0]
    table = {name: delta[:, index] for index, name in enumerate(WEIGHTS)}
    table["overall"] = delta @ np.asarray(list(WEIGHTS.values()))
    intervals = {}
    for name, array in table.items():
        valid = np.isfinite(array)
        intervals[name] = {
            "defined": int(valid.sum()),
            "undefined": int((~valid).sum()),
            "ci95_if_all_defined": np.quantile(array, [0.025, 0.975]).tolist()
            if valid.all()
            else None,
        }
    return {
        "unit": "acquisition_group",
        "seed": 12345,
        "replicates": 2000,
        "averaging": "per-group MiD across fixed seeds; no averaging of TTC",
        "training_seed_variability": (
            "Reported separately as metric sample std; not estimated by this bootstrap"
        ),
        "intervals": intervals,
    }


def load_head(seed: int, mode: str) -> tuple[RGBAnchorComplement, dict]:
    """Check every replica head's optimizer/sampler trajectory before inference."""
    endpoints = freeze_all()
    record = registered()
    recipe = {**record["head_recipes"][mode], "seed": seed}
    name = f"FUSION-{mode}-S{seed}"
    folder = DIRECTORY / "head_fits" / name
    payload = torch.load(folder / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    if (
        payload["identity"] != read(folder / "IDENTITY.json")
        or payload["completed_updates"] != 2500
        or payload["accumulation_index"] != 0
    ):
        raise ValueError("Replica fusion payload boundary changed")
    setup(seed)
    model = RGBAnchorComplement(
        28, 196, hidden=recipe["hidden"], mode=mode, residual_scale=recipe["residual_scale"]
    )
    final = payload["model_state_dict"]
    if final.keys() != model.state_dict().keys() or not all(
        torch.isfinite(value).all() for value in final.values()
    ):
        raise ValueError("Replica fusion tensor state differs")
    changed = [
        name for name, value in model.state_dict().items() if not torch.equal(value, final[name])
    ]
    if not changed:
        raise ValueError("Replica fusion did not learn")
    curve, timings = payload["cursor"]["curve"], payload["cursor"]["timings"]
    if len(curve) != 2500 or len(timings) != 2500:
        raise ValueError("Incomplete replica fusion curve")
    sampler = torch.Generator().manual_seed(seed)
    for step, (row, elapsed) in enumerate(zip(curve, timings, strict=True), 1):
        ids = torch.randint(18422, (recipe["batch_size"],), generator=sampler).tolist()
        if (
            row["ids"] != ids
            or row["update"] != step
            or elapsed < 0
            or not all(
                math.isfinite(value)
                for value in [elapsed, *[v for k, v in row.items() if k != "ids"]]
            )
        ):
            raise ValueError("Replica fusion trajectory differs")
    if not torch.equal(sampler.get_state(), payload["sampler_generator_state"]):
        raise ValueError("Replica fusion sampler differs")
    state = payload["optimizer_state_dict"]
    ids = [index for group in state["param_groups"] for index in group["params"]]
    names = [name for name, _ in model.named_parameters()]
    if len(ids) != len(names) or len(set(ids)) != len(ids) or set(ids) != set(state["state"]):
        raise ValueError("Replica fusion Adam coverage differs")
    for index, parameter_name in zip(ids, names, strict=True):
        entry = state["state"][index]
        if float(entry["step"]) != 2500 or any(
            entry[key].shape != final[parameter_name].shape or not torch.isfinite(entry[key]).all()
            for key in ("exp_avg", "exp_avg_sq")
        ):
            raise ValueError("Replica fusion Adam endpoint differs")
    if payload["scheduler_state_dict"]["last_epoch"] != 2500:
        raise ValueError("Replica fusion scheduler differs")
    proof = {
        "status": "PASS",
        "checkpoint_sha256": endpoints["heads"][name],
        "sample_ids_regenerated": 2500 * recipe["batch_size"],
        "changed_tensors": changed,
        "adam_step": 2500,
        "scheduler_step": 2500,
        "sampler_exact": True,
    }
    freeze(folder / "ENDPOINT_VERIFICATION.json", proof)
    model.load_state_dict(final, strict=True)
    return model.eval().requires_grad_(False), proof


def register() -> dict:
    """Bind the seed aggregation rule before any replica outcome is evaluated."""
    path = "operational/joint_measurement/fusion_replication_evaluate.py"
    sources = {path: sha(ROOT / path)}
    sources.update(
        {
            name: sha(ROOT / name)
            for name in (
                "operational/joint_measurement/fusion_replication_pipeline.py",
                "operational/joint_measurement/fusion_export.py",
                "operational/joint_measurement/geometry_export.py",
                "operational/joint_measurement/bootstrap.py",
                "operational/joint_measurement/verify_predictions.py",
                "operational/range_transfer/analyze.py",
                "operational/rgb_port/evaluate.py",
            )
        }
    )
    committed_sources(sources)
    record = {
        "replication_registration_sha256": sha(DIRECTORY / "REGISTRATION.json"),
        "sources": sources,
        "seeds": [7, 13, 23],
        "sample_std_ddof": 1,
        "gate": (
            "Original system gate on each seed and on seed-mean scalar metrics; p95 "
            "safeguard uses mean of per-seed p95, not pooled predictions"
        ),
        "bootstrap": (
            "2000 paired acquisition-group resamples of seed-mean per-group MiD; never TTC ensemble"
        ),
        "selection": "No best seed, no recipe rescue; retain all three seeds and ZERO controls",
        "references": ["R_CTX", "F_TRUE", "FUSION-ZERO"],
    }
    freeze(DIRECTORY / "evaluation/REGISTRATION.json", record)
    return record


def evaluate() -> dict:
    """Publish all complete cohorts, original gates and exports before interpreting replicas."""
    registration = register()
    freeze_all()
    h_blocks, h_rows, _ = inputs()
    original_registration = read(ORIGINAL / "REGISTRATION.json")
    if sha(ORIGINAL / "normalizers/rgb.json") != original_registration["normalizers"]["rgb"]:
        raise ValueError("Shared historical RGB normalizer changed")
    rgb_normalizer = FrozenNormalizer.load(ORIGINAL / "normalizers/rgb.json")
    v_rows = role_rows("V")
    v_cache = ORIGINAL / "inputs/V/R_CTX"
    v_rgb = load_feature_cache(
        v_cache / "features.npz",
        expected_sha256=read(v_cache / "COMPLETE.json")["sha256"],
        expected_identity=FeatureIdentity(**original_registration["inputs"]["rgb"]["identity"]),
        role="V",
        manifest_sha256=sha(owner() / "V_MANIFEST.json"),
        sample_tokens=v_rows.sample_token.to_numpy(dtype=str),
        feature_count=28,
    )
    frames, results, exports, proofs = {}, {}, {}, {}
    if sha(ORIGINAL / "evaluation/RESULT.json") != registered()["seed7_evidence_sha256"]:
        raise ValueError("The original seed7 interpretation changed after replication admission")
    seed7 = read(ORIGINAL / "evaluation/RESULT.json")
    for name in ("FUSION-TRUE-S7", "FUSION-ZERO-S7", "R_CTX", "F_TRUE"):
        folder = ORIGINAL / "predictions/V" / name
        if sha(folder / "predictions.parquet") != seed7["results"][f"V/{name}"]["sha256"]:
            raise ValueError("Seed7 comparison table changed")
        frames[name] = pd.read_parquet(folder / "predictions.parquet")
        results[name] = seed7["results"][f"V/{name}"]["metrics"]
    for seed in SEEDS:
        event_normalizer = FrozenNormalizer.load(DIRECTORY / "normalizers" / f"event_S{seed}.json")
        h_event = event_block(seed, "H")
        v_event = event_block(seed, "V")
        for mode in ("TRUE", "ZERO"):
            name = f"FUSION-{mode}-S{seed}"
            model, proofs[name] = load_head(seed, mode)
            for role, rows, rgb, event in (
                ("H", h_rows, h_blocks["rgb"], h_event),
                ("V", v_rows, v_rgb, v_event),
            ):
                bank = FusionTensorBank(
                    rgb,
                    event,
                    rgb_normalizer,
                    event_normalizer,
                    role=role,
                    manifest_sha256=sha(owner() / f"{role}_MANIFEST.json"),
                    sample_tokens=rows.sample_token.to_numpy(dtype=str),
                )
                estimates = []
                with torch.inference_mode():
                    for start in range(0, len(rows), 128):
                        batch = bank.batch(torch.arange(start, min(start + 128, len(rows))))
                        output = model(**batch)["ttc"]
                        missing = ~batch["rgb_available"]
                        if not torch.equal(output[missing], batch["event_ttc"][missing]):
                            raise ValueError("Replica fallback changed")
                        estimates.append(output.numpy())
                frame = rows[["sample_token", "group_id", "target_ttc"]].copy()
                frame["prediction_ttc"] = np.concatenate(estimates).astype(np.float64)
                folder = DIRECTORY / "predictions" / role / name
                freeze(
                    folder / "BINDING.json",
                    {
                        "evaluation_sha256": sha(DIRECTORY / "evaluation/REGISTRATION.json"),
                        "endpoint_sha256": proofs[name]["checkpoint_sha256"],
                        "role": role,
                    },
                )
                path = folder / "predictions.parquet"
                if path.exists():
                    pd.testing.assert_frame_equal(pd.read_parquet(path), frame)
                else:
                    frame.to_parquet(path, index=False)
                score = metrics(
                    frame.target_ttc.to_numpy(),
                    frame.prediction_ttc.to_numpy(),
                    frame.group_id.to_numpy(),
                )
                freeze(folder / "METRICS.json", score)
                freeze(
                    folder / "COMPLETE.json",
                    {"status": "COMPLETE", "rows": len(frame), "sha256": sha(path)},
                )
                verify(folder)
                if role == "V":
                    frames[name], results[name] = frame, score
            examples = [
                (
                    torch.from_numpy(h_blocks["rgb"].features[:4].copy()),
                    torch.from_numpy(h_event.features[:4].copy()),
                    torch.from_numpy(h_blocks["rgb"].prediction_ttc[:4].copy()),
                    torch.from_numpy(h_event.prediction_ttc[:4].copy()),
                    torch.tensor(mask),
                )
                for mask in ([True] * 4, [False] * 4, [True, False, True, False])
            ]
            exports[name] = export_head(
                DIRECTORY / "exports" / name,
                model,
                rgb_normalizer,
                event_normalizer,
                h_blocks["rgb"].identity,
                h_event.identity,
                examples,
                binding={"endpoint_sha256": proofs[name]["checkpoint_sha256"]},
            )
    means, deviations = {}, {}
    for mode in ("TRUE", "ZERO"):
        scores = [results[f"FUSION-{mode}-S{seed}"] for seed in (7, *SEEDS)]
        means[mode] = mean_metrics(scores)
        deviations[mode] = {
            "aggregate": float(np.std([value["overall_MiD"] for value in scores], ddof=1)),
            "crucial": float(
                np.std([value["bins"]["crucial"]["group_macro_mid"] for value in scores], ddof=1)
            ),
            "mae": float(np.std([value["diagnostics"]["mae_s"] for value in scores], ddof=1)),
        }
    gates, intervals = {}, {}
    for reference in ("R_CTX", "F_TRUE", "ZERO"):
        baseline = means["ZERO"] if reference == "ZERO" else results[reference]
        gates[f"mean_TRUE_vs_{reference}"] = gate(baseline, means["TRUE"])
        references = (
            [frames[f"FUSION-ZERO-S{seed}"] for seed in (7, *SEEDS)]
            if reference == "ZERO"
            else [frames[reference]]
        )
        intervals[reference] = loss_bootstrap(
            references, [frames[f"FUSION-TRUE-S{seed}"] for seed in (7, *SEEDS)]
        )
        for seed in (7, *SEEDS):
            ref = results[f"FUSION-ZERO-S{seed}"] if reference == "ZERO" else results[reference]
            gates[f"seed{seed}_TRUE_vs_{reference}"] = gate(ref, results[f"FUSION-TRUE-S{seed}"])
    for seed in SEEDS:
        base = DIRECTORY / "predictions/V"
        compare(
            base / f"FUSION-ZERO-S{seed}" / "predictions.parquet",
            base / f"FUSION-TRUE-S{seed}" / "predictions.parquet",
            DIRECTORY / "bootstrap" / f"TRUE_vs_ZERO_S{seed}",
        )
    result = {
        "status": "REPLICA_METRICS_READY_FOR_INTERPRETATION",
        "registration": registration,
        "metrics": results,
        "means": means,
        "metric_sample_std": deviations,
        "gates": gates,
        "group_bootstrap_seed_mean_loss": intervals,
        "head_endpoint_proofs": proofs,
        "exports": exports,
        "system_promotion": False,
        "official_comparison": False,
        "full_objective_complete": False,
    }
    write(DIRECTORY / "evaluation/RESULT.json", result)
    return {"status": result["status"], "gates": gates, "metric_sample_std": deviations}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    print(register() if args.register_only else evaluate())
