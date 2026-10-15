"""Evaluate the fixed fusion pair on complete H/V, preserving native system gates."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.rgb_port.normalization import FrozenNormalizer
from operational.range_transfer.analyze import gate, metrics
from operational.range_transfer.common import digest, freeze, read, sha, utc, write

from .bootstrap import compare
from .common import ROOT, RUN, owner
from .data import role_rows
from .fusion_data import FusionTensorBank
from .fusion_export import export_head
from .fusion_features import FeatureIdentity, load_feature_cache
from .fusion_model import RGBAnchorComplement
from .fusion_train import DIRECTORY, MODES, REGISTRATION, freeze_endpoints, inputs
from .system_compare import require_cohort
from .train import committed_sources, setup
from .verify_predictions import verify


def load_head(mode: str) -> tuple[RGBAnchorComplement, dict]:
    """Regenerate every sampler draw and verify actual Adam, scheduler and learned tensors."""
    endpoint = freeze_endpoints()
    record = read(REGISTRATION)
    fit = f"FUSION-{mode}-S7"
    folder = DIRECTORY / "fits" / fit
    recipe = record["recipes"][mode]
    setup(recipe["seed"])
    model = RGBAnchorComplement(
        28, 196, hidden=recipe["hidden"], residual_scale=recipe["residual_scale"], mode=mode
    )
    payload = torch.load(folder / "checkpoint_last.pt", map_location="cpu", weights_only=False)
    if (
        payload["identity"] != read(folder / "IDENTITY.json")
        or payload["completed_updates"] != recipe["updates"]
        or payload["accumulation_index"] != 0
    ):
        raise ValueError("Fusion payload endpoint/identity changed")
    final = payload["model_state_dict"]
    original = model.state_dict()
    if final.keys() != original.keys() or not all(
        final[key].shape == value.shape
        and final[key].dtype == value.dtype
        and torch.isfinite(final[key]).all()
        for key, value in original.items()
    ):
        raise ValueError("Fusion final weights are incompatible or nonfinite")
    changed = [key for key in original if not torch.equal(original[key], final[key])]
    if not changed:
        raise ValueError("Fusion scientific fit did not change any weights")
    generator = torch.Generator().manual_seed(recipe["seed"])
    curve, timings = payload["cursor"]["curve"], payload["cursor"]["timings"]
    if len(curve) != recipe["updates"] or len(timings) != len(curve):
        raise ValueError("Incomplete fusion trajectory")
    for step, (row, duration) in enumerate(zip(curve, timings, strict=True), 1):
        indices = torch.randint(record["population"], (recipe["batch_size"],), generator=generator)
        if (
            row["update"] != step
            or row["ids"] != indices.tolist()
            or duration < 0
            or not all(
                math.isfinite(value)
                for value in [duration, *[v for k, v in row.items() if k != "ids"]]
            )
        ):
            raise ValueError("Fusion sampling or trajectory differs")
    if not torch.equal(generator.get_state(), payload["sampler_generator_state"]):
        raise ValueError("Fusion sampler final state differs")
    optimizer = payload["optimizer_state_dict"]
    params = [index for group in optimizer["param_groups"] for index in group["params"]]
    names = [name for name, _ in model.named_parameters()]
    if (
        len(params) != len(names)
        or len(set(params)) != len(params)
        or set(params) != set(optimizer["state"])
    ):
        raise ValueError("Fusion Adam coverage differs")
    for index, name in zip(params, names, strict=True):
        state = optimizer["state"][index]
        if float(state["step"]) != recipe["updates"] or any(
            state[key].shape != final[name].shape or not torch.isfinite(state[key]).all()
            for key in ("exp_avg", "exp_avg_sq")
        ):
            raise ValueError("Fusion Adam states differ from final update")
    if payload["scheduler_state_dict"]["last_epoch"] != recipe["updates"]:
        raise ValueError("Fusion scheduler endpoint differs")
    model.load_state_dict(final, strict=True)
    model.eval().requires_grad_(False)
    proof = {
        "status": "PASS",
        "checkpoint_sha256": endpoint["endpoints"][fit],
        "sample_ids_regenerated": recipe["updates"] * recipe["batch_size"],
        "sampler_final_state_exact": True,
        "adam_states": len(params),
        "adam_step": recipe["updates"],
        "scheduler_step": recipe["updates"],
        "changed_tensors": changed,
        "optimizer_updates_in_verification": 0,
    }
    freeze(folder / "ENDPOINT_VERIFICATION.json", proof)
    return model, proof


def publish(
    role: str, name: str, rows: pd.DataFrame, prediction: np.ndarray, binding: dict
) -> dict:
    """Retain every signed prediction; independent NumPy arithmetic checks the scorer."""
    folder = DIRECTORY / "predictions" / role / name
    freeze(folder / "BINDING.json", binding)
    frame = rows[["sample_token", "group_id", "target_ttc"]].copy()
    if prediction.shape != (len(frame),):
        raise ValueError("Fusion prediction coverage differs")
    frame["prediction_ttc"] = prediction.astype(np.float64)
    path = folder / "predictions.parquet"
    if path.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(path), frame)
    else:
        frame.to_parquet(path, index=False)
    result = metrics(
        frame.target_ttc.to_numpy(), frame.prediction_ttc.to_numpy(), frame.group_id.to_numpy()
    )
    freeze(folder / "METRICS.json", result)
    freeze(
        folder / "COMPLETE.json",
        {
            "status": "COMPLETE",
            "rows": len(frame),
            "sha256": sha(path),
            "binding": digest(binding),
        },
    )
    return {"metrics": result, "verification": verify(folder), "sha256": sha(path)}


def evaluate() -> dict:
    """Both fixed endpoints and complete V inputs precede all outcome computations."""
    torch.set_num_threads(1)
    endpoints = freeze_endpoints()
    record = read(REGISTRATION)
    for path, expected in record["sources"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("Fusion registered training source changed")
    h_blocks, h_rows, _ = inputs()
    normalizers = {}
    for name in ("rgb", "event"):
        path = DIRECTORY / "normalizers" / f"{name}.json"
        if sha(path) != record["normalizers"][name]:
            raise ValueError("Fusion H normalizer changed")
        normalizers[name] = FrozenNormalizer.load(path)
    v_record = read(DIRECTORY / "inputs/V/REGISTRATION.json")
    if v_record["endpoints"] != endpoints:
        raise ValueError("V input extraction preceded or differs from paired endpoint freeze")
    for path, expected in v_record["sources"].items():
        if sha(ROOT / path) != expected:
            raise ValueError("V extraction source changed")
    v_manifest = owner() / "V_MANIFEST.json"
    if (
        sha(v_manifest) != v_record["manifest_sha256"]
        or sha(Path(read(v_manifest)["rows_path"])) != read(v_manifest)["rows_sha256"]
    ):
        raise ValueError("V manifest or rows changed")
    v_rows = role_rows("V")
    v_blocks = {}
    for name, folder in (("rgb", "R_CTX"), ("event", "event")):
        directory = DIRECTORY / "inputs/V" / folder
        receipt, binding = read(directory / "COMPLETE.json"), read(directory / "BINDING.json")
        caller = binding["input_binding"] if name == "rgb" else binding["input_binding"]["caller"]
        if (
            receipt["status"] != "COMPLETE"
            or receipt["binding"] != digest(binding)
            or caller != {"registration_sha256": sha(DIRECTORY / "inputs/V/REGISTRATION.json")}
            or binding["identity"] != record["inputs"][name]["identity"]
        ):
            raise ValueError("V feature caller or identity mismatch")
        v_blocks[name] = load_feature_cache(
            directory / "features.npz",
            expected_sha256=receipt["sha256"],
            expected_identity=FeatureIdentity(**record["inputs"][name]["identity"]),
            role="V",
            manifest_sha256=sha(v_manifest),
            sample_tokens=v_rows.sample_token.to_numpy(dtype=str),
            feature_count=28 if name == "rgb" else 196,
        )
    sources = {
        f"operational/joint_measurement/{name}.py": sha(
            ROOT / f"operational/joint_measurement/{name}.py"
        )
        for name in ("fusion_evaluate", "bootstrap", "verify_predictions", "fusion_export")
    }
    sources.update(
        {
            name: sha(ROOT / name)
            for name in (
                "operational/range_transfer/analyze.py",
                "operational/rgb_port/evaluate.py",
            )
        }
    )
    committed_sources(sources)
    admission = read(RUN / "verification/HISTORICAL_MULTIMODAL_ADMISSION.json")
    registration = {
        "endpoints": endpoints,
        "sources": sources,
        "H_scope": "in-sample sequential supervised diagnostics, never independent selection",
        "V_scope": "exposed internal development; unchanged system gates; no recipe rescue",
        "historical_admission_sha256": sha(
            RUN / "verification/HISTORICAL_MULTIMODAL_ADMISSION.json"
        ),
    }
    freeze(DIRECTORY / "evaluation/REGISTRATION.json", registration)
    models, proofs = {}, {}
    for mode in MODES:
        models[mode], proofs[mode] = load_head(mode)
    results, timings = {}, {}
    for role, rows, blocks in (("H", h_rows, h_blocks), ("V", v_rows, v_blocks)):
        manifest_sha = sha(owner() / f"{role}_MANIFEST.json")
        bank = FusionTensorBank(
            blocks["rgb"],
            blocks["event"],
            normalizers["rgb"],
            normalizers["event"],
            role=role,
            manifest_sha256=manifest_sha,
            sample_tokens=rows.sample_token.to_numpy(dtype=str),
        )
        for mode, model in models.items():
            predictions = []
            started = time.perf_counter()
            with torch.inference_mode():
                for start in range(0, len(rows), 128):
                    batch = bank.batch(torch.arange(start, min(start + 128, len(rows))))
                    output = model(**batch)["ttc"]
                    missing = ~batch["rgb_available"]
                    if not torch.equal(output[missing], batch["event_ttc"][missing]):
                        raise ValueError("Fusion inference changed exact event fallback")
                    predictions.append(output.numpy())
            fit = f"FUSION-{mode}-S7"
            timings[f"{role}/{fit}"] = time.perf_counter() - started
            results[f"{role}/{fit}"] = publish(
                role,
                fit,
                rows,
                np.concatenate(predictions),
                {
                    "evaluation": sha(DIRECTORY / "evaluation/REGISTRATION.json"),
                    "endpoint": endpoints["endpoints"][fit],
                    "manifest": manifest_sha,
                    "scope": registration[f"{role}_scope"],
                },
            )
        # Native anchor on H is only an in-sample diagnostic; on V also admit archived output below.
        if role == "H":
            results["H/R_CTX"] = publish(
                role,
                "R_CTX",
                rows,
                blocks["rgb"].prediction_ttc,
                {
                    "parent": blocks["rgb"].identity.endpoint_sha256,
                    "manifest": manifest_sha,
                    "scope": registration["H_scope"],
                },
            )
    for name in ("R_CTX", "F_TRUE"):
        path = owner() / "predictions/V" / f"{name}.csv"
        if sha(path) != admission["heads"][name]["V_prediction_sha256"]:
            raise ValueError("Historical V prediction bytes changed")
        # pandas supports round_trip; the installed pandas-stubs omit this literal.
        frame = pd.read_csv(path, float_precision="round_trip")  # pyright: ignore[reportCallIssue, reportArgumentType]
        require_cohort(frame, v_rows)
        if name == "R_CTX":
            np.testing.assert_array_equal(
                frame.prediction_ttc.to_numpy(), v_blocks["rgb"].prediction_ttc.astype(np.float64)
            )
        results[f"V/{name}"] = publish(
            "V",
            name,
            v_rows,
            frame.prediction_ttc.to_numpy(),
            {
                "historical_admission": registration["historical_admission_sha256"],
                "source_csv_sha256": sha(path),
            },
        )
    gates, bootstraps, exports = {}, {}, {}
    for mode, model in models.items():
        fit = f"FUSION-{mode}-S7"
        for reference in ("R_CTX", "F_TRUE", "FUSION-ZERO-S7"):
            if reference == fit:
                continue
            key = f"{fit}_vs_{reference}"
            gates[key] = gate(results[f"V/{reference}"]["metrics"], results[f"V/{fit}"]["metrics"])
            base = DIRECTORY / "predictions/V"
            bootstraps[key] = compare(
                base / reference / "predictions.parquet",
                base / fit / "predictions.parquet",
                DIRECTORY / "bootstrap/V" / key,
            )
        examples = []
        for mask in ([True] * 4, [False] * 4, [True, False, True, False]):
            examples.append(
                (
                    torch.from_numpy(h_blocks["rgb"].features[:4].copy()),
                    torch.from_numpy(h_blocks["event"].features[:4].copy()),
                    torch.from_numpy(h_blocks["rgb"].prediction_ttc[:4].copy()),
                    torch.from_numpy(h_blocks["event"].prediction_ttc[:4].copy()),
                    torch.tensor(mask),
                )
            )
        exports[fit] = export_head(
            DIRECTORY / "exports" / fit,
            model,
            normalizers["rgb"],
            normalizers["event"],
            h_blocks["rgb"].identity,
            h_blocks["event"].identity,
            examples,
            binding={
                "endpoint_sha256": endpoints["endpoints"][fit],
                "registration_sha256": sha(REGISTRATION),
            },
        )
    result = {
        "status": "FUSION_METRICS_READY_FOR_INTERPRETATION",
        "utc": utc(),
        "endpoints": endpoints,
        "endpoint_verification": proofs,
        "results": results,
        "gates": gates,
        "bootstrap": bootstraps,
        "exports": exports,
        "head_only_cpu_inference_seconds": timings,
        "full_pipeline_cost_complete": False,
        "system_promotion": False,
        "official_comparison": False,
        "full_objective_complete": False,
    }
    write(DIRECTORY / "evaluation/RESULT.json", result)
    return {
        "status": result["status"],
        "gates": gates,
        "metrics": {key: value["metrics"]["overall_MiD"] for key, value in results.items()},
    }


if __name__ == "__main__":
    print(evaluate())
