"""Prospective, zero-update gradient diagnostic on the historical A1-FULL endpoint.

This module uses TTC targets only for a descriptive P-panel diagnostic.  It is
not an inference path, a fit, a candidate export, or independent validation.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from . import common, training
from .losses import MASSES, training_mass

if TYPE_CHECKING:
    from torch import Tensor, nn

SEED = 7
MAX_PER_CELL = 4
MICROBATCH = 4
BANDS = ("crucial", "small", "long", "negative")
DIRECTORY = common.RUN / "diagnostics_gradients"
REGISTRATION = DIRECTORY / "REGISTRATION.json"


def _band(target: float) -> str:
    if 0.1 < target <= 3:
        return "crucial"
    if 3 < target <= 6:
        return "small"
    if 6 < target <= 10:
        return "long"
    if -10 < target < 0:
        return "negative"
    raise ValueError(f"Target outside the registered TTC bands: {target}")


def _rank(token: str, seed: int = SEED) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}|{token}".encode()).hexdigest()
    return digest, token


def select_panel(rows: pd.DataFrame, *, limit: int = MAX_PER_CELL) -> list[dict[str, Any]]:
    """Select a stable uniform-without-replacement hash panel per group/band cell."""
    required = {"sample_token", "group_id", "target_ttc"}
    if not required.issubset(rows.columns) or limit <= 0:
        raise ValueError("Panel requires token, group, target and a positive cell limit")
    frame = rows.reset_index(drop=True)
    if frame.sample_token.astype(str).duplicated().any():
        raise ValueError("P sample tokens must be unique")
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for index, row in frame.iterrows():
        target = float(row.target_ttc)
        token, group = str(row.sample_token), str(row.group_id)
        band = _band(target)
        candidates.setdefault((group, band), []).append(
            {
                "index": int(index),
                "sample_token": token,
                "group_id": group,
                "band": band,
                "target_ttc": target,
            }
        )
    panel: list[dict[str, Any]] = []
    for (_group, _band_name), values in sorted(candidates.items()):
        values.sort(key=lambda item: _rank(item["sample_token"]))
        chosen = values[:limit]
        population = len(values)
        for item in chosen:
            panel.append({**item, "cell_population": population, "cell_sample": len(chosen)})
    if not panel or len(panel) > len(candidates) * limit:
        raise ValueError("Invalid fixed panel")
    return panel


def parameter_blocks(names: Iterable[str]) -> dict[str, str]:
    """Map every A1 geometry parameter to the registered scientific block."""
    result: dict[str, str] = {}
    for name in names:
        if name.startswith("encoder."):
            block = "encoder"
        elif name.startswith("spatial."):
            block = "spatial_tower"
        elif name.startswith(("gates.", "spatial_weights.")):
            block = "head"
        else:
            raise ValueError(f"Unregistered GeometricPairMeasurement parameter: {name}")
        result[name] = block
    return result


def parameter_block_indices(names: Sequence[str]) -> dict[str, list[int]]:
    """Return an exhaustive, nonempty index partition for the three scientific blocks."""
    blocks = parameter_blocks(names)
    result = {
        block: [index for index, name in enumerate(names) if blocks[name] == block]
        for block in ("encoder", "spatial_tower", "head")
    }
    if any(not indices for indices in result.values()):
        raise ValueError("Encoder, spatial tower and head must all contain trainable parameters")
    if sorted(index for indices in result.values() for index in indices) != list(range(len(names))):
        raise ValueError("Parameter block partition is not exhaustive and unique")
    return result


def cosine(left: Sequence[Tensor], right: Sequence[Tensor], indices: Sequence[int]) -> float | None:
    """Return a block cosine, or None when either block gradient is exactly zero."""
    dot = sum(float((left[i].double() * right[i].double()).sum()) for i in indices)
    aa = sum(float(left[i].double().square().sum()) for i in indices)
    bb = sum(float(right[i].double().square().sum()) for i in indices)
    if aa == 0 or bb == 0:
        return None
    value = dot / math.sqrt(aa * bb)
    return max(-1.0, min(1.0, value))


def vector_summary(vectors: Sequence[Tensor], indices: Sequence[int]) -> dict[str, Any]:
    squares = [float(vectors[i].double().square().sum()) for i in indices]
    finite = all(bool(vectors[i].isfinite().all()) for i in indices)
    return {
        "l2_norm": math.sqrt(sum(squares)),
        "has_signal": sum(squares) > 0,
        "finite": finite,
        "parameter_tensors": len(indices),
        "zero_parameter_tensors": sum(s == 0 for s in squares),
    }


def clip_vectors(
    vectors: Sequence[Tensor], maximum: float = 1.0
) -> tuple[list[Tensor], float, float]:
    """Apply the registered global L2 clipping algebra without touching ``.grad``."""
    norm = math.sqrt(sum(float(value.double().square().sum()) for value in vectors))
    scale = min(1.0, maximum / (norm + 1e-6))
    return [value * scale for value in vectors], norm, scale


def adamw_delta(
    parameter: Tensor,
    gradient: Tensor,
    state: Mapping[str, object],
    group: Mapping[str, object],
) -> Tensor:
    """Analytically compute one PyTorch AdamW delta without mutating any input."""
    required = {"exp_avg", "exp_avg_sq", "step"}
    if not required.issubset(state) or group.get("amsgrad", False) or group.get("maximize", False):
        raise ValueError("Only complete standard non-AMSGrad AdamW state is admitted")
    beta1, beta2 = group["betas"]
    step_value = state["step"]
    step = int(step_value.item() if hasattr(step_value, "item") else step_value) + 1
    exp_avg = state["exp_avg"] * beta1 + gradient * (1 - beta1)
    exp_avg_sq = state["exp_avg_sq"] * beta2 + gradient.square() * (1 - beta2)
    lr = float(group["lr"])
    eps = float(group["eps"])
    decay = float(group["weight_decay"])
    denom = exp_avg_sq.sqrt() / math.sqrt(1 - beta2**step) + eps
    adaptive = -(lr / (1 - beta1**step)) * exp_avg / denom
    return adaptive - lr * decay * parameter


def ht_band_masses(
    panel: Sequence[Mapping[str, object]], masses: Sequence[float]
) -> dict[str, float]:
    """Return the exact panel Horvitz-Thompson estimate of each registered band mass."""
    result = dict.fromkeys(BANDS, 0.0)
    for item in panel:
        band = str(item["band"])
        if band not in result:
            raise ValueError(f"Unregistered panel band: {band}")
        index = int(item["index"])
        population = int(item["cell_population"])
        sample = int(item["cell_sample"])
        if not 0 <= index < len(masses) or population < sample or sample <= 0:
            raise ValueError("Invalid Horvitz-Thompson panel metadata")
        result[band] += float(masses[index]) * population / sample
    return result


def validate_adamw_state(
    parameter: Tensor, state: Mapping[str, object], *, expected_step: int
) -> None:
    """Fail closed on an incomplete, mis-shaped, nonfinite or wrong-step AdamW state."""
    required = {"exp_avg", "exp_avg_sq", "step"}
    if set(state) != required:
        raise ValueError("AdamW state keys are not exactly exp_avg/exp_avg_sq/step")
    first, second = state["exp_avg"], state["exp_avg_sq"]
    if not hasattr(first, "shape") or not hasattr(second, "shape"):
        raise ValueError("AdamW moments must be tensors")
    if first.shape != parameter.shape or second.shape != parameter.shape:
        raise ValueError("AdamW moment shape differs from its parameter")
    if not bool(first.isfinite().all()) or not bool(second.isfinite().all()):
        raise ValueError("AdamW moments must be finite")
    if bool((second < 0).any()):
        raise ValueError("AdamW second moment must be nonnegative")
    step_value = state["step"]
    step = int(step_value.item() if hasattr(step_value, "item") else step_value)
    if step != expected_step:
        raise ValueError(f"AdamW state step {step} differs from {expected_step}")


def validate_paired_rng_restores(restores: Sequence[bool]) -> None:
    """Require one successful identical CPU/CUDA RNG restoration per registered band."""
    if list(restores) != [True] * len(BANDS):
        raise ValueError("Paired CPU/CUDA RNG restoration failed")


def _sources() -> dict[str, str]:
    paths = [
        Path(__file__),
        common.ROOT / "tests/unit/test_decision_recovery_gradients.py",
        common.ROOT / "operational/decision_recovery/common.py",
        common.ROOT / "operational/decision_recovery/training.py",
        common.ROOT / "operational/decision_recovery/losses.py",
        common.ROOT / "operational/decision_recovery/sensor_runtime.py",
        common.ROOT / "operational/joint_measurement/geometry_model.py",
        common.ROOT / "operational/joint_measurement/phase_transport_loss.py",
        common.ROOT / "operational/joint_measurement/adaptation_depth.py",
    ]
    return {path.relative_to(common.ROOT).as_posix(): common.sha(path) for path in paths}


def _checkpoint() -> Path:
    return training.source_run() / "adaptive/A1_adaptation_depth/fits/A1-FULL-S7/checkpoint_last.pt"


def _receipt() -> Path:
    return _checkpoint().with_name("CHECKPOINT_RECEIPT.json")


def _canonical_sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _freeze(path: Path, value: dict[str, Any]) -> dict[str, Any]:
    if path.exists():
        if common.read_json(path) != value:
            raise ValueError(f"Frozen gradient evidence changed: {path.name}")
    else:
        common.write_json(path, value)
    return value


def prepare_registration() -> dict[str, Any]:
    """Freeze the P panel and signed A1 endpoint before diagnostic GPU work."""
    if REGISTRATION.exists():
        return registered()
    training_record = training.registered()
    sources = _sources()
    commit = training.committed_sources(sources)
    receipt = common.read_json(_receipt())
    endpoints = training.source_run() / "adaptive/A1_adaptation_depth/ENDPOINTS.json"
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("fit_id") != "A1-FULL-S7"
        or receipt.get("completed_updates") != 12000
        or receipt.get("complete_state") is not True
        or common.sha(_checkpoint()) != receipt.get("checkpoint_sha256")
        or common.read_json(endpoints).get("A1-FULL-S7") != receipt.get("checkpoint_sha256")
    ):
        raise ValueError("Signed complete historical A1-FULL endpoint required")
    rows = training.role_rows("P")
    if len(rows) != 53890:
        raise ValueError("Gradient diagnostic requires the complete P population")
    panel = select_panel(rows)
    groups = sorted(rows.group_id.astype(str).unique().tolist())
    cells = {(item["group_id"], item["band"]) for item in panel}
    expected_cells = {(group, band) for group in groups for band in BANDS}
    if (
        len(groups) != 24
        or cells != expected_cells
        or len(panel) != 384
        or any(item["cell_sample"] != MAX_PER_CELL for item in panel)
    ):
        raise ValueError("Expected four queries from all four bands in each of 24 P groups")
    masses = training_mass(rows.target_ttc.to_numpy(), rows.group_id.to_numpy(), "L0")
    estimated_band_masses = ht_band_masses(panel, masses)
    expected_band_masses = dict(zip(BANDS, MASSES["L0"], strict=True))
    if not math.isclose(float(masses.sum()), 1.0, rel_tol=0, abs_tol=1e-12) or any(
        not math.isclose(estimated_band_masses[band], expected, rel_tol=0, abs_tol=1e-12)
        for band, expected in expected_band_masses.items()
    ):
        raise ValueError("Panel HT masses do not recover the original L0 band masses")
    record = {
        "schema": "v19_a1_full_gradient_diagnostic_v1",
        "status": "REGISTERED_BEFORE_GPU_DIAGNOSTIC",
        "source_commit": commit,
        "sources": sources,
        "main_registration_sha256": common.sha(training.destination("REGISTRATION.json")),
        "training_registration_sha256": common.sha(
            training.destination("training/REGISTRATION.json")
        ),
        "training_registration_schema": training_record["schema"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "checkpoint_receipt_sha256": common.sha(_receipt()),
        "endpoints_sha256": common.sha(endpoints),
        "P_manifest_sha256": common.sha(training.manifest_path("P")),
        "P_population": len(rows),
        "P_groups": groups,
        "panel": panel,
        "panel_sha256": _canonical_sha(panel),
        "panel_policy": {
            "seed": SEED,
            "ranking": "SHA256(seed|sample_token), then sample_token",
            "sampling": "uniform without replacement within every observed P group/band cell",
            "maximum_per_cell": MAX_PER_CELL,
            "maximum_rows": 384,
        },
        "L0_mass_sha256": _canonical_sha(masses.tolist()),
        "L0_expected_band_masses": expected_band_masses,
        "L0_HT_panel_band_masses": estimated_band_masses,
        "gradient_estimators": {
            "unweighted": "equal weight over selected rows in each band",
            "L0_HT": "query L0 mass * cell_population/cell_sample; sums estimate global objective",
            "band_normalized": "L0_HT gradient divided by its estimated band mass",
        },
        "objective": "emitted plus raw; phase_terms raw already contains coefficient 0.1",
        "mode": "train with fixed seed7 RNG; dropout remains active as in adaptation",
        "rng_policy": (
            "snapshot CPU and every CUDA RNG after A1-FULL materialization; restore the same "
            "states before each 96-query band so dropout streams are position-paired"
        ),
        "optimizer_probe": (
            "global L0_HT panel gradient, norm-clipped at 1.0, hypothetical AdamW step+1 "
            "using stored LR/moments and decoupled weight decay; no scheduler or optimizer step"
        ),
        "interpretation": (
            "single descriptive P panel and one dropout realization; targets are diagnostic-only; "
            "not independent generalization or a deployable inference feature"
        ),
        "gpu_reservation_seconds": 900,
        "optimizer_updates": 0,
        "H_rows_opened": 0,
        "candidate_export": False,
    }
    _freeze(DIRECTORY / "PANEL.json", {"panel": panel, "sha256": record["panel_sha256"]})
    return _freeze(REGISTRATION, record)


def registered() -> dict[str, Any]:
    record = common.read_json(REGISTRATION)
    training_record = training.registered()
    endpoints = training.source_run() / "adaptive/A1_adaptation_depth/ENDPOINTS.json"
    if (
        record["sources"] != _sources()
        or common.sha(training.destination("REGISTRATION.json"))
        != record["main_registration_sha256"]
        or common.sha(training.destination("training/REGISTRATION.json"))
        != record["training_registration_sha256"]
        or training_record["schema"] != record["training_registration_schema"]
        or common.sha(_checkpoint()) != record["checkpoint_sha256"]
        or common.sha(_receipt()) != record["checkpoint_receipt_sha256"]
        or common.sha(endpoints) != record["endpoints_sha256"]
        or common.sha(training.manifest_path("P")) != record["P_manifest_sha256"]
        or _canonical_sha(record["panel"]) != record["panel_sha256"]
        or common.read_json(DIRECTORY / "PANEL.json")
        != {"panel": record["panel"], "sha256": record["panel_sha256"]}
    ):
        raise ValueError("Gradient diagnostic registration or inputs changed")
    return record


def _model() -> tuple[nn.Module, dict[str, object]]:
    import torch

    from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
    from operational.joint_measurement.adaptation_depth import configure
    from operational.joint_measurement.geometry_model import GeometricPairMeasurement
    from operational.joint_measurement.train import native_config

    model = GeometricPairMeasurement(CausalScaleTTC(native_config()[0]))
    payload = torch.load(_checkpoint(), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    configure(model, "FULL")
    return model, payload


def _state_sha(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _dot(left: Sequence[Tensor], right: Sequence[Tensor], indices: Sequence[int]) -> float:
    return sum(float((left[i].double() * right[i].double()).sum()) for i in indices)


def _admit_existing_result(
    result: dict[str, Any], record: dict[str, Any], output: Path
) -> dict[str, Any]:
    """Fail closed when replaying an already materialized zero-update result."""
    complete_path = output.with_name("COMPLETE.json")
    if not complete_path.is_file():
        raise ValueError("Gradient result has no COMPLETE receipt")
    complete = common.read_json(complete_path)
    if (
        result.get("status") != "COMPLETE_ZERO_UPDATE_DIAGNOSTIC"
        or result.get("registration_sha256") != common.sha(REGISTRATION)
        or result.get("checkpoint_sha256") != record["checkpoint_sha256"]
        or result.get("optimizer_updates") != 0
        or complete.get("status") != result["status"]
        or complete.get("optimizer_updates") != 0
        or complete.get("result_sha256") != common.sha(output)
    ):
        raise ValueError("Existing gradient result or COMPLETE receipt changed")
    return result


def run() -> dict[str, Any]:
    """Execute the registered zero-update diagnostic under the shared GPU locks."""
    import torch

    from operational.joint_measurement.phase_transport_loss import phase_terms
    from operational.joint_measurement.train import make_optimizer, setup

    from .sensor_runtime import make_evaluation_sensors

    record = registered()
    output = DIRECTORY / "RESULT.json"
    if output.exists():
        result = common.read_json(output)
        return _admit_existing_result(result, record, output)
    rows = training.role_rows("P")
    masses = training_mass(rows.target_ttc.to_numpy(), rows.group_id.to_numpy(), "L0")
    if _canonical_sha(masses.tolist()) != record["L0_mass_sha256"]:
        raise ValueError("Registered L0 masses changed")
    panel_by_band = {band: [] for band in BANDS}
    for item in record["panel"]:
        panel_by_band[item["band"]].append(item)
    observed_band_masses = ht_band_masses(record["panel"], masses)
    if (
        any(len(values) != 96 for values in panel_by_band.values())
        or observed_band_masses != record["L0_HT_panel_band_masses"]
        or any(
            not math.isclose(
                observed_band_masses[band],
                record["L0_expected_band_masses"][band],
                rel_tol=0,
                abs_tol=1e-12,
            )
            for band in BANDS
        )
    ):
        raise ValueError("Registered panel support or original L0 HT masses changed")

    with (
        training.file_lock(training.destination("training/DRIVER.lock")),
        training.gpu_charge("V19_A1_FULL_GRADIENT_DIAGNOSTIC", reserve=900) as deadline,
        training.runtime_context(),
    ):
        setup(SEED)
        model, checkpoint = _model()
        model = model.cuda().train()
        dropout_modules = sum(
            module.__class__.__name__.startswith("Dropout") for module in model.modules()
        )
        if dropout_modules == 0:
            raise ValueError("Registered train-mode diagnostic expected active dropout modules")
        before = _state_sha(model)
        paired_cpu_rng = torch.get_rng_state().clone()
        paired_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
        rng_restores = []
        named = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        names, parameters = [item[0] for item in named], [item[1] for item in named]
        block_indices = parameter_block_indices(names)
        vectors: dict[str, dict[str, dict[str, list[Any]]]] = {}
        values: dict[str, dict[str, dict[str, float]]] = {}
        sensors = make_evaluation_sensors("P")
        try:
            for band in BANDS:
                torch.set_rng_state(paired_cpu_rng)
                torch.cuda.set_rng_state_all(paired_cuda_rng)
                rng_restores.append(
                    torch.equal(torch.get_rng_state(), paired_cpu_rng)
                    and all(
                        torch.equal(left, right)
                        for left, right in zip(
                            torch.cuda.get_rng_state_all(), paired_cuda_rng, strict=True
                        )
                    )
                )
                selected = panel_by_band[band]
                count = len(selected)
                accum = {
                    estimator: {
                        term: [torch.zeros_like(p, device="cuda") for p in parameters]
                        for term in ("emitted", "raw")
                    }
                    for estimator in ("unweighted", "L0_HT")
                }
                scalar = {key: {term: 0.0 for term in ("emitted", "raw")} for key in accum}
                estimated_mass = 0.0
                for start in range(0, count, MICROBATCH):
                    if time.perf_counter() > deadline:
                        raise InterruptedError("Gradient diagnostic exceeded its GPU reservation")
                    chunk = selected[start : start + MICROBATCH]
                    indices = [item["index"] for item in chunk]
                    batch = sensors.batch(indices).to(torch.device("cuda"))
                    raw = model(batch.events, batch.delta_t_s.reshape(len(chunk)))["raw_phase"]
                    terms, valid = phase_terms(raw, batch.target_ttc_s)
                    if not bool(valid):
                        raise FloatingPointError("Panel phase domain guard failed")
                    unweighted = torch.full_like(raw, 1 / count)
                    ht_values = [
                        masses[item["index"]] * item["cell_population"] / item["cell_sample"]
                        for item in chunk
                    ]
                    ht = torch.tensor(ht_values, dtype=raw.dtype, device=raw.device)
                    estimated_mass += sum(ht_values)
                    requests = [
                        ("unweighted", "emitted", (terms["emitted"] * unweighted).sum()),
                        ("unweighted", "raw", (terms["raw"] * unweighted).sum()),
                        ("L0_HT", "emitted", (terms["emitted"] * ht).sum()),
                        ("L0_HT", "raw", (terms["raw"] * ht).sum()),
                    ]
                    for request_index, (estimator, term, loss) in enumerate(requests):
                        gradients = torch.autograd.grad(
                            loss,
                            parameters,
                            retain_graph=request_index + 1 < len(requests),
                            allow_unused=True,
                        )
                        scalar[estimator][term] += float(loss.detach())
                        for index, gradient in enumerate(gradients):
                            if gradient is not None:
                                accum[estimator][term][index].add_(gradient.detach())
                vectors[band] = {}
                values[band] = {}
                for estimator in accum:
                    term_vectors = {
                        term: [v.cpu() for v in accum[estimator][term]] for term in accum[estimator]
                    }
                    term_vectors["combined"] = [
                        left + right
                        for left, right in zip(
                            term_vectors["emitted"], term_vectors["raw"], strict=True
                        )
                    ]
                    vectors[band][estimator] = term_vectors
                    values[band][estimator] = {
                        **scalar[estimator],
                        "combined": scalar[estimator]["emitted"] + scalar[estimator]["raw"],
                    }
                vectors[band]["band_normalized"] = {
                    term: [value / estimated_mass for value in vectors[band]["L0_HT"][term]]
                    for term in ("emitted", "raw", "combined")
                }
                values[band]["band_normalized"] = {
                    term: values[band]["L0_HT"][term] / estimated_mass
                    for term in ("emitted", "raw", "combined")
                }
                values[band]["estimated_L0_mass"] = {"value": estimated_mass}
        finally:
            sensors.shutdown()

        summaries: dict[str, Any] = {}
        for band in BANDS:
            summaries[band] = {
                "rows": len(panel_by_band[band]),
                "loss_values": values[band],
                "gradients": {
                    estimator: {
                        term: {
                            block: vector_summary(vectors[band][estimator][term], indices)
                            for block, indices in block_indices.items()
                        }
                        for term in ("emitted", "raw", "combined")
                    }
                    for estimator in ("unweighted", "L0_HT", "band_normalized")
                },
            }
        cosines: dict[str, Any] = {}
        for estimator in ("unweighted", "L0_HT", "band_normalized"):
            cosines[estimator] = {}
            for left_index, left in enumerate(BANDS):
                for right in BANDS[left_index + 1 :]:
                    cosines[estimator][f"{left}__{right}"] = {
                        term: {
                            block: cosine(
                                vectors[left][estimator][term],
                                vectors[right][estimator][term],
                                indices,
                            )
                            for block, indices in block_indices.items()
                        }
                        for term in ("emitted", "raw", "combined")
                    }

        overall = [
            sum(
                (vectors[band]["L0_HT"]["combined"][i] for band in BANDS),
                torch.zeros_like(parameters[i], device="cpu"),
            )
            for i in range(len(parameters))
        ]
        overall_summary = {
            block: vector_summary(overall, indices) for block, indices in block_indices.items()
        }
        if any(item["finite"] is not True for item in overall_summary.values()):
            raise FloatingPointError("Global panel gradient is nonfinite in a model block")
        clipped, preclip_norm, clip_scale = clip_vectors(overall)
        optimizer, scheduler = make_optimizer(model, updates=12000)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        optimizer_parameters = [
            parameter for group in optimizer.param_groups for parameter in group["params"]
        ]
        optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
        expected_ids = [id(parameter) for parameter in parameters]
        if len(set(optimizer_ids)) != len(optimizer_ids) or optimizer_ids != expected_ids:
            raise ValueError("AdamW parameter mapping is not exhaustive, unique and ordered")
        if len(optimizer.param_groups) != 2:
            raise ValueError("Expected ordered encoder/readout AdamW groups")
        first_group_names = names[: len(optimizer.param_groups[0]["params"])]
        second_group_names = names[len(first_group_names) :]
        if (
            not first_group_names
            or not second_group_names
            or not all(name.startswith("encoder.") for name in first_group_names)
            or any(name.startswith("encoder.") for name in second_group_names)
        ):
            raise ValueError("AdamW groups are not ordered encoder then readout")
        recipe = training.recipe()["optimizer"]
        expected_initial_lrs = [recipe["backbone_lr"], recipe["readout_lr"]]
        for group, expected_lr in zip(optimizer.param_groups, expected_initial_lrs, strict=True):
            if (
                tuple(group["betas"]) != tuple(recipe["betas"])
                or float(group["eps"]) != float(recipe["eps"])
                or float(group["weight_decay"]) != float(recipe["weight_decay"])
                or float(group["initial_lr"]) != float(expected_lr)
                or not 0 < float(group["lr"]) <= float(expected_lr)
                or group.get("foreach") is not False
                or group.get("amsgrad") is not False
                or group.get("maximize") is not False
            ):
                raise ValueError("Stored AdamW group recipe or LR is invalid")
        group_by_id = {
            id(parameter): group
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        deltas = []
        steps = set()
        lrs = set()
        for parameter, gradient in zip(parameters, clipped, strict=True):
            state = optimizer.state.get(parameter)
            if state is None:
                raise ValueError("AdamW state is missing a registered parameter")
            validate_adamw_state(parameter, state, expected_step=12000)
            group = group_by_id[id(parameter)]
            steps.add(12000)
            lrs.add(float(group["lr"]))
            deltas.append(
                adamw_delta(
                    parameter.detach().cpu(),
                    gradient,
                    {
                        key: value.detach().cpu() if hasattr(value, "detach") else value
                        for key, value in state.items()
                    },
                    group,
                )
            )
        if set(optimizer.state) != set(parameters):
            raise ValueError("AdamW contains state outside the registered parameter set")
        optimizer_probe = {
            "status": "AVAILABLE_VALIDATED",
            "stored_steps": sorted(steps),
            "stored_lrs": sorted(lrs),
            "mapping": "exhaustive_unique_model_order; encoder_then_readout",
            "preclip_global_norm": preclip_norm,
            "clip_scale": clip_scale,
            "products": {
                band: {
                    **{
                        block: _dot(vectors[band]["L0_HT"]["combined"], deltas, indices)
                        for block, indices in block_indices.items()
                    },
                    "all": _dot(
                        vectors[band]["L0_HT"]["combined"],
                        deltas,
                        list(range(len(parameters))),
                    ),
                }
                for band in BANDS
            },
            "interpretation": "negative g_band dot delta is locally favorable only",
        }

        after = _state_sha(model)
        if before != after or any(parameter.grad is not None for parameter in parameters):
            raise ValueError("Diagnostic mutated model state or populated parameter .grad")
        validate_paired_rng_restores(rng_restores)
        if any(
            not entry["finite"]
            for band in summaries.values()
            for estimator in band["gradients"].values()
            for term in estimator.values()
            for entry in term.values()
        ):
            raise FloatingPointError("Nonfinite panel gradients")
        result = {
            "schema": "v19_a1_full_gradient_diagnostic_result_v1",
            "status": "COMPLETE_ZERO_UPDATE_DIAGNOSTIC",
            "registration_sha256": common.sha(REGISTRATION),
            "checkpoint_sha256": record["checkpoint_sha256"],
            "model_state_before_sha256": before,
            "model_state_after_sha256": after,
            "parameter_blocks": {
                block: [names[i] for i in indices] for block, indices in block_indices.items()
            },
            "bands": summaries,
            "L0_HT_panel_band_masses": observed_band_masses,
            "global_L0_HT_gradient": overall_summary,
            "cosines": cosines,
            "adamw_local_probe": optimizer_probe,
            "mode": record["mode"],
            "rng_policy": record["rng_policy"],
            "paired_rng_restores": rng_restores,
            "active_dropout_modules": dropout_modules,
            "interpretation": record["interpretation"],
            "optimizer_updates": 0,
            "H_rows_opened": 0,
            "candidate_export": False,
        }
        common.write_json(output, result)
        common.write_json(
            DIRECTORY / "COMPLETE.json",
            {
                "status": result["status"],
                "result_sha256": common.sha(output),
                "optimizer_updates": 0,
            },
        )
        del model
        del optimizer
        torch.cuda.empty_cache()
        return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    arguments = parser.parse_args()
    print(json.dumps({"register": prepare_registration, "run": run}[arguments.action]()))
