"""Replay real producer batches and diagnose temporal equivariance without labels."""

from unittest.mock import patch

import numpy as np
import torch
from numpy.typing import ArrayLike

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.phase_timewarp.capture import PAIRS, PRODUCERS, capture_calls
from operational.phase_timewarp.warp import filter_retained, warp_metadata, warp_pixels
from operational.rgb_port import transfer as engine
from operational.rgb_port_streaming.contracts import NativeObservation


def replay_calls(
    models: dict,
    calls: list[dict],
    source_arrays: dict,
    query_us: int,
    speed: float,
    device: str,
    *,
    only_ids: set[str] | None = None,
) -> dict:
    """Reexecute each retained original group; a1 never changes its FP32 dt bytes."""
    transformed = {"event": [], "rgb": []}
    for call in calls:
        modality = "event" if call["prefix"] == "E" else "rgb"
        for item in call["observations"]:
            if only_ids is None or item["observation_id"] in only_ids:
                transformed[modality].append(warp_metadata(item, query_us, speed))
    discarded = {}
    for modality in transformed:
        transformed[modality], discarded[modality] = filter_retained(
            transformed[modality], query_us
        )
    selected = {item["observation_id"]: item for items in transformed.values() for item in items}
    observations = {"event": [], "rgb": []}
    with (
        torch.inference_mode(),
        patch.object(engine, "ProducerObservation", NativeObservation),
        capture_calls(models, device) as (records, arrays),
    ):
        for original in calls:
            members = [
                item for item in original["observations"] if item["observation_id"] in selected
            ]
            if not members:
                continue
            key = f"call{original['call_id']}"
            modality = "event" if original["prefix"] == "E" else "rgb"
            pixels = np.stack(
                [
                    warp_pixels(
                        source_arrays[key + ".pixels"][item["batch_index"]],
                        modality,
                        item.get("windows_us"),
                        selected[item["observation_id"]].get("windows_us"),
                    )
                    for item in members
                ]
            )
            delta = (
                np.stack([source_arrays[key + ".delta"][item["batch_index"]] for item in members])
                if speed == 1
                else np.asarray(
                    [selected[item["observation_id"]]["delta_t_s"] for item in members], np.float32
                )
            )
            try:
                raw, experts = engine._expert_block(
                    models,
                    original["prefix"],
                    torch.from_numpy(pixels).to(device),
                    torch.from_numpy(delta).to(device),
                )
            except ValueError as error:
                if str(error) != "TTC lies outside the signed benchmark-phase domain":
                    raise
                records[-1]["domain_rejection"] = True
                records[-1]["rejection_scope"] = (
                    "Entire original retained batch; no singleton substitution"
                )
            else:
                for i, item in enumerate(members):
                    observations[modality].append(
                        {
                            **selected[item["observation_id"]],
                            "raw17": raw[i].cpu().numpy().copy(),
                            "expert3": experts[i].cpu().numpy().copy(),
                        }
                    )
            records[-1]["original_call_id"] = original["call_id"]
            records[-1]["observations"] = [
                {
                    **selected[item["observation_id"]],
                    "batch_index": i,
                    "original_batch_index": item["batch_index"],
                }
                for i, item in enumerate(members)
            ]
    return {
        "calls": records,
        "arrays": arrays,
        "observations": observations,
        "discarded": discarded,
        "selected_ids": list(selected),
        "accepted_ids": [
            item["observation_id"] for items in observations.values() for item in items
        ],
    }


def stage_traces(calls: list[dict], arrays: dict) -> dict:
    """Index diagnostics by stable observation identity, never by replay call number."""
    result = {}
    for call in calls:
        key = f"call{call['call_id']}"
        for item in call["observations"]:
            index = item["batch_index"]
            values = {"delta": arrays[key + ".delta"][index]}
            for name, value in arrays.items():
                if name.startswith(key + ".") and name not in {key + ".pixels", key + ".delta"}:
                    values[name[len(key) + 1 :]] = value[index]
            result[item["observation_id"]] = values
    return result


def finite_ratio(numerator: ArrayLike, denominator: ArrayLike) -> list:
    """Undefined/nonfinite ratios remain explicit null, rather than substituted errors."""
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    result = np.full(numerator.shape, np.nan, np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
    np.divide(numerator, denominator, out=result, where=valid)
    return [float(v) if np.isfinite(v) else None for v in result.reshape(-1)]


def finite_max_difference(actual: ArrayLike, expected: ArrayLike) -> float | None:
    """Preserve nonfinite trace differences as unknown; original arrays remain saved."""
    difference = np.abs(np.asarray(actual) - np.asarray(expected))
    return float(np.max(difference)) if np.all(np.isfinite(difference)) else None


def compare_stages(warped: dict, control: dict, speed: float) -> dict:
    """Report stage-local equivariance and clipping, not predictive accuracy."""
    report = {}
    w = stage_traces(warped["calls"], warped["arrays"])
    c = stage_traces(control["calls"], control["arrays"])
    for identity, values in w.items():
        if identity not in c:
            raise ValueError("Stage comparison lacks an identical clean observation")
        result: dict = {
            "measured_dt_control_over_warp": finite_ratio(c[identity]["delta"], values["delta"])
        }
        for name in PRODUCERS:
            if (
                name + ".inverse_ttc_mean" not in values
                or name + ".inverse_ttc_mean" not in c[identity]
            ):
                continue
            reference = c[identity]
            result[name] = {
                "inverse_ratio": finite_ratio(
                    values[name + ".inverse_ttc_mean"],
                    speed * reference[name + ".inverse_ttc_mean"],
                ),
                "ttc_ratio": finite_ratio(
                    speed * values[name + ".ttc_mean_seconds"],
                    reference[name + ".ttc_mean_seconds"],
                ),
                "pair_inverse_ratio": finite_ratio(
                    values[name + ".pair_inverse_ttc"],
                    speed * reference[name + ".pair_inverse_ttc"],
                ),
                "pair_inverse_ratio_measured_dt": finite_ratio(
                    values[name + ".pair_inverse_ttc"] * values["delta"],
                    reference[name + ".pair_inverse_ttc"] * reference["delta"],
                ),
                "pair_log_ratio_max_abs_difference": finite_max_difference(
                    values[name + ".pair_log_height_ratio"],
                    reference[name + ".pair_log_height_ratio"],
                ),
                "undefined_diagnostic_policy": "Nonfinite values are null; original traces in NPZ",
                "known": np.asarray(values[name + ".known_mask"]).tolist(),
                "sensor_support": np.asarray(values[name + ".sensor_support"]).tolist(),
                "ttc_at_cap60": bool(np.abs(values[name + ".ttc_mean_seconds"]) >= 60),
            }
        for name in PAIRS:
            if name + ".raw_phase" in values and "expert3" in values and "expert3" in c[identity]:
                phase = torch.tensor([values["expert3"][2]], dtype=torch.float32)
                original = torch.tensor([c[identity]["expert3"][2]], dtype=torch.float32)
                result[name] = {
                    "raw_phase": finite_ratio(values[name + ".raw_phase"], 1)[0],
                    "emitted_phase": finite_ratio(phase[0].numpy(), 1)[0],
                    "ttc_ratio": finite_ratio(
                        speed * phase_to_ttc(phase).numpy(), phase_to_ttc(original).numpy()
                    ),
                }
        report[identity] = result
    return report
