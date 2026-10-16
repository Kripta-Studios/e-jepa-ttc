"""Localize strict raw/cohort parity failures without changing any tolerance or weight."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.range_transfer.common import freeze, read, sha
from operational.rgb_port.cost_routes import _normalizers
from operational.rgb_port_streaming.contracts import NativeObservation

from .common import ROOT, RUN, gpu_charge, owner
from .data import SensorDataset
from .fusion_cost_native import admit, native_context, qa
from .fusion_cost_native_v2 import NativeRuntime, native_median_policy
from .fusion_history_extract import extract_history
from .fusion_pair_extract import extract_pair
from .geometry_evaluate import load_model
from .geometry_graph_probe import no_live_training_driver
from .train import committed_sources, setup

OUT = RUN / "fusion/cost/parity_diagnostic_v1"
FAILED = RUN / "fusion/cost/raw_e2e_native_v2"
KEYS = ("features", "timing", "valid", "expert_phase")


def difference(left: np.ndarray, right: np.ndarray) -> dict:
    """Describe all numerical differences; never convert a mismatch to an admission."""
    if left.shape != right.shape:
        return {
            "shape_equal": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
    return {
        "shape_equal": True,
        "exact": bool(np.array_equal(left, right)),
        "maximum_absolute_difference": float(delta.max()),
        "mean_absolute_difference": float(delta.mean()),
        "finite": bool(np.isfinite(left).all() and np.isfinite(right).all()),
    }


def arrays(path: Path, keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Read only selected numeric feature fields, with no TTC target column."""
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in keys}


def registration() -> dict:
    """Use the first four queries in the original fixed cost order, including the failure."""
    original = read(FAILED / "REGISTRATION.json")
    source = "operational/joint_measurement/fusion_cost_diagnostic.py"
    sources = {source: sha(ROOT / source)}
    committed_sources(sources)
    paths = {
        "R_HISTORY": owner() / "features/V/R_HISTORY.npz",
        "E_HISTORY": owner() / "features/V/E_HISTORY.npz",
        "R_CTX": RUN / "fusion/inputs/V/R_CTX/features.npz",
        "event": RUN / "fusion/inputs/V/event/features.npz",
    }
    history = read(RUN / "fusion/inputs/V/REGISTRATION.json")
    if sha(paths["R_HISTORY"]) != history["history_sha256"]:
        raise ValueError("Admitted RGB history changed")
    for key in ("R_CTX", "event"):
        if sha(paths[key]) != read(paths[key].parent / "COMPLETE.json")["sha256"]:
            raise ValueError("Admitted fusion feature bytes changed")
    e_receipt = read(paths["E_HISTORY"].with_suffix(".json"))
    if e_receipt["output_sha256"] != sha(paths["E_HISTORY"]):
        raise ValueError("Historical event history changed")
    record = {
        "indices": original["indices"][:4],
        "selection": "First four queries in already fixed raw-cost order; no new target selection",
        "failed_parity_sha256": sha(FAILED / "PARITY_FAILURE.json"),
        "cost_registration_sha256": sha(FAILED / "REGISTRATION.json"),
        "sources": sources,
        "arrays": {key: {"path": str(path), "sha256": sha(path)} for key, path in paths.items()},
        "R_CTX_checkpoint_sha256": sha(owner() / "fits/R_CTX/checkpoint_last.pt"),
        "rgb_original_batch": 128,
        "event_original_batch": 8,
        "optimizer_updates": 0,
        "tolerance_changes": False,
        "decision": "Diagnostic only; no automatic cost admission or system promotion",
    }
    freeze(OUT / "REGISTRATION.json", record)
    return record


def load_arrays(record: dict) -> dict:
    """Keep source bytes bound across CPU and later exclusive-GPU diagnostics."""
    result = {}
    for key, item in record["arrays"].items():
        path = Path(item["path"])
        if sha(path) != item["sha256"]:
            raise ValueError("Diagnostic input changed")
        result[key] = arrays(
            path, KEYS if key.endswith("HISTORY") else ("features", "prediction_ttc")
        )
    return result


def anchor() -> TemporalRefiner:
    """Use the actual frozen RGB anchor, with no head fitting or calibration."""
    model = TemporalRefiner(TemporalConfig(hidden=160))
    payload = torch.load(
        owner() / "fits/R_CTX/checkpoint_last.pt", map_location="cpu", weights_only=False
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval().requires_grad_(False)


def cpu() -> dict:
    """Isolate RGB anchor batch shape while keeping the historical input cache fixed."""
    record = registration()
    torch.set_num_threads(1)
    values = load_arrays(record)
    model = anchor()
    history = values["R_HISTORY"]
    rows = []
    with torch.inference_mode():
        for index in record["indices"]:
            start = index // 128 * 128
            stop = min(start + 128, len(history["features"]))
            single = [torch.from_numpy(history[key][index : index + 1]) for key in KEYS]
            batch = [torch.from_numpy(history[key][start:stop]) for key in KEYS]
            a, at = extract_history(model, *single)
            b, bt = extract_history(model, *batch)
            offset = index - start
            stored = values["R_CTX"]
            rows.append(
                {
                    "index": index,
                    "batch1_vs_stored_features": difference(
                        a, stored["features"][index : index + 1]
                    ),
                    "batch128_vs_stored_features": difference(
                        b[offset : offset + 1], stored["features"][index : index + 1]
                    ),
                    "batch1_vs_stored_ttc": difference(
                        at, stored["prediction_ttc"][index : index + 1]
                    ),
                    "batch128_vs_stored_ttc": difference(
                        bt[offset : offset + 1], stored["prediction_ttc"][index : index + 1]
                    ),
                }
            )
    result = {
        "status": "COMPLETE_DIAGNOSTIC",
        "registration_sha256": sha(OUT / "REGISTRATION.json"),
        "rows": rows,
        "cuda_initialized": torch.cuda.is_initialized(),
        "optimizer_updates": 0,
    }
    freeze(OUT / "CPU_RESULT.json", result)
    return result


def gpu() -> dict:
    """Compare pixels, clocks, features and head execution with one GPU owner."""
    no_live_training_driver()
    record = registration()
    values = load_arrays(record)
    native = admit()
    qa(native)
    setup(7)
    source = SensorDataset("V")
    rows = []
    provenance = read(owner() / "fits/E_A5_MATCHED/RUN_PROVENANCE.json")
    try:
        with native_context(native), gpu_charge("fusion_raw_parity_diagnostic", reserve=900):
            runtime = NativeRuntime(
                run=owner(),
                config=ROOT / "configs/rgb_port/producers.json",
                p_manifest=owner() / "P_MANIFEST.json",
                manifest=owner() / "V_MANIFEST.json",
                event_source_root=Path(provenance["source_identity"]["event_source_root"]),
                event_history=owner() / "features/V/E_RAW/history.npz",
                normalizers=_normalizers(FAILED / "NORMALIZERS.json"),
                device="cuda",
            )
            try:
                model = load_model().cuda().eval().requires_grad_(False)
                rgb_anchor = anchor()
                with (
                    torch.inference_mode(),
                    patch(
                        "operational.rgb_port.cost_routes.ProducerObservation", NativeObservation
                    ),
                ):
                    for index in record["indices"]:
                        raw = runtime.raw_event_source.batch([index], "event")
                        prepared = source.batch([index])
                        a, at = extract_pair(
                            model, raw.events[:, -2:].cuda(), raw.delta_t_s.reshape(-1).cuda()
                        )
                        b, bt = extract_pair(
                            model, prepared.events.cuda(), prepared.delta_t_s.reshape(-1).cuda()
                        )
                        start = index // 8 * 8
                        batch = source.batch(list(range(start, min(start + 8, source.population))))
                        wide, wt = extract_pair(
                            model, batch.events.cuda(), batch.delta_t_s.reshape(-1).cuda()
                        )
                        offset = index - start
                        item = {
                            "index": index,
                            "raw_pixels_exact": torch.equal(raw.events[:, -2:], prepared.events),
                            "raw_delta_exact": torch.equal(
                                raw.delta_t_s.reshape(-1), prepared.delta_t_s.reshape(-1)
                            ),
                            "event_raw_vs_prepared_features": difference(a, b),
                            "event_raw_vs_prepared_ttc": difference(at, bt),
                            "event_batch1_vs_stored": difference(
                                a, values["event"]["features"][index : index + 1]
                            ),
                            "event_batch8_vs_stored": difference(
                                wide[offset : offset + 1],
                                values["event"]["features"][index : index + 1],
                            ),
                            "event_batch8_ttc_vs_stored": difference(
                                wt[offset : offset + 1],
                                values["event"]["prediction_ttc"][index : index + 1],
                            ),
                        }
                        rgb = runtime.rgb_context(index)
                        if rgb is None:
                            raise ValueError("Registered diagnostic query lost RGB")
                        event = runtime.raw_call(index, "event_context")
                        for name, actual in (("R_HISTORY", rgb), ("E_HISTORY", event)):
                            cached = values[name]
                            valid = cached["valid"][index].astype(bool)
                            item[name] = {}
                            for key, tensor in zip(KEYS, actual, strict=True):
                                expected = cached[key][index : index + 1]
                                if key != "expert_phase":
                                    expected = expected[:, valid]
                                item[name][key] = difference(tensor.cpu().numpy(), expected)
                        rt, rtt = extract_history(rgb_anchor, *[value.cpu() for value in rgb])
                        item["raw_rgb28_vs_stored"] = difference(
                            rt, values["R_CTX"]["features"][index : index + 1]
                        )
                        item["raw_rgb_anchor_vs_stored"] = difference(
                            rtt, values["R_CTX"]["prediction_ttc"][index : index + 1]
                        )
                        cached_rgb = [
                            torch.from_numpy(values["R_HISTORY"][key][index : index + 1]).cuda()
                            for key in KEYS
                        ]
                        cached_event = [
                            torch.from_numpy(values["E_HISTORY"][key][index : index + 1]).cuda()
                            for key in KEYS
                        ]
                        item["R_CTX_same_cache_cpu_vs_gpu"] = difference(
                            extract_history(rgb_anchor, *[value.cpu() for value in cached_rgb])[1],
                            phase_to_ttc(runtime.heads["R_CTX"](*cached_rgb)["point_phase"])
                            .cpu()
                            .numpy(),
                        )
                        with native_median_policy():
                            original = runtime.heads["F_TRUE"](*cached_event, *cached_rgb[:3])[
                                "ttc"
                            ]
                            emitted = runtime.heads["F_TRUE"](*event, *rgb[:3])["ttc"]
                        item["F_TRUE_cached_vs_raw_contexts"] = difference(
                            original.cpu().numpy(), emitted.cpu().numpy()
                        )
                        rows.append(item)
            finally:
                runtime.close()
    finally:
        source.close()
    result = {
        "status": "COMPLETE_DIAGNOSTIC",
        "registration_sha256": sha(OUT / "REGISTRATION.json"),
        "rows": rows,
        "optimizer_updates": 0,
        "tolerance_changes": False,
        "cost_admitted": False,
    }
    freeze(OUT / "GPU_RESULT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    print(gpu() if args.gpu else cpu())
