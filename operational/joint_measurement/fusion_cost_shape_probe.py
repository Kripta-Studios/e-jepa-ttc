"""Diagnose historical batch-dependent numerics without fitting or changing tolerances."""

from __future__ import annotations

import argparse
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.rgb_port.data import decode_query
from e_jepa_ttc.rgb_port.features import (
    event_phase17,
    event_statistics_from_channels12,
    rgb_phase17,
)
from e_jepa_ttc.simplex_t.phase import emitted_phase
from operational.range_transfer.common import freeze, read, sha, write
from operational.rgb_port.cost_routes import _normalizers
from operational.rgb_port.infer_experts import delta_matrix
from operational.rgb_port.train_heads import pair_point_phase
from operational.rgb_port.train_producers import producer_features
from operational.rgb_port_streaming.contracts import NativeObservation

from .common import ROOT, RUN, gpu_charge, owner, require_resources
from .fusion_cost_backend_probe import convolution_policy
from .fusion_cost_batch_inventory import OUT as INVENTORY
from .fusion_cost_diagnostic import KEYS, difference, load_arrays
from .fusion_cost_diagnostic import registration as prior_registration
from .fusion_cost_native import admit, native_context, qa
from .fusion_cost_native_v2 import NativeRuntime, native_median_policy
from .garl_route_cost import guard_idle
from .train import committed_sources, setup

OUT = RUN / "fusion/cost/shape_diagnostic_v1"


def repeat_observation(
    sensor: torch.Tensor, delta: torch.Tensor, *, frames: int, batch: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Duplicate only the current input to the verified historical forward shape."""
    if (
        sensor.ndim != 5
        or sensor.shape[:2] != (1, frames)
        or delta.ndim != 2
        or delta.shape != (1, frames - 1)
        or frames not in (2, 3)
        or not 1 <= batch <= 32
    ):
        raise ValueError("Historical observation shape differs")
    if sensor.dtype != torch.float32 or delta.dtype != torch.float32:
        raise ValueError("Shape diagnostic requires FP32")
    return sensor.repeat(batch, 1, 1, 1, 1), delta.repeat(batch, 1)


class ShapeRuntime(NativeRuntime):
    """Reconstruct historical producer/PAIR batch shapes; retain only the first output."""

    shape_map: dict[str, dict[str, dict]]
    convolution_tf32 = False

    def _observation(
        self, sensor: torch.Tensor, delta: torch.Tensor, modality: str, identity: str
    ) -> tuple[np.ndarray, np.ndarray]:
        shape = self.shape_map[modality][identity]
        sensor, delta = repeat_observation(
            sensor, delta, frames=shape["input_frames"], batch=shape["forward_batch"]
        )
        names = ("E_A5_MATCHED", "E_C2F_MATCHED") if modality == "E" else ("R_A5", "R_C2F")
        pair_name = "PAIR_E_MATCHED" if modality == "E" else "PAIR_R"
        with torch.inference_mode(), convolution_policy(self.convolution_tf32):
            outputs = [producer_features(self.producers[name], sensor, delta) for name in names]
            pair_input = NativeObservation.from_output(outputs[0]).pair_input(delta[:, -1])
            pair = emitted_phase(pair_point_phase(self.pairs[pair_name], pair_input))
            phases = torch.stack((outputs[0]["point_phase"], outputs[1]["point_phase"], pair), -1)
            diagnostics = [
                torch.stack((item["flow"], item["margin"], item["log_variance"]), -1)
                for item in outputs
            ]
            if modality == "E":
                features = event_phase17(
                    event_statistics_from_channels12(sensor[:, -1]),
                    diagnostics[0],
                    diagnostics[1],
                    phases,
                )
            else:
                features = rgb_phase17(sensor[:, -1], diagnostics[0], diagnostics[1], phases)
        return features[0].cpu().numpy(), phases[0].cpu().numpy()

    def _event_observation(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        batch = self.event_source.batch([index], "event")
        sensor = batch.events.to(self.device).float()
        delta = delta_matrix(batch.delta_t_s, sensor.shape[1]).to(self.device)
        return self._observation(sensor, delta, "E", str(self.rows[index]["sample_token"]))

    def _rgb_observation(self, item: Mapping) -> tuple[np.ndarray, np.ndarray]:
        decoded = decode_query(item, eap_root=self.rgb_source.dataset.eap_root, reader=self.reader)
        sensor = torch.from_numpy(decoded["rgb"])[None].to(self.device)
        delta = torch.from_numpy(decoded["delta_t_s"])[None].to(self.device)
        return self._observation(sensor, delta, "R", str(item["observation_id"]))


def register() -> dict:
    """Fix reconstructed batch shapes and both prior backend settings before execution."""
    inventory = read(INVENTORY / "RESULT.json")
    original = prior_registration()
    if (
        inventory["status"] != "COMPLETE_METADATA_AUDIT"
        or inventory["indices"] != original["indices"]
    ):
        raise ValueError("Complete matching historical shape inventory required")
    sources = {
        f"operational/joint_measurement/{name}.py": sha(Path(__file__).with_name(name + ".py"))
        for name in ("fusion_cost_shape_probe", "fusion_cost_batch_inventory", "garl_route_cost")
    }
    committed_sources(sources)
    record = {
        "sources": sources,
        "inventory_sha256": sha(INVENTORY / "RESULT.json"),
        "prior_batch1_backend_result_sha256": sha(
            RUN / "fusion/cost/backend_diagnostic_v1/RESULT.json"
        ),
        "indices": original["indices"],
        "producer_convolution_tf32": [False, True],
        "factor": "Producer and PAIR forward batch shape, from admitted historical fragments",
        "input_policy": (
            "Repeat each query's own pixels/delta to its original batch shape; "
            "no other query inputs"
        ),
        "retained_prediction": "First row; all repeated rows represent the same observation",
        "precision": "FP32, matmul TF32 disabled, original weights and normalization",
        "interpretation": (
            "Engineering diagnostic of runtime parity; no model/seed/recipe selection"
        ),
        "original_ttc_tolerance": {"rtol": 1e-5, "atol": 1e-4},
        "tolerance_changes": False,
        "optimizer_updates": 0,
        "cost_admitted": False,
        "not_a_latency_measurement": True,
    }
    freeze(OUT / "REGISTRATION.json", record)
    return record


def run() -> dict:
    """Execute only after training stops; preserve the original fixed query population."""
    guard_idle()
    require_resources(starting=True)
    record = register()
    if (OUT / "RESULT.json").exists():
        return read(OUT / "RESULT.json")
    inventory = read(INVENTORY / "RESULT.json")
    shape_map = {
        modality: {item["id"]: item for query in data["queries"] for item in query["observations"]}
        for modality, data in inventory["modalities"].items()
    }
    values = load_arrays(prior_registration())
    native = admit()
    qa(native)
    setup(7)
    provenance = read(owner() / "fits/E_A5_MATCHED/RUN_PROVENANCE.json")
    progress_path = OUT / "PROGRESS.json"
    progress = read(progress_path) if progress_path.exists() else None
    if progress and progress["registration_sha256"] != sha(OUT / "REGISTRATION.json"):
        raise ValueError("Shape diagnostic progress belongs to another registration")
    measurements = progress["measurements"] if progress else []
    done = {(row["index"], row["producer_cudnn_tf32"]) for row in measurements}
    if len(done) != len(measurements):
        raise ValueError("Duplicate shape diagnostic measurements")
    with (
        native_context(native),
        gpu_charge("fusion_historical_batch_shape_probe", reserve=900) as deadline,
    ):
        runtime = ShapeRuntime(
            run=owner(),
            config=ROOT / "configs/rgb_port/producers.json",
            p_manifest=owner() / "P_MANIFEST.json",
            manifest=owner() / "V_MANIFEST.json",
            event_source_root=Path(provenance["source_identity"]["event_source_root"]),
            event_history=owner() / "features/V/E_RAW/history.npz",
            normalizers=_normalizers(RUN / "fusion/cost/raw_e2e_native_v2/NORMALIZERS.json"),
            device="cuda",
        )
        runtime.shape_map = shape_map
        try:
            with torch.inference_mode():
                for index in record["indices"]:
                    cached = {
                        name: [
                            torch.from_numpy(values[name][key][index : index + 1]).cuda()
                            for key in KEYS
                        ]
                        for name in ("E_HISTORY", "R_HISTORY")
                    }
                    with native_median_policy():
                        expected = runtime.heads["F_TRUE"](
                            *cached["E_HISTORY"], *cached["R_HISTORY"][:3]
                        )["ttc"]
                    for enabled in record["producer_convolution_tf32"]:
                        if (index, enabled) in done:
                            continue
                        if time.perf_counter() + 120 > deadline:
                            partial = {
                                "status": "PARTIAL_SCOPE",
                                "measurements": measurements,
                                "registration_sha256": sha(OUT / "REGISTRATION.json"),
                            }
                            write(OUT / "PARTIAL.json", partial)
                            return partial
                        runtime.convolution_tf32 = enabled
                        rgb = runtime.rgb_context(index)
                        event = runtime.raw_call(index, "event_context")
                        if rgb is None:
                            raise ValueError("Fixed diagnostic query lost RGB")
                        row = {"index": index, "producer_cudnn_tf32": enabled}
                        for name, actual in (("R_HISTORY", rgb), ("E_HISTORY", event)):
                            valid = values[name]["valid"][index].astype(bool)
                            row[name] = {}
                            for key, tensor in zip(KEYS, actual, strict=True):
                                reference = values[name][key][index : index + 1]
                                if key != "expert_phase":
                                    reference = reference[:, valid]
                                row[name][key] = difference(tensor.cpu().numpy(), reference)
                        with native_median_policy():
                            actual = runtime.heads["F_TRUE"](*event, *rgb[:3])["ttc"]
                        row["F_TRUE_raw_vs_cached_contexts"] = difference(
                            actual.cpu().numpy(), expected.cpu().numpy()
                        )
                        row["raw_ttc_s"], row["cached_ttc_s"] = float(actual[0]), float(expected[0])
                        row["original_tolerance_pass"] = bool(
                            torch.allclose(actual, expected, **record["original_ttc_tolerance"])
                        )
                        if torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32:
                            raise ValueError("Scoped backend policy leaked")
                        measurements.append(row)
                        write(
                            progress_path,
                            {
                                "registration_sha256": sha(OUT / "REGISTRATION.json"),
                                "measurements": measurements,
                            },
                        )
        finally:
            runtime.close()
    result = {
        "status": "COMPLETE_DIAGNOSTIC",
        "registration_sha256": sha(OUT / "REGISTRATION.json"),
        "measurements": measurements,
        "optimizer_updates": 0,
        "cost_admitted": False,
        "tolerance_changes": False,
        "full_population_cost_still_required": True,
    }
    freeze(OUT / "RESULT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    print(register() if parser.parse_args().register_only else run())
