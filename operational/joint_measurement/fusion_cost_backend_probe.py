"""Test historical convolution backend policy after strict raw/cohort parity failed."""

from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import torch

from operational.range_transfer.common import freeze, read, sha
from operational.rgb_port.cost_routes import _normalizers
from operational.rgb_port_streaming.contracts import NativeObservation

from .common import ROOT, RUN, gpu_charge, owner
from .fusion_cost_diagnostic import KEYS, difference, load_arrays
from .fusion_cost_diagnostic import registration as original_registration
from .fusion_cost_native import admit, native_context, qa
from .fusion_cost_native_v2 import NativeRuntime, native_median_policy
from .geometry_graph_probe import no_live_training_driver
from .train import committed_sources, setup

OUT = RUN / "fusion/cost/backend_diagnostic_v1"
FAILED = RUN / "fusion/cost/raw_e2e_native_v2"


@contextmanager
def convolution_policy(enabled: bool) -> Iterator[None]:
    """Alter only this process's cuDNN convolution TF32 flag, restoring it even on failure."""
    previous = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cudnn.allow_tf32 = enabled
        yield
    finally:
        torch.backends.cudnn.allow_tf32 = previous


class ConvolutionRuntime(NativeRuntime):
    """Scope backend changes to historical sensor producers; new C1/head paths are untouched."""

    convolution_tf32 = False

    def _event_observation(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        with convolution_policy(self.convolution_tf32):
            return super()._event_observation(index)

    def _rgb_observation(self, item: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        with convolution_policy(self.convolution_tf32):
            return super()._rgb_observation(item)


def register() -> dict:
    """Fix both backend settings and the original four diagnostic queries before execution."""
    original = original_registration()
    prior = RUN / "fusion/cost/parity_diagnostic_v1/GPU_RESULT.json"
    if read(prior)["status"] != "COMPLETE_DIAGNOSTIC":
        raise ValueError("Complete first strict-parity localization required")
    paths = (
        "fusion_cost_backend_probe",
        "fusion_cost_diagnostic",
        "fusion_cost_native",
        "fusion_cost_native_v2",
    )
    sources = {
        f"operational/joint_measurement/{name}.py": sha(
            ROOT / f"operational/joint_measurement/{name}.py"
        )
        for name in paths
    }
    committed_sources(sources)
    result = {
        "sources": sources,
        "prior_diagnostic_sha256": sha(prior),
        "indices": original["indices"],
        "settings": [False, True],
        "factor": "cuDNN convolution allow_tf32 on historical A5/C2F producers only",
        "fixed": "FP32 tensors, matmul TF32False, same weights/clocks/ROIs/normalizers/heads",
        "evidence": (
            "Historical infer_experts.main/build_producer set deterministic execution but "
            "do not set cuDNN allow_tf32; current V18 setup explicitly disables it. "
            "Stored features differ although clocks/validity match. This is a hypothesis, "
            "not a claim that the historical effective flag was recorded."
        ),
        "optimizer_updates": 0,
        "tolerance_changes": False,
        "cost_admitted": False,
        "label_use": "No target values used; numerical output/feature comparison only",
        "no_model_selection": True,
    }
    freeze(OUT / "REGISTRATION.json", result)
    return result


def run() -> dict:
    """Compare both settings with complete histories and cached-context F_TRUE inference."""
    no_live_training_driver()
    record = register()
    if (OUT / "RESULT.json").exists():
        return read(OUT / "RESULT.json")
    values = load_arrays(original_registration())
    native = admit()
    qa(native)
    setup(7)
    provenance = read(owner() / "fits/E_A5_MATCHED/RUN_PROVENANCE.json")
    measurements = []
    with (
        native_context(native),
        gpu_charge("fusion_historical_convolution_backend_probe", reserve=900),
    ):
        runtime = ConvolutionRuntime(
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
            with (
                torch.inference_mode(),
                patch("operational.rgb_port.cost_routes.ProducerObservation", NativeObservation),
            ):
                for index in record["indices"]:
                    cached_rgb = [
                        torch.from_numpy(values["R_HISTORY"][key][index : index + 1]).cuda()
                        for key in KEYS
                    ]
                    cached_event = [
                        torch.from_numpy(values["E_HISTORY"][key][index : index + 1]).cuda()
                        for key in KEYS
                    ]
                    with native_median_policy():
                        expected = runtime.heads["F_TRUE"](*cached_event, *cached_rgb[:3])["ttc"]
                    for enabled in record["settings"]:
                        runtime.convolution_tf32 = enabled
                        rgb = runtime.rgb_context(index)
                        event = runtime.raw_call(index, "event_context")
                        if rgb is None:
                            raise ValueError("Fixed query lost its RGB history")
                        row = {"index": index, "producer_cudnn_tf32": enabled}
                        for name, actual in (("R_HISTORY", rgb), ("E_HISTORY", event)):
                            cached = values[name]
                            valid = cached["valid"][index].astype(bool)
                            row[name] = {}
                            for key, tensor in zip(KEYS, actual, strict=True):
                                reference = cached[key][index : index + 1]
                                if key != "expert_phase":
                                    reference = reference[:, valid]
                                row[name][key] = difference(tensor.cpu().numpy(), reference)
                        with native_median_policy():
                            actual = runtime.heads["F_TRUE"](*event, *rgb[:3])["ttc"]
                        row["F_TRUE_raw_vs_cached_contexts"] = difference(
                            actual.cpu().numpy(), expected.cpu().numpy()
                        )
                        row["raw_ttc_s"] = float(actual[0])
                        row["cached_ttc_s"] = float(expected[0])
                        row["cudnn_flag_restored"] = not torch.backends.cudnn.allow_tf32
                        row["matmul_tf32_disabled"] = not torch.backends.cuda.matmul.allow_tf32
                        if not row["cudnn_flag_restored"] or not row["matmul_tf32_disabled"]:
                            raise ValueError("Backend scope leaked into the rest of the route")
                        measurements.append(row)
        finally:
            runtime.close()
    result = {
        "status": "COMPLETE_DIAGNOSTIC",
        "registration_sha256": sha(OUT / "REGISTRATION.json"),
        "measurements": measurements,
        "optimizer_updates": 0,
        "cost_admitted": False,
        "tolerance_changes": False,
        "historical_effective_flag_recorded": False,
    }
    freeze(OUT / "RESULT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register-only", action="store_true")
    print(register() if parser.parse_args().register_only else run())
