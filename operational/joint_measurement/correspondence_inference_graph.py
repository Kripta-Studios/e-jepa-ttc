"""Exact-output CUDA inference overlay; keep registered B1 evaluation and cost intact."""

from __future__ import annotations

import argparse
import statistics
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.range_transfer.common import freeze, read, sha, utc, write

from . import correspondence_evaluate as b1
from . import geometry_evaluate as evaluation
from .common import ROOT, RUN, gpu_charge
from .compute_guard import guard_idle
from .data import SensorDataset
from .geometry_capture import CaptureCore, checked_phase
from .geometry_model import GeometricPairMeasurement
from .train import committed_sources, setup

OUT = RUN / "runtime/B1_inference_graph_v2"


def validate_inputs(inputs: Tensor, dt: Tensor) -> None:
    """Keep original batch sizes, precision and scalar native clock layout."""
    if inputs.shape not in ((4, 2, 12, 128, 128), (2, 2, 12, 128, 128)):
        raise ValueError("Only registered B1 batches4 and tail2 are admitted")
    if dt.shape != (len(inputs),) or inputs.dtype != torch.float32 or dt.dtype != torch.float32:
        raise ValueError("FP32 pixels and original scalar clocks required")
    if inputs.device != dt.device:
        raise ValueError("Pixel and clock devices differ")


class InferenceGraph(nn.Module):
    """Capture the already verified tensor core without changing any parameter."""

    def __init__(self, model: GeometricPairMeasurement) -> None:
        super().__init__()
        self.model = model
        self.core = CaptureCore(model)
        self.graphs: dict[int, tuple] = {}
        self.eval().requires_grad_(False)

    def forward(self, inputs: Tensor, delta_t_s: Tensor) -> dict[str, Tensor]:
        """Replay fixed shapes and materialize the original finite/domain guard."""
        validate_inputs(inputs, delta_t_s)
        if inputs.device.type != "cuda" or self.training or torch.is_grad_enabled():
            raise ValueError("CUDA evaluation without autograd required")
        size = len(inputs)
        if size not in self.graphs:
            pixels, clocks = inputs.clone(), delta_t_s.clone()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.core(pixels, clocks)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = self.core(pixels, clocks)
            self.graphs[size] = (graph, pixels, clocks, output)
        graph, pixels, clocks, output = self.graphs[size]
        pixels.copy_(inputs)
        clocks.copy_(delta_t_s)
        graph.replay()
        checked = checked_phase(output)
        # The caller must own its phase after a subsequent replay overwrites buffers.
        return {"raw_phase": checked["raw_phase"].clone()}


def register() -> dict:
    """Freeze a zero-update runtime test before measuring or changing inference."""
    sources = {
        f"operational/joint_measurement/{name}.py": sha(
            ROOT / f"operational/joint_measurement/{name}.py"
        )
        for name in (
            "compute_guard",
            "correspondence_inference_graph",
            "geometry_capture",
            "geometry_model",
            "correspondence_evaluate",
            "geometry_evaluate",
        )
    }
    committed_sources(sources)
    record = {
        "supersedes_unexecuted_probe": sha(RUN / "runtime/B1_inference_graph_v1/REGISTRATION.json"),
        "revision": (
            "Exclude exact MLflow MCP service from CPU compute guard; "
            "GPU ownership remains enforced"
        ),
        "sources": sources,
        "endpoints_sha256": sha(b1.DIRECTORY / "ENDPOINTS.json"),
        "evaluation_registration_sha256": sha(b1.EVALUATION),
        "models": [b1.FIT, b1.CONTROL],
        "P_indices": list(range(64)) + [53888, 53889],
        "batches": [4, 2],
        "precision": "FP32_no_TF32",
        "parity": "Exact raw phase and native emitted TTC; no tolerance relaxation",
        "require": ["weights_unchanged", "rng_unchanged", "invalid_clock_rejected"],
        "maximum_median_graph_to_eager_ratio": 0.9,
        "optimizer_updates": 0,
        "scope": "Prediction execution only; original cost, export, scorer and selection unchanged",
        "resume": "Retain every verified existing fragment under its original binding",
        "documentation": "https://docs.pytorch.org/docs/2.14/notes/cuda.html#cuda-graphs",
    }
    freeze(OUT / "REGISTRATION.json", record)
    return record


def probe() -> dict:
    """Compare both frozen endpoints on P; fail closed before any H graph inference."""
    guard_idle()
    record = register()
    if (OUT / "RESULT.json").exists():
        return read(OUT / "RESULT.json")
    result: dict = {"utc": utc(), "registration_sha256": sha(OUT / "REGISTRATION.json")}
    try:
        setup(7)
        data = SensorDataset("P")
        measurements = []
        with gpu_charge("B1_exact_inference_graph_probe", reserve=900), torch.inference_mode():
            for name in record["models"]:
                with b1.predict_context(name):
                    model = evaluation.load_model().cuda().eval().requires_grad_(False)
                initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
                cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
                candidate = InferenceGraph(model)
                for ids in [list(range(i, i + 4)) for i in range(0, 64, 4)] + [[53888, 53889]]:
                    tick = time.perf_counter()
                    batch = data.batch(ids)
                    pixels, dt = batch.events.cuda(), batch.delta_t_s.cuda().reshape(-1)
                    torch.cuda.synchronize()
                    load_seconds = time.perf_counter() - tick
                    # First capture is measured separately from the paired warm forward.
                    tick = time.perf_counter()
                    candidate(pixels, dt)
                    torch.cuda.synchronize()
                    first_call_seconds = time.perf_counter() - tick
                    timings, phases = {}, {}
                    for route in ["eager", "graph"] if ids[0] % 8 == 0 else ["graph", "eager"]:
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        phase = (model if route == "eager" else candidate)(pixels, dt)["raw_phase"]
                        ttc = phase_to_ttc(phase)
                        torch.cuda.synchronize()
                        timings[route] = time.perf_counter() - tick
                        phases[route] = (phase.clone(), ttc.clone())
                    if not all(
                        torch.equal(a, b)
                        for a, b in zip(phases["eager"], phases["graph"], strict=True)
                    ):
                        raise ValueError(f"Exact inference parity failed: {name}, {ids}")
                    if name == b1.FIT and ids[0] < 64:
                        original = pd.read_parquet(
                            b1.DIRECTORY / "predictions/P" / name / "chunk_00000.parquet"
                        ).iloc[ids]
                        for column, value in zip(
                            ("raw_phase", "prediction_ttc"), phases["graph"], strict=True
                        ):
                            if not np.array_equal(
                                original[column].to_numpy(), value.cpu().double().numpy()
                            ):
                                raise ValueError("Graph output differs from durable eager fragment")
                    measurements.append(
                        {
                            "model": name,
                            "indices": ids,
                            "timings": timings,
                            "load_seconds": load_seconds,
                            "first_call_seconds": first_call_seconds,
                        }
                    )
                invalid = dt.clone()
                invalid[0] = 0
                try:
                    candidate(pixels, invalid)
                except ValueError as error:
                    if "guard failed" not in str(error):
                        raise
                else:
                    raise ValueError("Invalid native clock was not rejected")
                if not all(torch.equal(initial[k], v) for k, v in model.state_dict().items()):
                    raise ValueError("Inference changed model state")
                if not torch.equal(cpu_rng, torch.get_rng_state()) or not all(
                    torch.equal(a, b)
                    for a, b in zip(gpu_rng, torch.cuda.get_rng_state_all(), strict=True)
                ):
                    raise ValueError("Inference changed RNG state")
                del candidate, model, initial
                torch.cuda.empty_cache()
        data.close()
        ratio = statistics.median(
            x["timings"]["graph"] / x["timings"]["eager"] for x in measurements
        )
        result.update(
            status="ADMITTED"
            if ratio <= record["maximum_median_graph_to_eager_ratio"]
            else "REJECTED_SPEED",
            paired_ratio=ratio,
            measurements=measurements,
            exact_phase_and_ttc=True,
            stored_fragment_exact=True,
            weights_unchanged=True,
            rng_unchanged=True,
            invalid_clock_rejected=True,
            optimizer_updates=0,
        )
    except Exception as error:
        result.update(status="FAIL", error=repr(error), optimizer_updates=0)
        write(OUT / "RESULT.json", result)
        raise
    freeze(OUT / "RESULT.json", result)
    return result


def run() -> dict:
    """Use the original evaluator, substituting only its prediction-time model call."""
    guard_idle()
    register()
    proof = read(OUT / "RESULT.json")
    if proof["status"] != "ADMITTED" or proof["registration_sha256"] != sha(
        OUT / "REGISTRATION.json"
    ):
        raise ValueError("Exact numerical and speed admission required")
    original_predict, original_load = evaluation.predict, evaluation.load_model

    def predict(role: str) -> dict:
        directory = b1.DIRECTORY / "predictions" / role / evaluation.FIT
        runtime_path = directory / "INFERENCE_RUNTIME.json"
        record = {
            "registration_sha256": sha(OUT / "REGISTRATION.json"),
            "probe_sha256": sha(OUT / "RESULT.json"),
            "precision_batch_population_scorer_unchanged": True,
        }
        if runtime_path.exists():
            saved = read(runtime_path)
            if any(saved[k] != v for k, v in record.items()) or any(
                sha(directory / name) != expected
                for name, expected in saved["preserved_fragments"].items()
            ):
                raise ValueError("Inference runtime or original fragments changed")
        else:
            preserved = {}
            for path in sorted(directory.glob("chunk_*.parquet")):
                if not path.with_suffix(".json").exists():
                    continue
                expected = read(path.with_suffix(".json"))["sha256"]
                if sha(path) != expected:
                    raise ValueError("Durable prediction fragment changed")
                preserved[path.name] = expected
            freeze(runtime_path, {**record, "preserved_fragments": preserved})
        with patch.object(evaluation, "load_model", lambda: InferenceGraph(original_load())):
            return original_predict(role)

    with patch.object(evaluation, "predict", predict):
        return b1.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "probe", "run"))
    mode = parser.parse_args().mode
    print({"register": register, "probe": probe, "run": run}[mode]())
