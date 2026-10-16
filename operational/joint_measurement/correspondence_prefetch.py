"""Admit exact lossless sensor prefetch over the separately admitted inference graph."""

from __future__ import annotations

import argparse
import time
from unittest.mock import patch

import psutil
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.range_transfer.common import freeze, read, sha, utc, write

from . import correspondence_inference_graph as graph
from . import geometry_evaluate as evaluation
from .common import ROOT, RUN, gpu_charge
from .compute_guard import guard_idle
from .data import SensorDataset
from .prefetched_sensor import PrefetchedSensor
from .train import committed_sources, setup

OUT = RUN / "runtime/B1_sensor_prefetch_v2"


def register() -> dict:
    """Bind prospective runtime-only parity and speed gates before the admission probe."""
    sources = {
        f"operational/joint_measurement/{name}.py": sha(
            ROOT / f"operational/joint_measurement/{name}.py"
        )
        for name in ("correspondence_prefetch", "prefetched_sensor", "data", "compute_guard")
    }
    committed_sources(sources)
    record = {
        "supersedes_rejected_probe": sha(RUN / "runtime/B1_sensor_prefetch_v1/RESULT.json"),
        "revision": "Avoid inner NPY copies and repeated file loaders; retain the speed gate",
        "previous_source_commit": "41f398f",
        "sources": sources,
        "graph_registration_sha256": sha(graph.OUT / "REGISTRATION.json"),
        "graph_proof_sha256": sha(graph.OUT / "RESULT.json"),
        "endpoints_sha256": sha(graph.b1.DIRECTORY / "ENDPOINTS.json"),
        "workers": 8,
        "maximum_resident_packs": 2,
        "persistent_disk_cache_bytes": 0,
        "input_parity": "Exact FP32 pixels/clocks: all P rows0:512, P final2, H final2",
        "output_parity": "Both frozen endpoints: P rows0:64 and final2, exact raw phase and TTC",
        "performance": "P rows0:512, graph original/prefetch, including startup and drain",
        "maximum_end_to_end_ratio": 0.8,
        "optimizer_updates": 0,
        "scope": "Prediction reader only; original ordering, B4/tail2, scoring, costs and exports",
        "integrity": "SHA256 of bytes consumed per pack, ZIP CRC, unique members, native tokens",
        "sources_researched_as_of": "2026-10-16",
        "references": [
            "https://docs.ffcv.io/making_dataloaders.html",
            "https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html",
            "https://docs.nvidia.com/cuda/nvcomp/installation.html",
            "https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html",
            "https://arxiv.org/abs/2602.08190",
        ],
    }
    freeze(OUT / "REGISTRATION.json", record)
    return record


def probe() -> dict:
    """Quiescent paired full-pipeline benchmark with unchanged weights and exact outputs."""
    guard_idle()
    registration = register()
    graph.register()
    if read(graph.OUT / "RESULT.json")["status"] != "ADMITTED":
        raise ValueError("Graph admission missing")
    if (OUT / "RESULT.json").exists():
        return read(OUT / "RESULT.json")
    result: dict = {"utc": utc(), "registration_sha256": sha(OUT / "REGISTRATION.json")}
    try:
        setup(7)
        parity = []
        for role, batches in (
            ("P", [list(range(i, i + 4)) for i in range(0, 512, 4)] + [[53888, 53889]]),
            ("H", [[18420, 18421]]),
        ):
            original, candidate = SensorDataset(role), PrefetchedSensor(role, ahead=False)
            try:
                for indices in batches:
                    left, right = original.batch(indices), candidate.batch(indices)
                    if not torch.equal(left.events, right.events) or not torch.equal(
                        left.delta_t_s.reshape(-1), right.delta_t_s.reshape(-1)
                    ):
                        raise ValueError("Exact native pixel/clock parity failed")
                parity.append(
                    {"role": role, "rows": sum(map(len, batches)), "packs": candidate.timings}
                )
            finally:
                original.close()
                candidate.close()
        measurements = []
        with gpu_charge("B1_lossless_prefetch_probe", reserve=900), torch.inference_mode():
            for name in (graph.b1.FIT, graph.b1.CONTROL):
                with graph.b1.predict_context(name):
                    model = evaluation.load_model().cuda().eval().requires_grad_(False)
                candidate_model = graph.InferenceGraph(model)
                weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
                original, candidate = SensorDataset("P"), PrefetchedSensor("P", ahead=False)
                try:
                    for indices in [list(range(i, i + 4)) for i in range(0, 64, 4)] + [
                        [53888, 53889]
                    ]:
                        left, right = original.batch(indices), candidate.batch(indices)
                        expected = candidate_model(
                            left.events.cuda(), left.delta_t_s.cuda().reshape(-1)
                        )["raw_phase"]
                        actual = candidate_model(
                            right.events.cuda(), right.delta_t_s.cuda().reshape(-1)
                        )["raw_phase"]
                        if not torch.equal(expected, actual) or not torch.equal(
                            phase_to_ttc(expected), phase_to_ttc(actual)
                        ):
                            raise ValueError("Exact output parity failed")
                    if any(not torch.equal(v, model.state_dict()[k]) for k, v in weights.items()):
                        raise ValueError("Weights changed during inference")
                finally:
                    original.close()
                    candidate.close()
                # Graph already captured. Alternate reader order between the two model endpoints.
                routes = (
                    ("original", "prefetch") if name == graph.b1.FIT else ("prefetch", "original")
                )
                measured = {}
                for route in routes:
                    dataset = SensorDataset("P") if route == "original" else PrefetchedSensor("P")
                    waits, transfers, forwards = [], [], []
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    try:
                        for begin in range(0, 512, 4):
                            t = time.perf_counter()
                            batch = dataset.batch(list(range(begin, begin + 4)))
                            loaded = time.perf_counter()
                            pixels, dt = batch.events.cuda(), batch.delta_t_s.cuda().reshape(-1)
                            torch.cuda.synchronize()
                            copied = time.perf_counter()
                            raw = candidate_model(pixels, dt)["raw_phase"]
                            phase_to_ttc(raw).cpu().double().tolist()
                            ended = time.perf_counter()
                            waits.append(loaded - t)
                            transfers.append(copied - loaded)
                            forwards.append(ended - copied)
                    finally:
                        dataset.close()
                    measured[route] = {
                        "elapsed_s_including_drain": time.perf_counter() - start,
                        "batches": 128,
                        "load_wait_s": sum(waits),
                        "h2d_s": sum(transfers),
                        "forward_decode_d2h_s": sum(forwards),
                        "max_load_wait_s": max(waits),
                    }
                ratio = (
                    measured["prefetch"]["elapsed_s_including_drain"]
                    / measured["original"]["elapsed_s_including_drain"]
                )
                measurements.append({"model": name, "ratio": ratio, "routes": measured})
                print(measurements[-1], flush=True)
                del candidate_model, model, weights
                torch.cuda.empty_cache()
        result.update(
            status="ADMITTED"
            if all(m["ratio"] <= registration["maximum_end_to_end_ratio"] for m in measurements)
            else "REJECTED_SPEED",
            input_parity=parity,
            exact_pixels_clocks_phase_ttc=True,
            weights_unchanged=True,
            optimizer_updates=0,
            measurements=measurements,
            rss_bytes=psutil.Process().memory_info().rss,
            cache_scope=(
                "OS cache uncontrolled; paired warm-cache fixed P population; "
                "startup and future drain included"
            ),
        )
    except Exception as error:
        write(OUT / "RESULT.json", {**result, "status": "FAIL", "error": repr(error)})
        raise
    freeze(OUT / "RESULT.json", result)
    return result


def run() -> dict:
    """Resume original durable evaluation, composing two separately admitted overlays."""
    register()
    proof = read(OUT / "RESULT.json")
    if proof["status"] != "ADMITTED" or proof["registration_sha256"] != sha(
        OUT / "REGISTRATION.json"
    ):
        raise ValueError("Exact prefetch admission required")
    original_predict = evaluation.predict

    def predict(role: str) -> dict:
        directory = graph.b1.DIRECTORY / "predictions" / role / evaluation.FIT
        binding = {
            "registration_sha256": sha(OUT / "REGISTRATION.json"),
            "proof_sha256": sha(OUT / "RESULT.json"),
        }
        path = directory / "LOADER_RUNTIME.json"
        if path.exists():
            saved = read(path)
            if any(saved[k] != v for k, v in binding.items()) or any(
                sha(directory / name) != expected
                for name, expected in saved["preserved_fragments"].items()
            ):
                raise ValueError("Loader binding or preserved prediction changed")
        else:
            preserved = {}
            for fragment in sorted(directory.glob("chunk_*.parquet")):
                if fragment.with_suffix(".json").exists():
                    expected = read(fragment.with_suffix(".json"))["sha256"]
                    if sha(fragment) != expected:
                        raise ValueError("Durable prediction fragment changed")
                    preserved[fragment.name] = expected
            freeze(path, {**binding, "preserved_fragments": preserved})
        readers = []

        def make_reader(role: str) -> PrefetchedSensor:
            reader = PrefetchedSensor(role)
            readers.append(reader)
            return reader

        started = time.perf_counter()
        with patch.object(evaluation, "SensorDataset", make_reader):
            result = original_predict(role)
        write(
            directory / "LOADER_LATEST.json",
            {
                "utc": utc(),
                "elapsed_s": time.perf_counter() - started,
                "status": result["status"],
                "packs": [p for r in readers for p in r.timings],
                "registration_sha256": binding["registration_sha256"],
            },
        )
        return result

    with patch.object(evaluation, "predict", predict):
        return graph.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "probe", "run"))
    args = parser.parse_args()
    print({"register": register, "probe": probe, "run": run}[args.mode]())
