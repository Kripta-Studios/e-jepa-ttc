"""Paired batch-one raw-sensor cost for the frozen fusion and historical RGB systems."""

from __future__ import annotations

import copy
import socket
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import psutil
import torch

from e_jepa_ttc.rgb_port.normalization import FrozenNormalizer
from e_jepa_ttc.simplex_t.model import TemporalRefiner
from operational.range_transfer.common import digest, freeze, read, sha
from operational.rgb_port.cost_routes import RouteRuntime, _normalizers
from operational.rgb_port.infer_experts import _parents
from operational.rgb_port_streaming.contracts import NativeObservation

from .common import ROOT, RUN, gpu_charge, owner, require_resources
from .data import role_rows
from .fusion_evaluate import load_head
from .fusion_export import NormalizedFusion
from .fusion_history_extract import extract_history
from .fusion_pair_extract import extract_pair
from .fusion_train import DIRECTORY, freeze_endpoints, inputs
from .geometry_evaluate import load_model
from .raw_cost import indices_by_group
from .rgb_run import require_authorization
from .train import committed_sources, setup

OUT = DIRECTORY / "cost/raw_e2e"
NAMES = ("FUSION-TRUE-S7", "FUSION-ZERO-S7", "R_CTX", "F_TRUE")


def summary(values: list[float]) -> dict:
    """Keep distribution tails and throughput; do not infer cache-cold cost."""
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": len(array),
        "mean_seconds": float(array.mean()),
        "p50_seconds": float(np.quantile(array, 0.5)),
        "p95_seconds": float(np.quantile(array, 0.95)),
        "p99_seconds": float(np.quantile(array, 0.99)),
        "queries_per_second": float(1 / array.mean()),
    }


def measure() -> dict:
    """No concurrent GPU writer; charge loading, parity, first-call and warm measurements."""
    require_authorization()
    require_resources(starting=True)
    endpoints = freeze_endpoints()
    admission = read(RUN / "verification/HISTORICAL_MULTIMODAL_ADMISSION.json")
    rows = role_rows("V")
    indices = indices_by_group(rows)
    orders = [
        np.random.default_rng(261016 + i).permutation(NAMES).tolist() for i in range(len(indices))
    ]
    mapping = {
        name: str(owner() / f"features/H/{tag}_NORMALIZER.json")
        for name, tag in (("event", "E"), ("rgb", "R"))
    }
    freeze(OUT / "NORMALIZERS.json", mapping)
    parents = _parents(
        owner(), [*admission["expert_endpoints"], "R_CTX", "F_TRUE", "E_CTX_MATCHED"]
    )
    expected_parents = {
        **admission["expert_endpoints"],
        **{name: admission["heads"][name]["checkpoint_sha256"] for name in ("R_CTX", "F_TRUE")},
    }
    if any(parents[name] != expected for name, expected in expected_parents.items()):
        raise ValueError("Historical cost comparator parents changed")
    for name, key in (("event", "E_NORMALIZER.json"), ("rgb", "R_NORMALIZER.json")):
        if sha(Path(mapping[name])) != admission["normalizers"][key]:
            raise ValueError("Native historical cost normalizer changed")
    sources = {
        f"operational/joint_measurement/{name}.py": sha(
            ROOT / f"operational/joint_measurement/{name}.py"
        )
        for name in (
            "fusion_cost",
            "fusion_history_extract",
            "fusion_pair_extract",
            "fusion_export",
            "geometry_model",
        )
    }
    sources.update(
        {
            name: sha(ROOT / name)
            for name in (
                "operational/rgb_port/cost_routes.py",
                "operational/rgb_port_streaming/contracts.py",
                "operational/rgb_port_revision/raw_events.py",
            )
        }
    )
    committed_sources(sources)
    binding = {
        "endpoints": endpoints,
        "historical_parents": parents,
        "sources": sources,
        "indices": indices,
        "orders": orders,
        "tokens": rows.iloc[indices].sample_token.tolist(),
        "manifest_sha256": sha(owner() / "V_MANIFEST.json"),
        "normalizers": {name: sha(Path(path)) for name, path in mapping.items()},
        "scope": (
            "Raw event HDF5 and RGB TAR reads, native ROI/windows/history, all required"
            " producers, normalizers, transfers and signed TTC"
        ),
        "candidate_head_runtime": (
            "CPU FP32 head/history anchor; CUDA event and historical RGB producers; "
            "explicit transfer costs included"
        ),
        "baseline_runtime": "Unchanged historical native CUDA routes, existing native mask adapter",
        "ZERO_scope": (
            "Matched full producer computation retained; a skip-event optimized ZERO is"
            " not measured"
        ),
        "prepared_scope": "CPU feature-to-TTC head only; explicitly excludes producers/sensors",
        "first_call": (
            "First route invocation after model loading; OS cache uncontrolled, not a "
            "machine-cold measurement"
        ),
        "warmup_per_route": 20,
        "batch_size": 1,
        "precision": "FP32_no_TF32",
        "parity_rtol": 1e-5,
        "parity_atol": 1e-4,
        "optimizer_updates": 0,
        "system_promotion": False,
    }
    freeze(OUT / "REGISTRATION.json", binding)
    if (OUT / "COMPLETE.json").exists():
        result = read(OUT / "COMPLETE.json")
        if result["binding"] != digest(binding):
            raise ValueError("Fusion cost binding changed")
        return result
    blocks, _, _ = inputs()
    original_recipe = read(DIRECTORY / "REGISTRATION.json")
    for name in ("rgb", "event"):
        if sha(DIRECTORY / "normalizers" / f"{name}.json") != original_recipe["normalizers"][name]:
            raise ValueError("Fusion head normalizer changed before cost audit")
    normalizers = {
        name: FrozenNormalizer.load(DIRECTORY / "normalizers" / f"{name}.json")
        for name in ("rgb", "event")
    }
    references = {}
    for name in NAMES:
        folder = DIRECTORY / "predictions/V" / name
        if sha(folder / "predictions.parquet") != read(folder / "COMPLETE.json")["sha256"]:
            raise ValueError("Cost reference predictions changed")
        references[name] = pd.read_parquet(folder / "predictions.parquet").prediction_ttc.to_numpy()
    setup(7)
    with gpu_charge("fusion_raw_pipeline_cost", reserve=900) as deadline:
        manifest = read(owner() / "V_MANIFEST.json")
        provenance = read(owner() / "fits/E_A5_MATCHED/RUN_PROVENANCE.json")
        runtime = RouteRuntime(
            run=owner(),
            config=ROOT / "configs/rgb_port/producers.json",
            p_manifest=owner() / "P_MANIFEST.json",
            manifest=owner() / "V_MANIFEST.json",
            event_source_root=Path(provenance["source_identity"]["event_source_root"]),
            event_history=owner() / "features/V/E_RAW/history.npz",
            normalizers=_normalizers(OUT / "NORMALIZERS.json"),
            device="cuda",
        )
        try:
            if len(runtime.rows) != manifest["population_size"]:
                raise ValueError("Raw runtime cohort changed")
            anchor = copy.deepcopy(runtime.heads["R_CTX"]).cpu().eval().requires_grad_(False)
            if not isinstance(anchor, TemporalRefiner):
                raise TypeError("Native R_CTX must retain its temporal architecture")
            event = load_model().cuda().eval().requires_grad_(False)
            models = {}
            for mode in ("TRUE", "ZERO"):
                head, _ = load_head(mode)
                models[f"FUSION-{mode}-S7"] = NormalizedFusion(
                    head,
                    normalizers["rgb"],
                    normalizers["event"],
                    blocks["rgb"].identity,
                    blocks["event"].identity,
                ).eval()

            def call(name: str, index: int) -> torch.Tensor:
                with patch(
                    "operational.rgb_port.cost_routes.ProducerObservation", NativeObservation
                ):
                    if name == "R_CTX":
                        return runtime.full_rgb(index)
                    if name == "F_TRUE":
                        return runtime.raw_call(index, "full_fusion", "F_TRUE")
                    batch = runtime.raw_event_source.batch([index], "event")
                    features, estimate = extract_pair(
                        event, batch.events[:, -2:].cuda(), batch.delta_t_s.reshape(-1).cuda()
                    )
                    context = runtime.rgb_context(index)
                    available = context is not None
                    if available:
                        rgb_features, rgb_ttc = extract_history(
                            anchor, *[item.cpu() for item in context]
                        )
                    else:
                        rgb_features, rgb_ttc = (
                            np.full((1, 28), np.nan, np.float32),
                            np.full(1, np.nan, np.float32),
                        )
                    return models[name](
                        torch.from_numpy(rgb_features),
                        torch.from_numpy(features),
                        torch.from_numpy(rgb_ttc),
                        torch.from_numpy(estimate),
                        torch.tensor([available]),
                    )["ttc"]

            with torch.inference_mode():
                if not (OUT / "FIRST_CALL.json").exists():
                    first = {}
                    for name in NAMES:
                        torch.cuda.synchronize()
                        tick = time.perf_counter()
                        output = call(name, indices[0])
                        torch.cuda.synchronize()
                        first[name] = {
                            "seconds": time.perf_counter() - tick,
                            "ttc": float(output.item()),
                        }
                    freeze(OUT / "FIRST_CALL.json", {"binding": digest(binding), "results": first})
                if not (OUT / "PREPARED_HEAD_ONLY.json").exists():
                    args = (
                        torch.from_numpy(blocks["rgb"].features[:1]),
                        torch.from_numpy(blocks["event"].features[:1]),
                        torch.from_numpy(blocks["rgb"].prediction_ttc[:1]),
                        torch.from_numpy(blocks["event"].prediction_ttc[:1]),
                        torch.ones(1, dtype=torch.bool),
                    )
                    prepared = {}
                    for name, model in models.items():
                        for _ in range(20):
                            model(*args)
                        samples = []
                        for _ in range(100):
                            tick = time.perf_counter()
                            model(*args)
                            samples.append(time.perf_counter() - tick)
                        prepared[name] = summary(samples)
                    freeze(
                        OUT / "PREPARED_HEAD_ONLY.json",
                        {"binding": digest(binding), "E2E": False, "results": prepared},
                    )
                for name in NAMES:
                    for i in range(binding["warmup_per_route"]):
                        call(name, indices[i % len(indices)])
                for number, (index, order) in enumerate(zip(indices, orders, strict=True)):
                    path = OUT / f"row_{number:03d}.json"
                    if path.exists():
                        if read(path)["binding"] != digest(binding):
                            raise ValueError("Fusion raw cost fragment changed")
                        continue
                    if time.perf_counter() >= deadline:
                        return {"status": "PARTIAL", "next_row": number}
                    require_resources()
                    timings = {}
                    for name in order:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        allocated = torch.cuda.memory_allocated()
                        tick = time.perf_counter()
                        output = call(name, index)
                        torch.cuda.synchronize()
                        elapsed, value = time.perf_counter() - tick, float(output.item())
                        expected = float(references[name][index])
                        parity = bool(
                            np.isclose(
                                value,
                                expected,
                                rtol=binding["parity_rtol"],
                                atol=binding["parity_atol"],
                            )
                        )
                        timings[name] = {
                            "seconds": elapsed,
                            "prediction": value,
                            "reference": expected,
                            "parity": parity,
                            "absolute_difference": abs(value - expected),
                            "peak_extra_allocated_bytes": torch.cuda.max_memory_allocated()
                            - allocated,
                            "process_rss_bytes": psutil.Process().memory_info().rss,
                        }
                    freeze(path, {"binding": digest(binding), "index": index, "timings": timings})
                    if not all(value["parity"] for value in timings.values()):
                        freeze(
                            OUT / "PARITY_FAILURE.json",
                            {"binding": digest(binding), "row": number, "timings": timings},
                        )
                        raise ValueError("Raw fusion/native route parity failed; preserve evidence")
        finally:
            runtime.close()
    fragments = [read(OUT / f"row_{i:03d}.json") for i in range(len(indices))]
    if not all(row["timings"][name]["parity"] for row in fragments for name in NAMES):
        raise ValueError("Cost cannot complete with a failed parity fragment")
    result = {
        "status": "COMPLETE",
        "binding": digest(binding),
        "E2E": True,
        "results": {
            name: summary([row["timings"][name]["seconds"] for row in fragments]) for name in NAMES
        },
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(0),
        "parity": "All selected native signed TTC outputs within predeclared tolerance",
        "scope": binding["scope"],
        "system_promotion": False,
    }
    freeze(OUT / "COMPLETE.json", result)
    return result


if __name__ == "__main__":
    print(measure())
