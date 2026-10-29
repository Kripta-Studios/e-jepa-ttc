"""Paired raw-HDF5-to-TTC latency against the unchanged historical E_CTX pipeline."""

from __future__ import annotations

import argparse
import hashlib
import socket
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import psutil
import torch

from e_jepa_ttc.rgb_port.normalization import FrozenNormalizer
from operational.range_transfer.common import digest, freeze, read, sha, utc, write
from operational.rgb_port.cost_routes import RouteRuntime, _event_histories, _normalizers
from operational.rgb_port.infer_experts import _parents, _producer
from operational.rgb_port.train_heads import load_head_endpoint
from operational.rgb_port_streaming.contracts import NativeObservation

from .common import ROOT, RUN, config, gpu_charge, owner, require_resources
from .evaluate import freeze_primary
from .export import TTCOnly
from .raw_pair import RawPairSource
from .train import initialize, setup


class EventOnlyRuntime(RouteRuntime):
    """Reuse the full historical event route with exactly its four learned components."""

    def __init__(
        self, source: RawPairSource, normalizers: dict[str, FrozenNormalizer], device: str
    ) -> None:
        if source.role != "V":
            raise ValueError("Historical cost runtime requires its admitted V history")
        self.run = owner()
        self.device = torch.device(device)
        self.event_source = source.prepared
        self.raw_event_source = source.raw
        self.rows = source.rows.to_dict(orient="records")
        self.normalizers = normalizers
        self.producers = {
            name: _producer(
                owner(),
                ROOT / "configs/rgb_port/producers.json",
                owner() / "P_MANIFEST.json",
                name,
                device,
            )
            for name in ("E_A5_MATCHED", "E_C2F_MATCHED")
        }
        self.pairs = {
            "PAIR_E_MATCHED": load_head_endpoint(
                owner() / "fits/PAIR_E_MATCHED", "PAIR_E_MATCHED", self.device
            )
        }
        self.heads = {
            "E_CTX_MATCHED": load_head_endpoint(
                owner() / "fits/E_CTX_MATCHED", "E_CTX_MATCHED", self.device
            )
        }
        self.event_histories = _event_histories(owner() / "features/V/E_RAW/history.npz", self.rows)

    def close(self) -> None:
        """Input handles are borrowed and remain owned by the surrounding RawPairSource."""

    def _event_observation(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Use the existing native current-TTC mask adapter; leave all learned outputs intact."""
        with patch("operational.rgb_port.cost_routes.ProducerObservation", NativeObservation):
            return super()._event_observation(index)


def indices_by_group(rows: pd.DataFrame) -> list[int]:
    """Choose eight queries per group by token hash, without consulting targets."""
    if not rows.index.equals(pd.RangeIndex(len(rows))):
        raise ValueError("Raw-cost rows must retain their canonical ordinal index")
    selected = []
    for _, frame in rows.groupby("group_id", sort=True):
        ranked = sorted(
            frame.index.tolist(),
            key=lambda i: hashlib.sha256(str(rows.iloc[i].sample_token).encode()).hexdigest(),
        )
        if len(ranked) < 8:
            raise ValueError("Raw-cost group has fewer than eight queries")
        selected.extend(ranked[:8])
    return np.random.default_rng(20261012).permutation(selected).tolist()


def measure() -> dict:
    """Resume frozen raw routes; costs cannot change primary H admission retrospectively."""
    endpoints = freeze_primary()["endpoints"]
    selection_path = RUN / "PRIMARY_SELECTION_FINAL.json"
    selection = read(selection_path)
    if selection.get("system_promotion") is not False:
        raise ValueError("Complete prospective primary selection before opening V routes")
    require_resources(starting=True)
    directory = RUN / "cost/primary_raw_e2e"
    mappings = {
        name: str(owner() / f"features/H/{tag}_NORMALIZER.json")
        for name, tag in (("event", "E"), ("rgb", "R"))
    }
    freeze(directory / "NORMALIZERS.json", mappings)
    normalizers = _normalizers(directory / "NORMALIZERS.json")
    source = RawPairSource("V")
    names = ["A5-P2-FROZEN", *config()["fits"], "E_CTX_MATCHED"]
    indices = indices_by_group(source.rows)
    generator = np.random.default_rng(181012)
    orders = [generator.permutation(names).tolist() for _ in indices]
    historical = _parents(
        owner(), ["E_A5_MATCHED", "E_C2F_MATCHED", "PAIR_E_MATCHED", "E_CTX_MATCHED"]
    )
    history = owner() / "features/V/E_RAW/history.npz"
    binding = {
        "endpoints": endpoints,
        "primary_selection_sha256": sha(selection_path),
        "historical_parents": historical,
        "normalizer_hashes": {name: sha(Path(path)) for name, path in mappings.items()},
        "history_sha256": sha(history),
        "source_identity": source.identity,
        "evaluator_sha256": sha(Path(__file__)),
        "historical_routes_sha256": sha(ROOT / "operational/rgb_port/cost_routes.py"),
        "native_mask_contract_sha256": sha(ROOT / "operational/rgb_port_streaming/contracts.py"),
        "indices": indices,
        "orders": orders,
        "tokens": source.rows.iloc[indices].sample_token.tolist(),
        "group_ids": source.rows.iloc[indices].group_id.tolist(),
        "batch_size": 1,
        "warmup_per_route": 20,
        "scope": "Raw HDF5, common-ROI preprocessing, host-to-device, model and signed TTC",
        "weights_loading_in_route_latency": False,
        "weights_loading_in_gpu_budget": True,
        "prepared_voxels_in_timed_routes": False,
        "cache_policy": "Shared bounded two-reader HDF5 cache; OS cache uncontrolled",
        "context_difference": (
            "Primary models use P2; raw preprocessing still computes native T3 where present. "
            "Historical E_CTX retains its own native history and experts."
        ),
        "precision": "FP32_TF32_OFF",
    }
    try:
        freeze(directory / "REGISTRATION.json", binding)
        if (directory / "COMPLETE.json").exists():
            return read(directory / "COMPLETE.json")
        parity_path = directory / "RAW_PARITY.json"
        if parity_path.exists():
            if read(parity_path)["binding"] != digest(binding):
                raise ValueError("Raw preprocessing parity binding changed")
        else:
            for index in indices:
                require_resources()
                prepared = source.prepared.batch([index], "event")
                raw = source.raw.batch([index], "event")
                if not torch.equal(prepared.events, raw.events) or not torch.equal(
                    prepared.delta_t_s, raw.delta_t_s
                ):
                    raise ValueError(f"Raw historical preprocessing differs at V index {index}")
            freeze(
                parity_path,
                {
                    "status": "PASS",
                    "binding": digest(binding),
                    "queries": len(indices),
                    "full_native_tensors_exact": True,
                    "delta_exact": True,
                },
            )
        setup(7)
        with gpu_charge("primary_raw_e2e_cost", reserve=900) as deadline:
            models = {}
            for name in names[:-1]:
                model = initialize(name)
                if name != "A5-P2-FROZEN":
                    payload = torch.load(
                        RUN / "fits" / name / "checkpoint_last.pt",
                        map_location="cpu",
                        weights_only=False,
                    )
                    model.load_state_dict(payload["model_state_dict"], strict=True)
                    del payload
                models[name] = TTCOnly(model).cuda().eval().requires_grad_(False)
            runtime = EventOnlyRuntime(source, normalizers, device="cuda")
            try:

                def call(name: str, index: int) -> torch.Tensor:
                    if name == "E_CTX_MATCHED":
                        return runtime.raw_call(index, "full_event")
                    events, delta = source.batch([index])
                    return models[name](events.cuda(), delta.cuda())

                with torch.inference_mode():
                    warmup_start = time.perf_counter()
                    for name in names:
                        for position in range(binding["warmup_per_route"]):
                            call(name, indices[position % len(indices)])
                    torch.cuda.synchronize()
                    write(
                        directory / f"WARMUP_{time.time_ns()}.json",
                        {"utc": utc(), "seconds": time.perf_counter() - warmup_start},
                    )
                    for number, (index, order) in enumerate(zip(indices, orders, strict=True)):
                        path = directory / f"row_{number:03d}.json"
                        if path.exists():
                            if read(path)["binding"] != digest(binding):
                                raise ValueError("Raw-cost fragment binding changed")
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
                            duration = time.perf_counter() - tick
                            if not torch.isfinite(output).all():
                                raise ValueError(f"Nonfinite raw route output: {name}")
                            timings[name] = {
                                "seconds": duration,
                                "ttc_seconds": float(output.reshape(-1)[0]),
                                "peak_extra_allocated_bytes": (
                                    torch.cuda.max_memory_allocated() - allocated
                                ),
                                "process_rss_bytes": psutil.Process().memory_info().rss,
                            }
                        freeze(
                            path,
                            {
                                "binding": digest(binding),
                                "V_index": index,
                                "timings": timings,
                                "utc": utc(),
                            },
                        )
            finally:
                runtime.close()
        frames = [read(directory / f"row_{i:03d}.json") for i in range(len(indices))]
        summary = {}
        for name in names:
            seconds = np.array([row["timings"][name]["seconds"] for row in frames])
            summary[name] = {
                "p50_seconds": float(np.quantile(seconds, 0.5)),
                "p95_seconds": float(np.quantile(seconds, 0.95)),
                "p99_seconds": float(np.quantile(seconds, 0.99)),
                "mean_seconds": float(seconds.mean()),
                "n": len(seconds),
            }
        result = {
            "status": "COMPLETE",
            "binding": digest(binding),
            "results": summary,
            "E2E": True,
            "official_comparison": False,
            "system_promotion": False,
            "hardware": torch.cuda.get_device_name(0),
            "host": socket.gethostname(),
            "scope": binding["scope"],
        }
        freeze(directory / "COMPLETE.json", result)
        return result
    finally:
        source.close()


def verify_historical_cpu() -> dict:
    """Compare raw and prepared inputs through the unchanged historical event components."""
    require_resources(starting=True)
    torch.set_num_threads(1)
    path = RUN / "diagnostics/RAW_ECTX_NORMALIZERS.json"
    mappings = {
        name: str(owner() / f"features/H/{tag}_NORMALIZER.json")
        for name, tag in (("event", "E"), ("rgb", "R"))
    }
    write(path, mappings)
    source = RawPairSource("V")
    try:
        runtime = EventOnlyRuntime(source, _normalizers(path), "cpu")
        checks = []
        for row in read(RUN / "verification/raw_pair/V.json")["checks"]:
            require_resources()
            index = row["role_index"]
            with torch.inference_mode():
                prepared = runtime.full_event(index)
                raw = runtime.raw_call(index, "full_event")
            torch.testing.assert_close(raw, prepared, rtol=0, atol=0)
            checks.append(
                {
                    "V_index": index,
                    "group_id": row["group_id"],
                    "bit_exact": torch.equal(raw, prepared),
                }
            )
            print(f"RAW_ECTX_PARITY {index} PASS", flush=True)
        result = {
            "utc": utc(),
            "status": "PASS",
            "checks": checks,
            "source_sha256": sha(Path(__file__)),
            "optimizer_updates": 0,
            "scope": (
                "Raw versus prepared inputs through all four unchanged E_CTX components, "
                "selected V queries, CPU. Native mask adapter reused; no metric evaluation."
            ),
        }
        write(RUN / "verification/RAW_ECTX_CPU_PARITY.json", result)
        return result
    finally:
        source.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-historical-cpu", action="store_true")
    print(verify_historical_cpu() if parser.parse_args().verify_historical_cpu else measure())
