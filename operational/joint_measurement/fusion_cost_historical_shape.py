"""Measure raw fusion cost with the historical producer arithmetic and batch shapes."""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import patch

from operational.range_transfer.common import freeze, read, sha, write

from . import fusion_cost
from . import fusion_cost_batch_inventory as inventory
from .common import ROOT, RUN
from .compute_guard import guard_idle
from .fusion_cost_native import admit, native_context, qa
from .fusion_cost_shape_probe import ShapeRuntime
from .train import committed_sources

OUT = RUN / "fusion/cost/raw_e2e_historical_shape_v1"
INVENTORY = OUT / "inventory"
ORIGINAL = RUN / "fusion/cost/raw_e2e_native_v2/REGISTRATION.json"
DIAGNOSTIC = RUN / "fusion/cost/shape_diagnostic_v1/RESULT.json"


def sources() -> dict[str, str]:
    """Bind the overlay and unchanged numerical/runtime implementations."""
    return {
        f"operational/joint_measurement/{name}.py": sha(
            ROOT / f"operational/joint_measurement/{name}.py"
        )
        for name in (
            "fusion_cost_historical_shape",
            "fusion_cost",
            "fusion_cost_batch_inventory",
            "fusion_cost_shape_probe",
            "fusion_cost_backend_probe",
            "fusion_cost_native",
            "fusion_cost_native_v2",
            "compute_guard",
        )
    }


def shape_map(result: dict, expected: list[int]) -> dict[str, dict[str, dict]]:
    """Reject partial coverage and contradictory shapes for repeated observations."""
    if result["status"] != "COMPLETE_METADATA_AUDIT" or result["indices"] != expected:
        raise ValueError("Historical inventory does not cover the registered cost population")
    mapped = {}
    for modality in ("E", "R"):
        queries = result["modalities"][modality]["queries"]
        if [query["index"] for query in queries] != expected:
            raise ValueError("Historical query order/coverage changed")
        items = {}
        for query in queries:
            for item in query["observations"]:
                if item["id"] in items and items[item["id"]] != item:
                    raise ValueError("Conflicting historical observation shapes")
                if item["input_frames"] not in (2, 3) or not 1 <= item["forward_batch"] <= 32:
                    raise ValueError("Invalid historical producer shape")
                items[item["id"]] = item
        mapped[modality] = items
    return mapped


def register() -> dict:
    """Fix one faithful runtime, preserving every original cost query and tolerance."""
    committed_sources(sources())
    diagnostic = read(DIAGNOSTIC)
    rows = diagnostic["measurements"]
    restored = [row for row in rows if row["producer_cudnn_tf32"] is True]
    if (
        diagnostic["status"] != "COMPLETE_DIAGNOSTIC"
        or len(restored) != 4
        or len({row["index"] for row in restored}) != 4
        or not all(row["original_tolerance_pass"] for row in restored)
    ):
        raise ValueError("All four original diagnostic queries must pass unchanged tolerance")
    original = read(ORIGINAL)
    record = {
        "sources": sources(),
        "original_cost_registration_sha256": sha(ORIGINAL),
        "diagnostic_sha256": sha(DIAGNOSTIC),
        "indices": original["indices"],
        "tokens": original["tokens"],
        "orders": original["orders"],
        "precision": "FP32; historical producer cuDNN TF32 enabled; all other TF32 disabled",
        "internal_batch": "Repeat each observation to its historical producer batch, retain row0",
        "external_batch_size": 1,
        "tolerances": {key: original[key] for key in ("parity_rtol", "parity_atol")},
        "scope": original["scope"],
        "warmup_per_route": original["warmup_per_route"],
        "unchanged": "Weights, pixels, clocks, ROI, normalizers, heads, signed TTC and population",
        "interpretation": (
            "Cost of reproducing historical predictions, including repeated internal compute; "
            "not a speedup or minimum achievable batch-one latency. Four-query diagnostic "
            "does not admit this full-population cost; every measured output must pass."
        ),
        "optimizer_updates": 0,
        "system_promotion": False,
    }
    freeze(OUT / "RUNTIME_REGISTRATION.json", record)
    return record


def build_inventory() -> dict:
    """Reuse the fragment verifier with all original cost indices, never a favorable subset."""
    record = register()
    path = INVENTORY / "RESULT.json"
    if path.exists():
        result = read(path)
    else:
        prior = RUN / "fusion/cost/parity_diagnostic_v1/REGISTRATION.json"

        def expanded_read(candidate: Path) -> dict:
            value = read(candidate)
            if candidate == prior:
                value = {**value, "indices": record["indices"]}
            return value

        with (
            patch.object(inventory, "OUT", INVENTORY),
            patch.object(inventory, "read", expanded_read),
        ):
            result = inventory.run()
    shape_map(result, record["indices"])
    freeze(
        INVENTORY / "SCOPE.json",
        {
            "runtime_registration_sha256": sha(OUT / "RUNTIME_REGISTRATION.json"),
            "result_sha256": sha(path),
            "effective_indices": record["indices"],
            "original_four_query_registration_retained": True,
            "labels_read": False,
            "optimizer_updates": 0,
        },
    )
    return result


def bind_cost(binding: dict, registration_sha: str, inventory_sha: str) -> None:
    """Describe the actual backend and repeated work in the numerical cost binding."""
    binding.update(
        precision="FP32; historical producer cuDNN TF32 enabled; other TF32 disabled",
        baseline_runtime=(
            "Historical native arithmetic and per-observation original producer batch shape"
        ),
        producer_batch_policy=(
            "Repeat current input to historical batch; retain first output; charge all work"
        ),
        runtime_registration_sha256=registration_sha,
        historical_inventory_sha256=inventory_sha,
    )


def run() -> dict:
    """Measure resumable paired routes under the existing exclusive resource controls."""
    guard_idle()
    record = register()
    result = build_inventory()
    mapping = shape_map(result, record["indices"])
    native = admit()
    qa(native)

    class HistoricalRuntime(ShapeRuntime):
        convolution_tf32 = True
        shape_map = mapping

    def bound_freeze(path: Path, value: dict) -> None:
        if path == OUT / "REGISTRATION.json":
            if any(value[key] != record[key] for key in ("indices", "tokens", "orders")):
                raise ValueError("Original cost population or paired order changed")
            bind_cost(value, sha(OUT / "RUNTIME_REGISTRATION.json"), sha(INVENTORY / "RESULT.json"))
        freeze(path, value)

    with (
        native_context(native),
        patch.object(fusion_cost, "OUT", OUT),
        patch.object(fusion_cost, "RouteRuntime", HistoricalRuntime),
        patch.object(fusion_cost, "freeze", bound_freeze),
    ):
        measured = fusion_cost.measure()
    write(
        OUT / "LATEST.json",
        {
            "status": measured["status"],
            "runtime_registration_sha256": sha(OUT / "RUNTIME_REGISTRATION.json"),
            "cost_result": measured,
            "optimizer_updates": 0,
        },
    )
    return measured


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("register", "inventory", "run"))
    mode = parser.parse_args().mode
    result = {"register": register, "inventory": build_inventory, "run": run}[mode]()
    print({key: result.get(key) for key in ("status", "indices", "next_row")})
