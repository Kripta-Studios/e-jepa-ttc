"""Run frozen V12/V13/Garl checkpoints on prospectively declared author intervals."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from e_jepa_ttc.rgb_port.fusion import DualClockFusion
from operational.evttc_rgb_transfer.model import FullGarl
from operational.evttc_transfer.inputs import prepare as prepare_v12
from operational.evttc_transfer.models import FrozenModels
from operational.rgb_port import transfer as engine
from operational.rgb_port.accounting import atomic_write_json, sha256_file
from operational.rgb_port_external_fix.contracts import validate
from operational.rgb_port_external_fix.runtime import observation_contract
from operational.rgb_port_fusion_fix.fusion import deterministic_forward

from .inputs import PROTOCOL, build_manifest, garl_inputs, rgb_observations, validate_row


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    files = [
        *Path(__file__).parent.glob("*.py"),
        root / "operational/evttc_rgb_transfer/model.py",
        root / "operational/evttc_transfer/models.py",
        root / "operational/evttc_transfer/inputs.py",
        root / "src/e_jepa_ttc/data/garl_official_preprocessing.py",
    ]
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in files}


def own(run: Path, output: Path, device: str) -> dict:
    """Apply only input adapters to the exact existing V13 endpoint engine."""
    validate(run)
    manifest_path = output / "QUERY_MANIFEST.json"
    manifest = read(manifest_path)
    source = {**engine._source_bindings(), **source_hashes()}
    contract = {**PROTOCOL, "camera": "blackflys/left", "label_access": False}
    destination = output / "v13"
    started: dict[str, float] = {}
    prepare = engine.prepare_event
    fragment_values = engine._fragment_values

    def timed_prepare(row: dict) -> dict:
        validate_row(row)
        started[row["query_id"]] = time.perf_counter()
        return prepare(row)

    def timed_fragment(prediction: dict, history: list) -> dict:
        query = prediction["query_id"]
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started[query]) * 1000
        atomic_write_json(
            destination / "timings" / (query.replace(":", "_") + ".json"),
            {
                "query_id": query,
                "family_all_six_heads_total_ms": elapsed,
                "scope": "raw_inputs_through_all_six_heads_excludes_loading_and_fragment_write",
                "isolated_host": False,
            },
        )
        print(f"V13 {query} {elapsed:.1f}ms", flush=True)
        return fragment_values(prediction, history)

    with ExitStack() as stack:
        stack.enter_context(observation_contract())
        stack.enter_context(patch.object(DualClockFusion, "forward", deterministic_forward))
        stack.enter_context(
            patch.object(engine, "_manifest", lambda _: (manifest, manifest["rows"]))
        )
        stack.enter_context(patch.object(engine, "prepare_event", timed_prepare))
        stack.enter_context(patch.object(engine, "raw_rgb_observations", rgb_observations))
        stack.enter_context(patch.object(engine, "_source_bindings", lambda: source))
        stack.enter_context(patch.object(engine, "OWN_RGB_PREPROCESSING", contract))
        stack.enter_context(patch.object(engine, "_fragment_values", timed_fragment))
        return engine.predict_dev32(
            run=run,
            config=Path("configs/rgb_port/producers.json"),
            p_manifest=run / "P_MANIFEST.json",
            endpoint_freeze=run / "rounds/PRODUCERS_PAIR_ENDPOINT_FREEZE.json",
            query_manifest=manifest_path,
            event_normalizer=run / "features/H/E_NORMALIZER.json",
            rgb_normalizer=run / "features/H/R_NORMALIZER.json",
            output=destination,
            device=device,
        )


def finite(value: object) -> float | None:
    if not isinstance(value, (float, int)):
        raise TypeError("Expected scalar prediction")
    result = float(value)
    return result if math.isfinite(result) else None


def baselines(output: Path, v12: Path, public: Path, code_root: Path, device: str) -> dict:
    """Infer every declared query with native full Garl and V12; never load TTC."""
    manifest = read(output / "QUERY_MANIFEST.json")
    full = FullGarl(public, code_root, device=device)
    legacy = FrozenModels(v12 / "artifacts/train40_system_20261005", device=device)
    binding = {
        "manifest_sha256": sha256_file(output / "QUERY_MANIFEST.json"),
        "sources": source_hashes(),
        "Garl_full": full.bindings,
        "V12": legacy.bindings,
        "protocol": PROTOCOL,
        "device": device,
        "precision": "float32",
        "labels_read": False,
    }
    binding_path = output / "BASELINES_BINDING.json"
    if binding_path.exists() and read(binding_path) != binding:
        raise ValueError("Baseline binding changed; preserve run and use a new output directory")
    atomic_write_json(binding_path, binding)
    fragments = {}
    for index, row in enumerate(manifest["rows"]):
        path = output / "baselines" / f"query_{index:04d}.json"
        if path.exists():
            saved = read(path)
            if saved["query_id"] != row["query_id"] or saved["binding_sha256"] != sha256_file(
                binding_path
            ):
                raise ValueError("Baseline fragment identity changed")
        else:
            saved = {
                "query_id": row["query_id"],
                "binding_sha256": sha256_file(binding_path),
                "predictions": {},
                "failures": {},
                "timings_ms": {},
            }
            validate_row(row)
            started = time.perf_counter()
            try:
                rgb, events, meta = garl_inputs(row)
            except ValueError as exc:
                saved["failures"]["Garl_full"] = str(exc)
                saved["failures"]["Garl_event"] = str(exc)
            else:
                saved["input_audit"] = meta
                prepared = time.perf_counter()
                result = full.predict(rgb, events)
                finished = time.perf_counter()
                saved["predictions"]["Garl_full"] = finite(result["ttc"])
                saved["Garl_full_heights"] = result["heights"]
                saved["timings_ms"]["Garl_full_prepare"] = (prepared - started) * 1000
                saved["timings_ms"]["Garl_full_forward"] = (finished - prepared) * 1000
                saved["timings_ms"]["Garl_full_total"] = (finished - started) * 1000
                result = legacy.garl_predict(events.reshape(40, 128, 128))
                saved["predictions"]["Garl_event"] = finite(result["ttc"])
            started = time.perf_counter()
            try:
                prepared_v12 = prepare_v12(row)
            except ValueError as exc:
                saved["failures"]["V12_H8"] = str(exc)
            else:
                prediction = legacy.predict(prepared_v12["own_events"], valid=prepared_v12["valid"])
                saved["predictions"].update({k: finite(v) for k, v in prediction.items()})
                saved["predictions"]["H8_median3"] = finite(statistics.median(prediction.values()))
                saved["timings_ms"]["V12_H8_total"] = (time.perf_counter() - started) * 1000
            atomic_write_json(path, saved)
            print(f"Baselines {index + 1}/{len(manifest['rows'])} {row['query_id']}", flush=True)
        fragments[path.relative_to(output).as_posix()] = sha256_file(path)
    receipt = {
        "status": "COMPLETE",
        "population": len(manifest["rows"]),
        "labels_read": False,
        "optimizer_updates": 0,
        "binding_sha256": sha256_file(binding_path),
        "fragments": fragments,
    }
    atomic_write_json(output / "BASELINES_SEALED.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "v13", "baselines"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v12", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=Path("artifacts/rgb_port_20261008"))
    parser.add_argument("--garl-code", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "prepare":
        manifest = build_manifest(
            args.v12 / "artifacts/evttc_transfer_20261008/QUERY_MANIFEST.json",
            args.output / "QUERY_MANIFEST.json",
        )
        print(
            json.dumps(
                {"queries": len(manifest["rows"]), "candidates": len(manifest["candidates"])}
            )
        )
    elif args.stage == "v13":
        own(args.run, args.output, args.device)
    else:
        if args.garl_code is None:
            parser.error("--garl-code required for baselines")
        baselines(
            args.output,
            args.v12,
            args.v12 / "artifacts/evttc_rgb_transfer_20261008/public_garl",
            args.garl_code,
            args.device,
        )
    gc.collect()


if __name__ == "__main__":
    main()
