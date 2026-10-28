"""Frozen nine-component TRAIN40 inference on label-blind EvTTC sensor rows.

The portable seal fixes the weights and normalizers. The existing EvTTC sensor
adapter supplies dense causal histories; the final Test12 predictor supplies
the unchanged padded context, PHASE_ONLY head and event-only fallback.
"""

# ruff: noqa: E402 -- bind this checkout before importing its scientific modules

from __future__ import annotations

import argparse
import hashlib
import inspect
import io
import json
import os
import platform
import sys
import tarfile
import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.final_refit.recipe import canonical_sha256, load_recipe, stage
from operational.final_refit.test12 import FinalV16Train40Predictor
from operational.final_refit.training import _model
from operational.rgb_port import transfer as engine
from operational.rgb_port_streaming.contracts import NativeObservation

HEAD = "PHASE_ONLY_EVENT_S7_TRAIN40"
CANDIDATE = "PHASE_ONLY_EVENT_S7_TRAIN40_SCIENTIFIC_OOF"
MODEL_IDS = ("E_A5", "E_C2F", "R_A5", "R_C2F", "PAIR_E", "PAIR_R", "E_CTX_MATCHED", "R_CTX", HEAD)
FORBIDDEN = {
    "target",
    "target_phase",
    "target_ttc",
    "ttc",
    "gt_ttc",
    "ground_truth",
    "truth_ttc_seconds",
    "depth",
    "velocity",
}


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_once(path: Path, value: dict[str, Any]) -> None:
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"Refusing to replace a different frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def reject_targets(value: Any) -> None:
    if isinstance(value, dict):
        if FORBIDDEN.intersection(str(key).lower() for key in value):
            raise ValueError("Inference manifest contains a forbidden target field")
        for item in value.values():
            reject_targets(item)
    elif isinstance(value, list):
        for item in value:
            reject_targets(item)


def verify_candidate(root: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    """Bind local files to the immutable cloud seal without rewriting its paths."""
    root = root.resolve(strict=True)
    paths = read(root / "PORTABLE_PATHS.json")
    if paths.get("schema") != "ejepa_final_candidate_local_paths_v1":
        raise ValueError("Unknown portable candidate inventory")
    components = paths["components"]
    local: dict[str, Path] = {}
    needed = (
        *MODEL_IDS,
        "event_normalizer",
        "rgb_normalizer",
        "MODEL_SEAL.json",
        "OOF_RECIPE.json",
    )
    for name in needed:
        entry = components[name]
        file = Path(entry["local_path"]).resolve(strict=True)
        if not file.is_relative_to(root) or sha(file) != entry["sha256"]:
            raise ValueError(f"Portable component bytes or location differ: {name}")
        local[name] = file
    seal = read(local["MODEL_SEAL.json"])
    unsigned = {key: value for key, value in seal.items() if key != "seal_identity_sha256"}
    recipe = load_recipe(local["OOF_RECIPE.json"])
    if (
        seal.get("schema") != "scientific_oof_final_model_seal_v1"
        or seal.get("candidate") != CANDIDATE
        or seal.get("status") != "SEALED_NOT_EVALUATED"
        or seal.get("ancestry_domains_are_distinct") is not True
        or seal.get("scope", {}).get("private_targets_read") is not False
        or seal.get("seal_identity_sha256") != canonical_sha256(unsigned)
        or seal.get("recipe_sha256") != recipe.recipe_sha256
    ):
        raise ValueError("Portable candidate does not match its scientific seal")
    expected = dict(seal["deployment_dependencies"])
    expected[HEAD] = seal["trained_endpoints"][HEAD]["checkpoint_sha256"]
    for name in MODEL_IDS:
        if sha(local[name]) != expected[name]:
            raise ValueError(f"Checkpoint differs from sealed deployment: {name}")
    for modality in ("event", "rgb"):
        if sha(local[f"{modality}_normalizer"]) != seal["normalizers"][modality]:
            raise ValueError(f"Normalizer differs from sealed deployment: {modality}")
    binding = {
        "candidate": CANDIDATE,
        "model_seal_sha256": sha(local["MODEL_SEAL.json"]),
        "recipe_sha256": recipe.recipe_sha256,
        "components": {name: sha(path) for name, path in local.items()},
        "portable_paths_sha256": sha(root / "PORTABLE_PATHS.json"),
    }
    return local, binding


def load_models(paths: dict[str, Path], device: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Strict-load the exact nine sealed states using their embedded architecture."""
    recipe = load_recipe(paths["OOF_RECIPE.json"])
    models = {}
    aliases = {"E_A5": "E_A5_MATCHED", "E_C2F": "E_C2F_MATCHED", "PAIR_E": "PAIR_E_MATCHED"}
    for name in MODEL_IDS:
        payload = torch.load(paths[name], map_location="cpu", weights_only=False)
        if payload.get("status") != "COMPLETE":
            raise ValueError(f"Checkpoint is not complete: {name}")
        if name in {"E_A5", "E_C2F"}:
            config = payload["contract"]["model_config"]
            if payload.get("committed_updates") != payload["contract"]["updates_limit"]:
                raise ValueError("Event checkpoint has not reached its fixed endpoint")
            model = CausalScaleTTC(CausalScaleTTCConfig(**config))
        elif name in {"R_A5", "R_C2F"}:
            identity = payload["identity"]
            if (
                identity.get("fit_id") != name
                or payload["completed_updates"] != identity["updates"]
            ):
                raise ValueError("RGB endpoint identity differs")
            model = CausalScaleTTC(CausalScaleTTCConfig(**identity["model_config"]))
        else:
            identity = payload["identity"]
            fit = aliases.get(name, name)
            if (
                identity.get("stage") != fit
                or identity.get("technical") is not False
                or identity.get("recipe_sha256") != recipe.recipe_sha256
            ):
                raise ValueError("Cached head identity differs from the sealed recipe")
            model = _model(stage(recipe, fit))
        model.load_state_dict(payload["model_state_dict"], strict=True)
        models[aliases.get(name, name)] = model.float().eval().requires_grad_(False).to(device)
        del payload
    normalizers = {}
    for modality in ("event", "rgb"):
        norm = read(paths[f"{modality}_normalizer"])
        mean = torch.tensor(norm["mean"], dtype=torch.float32, device=device)
        scale = torch.tensor(norm["scale"], dtype=torch.float32, device=device)
        if mean.shape != (17,) or scale.shape != (17,) or not bool((scale > 0).all()):
            raise ValueError("Normalizer shape or scale differs")
        normalizers[modality] = mean, scale
    return models, normalizers


def verify_frozen_runtime(candidate: Path) -> dict[str, Any]:
    """Check inference source bytes against the runtime preserved from Pod2."""
    preservation = read(candidate.parent / "LOCAL_FINAL_PRESERVATION_COMPLETE.json")
    entry = preservation["groups"]["frozen_code"]
    archive = Path(entry["archive"])
    if sha(archive) != entry["sha256"]:
        raise ValueError("Preserved cloud source archive differs")
    names = (
        "src/e_jepa_ttc/models/causal_scale_ttc.py",
        "src/e_jepa_ttc/models/local_transport.py",
        "src/e_jepa_ttc/rgb_port/features.py",
        "src/e_jepa_ttc/simplex_t/model.py",
        "src/e_jepa_ttc/simplex_t/phase.py",
        "operational/final_refit/test12.py",
        "operational/final_refit/training.py",
        "operational/final_refit/features.py",
        "operational/rgb_port/train_producers.py",
        "operational/rgb_port/transfer.py",
        "operational/rgb_port_revision/transfer_inputs.py",
        "operational/rgb_port_streaming/contracts.py",
    )
    hashes = {}
    with tarfile.open(archive) as source:
        members = source.getmembers()
        for name in names:
            matches = [item for item in members if item.isfile() and item.name.endswith("/" + name)]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous frozen source: {name}")
            remote_bytes = source.extractfile(matches[0]).read()
            checksum = hashlib.sha256(remote_bytes).hexdigest()
            if sha(ROOT / name) != checksum:
                raise ValueError(
                    f"Local inference source differs from sealed cloud runtime: {name}"
                )
            hashes[name] = checksum
    return {"archive_sha256": entry["sha256"], "source_sha256": hashes}


def verify_saved_head_parity(candidate: Path, head: Any, samples: int = 8) -> dict[str, Any]:
    """Replay only the tiny head on saved Test12 states; never reopen sensors or GT."""
    preservation = read(candidate.parent / "LOCAL_FINAL_PRESERVATION_COMPLETE.json")
    entry = preservation["groups"]["test12_predictions_3RUZHwnCwK"]
    archive = Path(entry["archive"])
    if sha(archive) != entry["sha256"]:
        raise ValueError("Preserved prediction diagnostics archive changed")
    results = []
    with tarfile.open(archive) as source:
        members = {item.name: item for item in source.getmembers() if item.isfile()}
        for name in sorted(members):
            if "/diagnostics/" not in name or not name.endswith(".json"):
                continue
            metadata = json.loads(source.extractfile(members[name]).read())
            if metadata["fallback_source"] != "PHASE_ONLY":
                continue
            payload = source.extractfile(members[name.removesuffix(".json") + ".npz"]).read()
            if hashlib.sha256(payload).hexdigest() != metadata["npz_sha256"]:
                raise ValueError("Saved head diagnostic tensor changed")
            with np.load(io.BytesIO(payload), allow_pickle=False) as stored:
                state = torch.from_numpy(stored[f"{HEAD}.input_0"])
                widths = torch.from_numpy(stored[f"{HEAD}.input_1"])
                with torch.inference_mode():
                    prediction = head(state, widths)
                maximum = 0.0
                for key, tensor in prediction.items():
                    reference = torch.from_numpy(stored[f"{HEAD}.output_{key}"])
                    torch.testing.assert_close(tensor, reference, rtol=1e-5, atol=1e-5)
                    maximum = max(maximum, float((tensor - reference).abs().max()))
                if not np.isclose(
                    float(prediction["ttc"][0]), metadata["prediction"], rtol=1e-5, atol=1e-5
                ):
                    raise ValueError("Portable head differs from persisted Test12 TTC")
                results.append({"sample_token": metadata["sample_token"], "max_abs": maximum})
            if len(results) == samples:
                break
    if len(results) != samples:
        raise ValueError("Insufficient saved PHASE_ONLY diagnostic states")
    return {
        "status": "PASS",
        "scope": "HEAD_ONLY_SAVED_STATE340_AND_WIDTHS_NO_SENSOR_INFERENCE",
        "samples": results,
        "rtol": 1e-5,
        "atol": 1e-5,
        "device": "cpu",
        "archive_sha256": entry["sha256"],
        "targets_read": False,
        "optimizer_updates": 0,
    }


class FinalTransferPredictor:
    """Adapt EvTTC sensor observations to the unchanged final model context ABI."""

    def __init__(self, paths: dict[str, Path], device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.models, self.normalizers = load_models(paths, device)

    def predict(self, row: dict[str, Any]) -> dict[str, Any]:
        reject_targets(row)
        start = time.perf_counter()
        raw = engine.prepare_event(row)
        rgb, reason = engine.raw_rgb_observations(row)
        prepared = time.perf_counter()
        # Populate only physical observations computed from this query's causal
        # sensor windows; evaluation query subsampling never shortens history.
        native = FinalV16Train40Predictor.__new__(FinalV16Train40Predictor)
        native.device, native.models, native.normalizers = (
            self.device,
            self.models,
            self.normalizers,
        )
        native.event_values, native.rgb_values = {}, {}
        native.inputs = {"event_observations": {}, "rgb_observations": {}}
        query_us = int(row["anchor_us"])
        query = {"query_time_us": query_us, "event_history": [], "rgb_history": []}
        outputs = {}

        def capture(name: str) -> Any:
            def hook(module: Any, inputs: Any, result: Any) -> None:
                del module, inputs
                outputs[name] = result

            return hook

        hooks = [
            self.models[name].register_forward_hook(capture(name))
            for name in ("E_CTX_MATCHED", "R_CTX", HEAD)
        ]
        try:
            with (
                torch.inference_mode(),
                patch.object(engine, "ProducerObservation", NativeObservation),
            ):
                sensor = torch.from_numpy(raw["own_events"]).to(self.device)
                delta = torch.full((len(engine.EVENT_LAGS_US), 2), 0.1, device=self.device)
                eraw, experts = engine._expert_block(self.models, "E", sensor, delta)
                for index, lag in enumerate(engine.EVENT_LAGS_US):
                    identity = f"event:{query_us - int(lag)}"
                    query["event_history"].append(identity)
                    native.inputs["event_observations"][identity] = {
                        "anchor_us": query_us - int(lag),
                        "available_us": query_us,
                    }
                    native.event_values[identity] = (
                        eraw[index].cpu().numpy(),
                        experts[index].cpu().numpy(),
                    )
                timeline = []
                for pixels, metadata in rgb:
                    rraw, rexperts = engine._expert_block(
                        self.models,
                        "R",
                        torch.from_numpy(pixels)[None].to(self.device),
                        torch.tensor(
                            [metadata["delta_t_s"]], dtype=torch.float32, device=self.device
                        ),
                    )
                    timeline.append(
                        (
                            metadata["first_sensor_ts_us"],
                            metadata["anchor_us"],
                            metadata["available_us"],
                            rraw[0],
                            rexperts[0],
                        )
                    )
                history = engine._select_rgb_history(timeline, query_us)
                for item in history:
                    identity = f"rgb:{item[1]}"
                    query["rgb_history"].append(identity)
                    native.inputs["rgb_observations"][identity] = {
                        "anchor_us": int(item[1]),
                        "available_us": int(item[2]),
                    }
                    native.rgb_values[identity] = (item[3].cpu().numpy(), item[4].cpu().numpy())
                prediction = native.predict(query, native.inputs)
                event_ttc = float(phase_to_ttc(outputs["E_CTX_MATCHED"]["point_phase"])[0].cpu())
                rgb_ttc = (
                    float(phase_to_ttc(outputs["R_CTX"]["point_phase"])[0].cpu())
                    if "R_CTX" in outputs
                    else None
                )
        finally:
            for hook in hooks:
                hook.remove()
        return {
            "predictions": {
                "PHASE_ONLY_TRAIN40_S7": prediction,
                "E_CTX_TRAIN40": event_ttc,
                "R_CTX_TRAIN40": rgb_ttc,
            },
            "rgb_available": bool(query["rgb_history"]),
            "rgb_unavailable_reason": reason,
            "history_lengths": {
                "event": len(query["event_history"]),
                "rgb": len(query["rgb_history"]),
            },
            "seconds": {"preprocess": prepared - start, "total": time.perf_counter() - start},
            "targets_read": False,
            "optimizer_updates": 0,
        }


def run(candidate: Path, manifest: Path, output: Path, device: str, limit: int | None) -> None:
    paths, model_binding = verify_candidate(candidate)
    frozen_runtime = verify_frozen_runtime(candidate)
    value = read(manifest)
    rows = value["rows"]
    reject_targets(rows)
    ids = [str(row["query_id"]) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Evaluation query identities must be unique and nonempty")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(7)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    source_paths = [
        Path(__file__),
        Path(inspect.getfile(CausalScaleTTC)),
        Path(inspect.getfile(FinalV16Train40Predictor)),
        Path(inspect.getfile(engine)),
        Path(inspect.getfile(NativeObservation)),
        Path(inspect.getfile(engine.prepare_event)),
    ]
    binding = {
        "schema": "final_train40_transfer_inference_v1",
        "model": model_binding,
        "model_name": "PHASE_ONLY_TRAIN40_S7",
        "modality": "RGB+EVENTS",
        "model_seal_sha256": model_binding["model_seal_sha256"],
        "model_hashes": model_binding["components"],
        "frozen_runtime": frozen_runtime,
        "manifest_sha256": sha(manifest),
        "seed": 7,
        "device": device,
        "precision": "float32",
        "sources": {str(path.resolve()): sha(path) for path in source_paths},
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "inputs": engine.OWN_RGB_PREPROCESSING,
        "context": "FINAL_NATIVE_PADDED8_FROM_DENSE_SENSOR_HISTORY",
        "optimizer_updates": 0,
        "targets_read": False,
        "checkpoint_selection": False,
        "test12_submission": False,
    }
    write_once(output / "INFERENCE_FREEZE.json", binding)
    binding_sha = sha(output / "INFERENCE_FREEZE.json")
    predictor = FinalTransferPredictor(paths, device)
    fragments = {}
    fresh = 0
    for index, row in enumerate(rows):
        path = output / "own" / f"query_{index:05d}.json"
        if path.exists():
            old = read(path)
            if (
                old.get("query_id") != row["query_id"]
                or old.get("binding_sha256") != binding_sha
                or old.get("manifest_sha256") != sha(manifest)
            ):
                raise ValueError("Saved transfer prediction differs from frozen population/model")
        else:
            if (limit is not None and fresh >= limit) or (output / "STOP_REQUEST").exists():
                break
            result = predictor.predict(row)
            result.update(
                query_id=row["query_id"], manifest_sha256=sha(manifest), binding_sha256=binding_sha
            )
            write_once(path, result)
            fresh += 1
            print(
                json.dumps(
                    {
                        "completed": index + 1,
                        "total": len(rows),
                        "seconds": result["seconds"]["total"],
                    }
                ),
                flush=True,
            )
        fragments[str(path.relative_to(output))] = sha(path)
    if len(fragments) == len(rows):
        write_once(
            output / "PREDICTIONS_SEALED.json",
            {
                "schema": "final_train40_transfer_predictions_v1",
                "status": "COMPLETE",
                "manifest_sha256": sha(manifest),
                "binding_sha256": binding_sha,
                "queries": len(rows),
                "fragments": fragments,
                "targets_read": False,
                "optimizer_updates": 0,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        paths, binding = verify_candidate(args.candidate)
        runtime = verify_frozen_runtime(args.candidate)
        models, _ = load_models(paths, "cpu")
        parity = verify_saved_head_parity(args.candidate, models[HEAD])
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "model_binding": binding,
                    "frozen_runtime": runtime,
                    "saved_head_parity": parity,
                    "loaded_models": sorted(models),
                    "optimizer_updates": 0,
                }
            ),
            flush=True,
        )
    else:
        if args.manifest is None or args.output is None:
            parser.error("--manifest and --output are required for inference")
        run(args.candidate, args.manifest, args.output, args.device, args.limit)


if __name__ == "__main__":
    main()
