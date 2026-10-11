"""Bounded, resumable CPU attribution of the published V12 FCWD failures."""

from __future__ import annotations

import inspect
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC, CausalScaleTTCConfig
from e_jepa_ttc.training.stage61_pair_head import CachedPairDirectPhase
from operational.simplex_t_shared_route.adapter import extract
from operational.sota_eval.fcwd_inputs import prepare
from operational.train40_system.contracts import verified_endpoint

from .common import csv_write, digest, freeze, read, resources, sha, write


def _capture(output: Any) -> dict[str, np.ndarray]:  # noqa: ANN401 -- native output dataclass
    fields = (
        "ttc_mean_seconds",
        "ttc_log_variance",
        "log_height_ratio",
        "pair_log_height_ratio",
        "analytic_log_height_ratio",
        "residual_log_height_ratio",
        "visible_height_normalized",
        "sensor_support",
    )
    result = {key: getattr(output, key)[-8:].detach().cpu().numpy().copy() for key in fields}
    result.update(
        {
            key: value[-8:].detach().cpu().numpy().copy()
            for key, value in output.diagnostics.items()
            if value.ndim and len(value) == 16
        }
    )
    return result


def trace_raw(v12: Path, run: Path) -> dict[str, Any]:
    """Trace 20 published queries plus neighbours, with no learning or GPU access."""
    root = run / "v12_raw_attribution"
    manifest_path = v12 / "artifacts/sota_campaign_20261008/fcwd_inference/QUERY_MANIFEST.json"
    replay = v12 / "artifacts/ttc_revision_20261009/replay_fcwd"
    campaign = v12 / "artifacts/train40_system_20261005"
    replay_identity = read(replay / "REPLAY_IDENTITY.json")
    if not manifest_path.exists():
        result = {"status": "WAITING_RAW_MANIFEST", "path": str(manifest_path)}
        write(root / "RESULT.json", result)
        return result
    if sha(manifest_path) != replay_identity["manifest_sha256"]:
        raise ValueError("Raw FCWD manifest differs from historical replay")
    document = read(manifest_path)
    by_id = {r["query_id"]: r for r in document["rows"]}
    selected = pd.read_csv(run / "PUBLIC_DIAGNOSTIC_TRACES.csv").query_id.tolist()
    missing = sorted(
        {
            by_id[token]["raw_path"]
            for token in selected
            if not Path(by_id[token]["raw_path"]).exists()
        }
    )
    if missing:
        result = {"status": "WAITING_RAW_ASSETS", "paths": missing}
        write(root / "RESULT.json", result)
        return result
    source_paths = (
        "src/e_jepa_ttc/models/causal_scale_ttc.py",
        "operational/sota_eval/fcwd_inputs.py",
        "operational/simplex_t_shared_route/adapter.py",
    )
    loaded = (
        Path(inspect.getfile(CausalScaleTTC)),
        Path(inspect.getfile(prepare)),
        Path(inspect.getfile(inspect.unwrap(extract))),
    )
    for relative, loaded_path in zip(source_paths, loaded, strict=True):
        if sha(loaded_path) != sha(v12 / relative):
            raise ValueError(f"Historical raw inference source differs: {relative}")
    binding = {
        "manifest_sha256": sha(manifest_path),
        "selected_ids": selected,
        "source_hashes": {p: sha(v12 / p) for p in source_paths},
        "producer_bindings": replay_identity["models"]["producers"],
        "device": "cpu",
        "precision": "float32",
        "producer_batch": 16,
    }
    freeze(root / "BINDING.json", binding)
    fragments = root / "fragments"
    fragments.mkdir(parents=True, exist_ok=True)
    pending = [
        token for token in selected if not (fragments / f"{token.replace(':', '_')}.json").exists()
    ]
    models: dict[str, Any] = {}
    resource_state = resources(run, starting=True)
    write(root / "RESOURCE_START.json", resource_state)
    write(root / "RESOURCE_STATE.json", resource_state)
    if pending and resource_state["allowed"]:
        protocol = read(campaign / "TRAINING_PROTOCOL.json")
        if (
            sha(campaign / "TRAINING_PROTOCOL.json")
            != binding["producer_bindings"]["protocol_sha256"]
        ):
            raise ValueError("Historical producer protocol mismatch")
        for name, endpoint in (("a5", 49932), ("c2f", 49932), ("pair", 6840)):
            checkpoint, receipt = verified_endpoint(campaign, f"{name}_seed7", endpoint)
            if receipt["sha256"] != binding["producer_bindings"][name]["checkpoint_sha256"]:
                raise ValueError("Historical producer checkpoint mismatch")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if name == "pair":
                model = CachedPairDirectPhase(
                    dropout=payload["contract"]["model_config"]["dropout"]
                )
            else:
                model = CausalScaleTTC(
                    CausalScaleTTCConfig(**protocol["producers"][name]["model_config"])
                )
            model.load_state_dict(payload["model_state_dict"], strict=True)
            models[name.upper()] = model.float().eval().requires_grad_(False)
            del payload
    for number, token in enumerate(pending):
        if not models:
            break
        resource_state = resources(run)
        write(root / "RESOURCE_STATE.json", resource_state)
        if not resource_state["allowed"]:
            break
        row = by_id[token]
        captures: dict[str, dict[str, np.ndarray]] = {}
        hooks = []
        for name in ("A5", "C2F"):

            def hook(
                _module: torch.nn.Module,
                _args: tuple[object, ...],
                output: Any,  # noqa: ANN401 -- native producer dataclass
                name: str = name,
                captured: dict = captures,
            ) -> None:
                captured[name] = _capture(output)

            hooks.append(models[name].register_forward_hook(hook))
        started = time.perf_counter()
        try:
            with torch.inference_mode():
                prepared = prepare(row)
                events = torch.from_numpy(prepared["own_events"])
                events = torch.cat((torch.zeros_like(events), events), 0)
                delta = torch.full((16, 2), 0.1, dtype=torch.float32)
                features = extract("H8_SEED7", models, events, delta)[-8:]
        finally:
            for handle in hooks:
                handle.remove()
        selected_head_rows = pd.read_csv(run / "v12_cached_attribution/HEAD_TRACES.csv")
        # Store only diagnostic scalars/vectors, never whole event streams or GT inputs.
        rows = []
        for name, values in captures.items():
            for slot in range(8):
                entry = {
                    "query_id": token,
                    "producer": name,
                    "observation_slot": slot,
                    "anchor_us": row["anchor_us"],
                    "event_end_us": int(row["anchor_us"]) - (7 - slot) * 50000,
                    "roi_xyxy": json.dumps(row["square_xyxy"]),
                    "bbox_xyxy3": json.dumps(row["boxes_xyxy3"]),
                    "metadata_sha256": row["metadata_sha256"],
                }
                for key, array in values.items():
                    local = np.atleast_1d(array[slot]).ravel()
                    for offset, value in enumerate(local):
                        entry[f"{key}_{offset}"] = float(value) if np.isfinite(value) else None
                rows.append(entry)
        ordinal = sorted(
            by_id, key=lambda key: (by_id[key]["sequence_id"], by_id[key]["anchor_us"])
        ).index(token)
        saved_path = replay / "fragments" / f"{ordinal:06d}.npz"
        if sha(saved_path) != read(replay / "REPLAY_COMPLETE.json")["fragments"][saved_path.name]:
            raise ValueError("Historical feature reference changed")
        with np.load(saved_path, allow_pickle=False) as archive:
            difference = np.abs(features - archive["features"])
        fragment = {
            "binding_sha256": digest(binding),
            "query_id": token,
            "rows": rows,
            "seconds": time.perf_counter() - started,
            "feature_max_abs_cpu_vs_saved_gpu": float(difference.max()),
            "phase_max_abs_cpu_vs_saved_gpu": float(difference[:, 8:11].max()),
            "head_reference_available": token in set(selected_head_rows.query_id),
            "optimizer_updates": 0,
            "raw_preparation": "exact_native_no_warp",
        }
        write(fragments / f"{token.replace(':', '_')}.json", fragment)
        write(
            root / "PROGRESS.json",
            {
                "completed": len(selected) - len(pending) + number + 1,
                "total": len(selected),
                "last_query": token,
            },
        )
    rows, results = [], []
    for token in selected:
        path = fragments / f"{token.replace(':', '_')}.json"
        if path.exists():
            value = read(path)
            if value["binding_sha256"] != digest(binding):
                raise ValueError("Raw trace fragment identity changed")
            rows.extend(value["rows"])
            results.append(value)
    if rows:
        keys = sorted({key for row in rows for key in row})
        csv_write(root / "PRODUCER_TRACES.csv", [{k: row.get(k) for k in keys} for row in rows])
    result = {
        "status": "COMPLETE" if len(results) == len(selected) else "PAUSED_RESOURCE",
        "completed_queries": len(results),
        "requested_queries": len(selected),
        "producer_observations": len(rows),
        "optimizer_updates": 0,
        "raw_input_route": "native_label_free_exact_no_warp",
        "cost_scope": (
            "CPU_native_input_plus_producers_and_diagnostic_capture_not_isolated_benchmark"
        ),
        "seconds": sum(x["seconds"] for x in results),
        "max_feature_abs_cpu_vs_saved_gpu": max(
            (x["feature_max_abs_cpu_vs_saved_gpu"] for x in results), default=None
        ),
        "max_phase_abs_cpu_vs_saved_gpu": max(
            (x["phase_max_abs_cpu_vs_saved_gpu"] for x in results), default=None
        ),
        "V13_training_or_selection": False,
        "new_scientific_weights": False,
    }
    write(root / "RESULT.json", result)
    return result
