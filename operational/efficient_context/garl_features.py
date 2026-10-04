"""Native INNER-OOF/OUTER inputs, query-current ROI and explicit information budget."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

from .common import Campaign, atomic_bytes, atomic_json, digest, npz, read, release, spec


def build(c: Campaign, fold: int, role: str) -> dict:
    """Publish bounded per-query fragments; no evaluation before the12-producer seal."""
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    import torch

    from e_jepa_ttc.efficient_context.garl_input import inference_record
    from e_jepa_ttc.simplex_t.cached_event_reader import ReaderPool
    from e_jepa_ttc.simplex_t.training import state_digest

    from .budget import require
    from .native_garl import model_class, native_config

    if role not in ("inner_oof", "outer_dev"):
        raise ValueError("only the authorized TRAIN and OLD_DEV roles")
    seal = read(c.out / "garl/ENDPOINTS.json")
    if len(seal["fits"]) != 12 or not seal["all_twelve_frozen_before_evaluation"]:
        raise ValueError("all native producer endpoints required before family evaluation")
    protocol = read(c.out / "garl/PROTOCOL.json")
    if digest(c.out / "garl/PROTOCOL.json") != seal["protocol_sha256"]:
        raise ValueError("native producer freeze changed")
    for file in protocol["science_files"]:
        if digest(Path(file["path"])) != file["sha256"]:
            raise ValueError("native scientific implementation changed")
    sources = c.sources()
    parent = sources.source(spec(sources, fold, 8), role)
    folder = c.out / f"garl/features/fold{fold}/{role}"
    if role == "inner_oof":
        with np.load(c.out / f"garl/admission/outer{fold}_tokens.npz", allow_pickle=False) as z:
            tokens = z["tokens"].astype(str)
    else:
        from e_jepa_ttc.simplex_t.current_inputs import load_current_inputs

        tokens = load_current_inputs(
            sources.historical_root,
            fold,
            role,
            ancestry_sha256=sources.ancestry_sha256,
            allowed_sequences=sources.allowed_sequences,
        )["metadata"].sample_token.to_numpy(str)
    if len(tokens) != parent.population:
        raise ValueError("native source population differs from the paired head TRAIN/OLD_DEV")
    rows = pq.read_table(
        Path(c.local["garl_annotations_candidate"]),
        columns=["sample_token", "sequence_id", "event_windows_us", "boxes_xyxy"],
        filters=[("sample_token", "in", tokens.tolist())],
        use_threads=False,
    ).to_pylist()
    records = {row["sample_token"]: row for row in rows}
    if set(records) != set(tokens):
        raise ValueError("missing exact-token native query input; no query dropping permitted")
    admission = read(c.out / "garl/ADMISSION.json")
    endpoints = {v["key"]: v for v in seal["fits"]}
    inner_groups = [v for v in admission["fits"] if v["outer"] == fold and v["inner"] is not None]
    family = {}
    for token in tokens:
        sequence = records[token]["sequence_id"]
        candidates = [v["key"] for v in inner_groups if sequence in v["excluded_inner"]]
        if role == "inner_oof" and len(candidates) != 1:
            raise ValueError("query lacks a unique excluded INNER-OOF native producer")
        family[token] = candidates[0] if role == "inner_oof" else f"outer{fold}"
        if sequence in endpoints[family[token]]["train_sequences"]:
            raise ValueError("native comparator producer has seen the query's holdout")
    binding = {
        "role": role,
        "fold": fold,
        "parent_source_sha256": parent.identity_sha256,
        "producer_endpoints_sha256": digest(c.out / "garl/ENDPOINTS.json"),
        "input_metadata_sha256": digest(Path(c.local["garl_annotations_candidate"])),
        "implementation_sha256": digest(Path(__file__)),
        "native_encoding": (
            "40FP32channels,2nativeframes,floor-ms bounds; current2-frame supplied boxes"
        ),
        "count_rate": (
            "sensor-only PHASE17 columns0/1: exact current-query ROI and last100ms A5 window"
        ),
        "times": "unchanged H8 parent4channels and availability; no metadata scenario input",
        "native_history_lags_ms": [350, 300, 250, 200, 150, 100, 50, 0],
        "context_equivalent": False,
        "difference": (
            "native observation~200ms versus A5/C2F3-frame~300ms; native2-frame ROI centers"
        ),
        "RGB_DINO_or_A5_PAIR_outputs_used_as_head_features": False,
    }
    frozen = folder / "BINDING.json"
    if frozen.exists() and read(frozen) != binding:
        raise ValueError("native feature binding changed")
    if not frozen.exists():
        atomic_json(frozen, binding)
    pin = digest(frozen)
    pool = ReaderPool()
    model, active = None, None
    positions = {token: i for i, token in enumerate(tokens)}
    try:
        for token in sorted(tokens, key=lambda t: (family[t], records[t]["sequence_id"], t)):
            require(c)
            i = positions[token]
            path = folder / f"fragments/query{i:05d}.npz"
            receipt = path.with_suffix(".json")
            if receipt.exists():
                r = read(receipt)
                if r["binding_sha256"] != pin or digest(path) != r["sha256"]:
                    raise ValueError("native feature fragment changed")
                continue
            key = family[token]
            endpoint = endpoints[key]
            if active != key:
                model = None
                gc.collect()
                torch.cuda.empty_cache()
                if digest(Path(endpoint["checkpoint"])) != endpoint["sha256"]:
                    raise ValueError("native producer checkpoint changed")
                state = torch.load(endpoint["checkpoint"], map_location="cpu", weights_only=True)
                sha = state.pop("state_sha256")
                if state_digest(state) != sha or state["status"] != "COMPLETE":
                    raise ValueError("native producer is not a complete sealed endpoint")
                model = model_class(c)(native_config(c), is_train=False).float().cuda().eval()
                model.load_state_dict(state["model"], strict=True)
                del state
                active = key
            hist = parent.history[i, -8:]
            valid = hist >= 0
            safe = np.maximum(hist, 0)
            _, timing, _, _, _, _ = parent.gather(torch.tensor([i]))
            raw = np.zeros((8, 3), np.float32)
            native_ttc = np.zeros(8, np.float64)
            observation_keys = np.full(8, "", dtype="U64")
            query = records[token]
            anchor = int(parent.anchor_us[safe[-1]])
            original_windows = np.asarray(query["event_windows_us"], np.int64)
            for slot in np.flatnonzero(valid):
                lag = anchor - int(parent.anchor_us[safe[slot]])
                windows = original_windows - lag
                native = {
                    "sequence_id": query["sequence_id"],
                    "boxes_xyxy": query["boxes_xyxy"],
                    "event_windows_us": windows.tolist(),
                }
                raw_stat = (c.raw / query["sequence_id"] / "events.h5").stat()
                content = {
                    **native,
                    "producer": endpoint["sha256"],
                    "raw_bytes": raw_stat.st_size,
                    "raw_mtime_ns": raw_stat.st_mtime_ns,
                    "common_observation_id": int(safe[slot]),
                }
                observation_keys[slot] = hashlib.sha256(
                    json.dumps(content, sort_keys=True).encode()
                ).hexdigest()
                sensor = inference_record(native, pool, c.raw)
                if model is None:
                    raise ValueError("native producer was not loaded")
                with torch.inference_mode():
                    heights, _ = model(sensor[None].cuda())
                pair = heights.double().cpu().numpy()[0]
                ratio = pair[0] / pair[1]
                if not np.isfinite(ratio) or ratio <= 0:
                    raise ArithmeticError(
                        "native source produces invalid LHR phase; no dropping/repair"
                    )
                raw[slot] = -np.log(ratio), *parent.features[safe[slot], :2]
                native_ttc[slot] = np.inf if ratio == 1 else 0.1 / (1 - ratio)
                del sensor, heights
            npz(
                path,
                features=raw,
                times=timing.numpy()[0],
                valid=valid,
                native_ttc=native_ttc,
                observation_keys=observation_keys,
            )
            atomic_json(
                receipt,
                {
                    "sha256": digest(path),
                    "binding_sha256": pin,
                    "query": token,
                    "producer_key": key,
                    "producer_sha256": endpoint["sha256"],
                },
            )
            if i % 128 == 0:
                print(f"GARL_FEATURE_fold{fold}_{role}_{i}/{len(tokens)}", flush=True)
        compact, keys, history, times, native_ttc = [], {}, [], [], []
        for i in range(len(tokens)):
            with np.load(folder / f"fragments/query{i:05d}.npz", allow_pickle=False) as z:
                indices = np.full(8, -1, np.int64)
                for slot in np.flatnonzero(z["valid"]):
                    key = str(z["observation_keys"][slot])
                    value = z["features"][slot]
                    if key not in keys:
                        keys[key] = len(compact)
                        compact.append(value.copy())
                    elif not np.array_equal(compact[keys[key]], value):
                        raise ValueError("idempotent native observation cache differs")
                    indices[slot] = keys[key]
                history.append(indices)
                times.append(z["times"].copy())
                native_ttc.append(float(z["native_ttc"][-1]))
        npz(
            folder / "SOURCE.npz",
            features=np.asarray(compact, np.float32),
            history=np.asarray(history, np.int64),
            times=np.asarray(times, np.float32),
            truth=parent.target_phase,
            mass=parent.mass,
            native_ttc=np.asarray(native_ttc, np.float64),
            tokens=tokens,
        )
        metadata = pd.DataFrame(
            {"sample_token": tokens, "sequence_id": [records[t]["sequence_id"] for t in tokens]}
        )
        atomic_bytes(folder / "METADATA.csv", metadata.to_csv(index=False).encode())
        result = {
            "status": "COMPLETE",
            "binding_sha256": pin,
            "source_sha256": digest(folder / "SOURCE.npz"),
            "queries": len(tokens),
            "unique_observations": len(compact),
            "optimizer_updates": 0,
        }
        atomic_json(folder / "COMPLETE.json", result)
        return result
    finally:
        pool.close()
        release(sources)
        model = None
        gc.collect()
        torch.cuda.empty_cache()
