"""Export exact frozen head inputs/normalizers; diagnose cached LATENT only."""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from extras import ANALYSIS, CAMPAIGN, OUT, ROOT, check, record
from runtime import atomic_json, digest

from e_jepa_ttc.simplex_t.arms import resolve_arm
from e_jepa_ttc.simplex_t.cache import CachedQueries
from e_jepa_ttc.simplex_t.campaign_sources import CampaignSources
from e_jepa_ttc.simplex_t.compact_weights import load_compact_endpoint
from e_jepa_ttc.simplex_t.configuration_preflight import open_acknowledged_source_configuration
from e_jepa_ttc.simplex_t.context_sources import load_context_sources
from e_jepa_ttc.simplex_t.fixed_baseline_inputs import load_fixed_baselines
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from e_jepa_ttc.simplex_t.registry import FitSpec
from e_jepa_ttc.simplex_t.training import load_checkpoint


def preserve_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Retain an identical export; commit new arrays atomically."""
    if path.exists():
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays) or any(
                not np.array_equal(existing[name], array) for name, array in arrays.items()
            ):
                raise ValueError(f"preserve conflicting exported arrays: {path}")
        return
    pending = path.with_name(path.name + ".pending")
    with pending.open("wb") as stream:
        np.savez_compressed(stream, **arrays)  # pyright: ignore[reportArgumentType]
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def distribution(source: CachedQueries, role: str, fold: int) -> list[dict[str, Any]]:
    ids = np.unique(source.history[source.history >= 0])
    raw = np.asarray(source.features[ids], dtype=np.float64)
    normalized = (raw - source.normalizer.mean) / source.normalizer.scale
    rows = []
    for dim in range(raw.shape[1]):
        a, z = raw[:, dim], normalized[:, dim]
        rows.append(
            {
                "fold": fold,
                "role": role,
                "dimension": dim,
                "is_latent": dim >= 17,
                "unique_consumed_observations": len(ids),
                "normalizer_mean": float(source.normalizer.mean[dim]),
                "normalizer_scale": float(source.normalizer.scale[dim]),
                "raw_mean": float(a.mean()),
                "raw_std": float(a.std()),
                "raw_min": float(a.min()),
                "raw_max": float(a.max()),
                "raw_std_below_1e_minus8": bool(a.std() < 1e-8),
                "normalized_mean": float(z.mean()),
                "normalized_std": float(z.std()),
                "normalized_abs_max": float(np.abs(z).max()),
                "normalized_abs_above10_count": int((np.abs(z) > 10).sum()),
                "normalized_q01": float(np.quantile(z, 0.01)),
                "normalized_q99": float(np.quantile(z, 0.99)),
            }
        )
    return rows


def similarity_audit(sources: CampaignSources) -> list[dict[str, Any]]:
    """Use already cached identical current ROIs across folds, without fitting."""
    index_path = sources.index_root / "query_context_index.npz"
    with np.load(index_path, allow_pickle=False) as index:
        tokens = index["tokens"].copy()
        families = index["producer_family"].copy()
    embeddings = []
    for fold in range(3):
        pin = sources.folds[fold]
        manifest = record(pin.path / "COMPILED.json")
        if digest(pin.path / "COMPILED.json") != pin.sha256:
            raise ValueError("similarity compiled manifest changed")
        feature_path = pin.path / "features145.npy"
        if digest(feature_path) != manifest["arrays"]["features145"]:
            raise ValueError("similarity latent cache changed")
        with np.load(sources.dedup_root / f"outer{fold}.npz", allow_pickle=False) as dedup:
            current = dedup["history"][:, -1]
        features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
        embeddings.append(np.asarray(features[current, 17:], dtype=np.float64))
    rows = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        pairs = sorted(set(zip(families[a].tolist(), families[b].tolist(), strict=True)))
        for fa, fb in pairs:
            ids = np.flatnonzero((families[a] == fa) & (families[b] == fb))
            ids = ids[np.argsort(tokens[ids], kind="stable")]
            # Selection depends on identity only; no score-dependent filtering.
            ids = ids[np.linspace(0, len(ids) - 1, min(len(ids), 128), dtype=int)]
            x, y = embeddings[a][ids], embeddings[b][ids]
            x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
            y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
            gx, gy = x @ x.T, y @ y.T
            take = np.triu_indices(len(ids), k=1)
            vx, vy = gx[take], gy[take]
            rows.append(
                {
                    "fold_a": a,
                    "fold_b": b,
                    "family_a": int(fa),
                    "family_b": int(fb),
                    "shared_current_observations": len(ids),
                    "gram_cosine_MAE": float(np.abs(vx - vy).mean()),
                    "gram_cosine_pearson": float(np.corrcoef(vx, vy)[0, 1])
                    if vx.std() and vy.std()
                    else None,
                    "token_selection": (
                        "sorted identity; at most128 equally spaced; no label/score selection"
                    ),
                    "alignment_fitted": False,
                }
            )
    return rows


def export_sources() -> None:
    check()
    torch.set_num_threads(4)
    launch = record(CAMPAIGN / "launches/T6.json")
    freeze = record(Path(launch["freeze"]))
    sources, inspection = open_acknowledged_source_configuration(
        Path(launch["local_paths"]),
        Path(launch["source_configuration"]),
        launch["source_configuration_sha256"],
    )
    rows, publications = [], {}
    for stage, binding in launch["publications"].items():
        publications[stage] = record(Path(binding["publication"]))
        rows.extend(record(Path(binding["endpoints"]))["fits"])
    rows.sort(
        key=lambda row: (
            row["fit"]["fold"],
            row["model"]["feature_count"],
            resolve_arm(FitSpec(**row["fit"]), sources.graph).pool,
            row["key"],
        )
    )
    input_root = OUT / "head_inputs"
    input_root.mkdir(exist_ok=True)
    normalizer_root = OUT / "normalizers"
    normalizer_root.mkdir(exist_ok=True)
    exports, normalizers, distributions, curve_rows = {}, {}, [], []
    latent_done, input_done = set(), {}
    try:
        for row in rows:
            check()
            spec = FitSpec(**row["fit"])
            key = row["key"]
            train, dev = (sources.source(spec, role) for role in ("inner_oof", "outer_dev"))
            if {"inner_oof": train.identity_sha256, "outer_dev": dev.identity_sha256} != freeze[
                "source_identities"
            ][key]:
                raise ValueError(f"source identity changed: {key}")
            norm_id = digest_bytes(
                train.normalizer.mean.tobytes()
                + train.normalizer.scale.tobytes()
                + train.normalizer.consumed_ids_sha256.encode()
            )
            norm = normalizer_root / (norm_id + ".npz")
            preserve_npz(norm, {"mean": train.normalizer.mean, "scale": train.normalizer.scale})
            normalizers[key] = {
                "path": "normalizers/" + norm.name,
                "sha256": digest(norm),
                "TRAIN_consumed_ids_sha256": train.normalizer.consumed_ids_sha256,
                "train_source_sha256": train.identity_sha256,
                "dev_source_sha256": dev.identity_sha256,
            }
            if spec.name == "LATENT-D1-H8-C160" and spec.fold not in latent_done:
                for role, source in (("inner_oof", train), ("outer_dev", dev)):
                    distributions.extend(distribution(source, role, spec.fold))
                latent_done.add(spec.fold)
                pd.DataFrame(distributions).to_csv(
                    OUT / "LATENT_DISTRIBUTIONS.csv", index=False, float_format="%.17g"
                )
            publication_row = publications[spec.stage]["fits"][key]
            prediction_path = (
                Path(launch["publications"][spec.stage]["publication"]).parent
                / publication_row["path"]
            )
            if digest(prediction_path) != publication_row["sha256"]:
                raise ValueError("head output publication changed")
            frame = pd.read_parquet(prediction_path)
            metadata = pd.read_csv(
                sources.historical_root / "tables" / f"outer{spec.fold}_outer_dev.csv"
            )
            if not np.array_equal(frame.sample_token.to_numpy(), metadata.sample_token.to_numpy()):
                raise ValueError("head inputs/query publication order differs")
            if spec.name == "TPR-D0-H8-C160":
                original_sources = load_context_sources(
                    sources.folds[spec.fold].path,
                    sources.index_root,
                    sources.dedup_root,
                    sources.historical_root,
                    compiled_manifest_sha256=sources.folds[spec.fold].sha256,
                    ancestry_sha256=sources.ancestry_sha256,
                    allowed_sequences=sources.allowed_sequences,
                    feature_count=17,
                )
                original_dev = original_sources["outer_dev"]
                if not np.array_equal(original_dev.history, dev.history):
                    raise ValueError("baseline original cache and registered H8 histories differ")
                baseline = (
                    ROOT
                    / (f"artifacts/simplex_t/T1/fixed_baselines_outer{spec.fold}_fp64_emission")
                    / "BASELINES.json"
                )
                load_fixed_baselines(
                    baseline,
                    manifest_sha256=digest(baseline),
                    role="outer_dev",
                    source_sha256=original_dev.identity_sha256,
                    tokens=metadata.sample_token.to_numpy(),
                    sequences=metadata.sequence_id.to_numpy(),
                    history=original_dev.history[:, -8:],
                    validate_prerequisites=check,
                )
                del original_sources, original_dev
            compact = ANALYSIS / "compact_weights" / spec.stage / key
            manifest_hash = digest(compact / "WEIGHTS.json")
            model = load_compact_endpoint(compact, manifest_sha256=manifest_hash)
            if dev.identity_sha256 not in input_done:
                chunks = defaultdict_list()
                for start in range(0, dev.population, 128):
                    check()
                    inputs = dev.gather(torch.arange(start, min(start + 128, dev.population)))
                    for field, array in zip(
                        ("features", "times", "valid", "experts"), inputs[:4], strict=True
                    ):
                        chunks[field].append(array.numpy())
                arrays = {field: np.concatenate(parts) for field, parts in chunks.items()}
                arrays["sample_token"] = frame.sample_token.to_numpy(str)
                path = input_root / (dev.identity_sha256 + ".npz")
                preserve_npz(path, arrays)
                input_done[dev.identity_sha256] = {
                    "path": "head_inputs/" + path.name,
                    "sha256": digest(path),
                }
                del chunks, arrays
            input_pin = input_done[dev.identity_sha256]
            with np.load(OUT / input_pin["path"], allow_pickle=False) as archive:
                loaded = {name: archive[name] for name in ("features", "times", "valid", "experts")}
                actual = []
                with torch.inference_mode():
                    for start in range(0, dev.population, 128):
                        check()
                        inputs = [
                            torch.from_numpy(loaded[name][start : start + 128])
                            for name in ("features", "times", "valid", "experts")
                        ]
                        outputs = model(*inputs)
                        actual.append(outputs["point_phase"].numpy())
                point = np.concatenate(actual).astype(np.float64)
                if model.cfg.output_mode == "selector":
                    # Original selector emits the unchanged original expert TTC.
                    # Included input experts are phases, so compare its phase only.
                    expected = frame[
                        ["expert0_phase", "expert1_phase", "expert2_phase"]
                    ].to_numpy()[np.arange(len(frame)), frame.diagnostic_selector.to_numpy()]
                    error = float(np.max(np.abs(point - expected)))
                    if error > 1e-7:
                        raise ValueError(f"selector phase regeneration failed: {key}: {error}")
                    comparison = (
                        "expert phase tolerance1e-7; original expert TTC included in predictions"
                    )
                else:
                    predicted = phase_to_ttc(torch.from_numpy(point)).numpy()
                    error = float(np.max(np.abs(predicted - frame.prediction_ttc_s.to_numpy())))
                    if error > 1e-10:
                        raise ValueError(f"compact head output differs: {key}: {error}")
                    comparison = "emitted TTC tolerance1e-10, rtol0 fixed before inference"
            exports[key] = {
                "inputs": input_pin,
                "compact_manifest_sha256": manifest_hash,
                "compact_path": "compact_weights/" + spec.stage + "/" + key,
                "prediction_path": publication_row["path"],
                "prediction_sha256": publication_row["sha256"],
                "max_abs_regeneration_error": error,
                "comparison": comparison,
            }
            del loaded
            checkpoint = (
                Path(launch["publications"][spec.stage]["checkpoint_root"]) / row["checkpoint"]
            )
            if digest(checkpoint) != row["checkpoint_sha256"]:
                raise ValueError("curve endpoint changed")
            state = load_checkpoint(checkpoint)
            losses = state["losses"]
            if len(losses) != 2500 or state["completed_updates"] != 2500:
                raise ValueError("curve is not a fixed2500 endpoint")
            curve_rows.extend(
                {"fit": key, "update": i + 1, "TRAIN_loss": float(loss)}
                for i, loss in enumerate(losses)
            )
            del state, model, train, dev, frame
            gc.collect()
            atomic_json(
                OUT / "HEAD_EXPORT_PROGRESS.json",
                {"completed_heads": len(exports), "optimizer_updates": 0},
            )
            print(
                json.dumps(
                    {
                        "status": "COMPACT_INPUTS_AND_CURVE_VERIFIED",
                        "heads": len(exports),
                        "fit": key,
                    }
                ),
                flush=True,
            )
        similarities = similarity_audit(sources)
        pd.DataFrame(similarities).to_csv(
            OUT / "LATENT_SIMILARITY.csv", index=False, float_format="%.17g"
        )
        pd.DataFrame(curve_rows).to_parquet(
            OUT / "TRAINING_CURVES_ALL_180000_UPDATES.parquet", index=False
        )
        producer_manifest = record(sources.index_root / "INDEX_MANIFEST.json")
        atomic_json(
            OUT / "LATENT_PRODUCERS.json",
            {
                "families": producer_manifest["families"],
                "latent_dimensions": 128,
                "semantic_alignment_enforced": False,
                "alignment_hypothesis_demonstrated": False,
                "shared_observation_comparison": (
                    "existing identical current query ROI across compiled folds; "
                    "grouped by producer family pair"
                ),
                "all_three_normalizers_and_source_identities_verified": len(latent_done) == 3,
                "similarity_comparisons": len(similarities),
                "optimizer_updates": 0,
            },
        )
        ancestry = record(sources.historical_root / "NESTED_ANCESTRY_AUDIT.json")
        producer_by_hash = {p["checkpoint_sha256"]: p for p in ancestry["producers"]}
        experts = []
        for pin in ancestry["input_bindings"].values():
            if pin["sha256"] not in producer_by_hash:
                continue
            path = Path(pin["path"])
            if path.stat().st_size != pin["bytes"] or digest(path) != pin["sha256"]:
                raise ValueError("expert checkpoint inventory mismatch")
            producer = producer_by_hash[pin["sha256"]]
            experts.append(
                {
                    **pin,
                    "expert": producer["expert"],
                    "role": producer["role"],
                    "fold": producer["outer_fold"],
                    "bytes_in_bundle": False,
                }
            )
        if len(experts) != 36:
            raise ValueError("36 actual expert checkpoint identities required")
        atomic_json(
            OUT / "EXPERT_CHECKPOINT_INVENTORY.json",
            {"experts": experts, "raw_autonomous_regeneration": False},
        )
        atomic_json(
            OUT / "HEAD_EXPORT.json",
            {
                "status": "72_COMPACT_HEADS_NORMALIZERS_INPUTS_CURVES_VERIFIED",
                "heads": exports,
                "normalizers": normalizers,
                "inspection": inspection,
                "torch_threads": 4,
                "torch_version": str(torch.__version__),
                "optimizer_updates": 0,
            },
        )
    finally:
        sources.release()


def digest_bytes(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


def defaultdict_list() -> dict[str, list[np.ndarray]]:
    from collections import defaultdict

    return defaultdict(list)
