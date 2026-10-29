"""Execute one complete scientific 24/8/8 rotation, without selecting on generation."""

from __future__ import annotations

import argparse
import gc
import os
import traceback
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch
from torch import nn

from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC
from e_jepa_ttc.rgb_port.features import pair133
from operational.rgb_port.accounting import atomic_write_json, read_json_shared
from operational.rgb_port.infer_experts import delta_matrix
from operational.rgb_port.recipe import ProducerRecipe
from operational.rgb_port.train_producers import build_producer, producer_features

from .event_history import _historical_pair_input
from .features import _atomic_npz, frozen_context_state
from .fold_sources import load_fold_event_source, load_fold_rgb_source
from .fold_training import (
    FoldTrainingUnit,
    _stage,
    build_fold_training_unit,
    write_fold_training_unit,
)
from .materialize import FragmentStore
from .oof_stitch import _expected_keys, payload_sha256
from .producers import _final_memory_allowed
from .recipe import canonical_sha256, load_recipe, sha256
from .rgb import AllTrainRGBSource, load_all_train_source
from .scientific_authority import load_config
from .scientific_fold_heads import fit_scientific_head, verify_scientific_admission
from .scientific_fold_producers import FIT_NAMES, train
from .scientific_oof_features import (
    arrays,
    event_raw,
    normalize_histories,
    rgb_raw,
    role_rows,
    seal_role_cache,
    unique_normalizer,
)
from .shared_resources import SharedResourceMonitor
from .training import load_cached_endpoint


class ResourcePause(RuntimeError):  # noqa: N818 -- shared queue pause signal, not scientific failure
    """Recoverable resource/manual pause, distinct from scientific or I/O failures."""

    def __init__(self, resources: Mapping[str, Any], *, manual: bool = False) -> None:
        super().__init__("manual stop" if manual else "resource reserve unavailable")
        self.resources, self.manual = dict(resources), manual


def make_guard(run: Path) -> Callable[[], None]:
    """Keep authenticated inventory checks and both user-selected 2 GiB reserves."""
    monitor = SharedResourceMonitor(
        Path(__file__).resolve().parents[2],
        session_path=run / "RESOURCE_SESSION.json",
        snapshot_path=run / "RESOURCE_SNAPSHOT.json",
        run_root=run,
    )

    def guard() -> None:
        if (run / "STOP_REQUEST").exists():
            raise ResourcePause({}, manual=True)
        allowed, resources = monitor.guard(
            run,
            checkpoint_reservation=1024**3,
            completed_updates=0,
        )
        if not _final_memory_allowed(allowed, resources):
            raise ResourcePause(resources)

    return guard


def load_fold_model(directory: Path, device: str) -> nn.Module:
    """Load only an admitted scientific producer, from its embedded exact recipe."""
    admission = verify_scientific_admission(directory / "ENDPOINT_ADMISSION.json")
    saved = torch.load(admission["checkpoint_path"], map_location="cpu", weights_only=False)
    embedded = saved["identity"]["producer_recipe"]
    recipe = ProducerRecipe(**{k: embedded[k] for k in ProducerRecipe.__dataclass_fields__})
    model = build_producer(recipe)
    model.load_state_dict(saved["model_state_dict"], strict=True)
    return model.to(device).float().eval().requires_grad_(False)


def target_rows(base: AllTrainRGBSource, rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """Join TRAIN targets by exact native ordinal plus token and group checks."""
    records = [base.row(int(x)) for x in rows.source_ordinal]
    for native, record in zip(rows.to_dict("records"), records, strict=True):
        if str(native["sample_token"]) != str(record["sample_token"]) or str(
            native["group_id"]
        ) != str(record["group_id"]):
            raise ValueError("Training target join crossed a token or group")
    return {
        "target_phase": np.asarray([x["target_phase"] for x in records], np.float32),
        "sample_token": rows.sample_token.to_numpy(str),
        "group_id": rows.group_id.to_numpy(str),
        "sequence_id": np.asarray([x["sequence_id"] for x in records]),
    }


def pair_cache(
    config: Mapping[str, Any],
    rotation: int,
    unit: FoldTrainingUnit,
    base: AllTrainRGBSource,
    root: Path,
    name: str,
    guard: Callable[[], None],
) -> Path:
    """Regenerate PAIR133 from this fold's A5 and its producer-role rows only."""
    ancestor = "E_A5" if name == "PAIR_E" else "R_A5"
    admission = verify_scientific_admission(root / "fits" / ancestor / "ENDPOINT_ADMISSION.json")
    parents = {ancestor: admission["checkpoint_sha256"]}
    output = root / "caches" / f"{name}.npz"
    if output.exists() and output.with_suffix(".json").exists():
        from .features import validate_cache_receipt

        receipt = validate_cache_receipt(
            output, "final_refit_pair_cache_v1", expected_role="producer_pair"
        )
        if receipt["parents"] != parents:
            raise ValueError("PAIR cache parent changed")
        return output
    contract = Path(config["fold_contract"])
    rows = role_rows(contract, rotation, "producer")
    source = (
        load_fold_event_source(contract, rotation, Path(config["event_root"]))
        if name == "PAIR_E"
        else load_fold_rgb_source(
            Path(config["rgb_source"]), Path(config["teacher"]), contract, rotation
        )
    )
    model = load_fold_model(root / "fits" / ancestor, config["device"])
    store = FragmentStore(
        output.with_suffix(".parts"),
        {
            "schema": "scientific_fold_pair133_v1",
            "parents": parents,
            "rows_sha256": unit.roles["producer"]["sha256"],
            "precision": "float32",
        },
    )
    parts = []
    try:
        with torch.inference_mode():
            for part, start in enumerate(range(0, len(rows), 32)):
                saved = store.read(part)
                if saved is not None:
                    parts.append(saved)
                    continue
                guard()
                indices = list(range(start, min(start + 32, len(rows))))
                # RGB frame counts can differ. A single source batch must have one shape.
                features: dict[int, np.ndarray] = {}
                buckets: dict[int, list[int]] = {}
                for i in indices:
                    buckets.setdefault(int(source.frame_counts[i]), []).append(i)
                for selected in buckets.values():
                    batch = source.batch(selected, "event" if name == "PAIR_E" else "rgb")
                    sensor = batch.events.to(config["device"]).float()
                    delta = delta_matrix(batch.delta_t_s, sensor.shape[1]).to(config["device"])
                    result = producer_features(cast(CausalScaleTTC, model), sensor, delta)
                    support = result["support"]
                    values = (
                        _historical_pair_input(result, delta[:, -1])
                        if name == "PAIR_E"
                        else pair133(
                            result["token128"],
                            delta[:, -1],
                            torch.stack(
                                (support[:, -1], torch.minimum(support[:, -2], support[:, -1])), -1
                            ),
                        )
                    )
                    for i, array in zip(selected, values.cpu().numpy(), strict=True):
                        features[i] = array
                value = {"features": np.stack([features[i] for i in indices]).astype(np.float32)}
                store.write(part, value)
                parts.append(value)
    finally:
        source.close()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    values = {
        "features": np.concatenate([x["features"] for x in parts]),
        **target_rows(base, rows),
        "role": np.full(len(rows), "producer_pair"),
    }
    seal_role_cache(
        output,
        values,
        schema="final_refit_pair_cache_v1",
        role="producer_pair",
        parents=parents,
        rows_sha=unit.roles["producer"]["sha256"],
    )
    return output


def fit_head(
    config: Mapping[str, Any],
    root: Path,
    unit: FoldTrainingUnit,
    name: str,
    cache: Path,
    guard: Callable[[], None],
) -> None:
    """Run fixed exposure in checkpointed chunks, returning only at a full endpoint."""
    stage = _stage(unit, name)
    recipe_path = Path(config["recipe"])
    recipe = load_recipe(recipe_path)
    while True:
        guard()
        receipt = fit_scientific_head(
            unit_path=root / "FOLD_TRAINING_UNIT.json",
            recipe=recipe,
            recipe_path=recipe_path,
            stage_id=name,
            cache=cache,
            run=root,
            parent_receipts={
                k: root / "fits" / k / "CHECKPOINT_RECEIPT.json" for k in stage.depends_on
            },
            admissions={k: root / "fits" / k / "ENDPOINT_ADMISSION.json" for k in stage.depends_on},
            device=config["device"],
            max_updates=100,
        )
        if receipt["status"] == "COMPLETE":
            return


def write_fragment(
    path: Path, values: Mapping[str, np.ndarray], *, parents: Mapping[str, str]
) -> None:
    """Seal target-free output; recovery rejects changed contents or producer parents."""
    interface = {
        "E_CTX": "PHASE17_RAW_EVENT",
        "R_CTX": "PHASE17_RAW_RGB",
        "PHASE20": "PHASE20_CONTEXT",
    }[path.parent.name]
    if set(values) != _expected_keys(interface):
        raise ValueError("Generation fragment fields differ from the target-free interface")
    if path.exists():
        old = arrays(path)
        if set(old) != set(values) or any(not np.array_equal(old[k], v) for k, v in values.items()):
            raise ValueError("Generation fragment changed on deterministic replay")
    else:
        _atomic_npz(path, values, compressed=True)
    receipt = {
        "schema": "scientific_oof_fragment_v1",
        "status": "COMPLETE",
        "rotation": int(path.parent.parent.parent.name),
        "interface": interface,
        "role": "generation",
        "physical_payload_sha256": payload_sha256(values),
        "output_sha256": sha256(path),
        "parents": dict(parents),
        "targets_used": False,
        "rows": len(values["sample_token"]),
    }
    sidecar = path.with_suffix(".json")
    if sidecar.exists() and read_json_shared(sidecar) != receipt:
        raise ValueError("Generation fragment receipt changed")
    atomic_write_json(sidecar, receipt)


def context_branch(
    config: Mapping[str, Any],
    rotation: int,
    root: Path,
    unit: FoldTrainingUnit,
    base: AllTrainRGBSource,
    name: str,
    guard: Callable[[], None],
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one refiner on its eight groups, then generate physical states for eight others."""
    dependencies = _stage(unit, name).depends_on
    parents = {
        k: verify_scientific_admission(root / "fits" / k / "ENDPOINT_ADMISSION.json")[
            "checkpoint_sha256"
        ]
        for k in dependencies
    }
    models = {
        k: load_cached_endpoint(
            root / "fits" / k, _stage(unit, k).recipe(), torch.device(config["device"])
        )
        if k.startswith("PAIR")
        else load_fold_model(root / "fits" / k, config["device"])
        for k in dependencies
    }
    raws = {}
    try:
        for role in ("refiner", "generation"):
            guard()
            rows = role_rows(Path(config["fold_contract"]), rotation, role)
            output = root / role / name
            raw_path = (
                event_raw(
                    v12_run=Path(config["v12_run"]),
                    raw_root=Path(config["raw_root"]),
                    rows=rows,
                    models=models,
                    output=output,
                    parents=parents,
                    device=config["device"],
                    guard=guard,
                )
                if name == "E_CTX"
                else rgb_raw(
                    base,
                    rows,
                    models,
                    output,
                    device=config["device"],
                    parents=parents,
                    guard=guard,
                )
            )
            raws[role] = arrays(raw_path)
    finally:
        models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    normalizer = unique_normalizer(
        raws["refiner"],
        expected_groups=set(unit.roles["refiner"]["group_ids"]),
        expected_role_identity=unit.roles["refiner"],
    )
    normalizer.update(
        parents=parents,
        role="refiner",
        raw_sha256=sha256(root / "refiner" / name / "raw_phase17.npz"),
    )
    norm_path = root / "normalizers" / f"{name}.json"
    if norm_path.exists() and read_json_shared(norm_path) != normalizer:
        raise ValueError("Fold normalizer changed")
    atomic_write_json(norm_path, normalizer)
    mean, scale = (
        np.asarray(normalizer["mean"], np.float32),
        np.asarray(normalizer["scale"], np.float32),
    )
    train_raw = raws["refiner"]
    metadata = target_rows(base, role_rows(Path(config["fold_contract"]), rotation, "refiner"))
    metadata.pop("sequence_id")
    cache = root / "caches" / f"{name}.npz"
    seal_role_cache(
        cache,
        {
            "features": normalize_histories(train_raw, mean, scale),
            **{k: train_raw[k] for k in ("timing", "valid", "expert_phase")},
            **metadata,
            "role": np.full(len(metadata["sample_token"]), "refiner"),
        },
        schema="final_refit_context_cache_v1",
        role="refiner",
        parents=parents,
        rows_sha=unit.roles["refiner"]["sha256"],
        normalizers={name: sha256(norm_path)},
        details={"normalizer_fit_role": "refiner", "normalizer_group_ids": normalizer["groups"]},
    )
    fit_head(config, root, unit, name, cache, guard)
    generation = raws["generation"]
    write_fragment(root / "generation" / name / "fragment.npz", generation, parents=parents)
    model = load_cached_endpoint(
        root / "fits" / name, _stage(unit, name).recipe(), torch.device(config["device"])
    )
    features = normalize_histories(generation, mean, scale)
    states, widths = [], []
    for start in range(0, len(features), 128):
        guard()
        sl = slice(start, start + 128)
        block, width, _ = frozen_context_state(
            model,
            torch.from_numpy(features[sl]).to(config["device"]),
            *[
                torch.from_numpy(generation[k][sl]).to(config["device"])
                for k in ("timing", "valid", "expert_phase")
            ],
        )
        states.append(block[:, 160:170].cpu().numpy())
        widths.append(width.cpu().numpy())
    return np.concatenate(states), np.concatenate(widths)


def run_rotation(run: Path, rotation: int) -> dict[str, Any]:
    """Train producers, PAIR and CTX, then emit one exact excluded generation fold."""
    config = load_config(run)
    guard = make_guard(run)
    root = run / "rotations" / str(rotation)
    root.mkdir(parents=True, exist_ok=True)
    guard()
    sources = {str(p.resolve()): p for p in Path(__file__).parent.glob("*.py")}
    unit = build_fold_training_unit(
        Path(config["fold_contract"]), rotation=rotation, executed_sources=sources
    )
    write_fold_training_unit(root / "FOLD_TRAINING_UNIT.json", unit)
    monitor = SharedResourceMonitor(
        Path(__file__).resolve().parents[2],
        session_path=run / "RESOURCE_SESSION.json",
        snapshot_path=run / "RESOURCE_SNAPSHOT.json",
        run_root=run,
    )
    for name in FIT_NAMES:
        guard()
        if (root / "fits" / name / "ENDPOINT_ADMISSION.json").exists():
            verify_scientific_admission(root / "fits" / name / "ENDPOINT_ADMISSION.json")
            continue
        source = (
            load_fold_event_source(
                Path(config["fold_contract"]), rotation, Path(config["event_root"])
            )
            if name.startswith("E_")
            else load_fold_rgb_source(
                Path(config["rgb_source"]),
                Path(config["teacher"]),
                Path(config["fold_contract"]),
                rotation,
            )
        )
        try:
            result = train(
                source=source,
                fit=name,
                config=Path(config["config"]),
                contract=Path(config["fold_contract"]),
                run=run,
                monitor=monitor,
                device=config["device"],
            )
        finally:
            source.close()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if result["status"] != "COMPLETE":
            pause = read_json_shared(root / "fits" / name / "RESOURCE_PAUSE.json")
            raise ResourcePause(pause, manual=bool(pause.get("manual")))
    base = load_all_train_source(Path(config["rgb_source"]), Path(config["teacher"]))
    try:
        for name in ("PAIR_E", "PAIR_R"):
            cache = pair_cache(config, rotation, unit, base, root, name, guard)
            fit_head(config, root, unit, name, cache, guard)
        event, _ = context_branch(config, rotation, root, unit, base, "E_CTX", guard)
        rgb, widths = context_branch(config, rotation, root, unit, base, "R_CTX", guard)
        rows = role_rows(Path(config["fold_contract"]), rotation, "generation")
        parents = {
            k: verify_scientific_admission(root / "fits" / k / "ENDPOINT_ADMISSION.json")[
                "checkpoint_sha256"
            ]
            for k in ("E_CTX", "R_CTX")
        }
        write_fragment(
            root / "generation" / "PHASE20" / "fragment.npz",
            {
                "features": np.concatenate((rgb, event), 1),
                "rgb_width_logits": widths,
                "sample_token": rows.sample_token.to_numpy(str),
                "group_id": rows.group_id.to_numpy(str),
                "source_ordinal": rows.source_ordinal.to_numpy(np.int64),
            },
            parents=parents,
        )
    finally:
        base.close()
    fragments = {str(p.resolve()): sha256(p) for p in root.glob("generation/*/fragment.npz")}
    return {
        "scientific_endpoint": True,
        "admissions_sha256": {
            str(p.resolve()): sha256(p) for p in root.glob("fits/*/ENDPOINT_ADMISSION.json")
        },
        "fragment_files_sha256": fragments,
        "fragment_sha256": canonical_sha256(fragments),
        "phase_feature_order": "R10_then_E10",
    }


def stage_receipt(root: Path, stage: str, status: str, **details: object) -> dict[str, Any]:
    """Record durable progress across the rotation, including clean resource pauses."""
    completed = durable = pending = 0
    for path in root.glob("fits/*/UPDATE_JOURNAL.json"):
        state = read_json_shared(path)
        completed += int(state.get("completed_updates", 0))
        durable += int(state.get("durable_updates", 0))
        pending += int(state.get("pending_update_upper", 0))
    value = {
        "schema": "scientific_oof_stage_receipt_v1",
        "stage_id": stage,
        "status": status,
        "pid": os.getpid(),
        "updated_utc": datetime.now(UTC).isoformat(),
        "checkpoint_updates": 100,
        "completed_updates": completed,
        "durable_updates": durable,
        "pending_update_upper": pending,
        "manual_stop": False,
        **details,
    }
    atomic_write_json(root / "PIPELINE_RECEIPT.json", value)
    return value


def main() -> int:
    """CLI consumed by the recoverable night queue; exit 3 denotes an explicit pause."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rotation", type=int, choices=range(5), required=True)
    args = parser.parse_args()
    root = args.run / "rotations" / str(args.rotation)
    name = f"ROTATION_{args.rotation}"
    try:
        result = run_rotation(args.run, args.rotation)
        stage_receipt(root, name, "COMPLETE", **result)
        return 0
    except ResourcePause as exc:
        stage_receipt(
            root, name, "PAUSED_RESOURCE", resources=exc.resources, manual_stop=exc.manual
        )
        return 3
    except Exception:
        stage_receipt(root, name, "ERROR", traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    raise SystemExit(main())
