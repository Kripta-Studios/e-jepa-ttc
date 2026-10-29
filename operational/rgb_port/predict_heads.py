"""Generate cohort-complete V predictions from all six frozen RGB-PORT heads."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.rgb_port.accounting import (
    atomic_write_bytes,
    atomic_write_json,
    read_json_shared,
    sha256_file,
)
from operational.rgb_port.evaluate import HEAD_IDS
from operational.rgb_port.profile import profile_callable
from operational.rgb_port.train_heads import load_head_endpoint

TEMPORAL_IDS = {"E_H1_MATCHED", "E_CTX_MATCHED", "R_H1", "R_CTX"}
FUSION_IDS = {"F_TRUE", "F_ZERO"}
CSV_FIELDS = (
    "sample_token",
    "group_id",
    "target_ttc",
    "prediction_ttc",
    "phase",
    "prediction_phase",
    "raw_location_phase",
    "q10_ttc",
    "q90_ttc",
    "ttc_interval_low",
    "ttc_interval_high",
    "ttc_interval_status",
    "q10_phase",
    "q90_phase",
    "weight",
    "mass",
    "rgb_available",
    "failure_cause",
    "normalizer_sha256",
    "scoped_parents_sha256",
)


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _scoped_parent_sha(value: object) -> str:
    if isinstance(value, str) and len(value) == 64:
        return value
    return _canonical_sha(value)


def _load_inputs(path: Path) -> dict[str, Path]:
    value = read_json_shared(path)
    if set(value) != HEAD_IDS or any(not isinstance(item, str) for item in value.values()):
        raise ValueError(f"inputs JSON must map exactly these fit IDs: {sorted(HEAD_IDS)}")
    return {fit_id: Path(source).resolve(strict=True) for fit_id, source in value.items()}


def _frozen_heads(run: Path) -> tuple[dict[str, nn.Module], dict[str, dict[str, Any]]]:
    models: dict[str, nn.Module] = {}
    bindings: dict[str, dict[str, Any]] = {}
    for fit_id in sorted(HEAD_IDS):
        fit_dir = run / "fits" / fit_id
        receipt = read_json_shared(fit_dir / "CHECKPOINT_RECEIPT.json")
        if (
            receipt.get("schema") != "rgb_port_checkpoint_receipt_v1"
            or receipt.get("fit_id") != fit_id
            or receipt.get("status") != "COMPLETE"
            or receipt.get("completed_updates") != 2500
            or not receipt.get("scientific_endpoint")
            or not receipt.get("complete_state")
            or receipt.get("accumulation_index") != 0
        ):
            raise ValueError(f"{fit_id} is not an actual complete head endpoint")
        checkpoint = Path(receipt["checkpoint_path"]).resolve(strict=True)
        if receipt.get("checkpoint_sha256") != sha256_file(checkpoint):
            raise ValueError(f"{fit_id} checkpoint changed")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        identity = payload.get("identity")
        if not isinstance(identity, dict) or identity.get("fit_id") != fit_id:
            raise ValueError(f"{fit_id} checkpoint identity differs")
        if (
            not isinstance(identity.get("normalizer_sha256"), str)
            or identity.get("parent_sha256") is None
        ):
            raise ValueError(f"{fit_id} lacks frozen normalizer/parent ancestry")
        models[fit_id] = load_head_endpoint(fit_dir, fit_id, torch.device("cpu"))
        bindings[fit_id] = {
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "identity_sha256": receipt["identity_sha256"],
            "normalizer_sha256": identity.get("normalizer_sha256"),
            "parent_sha256": identity.get("parent_sha256"),
        }
    return models, bindings


def _load_cache(path: Path, fit_id: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name] for name in archive.files}
    common = {"target_phase", "mass", "sample_token", "group_id", "role"}
    required = (
        common | {"features", "timing", "valid", "expert_phase"}
        if fit_id in TEMPORAL_IDS
        else common
        | {
            "event_features",
            "event_timing",
            "event_valid",
            "event_expert_phase",
            "rgb_features",
            "rgb_timing",
            "rgb_valid",
        }
    )
    optional = {"rgb_available"} if fit_id in FUSION_IDS else set()
    if not required.issubset(values) or set(values) - required - optional:
        raise ValueError(f"{fit_id} V cache schema differs")
    count = len(values["sample_token"])
    if count < 1 or any(len(value) != count for value in values.values()):
        raise ValueError(f"{fit_id} V cache is empty or row-misaligned")
    if set(np.asarray(values["role"]).astype(str)) != {"V"}:
        raise ValueError(f"{fit_id} prediction cache is not V-only")
    tokens = np.asarray(values["sample_token"]).astype(str)
    if len(set(tokens.tolist())) != count:
        raise ValueError(f"{fit_id} V cache duplicates sample identities")
    if fit_id in FUSION_IDS and "rgb_available" in values:
        available = np.asarray(values["rgb_available"], dtype=np.bool_)
        values = {name: value[available] for name, value in values.items()}
        if not len(values["sample_token"]):
            raise ValueError(f"{fit_id} has no real RGB row to evaluate/profile")
    return values


def _to(device: torch.device, values: np.ndarray, *, dtype: torch.dtype) -> Tensor:
    return torch.as_tensor(values, dtype=dtype, device=device)


def _forward(
    model: nn.Module, fit_id: str, values: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    count = len(values["sample_token"])
    collected = {name: [] for name in ("point_phase", "raw_location", "q10", "q90")}
    for start in range(0, count, 128):
        selected = slice(start, min(start + 128, count))
        with torch.inference_mode():
            if fit_id in TEMPORAL_IDS:
                features = _to(device, values["features"][selected], dtype=torch.float32)
                timing = _to(device, values["timing"][selected], dtype=torch.float32)
                valid = _to(device, values["valid"][selected], dtype=torch.bool)
                experts = _to(device, values["expert_phase"][selected], dtype=torch.float32)
                if fit_id in {"E_H1_MATCHED", "R_H1"}:
                    features, timing, valid = features[:, -1:], timing[:, -1:], valid[:, -1:]
                output = model(features, timing, valid, experts)
            else:
                output = model(
                    _to(device, values["event_features"][selected], dtype=torch.float32),
                    _to(device, values["event_timing"][selected], dtype=torch.float32),
                    _to(device, values["event_valid"][selected], dtype=torch.bool),
                    _to(device, values["event_expert_phase"][selected], dtype=torch.float32),
                    _to(device, values["rgb_features"][selected], dtype=torch.float32),
                    _to(device, values["rgb_timing"][selected], dtype=torch.float32),
                    _to(device, values["rgb_valid"][selected], dtype=torch.bool),
                )
        for name in collected:
            collected[name].append(output[name].detach().cpu().numpy().astype(np.float64))
    result = {name: np.concatenate(parts) for name, parts in collected.items()}
    for phase_name, ttc_name in (
        ("point_phase", "prediction_ttc"),
        ("q10", "q10_ttc"),
        ("q90", "q90_ttc"),
    ):
        result[ttc_name] = (
            phase_to_ttc(torch.from_numpy(result[phase_name]).float()).numpy().astype(np.float64)
        )
    from operational.rgb_port_revision.outputs import interval_from_phase

    result["q10_ttc"], result["q90_ttc"], _ = interval_from_phase(result["q10"], result["q90"])
    return result


def _batch_one_call(
    model: nn.Module, fit_id: str, values: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    one = {name: value[:1] for name, value in values.items()}
    return _forward(model, fit_id, one)


def _metadata(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import pandas as pd

    manifest = read_json_shared(manifest_path)
    if manifest.get("status") != "COMPLETE" or manifest.get("role") != "V":
        raise ValueError("manifestV is not the complete frozen V role")
    rows_path = Path(manifest["rows_path"]).resolve(strict=True)
    if manifest.get("rows_sha256") != sha256_file(rows_path):
        raise ValueError("current V rows differ from manifestV")
    frame = pd.read_parquet(rows_path)
    required = {"sample_token", "group_id", "target_ttc"}
    if not required.issubset(frame.columns) or len(frame) != int(manifest["population_size"]):
        raise ValueError("V metadata population/schema differs")
    tokens = frame["sample_token"].astype(str).tolist()
    if len(set(tokens)) != len(tokens):
        raise ValueError("V metadata sample identities are not unique")
    target = frame["target_ttc"].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(target).all():
        raise ValueError("original V ground truth must be finite float64")
    rows = [
        {"sample_token": token, "group_id": str(group), "target_ttc": float(truth)}
        for token, group, truth in zip(tokens, frame["group_id"].astype(str), target, strict=True)
    ]
    return manifest, rows


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    from operational.rgb_port_revision.outputs import interval_from_phase

    low, high, status = interval_from_phase(
        np.array([row["q10_phase"] for row in rows]), np.array([row["q90_phase"] for row in rows])
    )
    for index, row in enumerate(rows):
        row.update(
            ttc_interval_low=low[index],
            ttc_interval_high=high[index],
            ttc_interval_status=status[index],
        )
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)  # pyright: ignore[reportArgumentType]
    atomic_write_bytes(path, stream.getvalue().encode("utf-8"))


def _prediction_source(
    fit_id: str,
    token: str,
    indexed: Mapping[str, Mapping[str, int]],
) -> tuple[str, int | None, bool]:
    """Select a direct head row or the exact E_CTX fallback for missing fusion RGB."""

    direct_index = indexed[fit_id].get(token)
    if fit_id in FUSION_IDS and direct_index is None:
        fallback_index = indexed["E_CTX_MATCHED"].get(token)
        return "E_CTX_MATCHED", fallback_index, True
    return fit_id, direct_index, False


def predict_heads(
    run: Path, manifest_path: Path, inputs_path: Path, output_dir: Path
) -> dict[str, Any]:
    """Predict all V rows once with six immutable head endpoints."""
    inputs = _load_inputs(inputs_path)
    models, bindings = _frozen_heads(run)
    completed_path = output_dir / "HEAD_PREDICTIONS.json"
    if completed_path.is_file():
        completed = read_json_shared(completed_path)
        if (
            completed.get("status") != "COMPLETE"
            or completed.get("manifest_v_sha256") != sha256_file(manifest_path)
            or completed.get("inputs_json_sha256") != sha256_file(inputs_path)
            or completed.get("endpoint_bindings") != bindings
        ):
            raise ValueError("existing head predictions have different frozen inputs")
        for artifact in completed.get("artifacts", {}).values():
            for kind in ("csv", "profile"):
                path = Path(artifact[f"{kind}_path"])
                if artifact[f"{kind}_sha256"] != sha256_file(path):
                    raise ValueError("completed head prediction artifact changed")
        return completed
    # Open V ground truth only after every one of the six endpoints is immutable.
    manifest, metadata = _metadata(manifest_path)
    caches = {fit_id: _load_cache(inputs[fit_id], fit_id) for fit_id in HEAD_IDS}
    predictions = {fit_id: _forward(models[fit_id], fit_id, caches[fit_id]) for fit_id in HEAD_IDS}
    token_order = [row["sample_token"] for row in metadata]
    metadata_by_token = {row["sample_token"]: row for row in metadata}
    indexed = {
        fit_id: {
            token: index
            for index, token in enumerate(np.asarray(cache["sample_token"]).astype(str))
        }
        for fit_id, cache in caches.items()
    }
    if set(indexed["E_CTX_MATCHED"]) != set(token_order) or set(indexed["E_H1_MATCHED"]) != set(
        token_order
    ):
        raise ValueError("event V heads must cover the complete manifest population")
    manifest_tokens = set(token_order)
    for fit_id, cache_index in indexed.items():
        if not set(cache_index).issubset(manifest_tokens):
            raise ValueError(f"{fit_id} cache contains rows outside manifestV")
        groups = np.asarray(caches[fit_id]["group_id"]).astype(str)
        tokens = np.asarray(caches[fit_id]["sample_token"]).astype(str)
        if any(
            metadata_by_token[token]["group_id"] != group
            for token, group in zip(tokens, groups, strict=True)
        ):
            raise ValueError(f"{fit_id} cache group identity differs from manifestV")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}
    for fit_id in sorted(HEAD_IDS):
        normalizer_sha = bindings[fit_id]["normalizer_sha256"]
        parent_sha = _scoped_parent_sha(bindings[fit_id]["parent_sha256"])
        rows: list[dict[str, Any]] = []
        for token in token_order:
            source_fit, source_index, fallback = _prediction_source(fit_id, token, indexed)
            meta = metadata_by_token[token]
            rgb_available = token in indexed["R_CTX"]
            if source_index is None:
                if fit_id not in {"R_H1", "R_CTX"}:
                    raise ValueError(f"{fit_id} unexpectedly lacks V row {token}")
                point = raw = q10 = q90 = prediction = q10_ttc = q90_ttc = float("nan")
                event_index = indexed["E_CTX_MATCHED"][token]
                mass = float(caches["E_CTX_MATCHED"]["mass"][event_index])
                failure = "MISSING_RGB_INPUT"
            else:
                value = predictions[source_fit]
                point = float(value["point_phase"][source_index])
                raw = float(value["raw_location"][source_index])
                q10 = float(value["q10"][source_index])
                q90 = float(value["q90"][source_index])
                prediction = float(value["prediction_ttc"][source_index])
                q10_ttc = float(value["q10_ttc"][source_index])
                q90_ttc = float(value["q90_ttc"][source_index])
                mass = float(caches[source_fit]["mass"][source_index])
                failure = "RGB_UNAVAILABLE_FALLBACK_E_CTX" if fallback else ""
            rows.append(
                {
                    **meta,
                    "prediction_ttc": prediction,
                    "phase": point,
                    "prediction_phase": point,
                    "raw_location_phase": raw,
                    "q10_ttc": q10_ttc,
                    "q90_ttc": q90_ttc,
                    "q10_phase": q10,
                    "q90_phase": q90,
                    "weight": mass,
                    "mass": mass,
                    "rgb_available": rgb_available,
                    "failure_cause": failure,
                    "normalizer_sha256": normalizer_sha,
                    "scoped_parents_sha256": parent_sha,
                }
            )
        csv_path = output_dir / f"{fit_id}.csv"
        _atomic_csv(csv_path, rows)
        profile = profile_callable(
            lambda model=models[fit_id], name=fit_id, cache=caches[fit_id]: _batch_one_call(
                model, name, cache
            ),
            scope=f"HEAD_ONLY:{fit_id}:excludes_producers_cache_io_and_preprocessing",
            device="cpu",
            warm_iterations=20,
        )
        profile.update(
            fit_id=fit_id,
            checkpoint_sha256=bindings[fit_id]["checkpoint_sha256"],
            cache_sha256=sha256_file(inputs[fit_id]),
        )
        profile_path = output_dir / f"{fit_id}.HEAD_ONLY_PROFILE.json"
        atomic_write_json(profile_path, profile)
        artifacts[fit_id] = {
            "csv_path": str(csv_path.resolve()),
            "csv_sha256": sha256_file(csv_path),
            "profile_path": str(profile_path.resolve()),
            "profile_sha256": sha256_file(profile_path),
            "rows": len(rows),
            "native_unfiltered": True,
        }
    result = {
        "schema": "rgb_port_head_predictions_v1",
        "status": "COMPLETE",
        "scope": "HEAD_ONLY_predictions_and_latency",
        "does_not_claim_full_pipeline_cost": True,
        "manifest_v_sha256": sha256_file(manifest_path),
        "rows_sha256": manifest["rows_sha256"],
        "population": len(metadata),
        "target_ttc_source_dtype": "float64_from_manifest_rows",
        "ttc_interval_semantics": (
            "image_of_phase_q10_q90_interval_on_one_monotone_branch; "
            "legacy_q10_ttc_q90_ttc_are_bound_aliases_not_guaranteed_marginal_quantiles; "
            "zero_crossings_unavailable_with_phase_bounds_retained"
        ),
        "row_order_sha256": hashlib.sha256(
            "\n".join(f"{row['sample_token']}\0{row['group_id']}" for row in metadata).encode()
        ).hexdigest(),
        "inputs_json_sha256": sha256_file(inputs_path),
        "endpoint_bindings": bindings,
        "artifacts": artifacts,
    }
    atomic_write_json(output_dir / "HEAD_PREDICTIONS.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--manifestV", type=Path, required=True)
    parser.add_argument("--inputsJSON", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = predict_heads(
        args.run.resolve(strict=True),
        args.manifestV.resolve(strict=True),
        args.inputsJSON.resolve(strict=True),
        args.output_dir.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["predict_heads", "main"]
