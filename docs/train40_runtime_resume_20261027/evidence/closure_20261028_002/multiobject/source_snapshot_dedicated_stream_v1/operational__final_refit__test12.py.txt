"""Seal-bound, resumable inference for the public GarlTTC test12 inputs.

The input inventory is label-free and may be reused from V16, but every sensor
observation and prediction is recomputed with the completed TRAIN40 chain.  No
submission is uploaded by this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn

from e_jepa_ttc.rgb_port.data import TarFrameReader, decode_query
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.rgb_port import transfer as rgb_engine
from operational.rgb_port.infer_experts import clock_features, delta_matrix

from .event_parity import load_v12_event_producer
from .export import (
    HEAD_STAGE,
    MODEL_SEAL_SCHEMA,
    OFFICIAL_TEST_QUERIES,
    PREDICTION_SCHEMA,
)
from .features import frozen_context_state
from .producers import build_rgb_recipe, load_rgb_endpoint
from .recipe import FinalRefitRecipe, StageRecipe, canonical_sha256, load_recipe, sha256, stage
from .rgb import load_all_train_source
from .training import load_cached_endpoint

FALLBACK_POLICY = "E_CTX_WHEN_RGB_HISTORY_UNAVAILABLE"
FRAGMENT_SCHEMA = "final_refit_test12_prediction_fragment_v1"
TRUSTED_INPUTS_SHA256 = "b78730abe68de7fbe06c6c98b25d5fa025c04d9303720b680ce0c0f575d2e526"
TRUSTED_OFFICIAL_TEST_INPUTS_SHA256 = (
    "aedb9da5484ceedb642ef1cbc7cf6115dcf7f368e4444b55de947989fc19a362"
)
TRUSTED_MODEL_INPUT_FREEZE_SHA256 = (
    "85fad0dfb8bac0fa5a09e6f4090818ea2b534908ef3d018e8f88f0805e825b1a"
)
TRUSTED_EVENT_INPUT_FREEZE_SHA256 = (
    "b3c4514bebb54112e94ad9ab52f997e5676e3ee89139294ab1b265ee9a650934"
)
FORBIDDEN_INPUT_KEYS = {
    "target",
    "target_phase",
    "target_ttc",
    "ttc",
    "label",
    "labels",
    "ground_truth",
    "ground_truth_ttc",
    "gt_ttc",
    "truth_ttc_seconds",
    "depth",
    "distance",
    "velocity",
    "phase",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _contains_private_key(value: object) -> bool:
    if isinstance(value, dict):
        if FORBIDDEN_INPUT_KEYS.intersection(str(key).lower() for key in value):
            return True
        return any(_contains_private_key(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_private_key(item) for item in value)
    return False


def _history_is_causal(
    identities: Sequence[str],
    observations: Mapping[str, Mapping[str, Any]],
    query: Mapping[str, Any],
    query_tokens: Mapping[str, Mapping[str, Any]],
    *,
    modality: str,
) -> None:
    if not 1 <= len(identities) <= 8 or len(set(identities)) != len(identities):
        raise ValueError("History must contain one to eight unique observations")
    previous = -1
    query_us = int(query["query_time_us"])
    for identity in identities:
        if identity not in observations:
            raise ValueError(f"History references unknown observation {identity}")
        item = observations[identity]
        owner_token = str(item.get("sample_token", ""))
        owner = query_tokens.get(owner_token)
        if (
            owner is None
            or str(owner.get("sequence_id")) != str(query["sequence_id"])
            or str(owner.get("track_id")) != str(query["track_id"])
        ):
            raise ValueError("History observation crosses an object or sequence boundary")
        anchor = int(item["anchor_us"])
        available = int(item["available_us"])
        sensor_times = [int(value) for value in item.get("producer_sensor_times_us", [])]
        if "input_span_us" in item:
            span = int(item["input_span_us"])
        else:
            if len(sensor_times) < 2 or any(
                later <= earlier
                for earlier, later in zip(sensor_times[:-1], sensor_times[1:], strict=True)
            ):
                raise ValueError("Observation has no valid native sensor span")
            span = sensor_times[-1] - sensor_times[0]
        if modality == "event":
            windows = np.asarray(item.get("windows_us", []), dtype=np.int64)
            sequence = str(query["sequence_id"])
            raw_parts = Path(str(item.get("raw_path", ""))).parts
            if (
                item.get("sequence_id") != sequence
                or item.get("track_id") != query["track_id"]
                or windows.shape != (3, 2)
                or np.any(windows[:, 1] <= windows[:, 0])
                or np.any(np.diff(windows, axis=0) < 0)
                or int(item.get("sensor_anchor_us", -1)) != int(windows[-1, 1])
                or np.any(windows[:, 1] > int(item.get("sensor_anchor_us", -1)))
                or span != int(windows[-1, 1] - windows[0, 0])
                or tuple(raw_parts[-4:]) != ("data", "test", sequence, "events.h5")
                or abs(
                    float(item.get("delta_t_s", np.nan))
                    - float(windows[-1, 0] - windows[-2, 0]) * 1e-6
                )
                > 2.1e-6
            ):
                raise ValueError("Event observation windows or sensor clock differ")
        elif modality == "rgb":
            aligned = [
                item.get(name, [])
                for name in (
                    "frame_ids",
                    "producer_times_us",
                    "producer_available_us",
                    "producer_sensor_times_us",
                    "producer_shards",
                    "producer_members",
                    "producer_boxes_xyxy",
                )
            ]
            lengths = {len(value) for value in aligned}
            times = [int(value) for value in item.get("producer_times_us", [])]
            arrivals = [int(value) for value in item.get("producer_available_us", [])]
            sequence = str(query["sequence_id"])
            track = str(query["track_id"])
            if (
                lengths != {len(times)}
                or len(times) < 2
                or any(b <= a for a, b in zip(times[:-1], times[1:], strict=True))
                or any(b <= a for a, b in zip(arrivals[:-1], arrivals[1:], strict=True))
                or any(b <= a for a, b in zip(sensor_times[:-1], sensor_times[1:], strict=True))
                or not np.allclose(
                    np.diff(np.asarray(times, dtype=np.int64)),
                    np.diff(np.asarray(sensor_times, dtype=np.int64)),
                    rtol=0.0,
                    atol=2.0,
                )
                or any(
                    arrival < time or arrival > query_us
                    for time, arrival in zip(times, arrivals, strict=True)
                )
                or any(time > query_us for time in times)
                or anchor != times[-1]
                or available != max(arrivals)
                or int(item.get("query_time_us", -1)) != anchor
                or int(item.get("query_age_us", -1)) != 0
                or any(not str(value).startswith(f"{track}|") for value in item["frame_ids"])
                or any(
                    not str(value).replace("\\", "/").startswith(f"data/test/{sequence}/")
                    for value in item["producer_shards"]
                )
            ):
                raise ValueError("RGB observation identity or clocks differ")
        else:
            raise ValueError("Unknown test12 modality")
        if (
            anchor <= previous
            or anchor > query_us
            or available < anchor
            or available > query_us
            or span <= 0
            or query_us - anchor + span > 650_000
        ):
            raise ValueError("History violates the native causal <=650 ms contract")
        previous = anchor


def validate_test12_inputs(
    inputs_path: Path,
    *,
    official_test_inputs_path: Path,
    expected_inputs_sha256: str = TRUSTED_INPUTS_SHA256,
    expected_official_test_inputs_sha256: str = TRUSTED_OFFICIAL_TEST_INPUTS_SHA256,
    expected_model_input_freeze_sha256: str = TRUSTED_MODEL_INPUT_FREEZE_SHA256,
    expected_event_input_freeze_sha256: str = TRUSTED_EVENT_INPUT_FREEZE_SHA256,
    verify_rgb_sources: bool = True,
    model_input_freeze_path: Path | None = None,
    event_input_freeze_path: Path | None = None,
) -> dict[str, Any]:
    """Validate exact coverage, physical aliases, clocks and absence of private labels."""

    inputs_path = inputs_path.resolve(strict=True)
    official_test_inputs_path = official_test_inputs_path.resolve(strict=True)
    if sha256(inputs_path) != expected_inputs_sha256:
        raise ValueError("Test12 INPUTS.json differs from the trusted frozen inventory")
    if sha256(official_test_inputs_path) != expected_official_test_inputs_sha256:
        raise ValueError("Official test_inputs.parquet differs from the trusted release")
    value = _read(inputs_path)
    if _contains_private_key(value):
        raise ValueError("Public test input inventory contains a target/label field")
    queries = value.get("queries")
    event = value.get("event_observations")
    rgb = value.get("rgb_observations")
    aliases = value.get("physical_aliases")
    if not isinstance(queries, list):
        raise ValueError("Public test input query inventory differs")
    if not isinstance(event, dict) or not isinstance(rgb, dict) or not isinstance(aliases, dict):
        raise ValueError("Public test input inventory fields differ")
    event = cast(dict[str, Mapping[str, Any]], event)
    rgb = cast(dict[str, Mapping[str, Any]], rgb)
    aliases = cast(dict[str, str], aliases)
    tokens = [str(row.get("sample_token", "")) for row in queries]
    if len(tokens) != OFFICIAL_TEST_QUERIES or len(set(tokens)) != OFFICIAL_TEST_QUERIES:
        raise ValueError("Official test12 requires exactly 6,762 unique sample tokens")
    official_frame = pd.read_parquet(official_test_inputs_path, columns=["sample_token"])
    official_tokens = official_frame["sample_token"].astype(str).tolist()
    if (
        len(official_tokens) != OFFICIAL_TEST_QUERIES
        or len(set(official_tokens)) != OFFICIAL_TEST_QUERIES
        or set(tokens) != set(official_tokens)
    ):
        raise ValueError("INPUTS.json tokens differ from official test_inputs.parquet")
    if set(aliases) != set(tokens) or not set(aliases.values()).issubset(set(tokens)):
        raise ValueError("Physical alias map must cover exactly every official token")
    by_token = {str(row["sample_token"]): row for row in queries}
    fallback_count = 0
    for row in queries:
        token = str(row["sample_token"])
        canonical = str(aliases[token])
        reference = by_token[canonical]
        for key in ("sequence_id", "track_id", "query_time_us", "event_history", "rgb_history"):
            if row.get(key) != reference.get(key):
                raise ValueError("Physical aliases differ in native sensor history")
        _history_is_causal(
            [str(item) for item in row["event_history"]],
            event,
            row,
            by_token,
            modality="event",
        )
        rgb_history = [str(item) for item in row["rgb_history"]]
        if rgb_history:
            _history_is_causal(rgb_history, rgb, row, by_token, modality="rgb")
        else:
            fallback_count += 1
    media_root_raw = value.get("media_root")
    media_root = (
        Path(str(media_root_raw)).resolve(strict=True) if media_root_raw is not None else None
    )
    rgb_metadata_verified = 0
    if media_root is not None:
        for item in event.values():
            raw_path = Path(str(item.get("raw_path", ""))).resolve(strict=True)
            if media_root not in raw_path.parents:
                raise ValueError("Event HDF5 escapes the public media root")
    if verify_rgb_sources:
        if media_root is None:
            raise ValueError("Public media root is missing")
        sources = value.get("rgb_sources")
        if not isinstance(sources, dict):
            raise ValueError("RGB source inventory is missing")
        for relative, raw_receipt in sources.items():
            if not isinstance(raw_receipt, dict):
                raise ValueError("RGB source receipt differs")
            receipt = cast(dict[str, Any], raw_receipt)
            path = (media_root / str(relative)).resolve(strict=True)
            if media_root not in path.parents:
                raise ValueError("RGB source escapes the public media root")
            stat = path.stat()
            if stat.st_size != int(receipt["size_bytes"]) or stat.st_mtime_ns != int(
                receipt["mtime_ns"]
            ):
                raise ValueError(f"RGB public source changed: {relative}")
            metadata = media_root / ".cache/huggingface/download" / f"{relative}.metadata"
            lines = metadata.read_text(encoding="utf-8").splitlines()
            if (
                len(lines) < 2
                or lines[0] != receipt.get("revision")
                or lines[1] != receipt.get("sha256")
            ):
                raise ValueError(f"RGB Hugging Face source metadata changed: {relative}")
            rgb_metadata_verified += 1
    provenance: dict[str, Any] = {
        "verified": False,
        "model_input_freeze_sha256": None,
        "event_input_freeze_sha256": None,
        "event_tensor_receipts_verified": 0,
        "rgb_hf_metadata_receipts_verified": rgb_metadata_verified,
        "event_raw_h5_rehashed": False,
        "rgb_tar_shards_rehashed": False,
    }
    if (model_input_freeze_path is None) != (event_input_freeze_path is None):
        raise ValueError("Both frozen input provenance files are required together")
    if model_input_freeze_path is not None and event_input_freeze_path is not None:
        model_freeze = _read(model_input_freeze_path.resolve(strict=True))
        event_freeze_path = event_input_freeze_path.resolve(strict=True)
        event_freeze = _read(event_freeze_path)
        if (
            sha256(model_input_freeze_path) != expected_model_input_freeze_sha256
            or sha256(event_freeze_path) != expected_event_input_freeze_sha256
            or model_freeze.get("inputs_sha256") != sha256(inputs_path)
            or model_freeze.get("test_targets_read") is not False
            or model_freeze.get("external_submission") is not False
            or event_freeze.get("schema") != "garlttc_native_h8_test_inputs_v1"
            or event_freeze.get("test_labels_read") is not False
            or model_freeze.get("test_input_sha256") != event_freeze.get("input_sha256")
        ):
            raise ValueError("Frozen public input provenance differs")
        frozen_rows = {str(row["sample_token"]): row for row in event_freeze.get("rows", [])}
        event_freeze_sha = sha256(event_freeze_path)
        raw_claims: dict[str, str] = {}
        for identity, item in event.items():
            frozen = frozen_rows.get(identity)
            if frozen is None or (
                item.get("sample_token") != identity
                or item.get("sequence_id") != frozen.get("sequence")
                or item.get("windows_us") != frozen.get("windows")
                or item.get("square_xyxy") != frozen.get("square")
                or item.get("raw_path") != frozen.get("path")
                or int(item.get("sensor_anchor_us", -1)) != int(frozen.get("anchor", -2))
                or np.float32(item.get("delta_t_s")) != np.float32(frozen.get("delta"))
            ):
                raise ValueError("Event observation differs from the frozen native input row")
            tensor = Path(str(item.get("current_tensor_path", ""))).resolve(strict=True)
            tensor_receipt = _read(tensor.with_suffix(".json"))
            if (
                sha256(tensor) != item.get("current_tensor_sha256")
                or tensor_receipt.get("input_freeze_sha256") != event_freeze_sha
                or tensor_receipt.get("sample_token") != identity
                or tensor_receipt.get("sha256") != item.get("current_tensor_sha256")
                or not isinstance(tensor_receipt.get("raw_sha256"), str)
                or len(tensor_receipt["raw_sha256"]) != 64
            ):
                raise ValueError("Event tensor receipt differs from frozen public provenance")
            raw_path = str(item["raw_path"])
            prior = raw_claims.setdefault(raw_path, str(tensor_receipt["raw_sha256"]))
            if prior != tensor_receipt["raw_sha256"]:
                raise ValueError("Event raw-source hash claims disagree within one HDF5")
        provenance.update(
            verified=True,
            model_input_freeze_sha256=sha256(model_input_freeze_path),
            event_input_freeze_sha256=event_freeze_sha,
            event_tensor_receipts_verified=len(event),
            event_raw_h5_hash_claims=len(raw_claims),
        )
    return {
        "inputs": value,
        "tokens": tokens,
        "canonical_tokens": list(dict.fromkeys(str(aliases[token]) for token in tokens)),
        "fallback_count": fallback_count,
        "inputs_sha256": sha256(inputs_path),
        "official_test_inputs_sha256": sha256(official_test_inputs_path),
        "input_provenance": provenance,
        "input_provenance_verified": provenance["verified"],
    }


class Test12Predictor(Protocol):
    """One immutable end-to-end predictor over a validated canonical query."""

    def predict(self, query: Mapping[str, Any], inputs: Mapping[str, Any]) -> float: ...

    def close(self) -> None: ...


class PredictionFragments:
    """Durable per-physical-query fragments bound to model and input hashes."""

    def __init__(self, root: Path, binding_sha256: str) -> None:
        self.root = root
        self.binding_sha256 = binding_sha256

    def _path(self, token: str) -> Path:
        return self.root / f"{hashlib.sha256(token.encode()).hexdigest()}.json"

    def read(self, token: str) -> float | None:
        path = self._path(token)
        if not path.exists():
            return None
        value = _read(path)
        if (
            set(value)
            != {
                "schema",
                "sample_token",
                "prediction",
                "binding_sha256",
                "private_targets_read",
            }
            or value.get("schema") != FRAGMENT_SCHEMA
            or value.get("sample_token") != token
            or value.get("binding_sha256") != self.binding_sha256
            or value.get("private_targets_read") is not False
            or not np.isfinite(float(value.get("prediction", np.nan)))
        ):
            raise ValueError(f"Prediction fragment differs from this sealed run: {token}")
        return float(value["prediction"])

    def write(self, token: str, prediction: float) -> None:
        if not np.isfinite(prediction):
            raise ValueError("Test12 prediction must be finite")
        path = self._path(token)
        if path.exists():
            raise FileExistsError(path)
        _atomic_json(
            path,
            {
                "schema": FRAGMENT_SCHEMA,
                "sample_token": token,
                "prediction": float(prediction),
                "binding_sha256": self.binding_sha256,
                "private_targets_read": False,
            },
        )


def _seal_identity(model_seal_path: Path) -> tuple[dict[str, Any], str]:
    seal = _read(model_seal_path.resolve(strict=True))
    if (
        seal.get("schema") != MODEL_SEAL_SCHEMA
        or seal.get("status") != "SEALED_NOT_EVALUATED"
        or seal.get("candidate") != HEAD_STAGE
        or seal.get("scope", {}).get("private_targets_read") is not False
    ):
        raise ValueError("Inference requires the complete unevaluated final TRAIN40 seal")
    return seal, sha256(model_seal_path)


def _validate_cached_stage_identity(
    checkpoint: Path,
    component: Mapping[str, Any],
    declared: StageRecipe,
    recipe: FinalRefitRecipe,
    checkpoint_hashes: Mapping[str, str],
) -> None:
    """Reject a valid-looking endpoint detached from this recipe or its exact parents."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    identity = payload.get("identity")
    if not isinstance(identity, dict):
        raise ValueError(f"Final endpoint has no training identity: {declared.stage_id}")
    unhashed = dict(identity)
    claimed = unhashed.pop("identity_sha256", None)
    expected_parents = {name: checkpoint_hashes[name] for name in declared.depends_on}
    if (
        payload.get("schema") != "rgb_port_producer_checkpoint_v1"
        or payload.get("identity_sha256") != claimed
        or component.get("identity_sha256") != claimed
        or claimed != canonical_sha256(unhashed)
        or identity.get("schema") != "final_refit_cached_stage_identity_v1"
        or identity.get("fit_id") != declared.stage_id
        or identity.get("kind") != declared.kind
        or identity.get("recipe_sha256") != recipe.recipe_sha256
        or identity.get("parent_checkpoint_sha256") != expected_parents
        or identity.get("technical") is not False
    ):
        raise ValueError(f"Final endpoint ancestry differs: {declared.stage_id}")


def run_test12_inference(
    *,
    inputs_path: Path,
    official_test_inputs_path: Path,
    model_seal_path: Path,
    predictor: Test12Predictor,
    work_directory: Path,
    predictions_output: Path,
    receipt_output: Path,
    verify_rgb_sources: bool = True,
    validated_inputs: dict[str, Any] | None = None,
    expected_inputs_sha256: str = TRUSTED_INPUTS_SHA256,
    expected_official_test_inputs_sha256: str = TRUSTED_OFFICIAL_TEST_INPUTS_SHA256,
    expected_model_input_freeze_sha256: str = TRUSTED_MODEL_INPUT_FREEZE_SHA256,
    expected_event_input_freeze_sha256: str = TRUSTED_EVENT_INPUT_FREEZE_SHA256,
    model_input_freeze_path: Path | None = None,
    event_input_freeze_path: Path | None = None,
) -> dict[str, Any]:
    """Run or resume all public queries and emit the receipt consumed by export.py."""

    if receipt_output.exists():
        raise FileExistsError("Final prediction receipt already exists")
    validated = (
        validate_test12_inputs(
            inputs_path,
            official_test_inputs_path=official_test_inputs_path,
            expected_inputs_sha256=expected_inputs_sha256,
            expected_official_test_inputs_sha256=expected_official_test_inputs_sha256,
            expected_model_input_freeze_sha256=expected_model_input_freeze_sha256,
            expected_event_input_freeze_sha256=expected_event_input_freeze_sha256,
            verify_rgb_sources=verify_rgb_sources,
            model_input_freeze_path=model_input_freeze_path,
            event_input_freeze_path=event_input_freeze_path,
        )
        if validated_inputs is None
        else validated_inputs
    )
    if validated.get("inputs_sha256") != sha256(inputs_path.resolve(strict=True)):
        raise ValueError("Prevalidated test12 inputs changed before inference")
    if (
        validated.get("inputs_sha256") != expected_inputs_sha256
        or validated.get("official_test_inputs_sha256") != expected_official_test_inputs_sha256
        or sha256(official_test_inputs_path.resolve(strict=True))
        != expected_official_test_inputs_sha256
    ):
        raise ValueError("Prevalidated test12 inventory is not the trusted official population")
    if validated.get("input_provenance_verified") is not True:
        raise ValueError("Final test12 inference requires frozen public-input provenance")
    validated_provenance = validated.get("input_provenance", {})
    if (
        validated_provenance.get("model_input_freeze_sha256") != expected_model_input_freeze_sha256
        or validated_provenance.get("event_input_freeze_sha256")
        != expected_event_input_freeze_sha256
    ):
        raise ValueError("Prevalidated test12 provenance is not the trusted frozen lineage")
    _, seal_sha = _seal_identity(model_seal_path)
    inference_source_sha = sha256(Path(__file__).resolve())
    binding = canonical_sha256(
        {
            "model_seal_sha256": seal_sha,
            "inputs_sha256": validated["inputs_sha256"],
            "inference_source_sha256": inference_source_sha,
        }
    )
    fragments = PredictionFragments(work_directory.resolve() / "prediction_fragments", binding)
    inputs = cast(dict[str, Any], validated["inputs"])
    aliases = cast(dict[str, str], inputs["physical_aliases"])
    by_token = {str(row["sample_token"]): row for row in inputs["queries"]}
    physical: dict[str, float] = {}
    try:
        for canonical in validated["canonical_tokens"]:
            restored = fragments.read(canonical)
            if restored is None:
                restored = float(predictor.predict(by_token[canonical], inputs))
                fragments.write(canonical, restored)
            physical[canonical] = restored
    finally:
        predictor.close()
    records = []
    for token in validated["tokens"]:
        query = by_token[token]
        rgb_available = bool(query["rgb_history"])
        records.append(
            {
                "sample_token": token,
                "prediction": physical[str(aliases[token])],
                "rgb_available": rgb_available,
                "fallback_source": "PHASE_ONLY" if rgb_available else "E_CTX",
            }
        )
    frame = pd.DataFrame.from_records(records)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = predictions_output.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp.csv")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    if predictions_output.exists():
        if predictions_output.read_bytes() != temporary.read_bytes():
            temporary.unlink()
            raise ValueError("Orphan prediction CSV differs from resumed sealed fragments")
        temporary.unlink()
    else:
        os.replace(temporary, predictions_output)
    receipt: dict[str, Any] = {
        "schema": PREDICTION_SCHEMA,
        "status": "COMPLETE",
        "candidate": HEAD_STAGE,
        "model_seal_sha256": seal_sha,
        "inputs_sha256": validated["inputs_sha256"],
        "predictions_sha256": sha256(predictions_output),
        "population": OFFICIAL_TEST_QUERIES,
        "physical_queries": len(validated["canonical_tokens"]),
        "computed_under_seal": True,
        "private_targets_read": False,
        "external_submission": False,
        "fallback_policy": FALLBACK_POLICY,
        "fallback_count": validated["fallback_count"],
        "causal_history_contract": "native same-object suffix, <=8 observations, <=650ms",
        "run_binding_sha256": binding,
        "inference_source_sha256": inference_source_sha,
        "input_provenance": validated["input_provenance"],
    }
    _atomic_json(receipt_output, receipt)
    return receipt


class FinalV16Train40Predictor:
    """Actual sensor-to-TTC backend for the sealed final V16-derived chain."""

    def __init__(
        self,
        *,
        recipe_path: Path,
        model_seal_path: Path,
        run_root: Path,
        v12_run: Path,
        v12_protocol_path: Path,
        rgb_source_descriptor: Path,
        teacher_manifest: Path,
        producer_config: Path,
        work_directory: Path,
        inputs: Mapping[str, Any],
        device: str,
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable")
        self.device = torch.device(device)
        self.work_directory = work_directory.resolve()
        self.inputs = inputs
        self.seal, self.seal_sha = _seal_identity(model_seal_path)
        self.recipe = load_recipe(recipe_path.resolve(strict=True))
        if self.seal.get("recipe_sha256") != self.recipe.recipe_sha256:
            raise ValueError("Model seal and final recipe differ")
        components = {
            str(item["stage"]): item for item in self.seal["components"]["trained_stages"]
        }
        external = {
            str(item["stage"]): item for item in self.seal["components"]["external_event_parents"]
        }
        expected_stages = {item.stage_id for item in self.recipe.stages}
        if set(components) != expected_stages or set(external) != {"E_A5", "E_C2F"}:
            raise ValueError("Model seal component graph differs from the final recipe")
        checkpoint_hashes = {
            **{name: str(item["checkpoint_sha256"]) for name, item in components.items()},
            **{name: str(item["checkpoint_sha256"]) for name, item in external.items()},
        }
        self.models: dict[str, nn.Module] = {}
        protocol = _read(v12_protocol_path.resolve(strict=True))
        for arm, seal_name, runtime_name in (
            ("a5", "E_A5", "E_A5_MATCHED"),
            ("c2f", "E_C2F", "E_C2F_MATCHED"),
        ):
            model, identity = load_v12_event_producer(v12_run.resolve(strict=True), protocol, arm)
            if identity["checkpoint_sha256"] != external[seal_name]["checkpoint_sha256"]:
                raise ValueError(f"Sealed event producer differs: {seal_name}")
            self.models[runtime_name] = model.to(self.device)
        source = load_all_train_source(
            rgb_source_descriptor.resolve(strict=True), teacher_manifest.resolve(strict=True)
        )
        try:
            for fit_id in ("R_A5", "R_C2F"):
                producer_recipe = build_rgb_recipe(
                    producer_config.resolve(strict=True),
                    fit_id,
                    rgb_source_descriptor.resolve(strict=True),
                    source,
                )
                fit = run_root.resolve(strict=True) / "fits" / fit_id
                if sha256(fit / "checkpoint_last.pt") != components[fit_id]["checkpoint_sha256"]:
                    raise ValueError(f"Sealed RGB producer differs: {fit_id}")
                self.models[fit_id] = load_rgb_endpoint(
                    fit, producer_recipe, device=str(self.device)
                )
        finally:
            source.close()
        for fit_id in ("PAIR_E_MATCHED", "PAIR_R", "E_CTX_MATCHED", "R_CTX", HEAD_STAGE):
            fit = run_root.resolve(strict=True) / "fits" / fit_id
            checkpoint = fit / "checkpoint_last.pt"
            if sha256(checkpoint) != components[fit_id]["checkpoint_sha256"]:
                raise ValueError(f"Sealed final endpoint differs: {fit_id}")
            _validate_cached_stage_identity(
                checkpoint,
                components[fit_id],
                stage(self.recipe, fit_id),
                self.recipe,
                checkpoint_hashes,
            )
            self.models[fit_id] = load_cached_endpoint(fit, stage(self.recipe, fit_id), self.device)
        self.normalizers: dict[str, tuple[Tensor, Tensor]] = {}
        for item in self.seal["components"]["normalizers"]:
            modality = str(item["modality"])
            path = Path(str(item["path"])).resolve(strict=True)
            if sha256(path) != item["sha256"]:
                raise ValueError(f"Sealed {modality} normalizer changed")
            payload = _read(path)
            self.normalizers[modality] = (
                torch.tensor(payload["mean"], dtype=torch.float32, device=self.device),
                torch.tensor(payload["scale"], dtype=torch.float32, device=self.device),
            )
        self.reader = TarFrameReader(
            Path(str(inputs["media_root"])), max_open=2, max_frame_cache_bytes=128 * 1024**2
        )
        self.event_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.rgb_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def _event_observation(self, identity: str) -> tuple[np.ndarray, np.ndarray]:
        if identity not in self.event_values:
            item = self.inputs["event_observations"][identity]
            cache = Path(str(item["current_tensor_path"])).resolve(strict=True)
            if sha256(cache) != item["current_tensor_sha256"]:
                raise ValueError(f"Frozen public event tensor changed: {identity}")
            with np.load(cache, allow_pickle=False) as archive:
                if not bool(archive["valid"][-1]):
                    raise ValueError("Current event observation is unavailable")
                sensor = torch.from_numpy(archive["events"][-1:]).to(self.device)
            delta = delta_matrix(torch.tensor([item["delta_t_s"]]), 3).to(self.device)
            with torch.inference_mode():
                raw, experts = rgb_engine._expert_block(self.models, "E", sensor, delta)
            self.event_values[identity] = (raw[0].cpu().numpy(), experts[0].cpu().numpy())
        return self.event_values[identity]

    def _rgb_observation(self, identity: str) -> tuple[np.ndarray, np.ndarray]:
        if identity not in self.rgb_values:
            item = self.inputs["rgb_observations"][identity]
            decoded = decode_query(
                item, eap_root=Path(str(self.inputs["media_root"])), reader=self.reader
            )
            sensor = torch.from_numpy(decoded["rgb"])[None].to(self.device)
            delta = torch.from_numpy(decoded["delta_t_s"])[None].to(self.device)
            with torch.inference_mode():
                raw, experts = rgb_engine._expert_block(self.models, "R", sensor, delta)
            self.rgb_values[identity] = (raw[0].cpu().numpy(), experts[0].cpu().numpy())
        return self.rgb_values[identity]

    def _context(
        self, identities: Sequence[str], modality: str, query_us: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        observations = self.inputs[f"{modality}_observations"]
        values = [
            self._event_observation(identity)
            if modality == "event"
            else self._rgb_observation(identity)
            for identity in identities
        ]
        count = len(identities)
        features = torch.zeros((1, 8, 17), dtype=torch.float32, device=self.device)
        raw = torch.from_numpy(np.stack([item[0] for item in values])).to(self.device)
        mean, scale = self.normalizers[modality]
        features[0, -count:] = (raw - mean) / scale
        anchors = [int(observations[identity]["anchor_us"]) for identity in identities]
        available = [int(observations[identity]["available_us"]) for identity in identities]
        timing = torch.zeros((1, 8, 4), dtype=torch.float32, device=self.device)
        timing[0, -count:] = torch.from_numpy(clock_features(anchors, available, query_us)).to(
            self.device
        )
        valid = torch.zeros((1, 8), dtype=torch.bool, device=self.device)
        valid[0, -count:] = True
        experts = torch.from_numpy(values[-1][1])[None].to(self.device)
        return features, timing, valid, experts

    def predict(self, query: Mapping[str, Any], inputs: Mapping[str, Any]) -> float:
        del inputs
        query_us = int(query["query_time_us"])
        event_context = self._context(
            [str(item) for item in query["event_history"]], "event", query_us
        )
        with torch.inference_mode():
            event_block, _, event_output = frozen_context_state(
                self.models["E_CTX_MATCHED"], *event_context
            )
            if not query["rgb_history"]:
                return float(phase_to_ttc(event_output["point_phase"])[0].cpu())
            rgb_context = self._context(
                [str(item) for item in query["rgb_history"]], "rgb", query_us
            )
            rgb_block, width, _ = frozen_context_state(self.models["R_CTX"], *rgb_context)
            output = self.models[HEAD_STAGE](torch.cat((rgb_block, event_block), -1), width)
            return float(output["ttc"][0].cpu())

    def close(self) -> None:
        self.reader.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--official-test-inputs", type=Path, required=True)
    parser.add_argument("--model-seal", type=Path, required=True)
    parser.add_argument("--model-input-freeze", type=Path, required=True)
    parser.add_argument("--event-input-freeze", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--v12-run", type=Path, required=True)
    parser.add_argument("--v12-protocol", type=Path, required=True)
    parser.add_argument("--rgb-source", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--producer-config", type=Path, required=True)
    parser.add_argument("--work-directory", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    validated = validate_test12_inputs(
        args.inputs.resolve(strict=True),
        official_test_inputs_path=args.official_test_inputs.resolve(strict=True),
        model_input_freeze_path=args.model_input_freeze.resolve(strict=True),
        event_input_freeze_path=args.event_input_freeze.resolve(strict=True),
    )
    predictor = FinalV16Train40Predictor(
        recipe_path=args.recipe,
        model_seal_path=args.model_seal,
        run_root=args.run_root,
        v12_run=args.v12_run,
        v12_protocol_path=args.v12_protocol,
        rgb_source_descriptor=args.rgb_source,
        teacher_manifest=args.teacher,
        producer_config=args.producer_config,
        work_directory=args.work_directory,
        inputs=validated["inputs"],
        device=args.device,
    )
    receipt = run_test12_inference(
        inputs_path=args.inputs,
        official_test_inputs_path=args.official_test_inputs,
        model_seal_path=args.model_seal,
        predictor=predictor,
        work_directory=args.work_directory,
        predictions_output=args.predictions,
        receipt_output=args.receipt,
        validated_inputs=validated,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FinalV16Train40Predictor",
    "PredictionFragments",
    "Test12Predictor",
    "run_test12_inference",
    "validate_test12_inputs",
]
