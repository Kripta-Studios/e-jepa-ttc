"""Sequence-at-a-time public Test12 inference with exact frozen tensor parity.

This adapter replaces only simultaneous raw-file residency/host mtime checks.
It retains the frozen validator's labels, identities, aliases, clocks, windows,
ROI and freeze bindings, then separately hashes every consumed tensor and RGB
shard. Evicted H5 bytes are represented by the converter's durable verification
receipt and the exact original tensor SHA, never by an unverified download.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

MEDIA_REVISION = "a814a946e60d37f7242358af5f344542177e50c5"
INVENTORY_SHA256 = "ab749a052855ea6e68e6eb4e6e1a19a27b6e0524d93c8ccce0952a90f77f4a95"
INVENTORY_PATH = Path("/workspace/ejepa_test12_probe/TEST12_EXACT_INVENTORY.json")


def read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Not a JSON object: {path}")
    return value


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            checksum.update(block)
    return checksum.hexdigest()


def canonical(value: Any) -> str:
    # Use the frozen implementation for scientific bindings, this for receipts only.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def contained(root: Path, relative: str) -> Path:
    normalized = relative.replace("\\", "/")
    path = (root / normalized).resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("Staged source escapes its root")
    return path


def structural_validator(original: Any) -> Any:
    """Retain frozen scientific checks, defer only physical media verification.

    AST guards fail closed if the specific frozen function shape changes. The
    resulting validation is explicitly structural and cannot admit inference.
    """
    tree = ast.parse(inspect.getsource(original))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    replacements = {"media_root": 0, "raw_residency": 0, "rgb_residency": 0, "tensor": 0}
    for index, node in enumerate(function.body):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "media_root" for target in node.targets
        ):
            node.value = ast.Constant(None)
            replacements["media_root"] += 1
        if isinstance(node, ast.If):
            expression = ast.unparse(node.test)
            if expression == "media_root is not None":
                function.body[index] = ast.Pass()
                replacements["raw_residency"] += 1
            elif expression == "verify_rgb_sources":
                function.body[index] = ast.Pass()
                replacements["rgb_residency"] += 1
    for node in ast.walk(function):
        if not isinstance(node, ast.For) or ast.unparse(node.target) != "(identity, item)":
            continue
        if ast.unparse(node.iter) != "event.items()":
            continue
        positions = [
            index
            for index, child in enumerate(node.body)
            if isinstance(child, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "tensor" for target in child.targets
            )
        ]
        if len(positions) != 1 or positions[0] != 2:
            raise ValueError("Frozen tensor validation shape changed")
        # Preserve frozen row lookup and all identity/window/ROI/clock comparisons.
        node.body = node.body[: positions[0]]
        replacements["tensor"] += 1
    if set(replacements.values()) != {1}:
        raise ValueError(f"Frozen validator shape changed: {replacements}")
    function.name = "_structural_test12"
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, "<staged-test12-structural-validator>", "exec"), namespace)
    return namespace[function.name]


def validate_structure(final: Any, run: Any) -> dict[str, Any]:
    if digest(INVENTORY_PATH) != INVENTORY_SHA256:
        raise ValueError("Pinned Hugging Face media inventory changed")
    inventory = read(INVENTORY_PATH)
    if inventory.get("revision") != MEDIA_REVISION:
        raise ValueError("Pinned public media revision changed")
    cfg = final._config(run)
    original = final.validate_test12_inputs
    result = structural_validator(original)(
        cfg["test12_inputs"],
        official_test_inputs_path=cfg["official_test_inputs"],
        model_input_freeze_path=cfg["model_input_freeze"],
        event_input_freeze_path=cfg["event_input_freeze"],
    )
    result["input_provenance_verified"] = False
    result["input_provenance"].update(
        verified=False,
        event_tensor_receipts_verified=0,
        structural_provenance_verified=True,
        media_verification="DEFERRED_PER_SEQUENCE",
    )
    result["frozen_validator_source_sha256"] = hashlib.sha256(
        inspect.getsource(original).encode()
    ).hexdigest()
    result["media_inventory"] = {entry["path"]: entry for entry in inventory["files"]}
    return result


def verify_raw_receipt(
    receipt: dict[str, Any],
    sequence: str,
    inputs_sha: str,
    freeze_sha: str,
    pin: dict[str, Any],
) -> str:
    expected = f"data/test/{sequence}/events.h5"
    sha = receipt.get("sha256", "")
    if (
        receipt.get("schema") != "test12_raw_source_verified_v1"
        or receipt.get("status") != "SHA256_VERIFIED"
        or receipt.get("relative_path") != expected
        or receipt.get("inputs_sha256") != inputs_sha
        or receipt.get("input_freeze_sha256") != freeze_sha
        or receipt.get("storage_state") not in {"PRESENT", "EVICTED_AFTER_TENSOR_VERIFICATION"}
        or not isinstance(sha, str)
        or len(sha) != 64
        or any(character not in "0123456789abcdef" for character in sha)
        or int(receipt.get("size_bytes", 0)) <= 0
        or receipt.get("revision") != MEDIA_REVISION
        or pin.get("path") != expected
        or pin.get("sha256") != sha
        or int(pin.get("bytes", -1)) != int(receipt.get("size_bytes", 0))
    ):
        raise ValueError("Unverified or mismatched staged raw H5 receipt")
    return sha


def verify_rgb_receipt(
    receipt: dict[str, Any], relative: str, frozen: dict[str, Any], inputs_sha: str
) -> None:
    if (
        receipt.get("schema") != "ejepa_test12_rgb_shard_receipt_v1"
        or receipt.get("status") != "PASS"
        or receipt.get("relative_path") != relative
        or receipt.get("inputs_sha256") != inputs_sha
        or receipt.get("sha256") != frozen["sha256"]
        or int(receipt.get("size_bytes", -1)) != int(frozen["size_bytes"])
        or receipt.get("revision") != frozen["revision"]
    ):
        raise ValueError(f"RGB source receipt differs: {relative}")


def admit_sequence(
    validated: dict[str, Any], sequence: str, event_root: Path, rgb_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Hash all sequence inputs before use; return a path-only relocated inventory."""
    import numpy as np

    inputs = validated["inputs"]
    known = {str(row["sequence_id"]) for row in inputs["queries"]}
    if sequence not in known:
        raise ValueError("Sequence is outside the frozen official inputs")
    inputs_sha = validated["inputs_sha256"]
    freeze_sha = validated["input_provenance"]["event_input_freeze_sha256"]
    sequence_path = contained(event_root, f"sequences/{sequence}.json")
    manifest = read(sequence_path)
    if manifest.get("status") != "COMPLETE":
        raise ValueError("Event sequence is not complete")
    raw_path = Path(manifest["raw_receipt_path"]).resolve(strict=True)
    if not raw_path.is_relative_to(event_root.resolve()):
        raise ValueError("Raw receipt outside event staging root")
    raw_sha = verify_raw_receipt(
        read(raw_path),
        sequence,
        inputs_sha,
        freeze_sha,
        validated["media_inventory"][f"data/test/{sequence}/events.h5"],
    )
    expected = {
        identity: item
        for identity, item in inputs["event_observations"].items()
        if item["sequence_id"] == sequence
    }
    if set(manifest["tensors"]) != set(expected):
        raise ValueError("Staged event sequence tensor coverage differs")
    relocated = copy.deepcopy(inputs)
    relocated["media_root"] = str(rgb_root.resolve(strict=True))
    proofs: dict[str, Any] = {}
    for identity, item in expected.items():
        entry = manifest["tensors"][identity]
        path = Path(entry["path"]).resolve(strict=True)
        sidecar = Path(entry["receipt_path"]).resolve(strict=True)
        if not path.is_relative_to(event_root.resolve()) or not sidecar.is_relative_to(
            event_root.resolve()
        ):
            raise ValueError("Tensor or receipt outside event staging root")
        receipt = read(sidecar)
        tensor_sha = item["current_tensor_sha256"]
        if (
            entry.get("sha256") != tensor_sha
            or digest(path) != tensor_sha
            or receipt.get("sha256") != tensor_sha
            or receipt.get("sample_token") != identity
            or receipt.get("input_freeze_sha256") != freeze_sha
            or receipt.get("inputs_sha256") != inputs_sha
            or receipt.get("raw_relative_path") != f"data/test/{sequence}/events.h5"
            or receipt.get("raw_sha256") != raw_sha
        ):
            raise ValueError(f"Tensor lacks exact frozen parity: {identity}")
        with np.load(path, allow_pickle=False) as archive:
            if not bool(archive["valid"][-1]):
                raise ValueError("Frozen current event observation is invalid")
        relocated["event_observations"][identity]["current_tensor_path"] = str(path)
        proofs[identity] = {
            "tensor_sha256": tensor_sha,
            "receipt_sha256": digest(sidecar),
            "path": str(path),
            "receipt_path": str(sidecar),
        }
    rgb_manifest_path = contained(rgb_root, f"sequences/{sequence}.json")
    rgb_manifest = read(rgb_manifest_path)
    if rgb_manifest.get("status") != "READY":
        raise ValueError("RGB sequence is not ready")
    expected_rgb = {
        relative: source
        for relative, source in inputs["rgb_sources"].items()
        if relative.replace("\\", "/").startswith(f"data/test/{sequence}/")
    }
    if set(rgb_manifest["source_receipts"]) != set(expected_rgb):
        raise ValueError("RGB sequence shard coverage differs")
    rgb_proofs: dict[str, Any] = {}
    for relative, source in expected_rgb.items():
        receipt_path = Path(rgb_manifest["source_receipts"][relative]).resolve(strict=True)
        if not receipt_path.is_relative_to(rgb_root.resolve()):
            raise ValueError("RGB receipt outside staging root")
        verify_rgb_receipt(read(receipt_path), relative, source, inputs_sha)
        path = contained(rgb_root, relative)
        stat = path.stat()
        if stat.st_size != source["size_bytes"] or digest(path) != source["sha256"]:
            raise ValueError(f"RGB shard bytes differ: {relative}")
        rgb_proofs[relative] = {
            "sha256": source["sha256"],
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "path": str(path),
            "receipt_path": str(receipt_path),
            "receipt_sha256": digest(receipt_path),
        }
    admission = {
        "schema": "test12_staged_sequence_admission_v1",
        "status": "PASS",
        "sequence_id": sequence,
        "inputs_sha256": inputs_sha,
        "frozen_validator_source_sha256": validated["frozen_validator_source_sha256"],
        "adapter_source_sha256": digest(Path(__file__)),
        "media_inventory_sha256": INVENTORY_SHA256,
        "media_revision": MEDIA_REVISION,
        "event_freeze_sha256": freeze_sha,
        "event_tensors": proofs,
        "rgb_sources": rgb_proofs,
        "raw_receipt_path": str(raw_path),
        "raw_receipt_sha256": digest(raw_path),
        "raw_sha256": raw_sha,
        "event_sequence_manifest_sha256": digest(sequence_path),
        "rgb_sequence_manifest_sha256": digest(rgb_manifest_path),
        "equivalence": "EXACT_FROZEN_NPZ_BYTES_AND_RGB_SHARD_BYTES_PATH_ONLY_RELOCATION",
        "raw_h5_residency_required": False,
        "raw_h5_rehashed_by_this_adapter": False,
        "private_targets_read": False,
        "external_submission": False,
    }
    return relocated, admission


def assert_rgb_unchanged(admission: dict[str, Any]) -> None:
    for proof in admission["rgb_sources"].values():
        stat = Path(proof["path"]).stat()
        if (stat.st_size, stat.st_mtime_ns) != (proof["size_bytes"], proof["mtime_ns"]):
            raise ValueError("Admitted RGB source changed during inference")


def save_npz(path: Path, arrays: dict[str, Any]) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def attach_diagnostics(predictor: Any, captured: dict[str, Any]) -> list[Any]:
    """Observe existing forward inputs/outputs without changing tensor operations."""
    import torch

    def hook(name: str) -> Any:
        def capture(module: Any, inputs: Any, output: Any) -> None:
            del module
            for index, value in enumerate(inputs):
                if isinstance(value, torch.Tensor):
                    captured[f"{name}.input_{index}"] = value.detach()
            if isinstance(output, dict):
                for key, value in output.items():
                    if isinstance(value, torch.Tensor):
                        captured[f"{name}.output_{key}"] = value.detach()

        return capture

    return [
        predictor.models[name].register_forward_hook(hook(name))
        for name in ("E_CTX_MATCHED", "R_CTX", "PHASE_ONLY_EVENT_S7_TRAIN40")
    ]


def diagnostic_paths(directory: Path, token: str) -> tuple[Path, Path]:
    name = hashlib.sha256(token.encode()).hexdigest()
    return directory / "diagnostics" / f"{name}.npz", directory / "diagnostics" / f"{name}.json"


def verify_diagnostic(directory: Path, token: str, binding: str, prediction: float) -> list[Path]:
    npz, metadata = diagnostic_paths(directory, token)
    receipt = read(metadata)
    if (
        receipt.get("schema") != "test12_staged_query_diagnostic_v1"
        or receipt.get("sample_token") != token
        or receipt.get("run_binding_sha256") != binding
        or receipt.get("prediction") != prediction
        or receipt.get("npz_sha256") != digest(npz)
    ):
        raise ValueError("Existing prediction diagnostic is missing or inconsistent")
    return [npz, metadata]


def verify_completion(
    final: Any, directory: Path, sequence: str, tokens: list[str], fragments: Any, binding: str
) -> None:
    complete = read(directory / f"{sequence}.complete.json")
    admission_path = directory / f"{sequence}.admission.json"
    admission = read(admission_path)
    values = {token: fragments.read(token) for token in tokens}
    if (
        complete.get("schema") != "test12_staged_sequence_complete_v1"
        or complete.get("status") != "COMPLETE"
        or complete.get("private_targets_read") is not False
        or complete.get("external_submission") is not False
        or admission.get("schema") != "test12_staged_sequence_admission_v1"
        or admission.get("status") != "PASS"
        or admission.get("sequence_id") != sequence
        or admission.get("run_binding_sha256") != binding
        or admission.get("private_targets_read") is not False
        or admission.get("external_submission") is not False
        or complete.get("run_binding_sha256") != binding
        or complete.get("sequence_id") != sequence
        or complete.get("admission_sha256") != digest(admission_path)
        or complete.get("canonical_tokens") != tokens
        or any(value is None for value in values.values())
        or complete.get("predictions_sha256") != final.canonical_sha256(values)
        or complete.get("feature_cache_sha256") != digest(directory / f"{sequence}.features.npz")
    ):
        raise ValueError("Completed sequence differs from its durable artifacts")
    for token, value in values.items():
        verify_diagnostic(directory, token, binding, value)


def ensure_consumed(
    final: Any,
    directory: Path,
    sequence: str,
    tokens: list[str],
    fragments: Any,
    binding: str,
    rgb_root: Path,
) -> None:
    """Reconcile the eviction marker after any crash, using only durable outputs."""
    verify_completion(final, directory, sequence, tokens, fragments, binding)
    admission_path = directory / f"{sequence}.admission.json"
    admission = read(admission_path)
    outputs = [
        admission_path,
        directory / f"{sequence}.complete.json",
        directory / f"{sequence}.features.npz",
    ]
    outputs.extend(Path(str(fragments._path(token))) for token in tokens)
    for token in tokens:
        outputs.extend(verify_diagnostic(directory, token, binding, fragments.read(token)))
    atomic(
        rgb_root / "consumed" / f"{sequence}.json",
        {
            "schema": "test12_staged_rgb_consumed_v1",
            "status": "PASS",
            "sequence_id": sequence,
            "source_receipts": {
                relative: proof["receipt_path"]
                for relative, proof in admission["rgb_sources"].items()
            },
            "outputs": [
                {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
                for path in outputs
            ],
            "run_binding_sha256": binding,
            "private_targets_read": False,
        },
    )


def predict_sequence(
    final: Any,
    run: Any,
    validated: dict[str, Any],
    sequence: str,
    event_root: Path,
    rgb_root: Path,
    deadline: float,
    device: str,
) -> None:
    from unittest.mock import patch

    from operational.final_refit.scientific_oof_pipeline import make_guard
    from operational.final_refit.test12 import PredictionFragments
    from operational.rgb_port import transfer
    from operational.rgb_port_streaming.contracts import NativeObservation

    _, seal_sha = final._scientific_seal(run / "final/MODEL_SEAL.json")
    binding = final.canonical_sha256(
        {"model_seal_sha256": seal_sha, "inputs_sha256": validated["inputs_sha256"]}
    )
    directory = Path(str(run)) / "final/test12/staged_sequences"
    original_rows = {str(row["sample_token"]): row for row in validated["inputs"]["queries"]}
    tokens = [
        token
        for token in validated["canonical_tokens"]
        if original_rows[token]["sequence_id"] == sequence
    ]
    if not tokens:
        raise ValueError("No frozen physical queries for sequence")
    fragments = PredictionFragments(run / "final/test12/fragments", binding)
    if (directory / f"{sequence}.complete.json").exists():
        ensure_consumed(final, directory, sequence, tokens, fragments, binding, rgb_root)
        print(json.dumps({"sequence": sequence, "status": "ALREADY_COMPLETE_VERIFIED"}), flush=True)
        return
    inputs, admission = admit_sequence(validated, sequence, event_root, rgb_root)
    admission["producer_observation_boundary"] = {
        "implementation": "operational.rgb_port_streaming.contracts.NativeObservation",
        "source_sha256": digest(Path(inspect.getfile(NativeObservation))),
        "known_semantics": "AGGREGATE_CURRENT_QUERY_BOOL_PRESERVED_UNCHANGED",
        "pair_input_operator": "UNCHANGED_ProducerObservation.pair_input",
        "remedy_reference": "operational.final_refit.event_parity.reuse_decision",
    }
    admission["model_seal_sha256"] = seal_sha
    admission["run_binding_sha256"] = binding
    admission_path = directory / f"{sequence}.admission.json"
    atomic(admission_path, admission)
    by_token = {str(row["sample_token"]): row for row in inputs["queries"]}
    tokens = [
        token
        for token in validated["canonical_tokens"]
        if by_token[token]["sequence_id"] == sequence
    ]
    guard = make_guard(run)
    guard()
    predictor = final.ScientificOOFFinalPredictor(
        run=run,
        model_seal_path=run / "final/MODEL_SEAL.json",
        inputs=inputs,
        device=device,
    )
    captured: dict[str, Any] = {}
    hooks = attach_diagnostics(predictor, captured)
    # Already-declared V12/V13 ABI adapter: validation copy only, actual mask and
    # pair_input arithmetic remain untouched. No temporal availability is invented.
    observation_boundary = patch.object(transfer, "ProducerObservation", NativeObservation)
    observation_boundary.start()
    try:
        for index, token in enumerate(tokens):
            if time.time() >= deadline or (run / "STOP_REQUEST").exists():
                raise TimeoutError("Authorized deadline or STOP reached; fragments are durable")
            guard()
            assert_rgb_unchanged(admission)
            existing = fragments.read(token)
            if existing is not None:
                verify_diagnostic(directory, token, binding, existing)
            else:
                import numpy as np

                query = by_token[token]
                before = {"event": set(predictor.event_values), "rgb": set(predictor.rgb_values)}
                cache_counts = {
                    modality: {
                        "hits": sum(
                            identity in before[modality]
                            for identity in query[f"{modality}_history"]
                        ),
                        "history_length": len(query[f"{modality}_history"]),
                    }
                    for modality in ("event", "rgb")
                }
                captured.clear()
                started = time.perf_counter()
                prediction = float(predictor.predict(query, inputs))
                inference_seconds = time.perf_counter() - started
                arrays = {key: value.cpu().numpy() for key, value in captured.items()}
                for modality, values in (
                    ("event", predictor.event_values),
                    ("rgb", predictor.rgb_values),
                ):
                    new = sorted(set(values) - before[modality])
                    arrays[f"{modality}_new_ids"] = np.asarray(new, dtype=str)
                    if new:
                        arrays[f"{modality}_new_raw"] = np.stack([values[x][0] for x in new])
                        arrays[f"{modality}_new_experts"] = np.stack([values[x][1] for x in new])
                diagnostic, metadata = diagnostic_paths(directory, token)
                save_npz(diagnostic, arrays)
                atomic(
                    metadata,
                    {
                        "schema": "test12_staged_query_diagnostic_v1",
                        "sample_token": token,
                        "sequence_id": sequence,
                        "track_id": query["track_id"],
                        "query_time_us": query["query_time_us"],
                        "prediction": prediction,
                        "run_binding_sha256": binding,
                        "npz_sha256": digest(diagnostic),
                        "event_history": query["event_history"],
                        "rgb_history": query["rgb_history"],
                        "fallback_source": "PHASE_ONLY" if query["rgb_history"] else "E_CTX",
                        "cache": cache_counts,
                        "instrumented_eager_wall_seconds": inference_seconds,
                        "timing_scope": (
                            "PREDICT_WITH_CACHE_AND_READ_ONLY_HOOKS_EXCLUDES_DIAGNOSTIC_IO"
                        ),
                        "gpu_kernel_time_measured": False,
                        "latency_benchmark": False,
                        "output_units": "TTC seconds; point_phase/q10/q90 in native phase units",
                        "intervals_calibrated_on_test": False,
                        "private_targets_read": False,
                    },
                )
                fragments.write(token, prediction)
            if (index + 1) % 20 == 0:
                print(
                    json.dumps(
                        {"sequence": sequence, "completed": index + 1, "total": len(tokens)}
                    ),
                    flush=True,
                )
    finally:
        observation_boundary.stop()
        for hook in hooks:
            hook.remove()
        predictor.close()
    import numpy as np

    feature_path = directory / f"{sequence}.features.npz"
    arrays: dict[str, Any] = {}
    for modality, values in (("event", predictor.event_values), ("rgb", predictor.rgb_values)):
        identities = sorted(values)
        arrays[f"{modality}_identities"] = np.asarray(identities, dtype=str)
        if identities:
            arrays[f"{modality}_raw"] = np.stack([values[identity][0] for identity in identities])
            arrays[f"{modality}_experts"] = np.stack(
                [values[identity][1] for identity in identities]
            )
    # Preserve computed physical observations for diagnostics; no second inference.
    save_npz(feature_path, arrays)
    # Every prediction is durable before the sequence can be evicted.
    values = {token: fragments.read(token) for token in tokens}
    if any(value is None for value in values.values()):
        raise ValueError("Sequence prediction coverage incomplete")
    atomic(
        directory / f"{sequence}.complete.json",
        {
            "schema": "test12_staged_sequence_complete_v1",
            "status": "COMPLETE",
            "sequence_id": sequence,
            "run_binding_sha256": binding,
            "admission_sha256": digest(admission_path),
            "canonical_tokens": tokens,
            "predictions_sha256": final.canonical_sha256(values),
            "feature_cache_sha256": digest(feature_path),
            "feature_cache_scope": "OBSERVATIONS_COMPUTED_IN_THIS_INVOCATION_ONLY",
            "private_targets_read": False,
            "external_submission": False,
        },
    )
    ensure_consumed(final, directory, sequence, tokens, fragments, binding, rgb_root)


def package(final: Any, run: Any, validated: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    from operational.final_refit.test12 import PredictionFragments

    _, seal_sha = final._scientific_seal(run / "final/MODEL_SEAL.json")
    binding = final.canonical_sha256(
        {"model_seal_sha256": seal_sha, "inputs_sha256": validated["inputs_sha256"]}
    )
    fragments = PredictionFragments(run / "final/test12/fragments", binding)
    rows = {str(row["sample_token"]): row for row in validated["inputs"]["queries"]}
    directory = Path(str(run)) / "final/test12/staged_sequences"
    completion_hashes = {}
    for sequence in sorted({str(row["sequence_id"]) for row in rows.values()}):
        path = directory / f"{sequence}.complete.json"
        complete = read(path)
        admission = directory / f"{sequence}.admission.json"
        expected = [
            token
            for token in validated["canonical_tokens"]
            if rows[token]["sequence_id"] == sequence
        ]
        verify_completion(final, directory, sequence, expected, fragments, binding)
        values = {token: fragments.read(token) for token in expected}
        if (
            complete.get("status") != "COMPLETE"
            or complete.get("run_binding_sha256") != binding
            or complete.get("admission_sha256") != digest(admission)
            or complete.get("canonical_tokens") != expected
            or any(value is None for value in values.values())
            or complete.get("predictions_sha256") != final.canonical_sha256(values)
        ):
            raise ValueError("Sequence completion is not bound to all exact predictions")
        completion_hashes[sequence] = digest(path)
    # The official package validator requires the exact parquet order.
    official = pd.read_parquet(final._config(run)["official_test_inputs"], columns=["sample_token"])
    tokens = official.sample_token.astype(str).tolist()
    aliases = validated["inputs"]["physical_aliases"]
    records = [
        {
            "sample_token": token,
            "prediction": fragments.read(str(aliases[token])),
            "rgb_available": bool(rows[token]["rgb_history"]),
            "fallback_source": "PHASE_ONLY" if rows[token]["rgb_history"] else "E_CTX",
        }
        for token in tokens
    ]
    output = Path(str(run)) / "final/test12/predictions.csv"
    temporary = output.with_suffix(".pending.csv")
    pd.DataFrame.from_records(records).to_csv(temporary, index=False, lineterminator="\n")
    if output.exists() and output.read_bytes() != temporary.read_bytes():
        temporary.unlink()
        raise ValueError("Existing predictions differ")
    os.replace(temporary, output)
    receipt = {
        "schema": final.PREDICTION_SCHEMA,
        "status": "COMPLETE",
        "candidate": final.self_candidate(),
        "model_seal_sha256": seal_sha,
        "inputs_sha256": validated["inputs_sha256"],
        "predictions_sha256": digest(output),
        "population": len(tokens),
        "physical_queries": len(validated["canonical_tokens"]),
        "computed_under_seal": True,
        "private_targets_read": False,
        "external_submission": False,
        "fallback_count": validated["fallback_count"],
        "run_binding_sha256": binding,
        "staged_sequence_completion_sha256": completion_hashes,
    }
    final._write_once(run / "final/test12/PREDICTION_RECEIPT.json", receipt)
    return final.package_scientific_predictions(run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "predict", "package"])
    parser.add_argument("--sequence", action="append", default=[])
    parser.add_argument("--event-root", type=Path, default=Path("/workspace/ejepa_test12_events"))
    parser.add_argument("--rgb-root", type=Path, default=Path("/workspace/ejepa_test12_rgb"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import runpod_final_bootstrap as bootstrap

    if time.time() >= bootstrap.DEADLINE:
        raise TimeoutError("Authorized deadline reached")
    mapping = bootstrap.setup()
    import runpod_final_cache_compat

    runpod_final_cache_compat.install(mapping)
    from operational.final_refit import scientific_oof_final as final

    run = final.Path(str(bootstrap.RUN))
    validated = validate_structure(final, run)
    if args.command == "package":
        print(json.dumps(package(final, run, validated)), flush=True)
        return
    sequences = args.sequence or sorted(
        {str(row["sequence_id"]) for row in validated["inputs"]["queries"]}
    )
    for sequence in sequences:
        if args.command == "preflight":
            _, admission = admit_sequence(validated, sequence, args.event_root, args.rgb_root)
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "sequence": sequence,
                        "event_tensors": len(admission["event_tensors"]),
                        "rgb_sources": len(admission["rgb_sources"]),
                    }
                ),
                flush=True,
            )
        else:
            predict_sequence(
                final,
                run,
                validated,
                sequence,
                args.event_root,
                args.rgb_root,
                bootstrap.DEADLINE,
                args.device,
            )


if __name__ == "__main__":
    main()
