"""Resume label-free FCWD inference with frozen TRAIN40 and published Garl weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from operational.efficient_context.common import Lease, atomic_bytes, atomic_json, digest

ROOT = Path(__file__).resolve().parents[2]
METHODS = (
    "H8_seed7",
    "H8_seed13",
    "H8_seed23",
    "public_Garl_event_lhr",
    "public_Garl_rgb_event_full",
)
SOURCE_FILES = (
    "operational/sota_eval/fcwd_run.py",
    "operational/sota_eval/fcwd_score.py",
    "operational/sota_eval/fcwd_inputs.py",
    "operational/sota_eval/scoring.py",
    "operational/evttc_transfer/models.py",
    "operational/evttc_rgb_transfer/model.py",
    "operational/efficient_context/common.py",
    "operational/simplex_t_closure/runtime.py",
    "operational/simplex_t_shared_route/adapter.py",
    "operational/train40_system/contracts.py",
    "operational/train40_system/garl_predictions.py",
    "src/e_jepa_ttc/efficient_context/mapped_union.py",
    "src/e_jepa_ttc/efficient_context/garl_input.py",
    "src/e_jepa_ttc/data/garl_official_preprocessing.py",
    "src/e_jepa_ttc/data/event_v4_geometry.py",
    "src/e_jepa_ttc/models/causal_scale_ttc.py",
    "src/e_jepa_ttc/training/stage61_pair_head.py",
    "src/e_jepa_ttc/simplex_t/model.py",
    "src/e_jepa_ttc/simplex_t/phase.py",
)
FORBIDDEN_VALUE_KEYS = {
    "truth_ttc_seconds",
    "target_ttc_seconds",
    "target_ttc_s",
    "ground_truth_ttc",
    "gt_ttc",
    "gt_depth",
    "future_events",
    "labels",
    "targets",
}


class OwnModel(Protocol):
    """Prediction-only boundary, also used by synthetic recovery tests."""

    bindings: dict[str, Any]

    def predict(
        self,
        own_events: np.ndarray,
        delta_t_s: float,
        valid: np.ndarray,
        *,
        availability_lag_s: float = 0.0,
    ) -> dict[str, float]: ...

    def garl_predict(self, sensor: np.ndarray) -> dict[str, Any]: ...


class FullModel(Protocol):
    """Published full-model prediction boundary."""

    bindings: dict[str, Any]

    def predict_sensor(self, sensor: np.ndarray) -> dict[str, Any]: ...


def stamp() -> str:
    """UTC wall-clock timestamp for operational receipts."""
    return datetime.now(UTC).isoformat()


def read(path: Path) -> dict[str, Any]:
    """Read a JSON receipt, never a target table."""
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: float) -> float | None:
    """Encode nonfinite model output as an explicit unavailable value."""
    return float(value) if math.isfinite(float(value)) else None


def reject_targets(value: object) -> None:
    """Reject known target-value fields recursively in inference rows/metadata."""
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_VALUE_KEYS:
                raise ValueError(f"Target or future-input field forbidden during inference: {key}")
            reject_targets(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            reject_targets(child)


def validate_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Require the fixed FCWD-only population without inspecting any GT asset."""
    if manifest.get("schema") != "fcwd_label_free_population_v1":
        raise ValueError("Expected the FCWD label-free input manifest schema")
    if manifest.get("status") != "FROZEN_LABEL_FREE":
        raise ValueError("FCWD query manifest must be FROZEN_LABEL_FREE")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Empty FCWD population")
    seen: set[str] = set()
    for row in rows:
        reject_targets(row)
        if row["sequence_id"] not in {"FCWD1", "FCWD2", "FCWD3"}:
            raise ValueError("Only FCWD1/2/3 inputs are admitted")
        if row["query_id"] in seen:
            raise ValueError("Duplicate query ID in FCWD population")
        seen.add(row["query_id"])
        path = Path(row["raw_path"])
        forbidden_parts = {
            "sealed_labels",
            "stage76",
            "test12",
            "private_test",
            "public_validation",
        }
        if (
            path.suffix.lower() not in {".hdf5", ".h5"}
            or path.name.lower() == "gt.hdf5"
            or any(part.lower() in forbidden_parts for part in path.parts)
        ):
            raise ValueError("FCWD raw input must be a sensor HDF5, outside sealed labels")
        expected = row["raw_stat"]
        stat = path.stat()
        if stat.st_size != expected["size_bytes"] or stat.st_mtime_ns != expected["mtime_ns"]:
            raise ValueError(f"FCWD raw source stat changed: {path}")
    return rows


def source_binding(
    manifest_path: Path, campaign: Path, public_full: Path, code_root: Path, device: str
) -> dict[str, Any]:
    """Capture an explicit source list before any model is allocated on CUDA."""
    files = {name: digest(ROOT / name) for name in SOURCE_FILES}
    campaign_files = ["TRAINING_PROTOCOL.json", "DELIVERY_FREEZE.json", "H8_FEATURE_MANIFEST.json"]
    campaign_files += [
        f"fits/{fit}/CHECKPOINT_RECEIPT.json"
        for fit in (
            "a5_seed7",
            "c2f_seed7",
            "pair_seed7",
            "h8_seed7",
            "h8_seed13",
            "h8_seed23",
        )
    ]
    return {
        "schema": "fcwd_pre_gpu_source_freeze_v1",
        "manifest_sha256": digest(manifest_path),
        "sources": files,
        "campaign_receipts": {name: digest(campaign / name) for name in campaign_files},
        "campaign": str(campaign.resolve()),
        "public_full_dir": str(public_full.resolve()),
        "code_root": str(code_root.resolve()),
        "device": device,
        "precision": "float32_no_tf32",
        "threads": 2,
        "methods": list(METHODS),
        "optimizer_updates": 0,
        "target_reads_allowed": False,
    }


def gpu_worker_command(arguments: Sequence[str]) -> bool:
    """Identify actual GPU worker entrypoints, excluding CPU-only planning wrappers."""
    values = [value.replace("\\", "/").lower() for value in arguments]
    if "--score" in values or "--device=cpu" in values:
        return False
    if any(
        left == "--device" and right == "cpu"
        for left, right in zip(values, values[1:], strict=False)
    ):
        return False
    modules = (
        "operational.evttc_transfer.run",
        "operational.evttc_rgb_transfer.run",
        "operational.sota_eval.fcwd_run",
        "operational.sota_eval.prefetch",
        "operational.sota_eval.full_prefetch",
        "operational.sota_eval.cost",
        "operational.sota_eval.r1_resume",
        "operational.efficient_context.r1_gib_measure",
    )
    for index, token in enumerate(values):
        for module in modules:
            script = module.replace(".", "/") + ".py"
            if token != module and token != script and not token.endswith("/" + script):
                continue
            if module.endswith((".prefetch", ".full_prefetch")):
                return index + 1 < len(values) and values[index + 1] == "run"
            if module.endswith(".r1_resume"):
                return "--execute" in values[index + 1 :]
            return True
    training_prefixes = (
        "operational.train40_system.engine",
        "operational/train40_system/engine",
        "operational.train40_system.h8_features",
        "operational/train40_system/h8_features",
    )
    # campaign and r1_resume_supervisor only supervise/wait. Their real workers
    # above are scanned independently; the wrapper is not a compute client.
    return any(
        any(token.startswith(prefix) or "/" + prefix in token for prefix in training_prefixes)
        for token in values
    )


def assert_no_competing_gpu_processes() -> None:
    """Reject known active CUDA evaluators/trainers, without blocking CPU audit jobs."""
    import psutil

    conflicts = []
    # Windows venv launchers retain an ancestor python.exe with the same CLI.
    own_process_tree = {os.getpid(), *(process.pid for process in psutil.Process().parents())}
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if (
                process.pid in own_process_tree
                or "python" not in (process.info["name"] or "").lower()
            ):
                continue
            arguments = process.info["cmdline"] or []
            if gpu_worker_command(arguments):
                conflicts.append(process.pid)
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied as exc:
            raise RuntimeError("Cannot verify a Python process for the single-GPU guard") from exc
    if conflicts:
        raise RuntimeError(f"Another GPU job is active: PIDs {conflicts}")


def tensor_receipt(value: np.ndarray | None) -> dict[str, Any] | None:
    """Bind bytes, dtype and shape of an input tensor without storing targets."""
    if value is None:
        return None
    array = np.ascontiguousarray(value)
    return {
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "bytes": array.nbytes,
    }


def measure_query(
    row: dict[str, Any],
    own: OwnModel,
    full: FullModel | None,
    prepare: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Predict all methods on the same query; preserve nonfinite/unavailable outcomes."""
    started = time.perf_counter()
    prepared = prepare(row)
    reject_targets(prepared.get("metadata", {}))
    prepared_at = time.perf_counter()
    predictions: dict[str, float | None] = dict.fromkeys(METHODS)
    reasons: dict[str, str] = {}
    try:
        own_result = own.predict(
            prepared["own_events"],
            delta_t_s=prepared["delta_t_s"],
            valid=prepared["valid"],
            availability_lag_s=prepared.get("availability_lag_s", 0.0),
        )
        if set(own_result) != set(METHODS[:3]):
            raise ValueError("Own model did not return the exact three frozen head identities")
        predictions.update({name: finite(value) for name, value in own_result.items()})
    except ValueError as exc:
        if str(exc) != "Frozen H8 head emitted an invalid prediction":
            raise
        reasons.update({name: "native_nonfinite_H8_prediction" for name in METHODS[:3]})
    own_at = time.perf_counter()
    heights: dict[str, list[float | None]] = {}
    for method, sensor, predictor in (
        (METHODS[3], prepared.get("garl_events"), own.garl_predict),
        (METHODS[4], prepared.get("garl_full_sensor"), full.predict_sensor if full else None),
    ):
        if sensor is None:
            reason_key = (
                "garl_unavailable_reason"
                if method == METHODS[3]
                else "garl_full_unavailable_reason"
            )
            reasons[method] = str(
                prepared.get("unavailable_reasons", {}).get(
                    method, prepared.get(reason_key) or "input_contract_unavailable"
                )
            )
            heights[method] = []
            continue
        if predictor is None:
            raise RuntimeError("Full sensor became available without a frozen full-model binding")
        try:
            result = predictor(sensor)
        except ValueError as exc:
            if "emitted invalid" not in str(exc) or "height" not in str(exc):
                raise
            result = {"ttc": math.nan, "heights": []}
            reasons[method] = "native_nonfinite_heights"
        predictions[method] = finite(result["ttc"])
        heights[method] = [finite(value) for value in result["heights"]]
    ended = time.perf_counter()
    for method, value in predictions.items():
        if value is None:
            reasons.setdefault(method, "native_nonfinite_ttc")
    return {
        "status": "PREDICTED" if not reasons else "UNAVAILABLE_OR_NONFINITE_RETAINED",
        "ttc": predictions,
        "unavailable_reasons": reasons,
        "garl_heights": heights,
        "input_tensors": {
            key: tensor_receipt(prepared.get(key))
            for key in (
                "own_events",
                "garl_events",
                "garl_full_sensor",
                "valid",
            )
        },
        "input_metadata": prepared.get("metadata", {}),
        "seconds": {
            "prepare": prepared_at - started,
            "own": own_at - prepared_at,
            "garl_combined": ended - own_at,
            "total": ended - started,
        },
        "targets_read": False,
        "optimizer_updates": 0,
    }


def fragment(
    output: Path, index: int, row: dict[str, Any], binding_sha: str
) -> dict[str, Any] | None:
    """Read only committed fragments; detect corruption and retain incomplete writes."""
    path = output / "predictions" / f"query_{index:05d}.json"
    checksum = path.with_suffix(".sha256")
    if not path.exists():
        if checksum.exists():
            raise ValueError("Prediction checksum exists without its fragment")
        return None
    if not checksum.exists():
        # The atomic JSON finished but its commit marker did not. Preserve it;
        # re-inference is cheap and avoids blessing bytes without a prior hash.
        return None
    if checksum.read_text(encoding="ascii").strip() != digest(path):
        raise ValueError(f"Prediction fragment checksum mismatch: {path.name}")
    saved = read(path)
    if saved["binding_sha256"] != binding_sha or saved["query_id"] != row["query_id"]:
        raise ValueError("Saved FCWD prediction identity/binding mismatch")
    if saved["sequence_id"] != row["sequence_id"] or set(saved["ttc"]) != set(METHODS):
        raise ValueError("Saved FCWD prediction sequence/method coverage changed")
    if saved.get("targets_read") is not False:
        raise ValueError("Saved prediction does not assert label-free inference")
    return saved


def execute_queries(
    rows: Sequence[dict[str, Any]],
    output: Path,
    binding_sha: str,
    own: OwnModel,
    full: FullModel | None,
    prepare: Callable[[dict[str, Any]], dict[str, Any]],
    limit: int | None = None,
) -> bool:
    """Save one atomic prediction per query and resume verified fragments exactly."""
    start = time.perf_counter()
    fresh = 0
    completed = 0
    for index, row in enumerate(rows):
        if fragment(output, index, row, binding_sha) is not None:
            completed += 1
            continue
        if (output / "STOP_REQUEST").exists() or (limit is not None and fresh >= limit):
            atomic_json(
                output / "STATE.json",
                {
                    "status": "PAUSED_PRESERVED",
                    "completed_queries": completed,
                    "total_queries": len(rows),
                    "checked_utc": stamp(),
                },
            )
            return False
        path = output / "predictions" / f"query_{index:05d}.json"
        if path.exists():
            orphan = output / "orphaned" / f"{path.stem}_{digest(path)}.json"
            atomic_bytes(orphan, path.read_bytes())
        try:
            measured = measure_query(row, own, full, prepare)
            measured.update(
                query_id=row["query_id"],
                sequence_id=row["sequence_id"],
                anchor_us=row["anchor_us"],
                binding_sha256=binding_sha,
                finished_utc=stamp(),
            )
            atomic_json(path, measured)
            atomic_bytes(path.with_suffix(".sha256"), (digest(path) + "\n").encode("ascii"))
        except Exception as exc:
            failure = {
                "status": "FAILED_PRESERVED",
                "query_id": row["query_id"],
                "error_type": type(exc).__name__,
                "error": str(exc),
                "binding_sha256": binding_sha,
                "checked_utc": stamp(),
                "resume": "Rerun the same CLI; committed fragments will be verified and reused",
            }
            atomic_json(output / "failures" / f"{index:05d}_{time.time_ns()}.json", failure)
            atomic_json(output / "STATE.json", failure)
            raise
        fresh += 1
        completed += 1
        elapsed = time.perf_counter() - start
        atomic_json(
            output / "STATE.json",
            {
                "status": "RUNNING",
                "completed_queries": completed,
                "total_queries": len(rows),
                "fresh_queries": fresh,
                "elapsed_seconds": elapsed,
                "fresh_queries_per_minute": fresh / elapsed * 60,
                "eta_seconds_at_current_run_rate": (len(rows) - completed) * elapsed / fresh,
                "checked_utc": stamp(),
                "optimizer_updates": 0,
            },
        )
    return True


def seal_predictions(
    output: Path, manifest_path: Path, rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Publish a complete population seal only after validating every commit marker."""
    binding_sha = digest(output / "INFERENCE_FREEZE.json")
    fragments = {}
    for index, row in enumerate(rows):
        if fragment(output, index, row, binding_sha) is None:
            raise ValueError("Cannot seal incomplete FCWD predictions")
        name = f"predictions/query_{index:05d}.json"
        fragments[name] = digest(output / name)
    seal = {
        "status": "COMPLETE",
        "queries": len(rows),
        "methods": list(METHODS),
        "binding_sha256": binding_sha,
        "source_freeze_sha256": digest(output / "SOURCE_FREEZE.json"),
        "manifest_sha256": digest(manifest_path),
        "fragments": fragments,
        "targets_read": False,
        "optimizer_updates": 0,
        "sealed_utc": stamp(),
    }
    atomic_json(output / "PREDICTIONS_SEALED.json", seal)
    atomic_json(
        output / "STATE.json",
        {"status": "PREDICTIONS_COMPLETE", "queries": len(rows), "checked_utc": stamp()},
    )
    return seal


def verify_seal(output: Path, manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify the complete label-free seal before a caller may open any GT CSV."""
    seal = read(output / "PREDICTIONS_SEALED.json")
    manifest = read(manifest_path)
    rows = validate_manifest(manifest)
    if seal["status"] != "COMPLETE" or seal["queries"] != len(rows):
        raise ValueError("FCWD prediction population is incomplete")
    if seal["manifest_sha256"] != digest(manifest_path):
        raise ValueError("FCWD query manifest changed after inference")
    if seal["source_freeze_sha256"] != digest(output / "SOURCE_FREEZE.json"):
        raise ValueError("FCWD source freeze changed after inference")
    if seal["binding_sha256"] != digest(output / "INFERENCE_FREEZE.json"):
        raise ValueError("FCWD model binding changed after inference")
    if len(seal["fragments"]) != len(rows) or seal.get("targets_read") is not False:
        raise ValueError("FCWD seal does not cover the complete label-free population")
    predictions = []
    for index, row in enumerate(rows):
        name = f"predictions/query_{index:05d}.json"
        if seal["fragments"].get(name) != digest(output / name):
            raise ValueError("Sealed FCWD fragment changed")
        saved = fragment(output, index, row, seal["binding_sha256"])
        if saved is None:
            raise ValueError("FCWD prediction lost its commit marker")
        predictions.append(saved)
    return manifest, predictions


def run(
    manifest_path: Path,
    output: Path,
    campaign: Path,
    public_full: Path,
    code_root: Path,
    device: str,
    *,
    limit: int | None = None,
    gpu_guard_root: Path | None = None,
) -> None:
    """Freeze CPU-side source identity, then load existing weights and infer only."""
    manifest_path, output = manifest_path.resolve(), output.resolve()
    if output == campaign.resolve() or campaign.resolve() in output.parents:
        raise ValueError("FCWD outputs must be separate from the historical TRAIN40 campaign")
    rows = validate_manifest(read(manifest_path))
    output.mkdir(parents=True, exist_ok=True)
    guard_root = gpu_guard_root or ROOT / "artifacts/sota_campaign_20261008/GPU_GUARD"
    guard_root.mkdir(parents=True, exist_ok=True)
    binding = source_binding(manifest_path, campaign, public_full, code_root, device)
    output_lease_entered = False
    try:
        with Lease(guard_root), Lease(output):
            output_lease_entered = True
            if device == "cuda":
                assert_no_competing_gpu_processes()
            source_freeze = output / "SOURCE_FREEZE.json"
            if source_freeze.exists() and read(source_freeze) != binding:
                raise ValueError("Source/config freeze changed; use a separately declared new run")
            if not source_freeze.exists():
                atomic_json(source_freeze, binding)
            if (output / "PREDICTIONS_SEALED.json").exists():
                verify_seal(output, manifest_path)
                return
            import torch

            from operational.evttc_rgb_transfer.model import FullGarl
            from operational.evttc_transfer.models import FrozenModels

            from .fcwd_inputs import prepare

            # The repository's narrow Torch stubs omit these official runtime APIs.
            runtime = cast(Any, torch)
            runtime.set_num_threads(2)
            runtime.set_num_interop_threads(2)
            runtime.backends.cuda.matmul.allow_tf32 = False
            runtime.backends.cudnn.allow_tf32 = False
            runtime.backends.cudnn.benchmark = False
            own = FrozenModels(campaign, device=device)
            full = FullGarl(public_full, code_root, device=device)
            freeze = {
                "source_freeze_sha256": digest(source_freeze),
                "manifest_sha256": digest(manifest_path),
                "own_models": own.bindings,
                "full_model": full.bindings,
                "precision": "float32_no_tf32",
                "device": device,
                "methods": list(METHODS),
                "targets_read": False,
                "optimizer_updates": 0,
            }
            path = output / "INFERENCE_FREEZE.json"
            if path.exists() and read(path) != freeze:
                raise ValueError("Frozen model bindings changed; predictions remain preserved")
            if not path.exists():
                atomic_json(path, freeze)
            complete = execute_queries(rows, output, digest(path), own, full, prepare, limit)
            if source_binding(manifest_path, campaign, public_full, code_root, device) != binding:
                raise ValueError("Sources changed during inference; cannot seal")
            if complete:
                seal_predictions(output, manifest_path, rows)
    except Exception as exc:
        if output_lease_entered:
            atomic_json(
                output / "STATE.json",
                {
                    "status": "FAILED_PRESERVED",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "checked_utc": stamp(),
                    "owned_leases_released_on_exit": True,
                    "optimizer_updates": 0,
                },
            )
        raise


def main() -> None:
    """Run/resume inference, or explicitly score only a verified complete seal."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument(
        "--public-full-dir",
        type=Path,
        default=Path("artifacts/evttc_rgb_transfer_20261008/public_garl"),
    )
    parser.add_argument("--code-root", type=Path, default=Path("E:/Garl-TTC"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gpu-guard-root", type=Path)
    parser.add_argument("--score", action="store_true")
    parser.add_argument(
        "--asset-manifest",
        type=Path,
        default=Path("artifacts/sota_campaign_20261008/fcwd/ASSET_MANIFEST.json"),
    )
    args = parser.parse_args()
    if args.score:
        from .fcwd_score import score_sealed

        score_sealed(args.manifest, args.output, args.asset_manifest)
    else:
        if args.campaign is None:
            parser.error("--campaign is required for frozen inference")
        run(
            args.manifest,
            args.output,
            args.campaign,
            args.public_full_dir,
            args.code_root,
            args.device,
            limit=args.limit,
            gpu_guard_root=args.gpu_guard_root,
        )


if __name__ == "__main__":
    main()
