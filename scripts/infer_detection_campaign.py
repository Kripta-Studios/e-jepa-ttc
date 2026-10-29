"""Replay an S1 campaign checkpoint using only its frozen label-free runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_SCHEMA = "eap_detection_campaign_checkpoint_v1"


def sha256_file(path: Path) -> str:
    """Hash a runtime artifact incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    """Hash a JSON-compatible value canonically."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def verify_loader_identity(prereg: Mapping[str, Any], current_path: Path = Path(__file__)) -> None:
    """Bind the outer verifier itself, before trusting its runtime decisions."""
    expected = prereg.get("code_sha256", {}).get("scripts/infer_detection_campaign.py")
    if not isinstance(expected, str) or sha256_file(current_path) != expected:
        raise ValueError("Executing campaign inference loader differs from preregistration")


def verify_checkpoint_runtime(
    checkpoint_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Verify checkpoint, preregistration, code and patched source before execution."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Unsupported campaign checkpoint schema")
    if checkpoint.get("producer") != "scripts/train_detection_campaign.py":
        raise ValueError("Unsupported campaign checkpoint producer")
    if checkpoint.get("inference_contract") != "campaign_v1_strict_loader_required":
        raise ValueError("Unsupported campaign inference contract")
    provenance = checkpoint.get("runtime_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Checkpoint lacks runtime provenance")
    if checkpoint.get("runtime_provenance_sha256") != canonical_sha256(provenance):
        raise ValueError("Checkpoint runtime provenance hash is inconsistent")
    if checkpoint_path.parent.name != "checkpoints":
        raise ValueError("Campaign checkpoint must remain inside its checkpoints directory")
    run_root = checkpoint_path.parent.parent
    prereg_path = run_root / "runtime_preregistration.json"
    if sha256_file(prereg_path) != provenance.get("runtime_preregistration_sha256"):
        raise ValueError("Runtime preregistration bytes differ from checkpoint")
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    verify_loader_identity(prereg)
    checks = {
        "source_receipt_sha256": canonical_sha256(prereg["source_receipt"]),
        "effective_model_sha256": canonical_sha256(prereg["effective_model"]),
        "code_sha256_manifest": canonical_sha256(prereg["code_sha256"]),
        "portable_source_sha256_manifest": canonical_sha256(prereg["portable_source_sha256"]),
        "matmul_precision": prereg["matmul_precision"],
    }
    if any(provenance.get(key) != value for key, value in checks.items()):
        raise ValueError("Runtime preregistration content differs from checkpoint provenance")
    for relative, expected in prereg["code_sha256"].items():
        path = run_root / "code" / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Frozen runtime dependency differs: {relative}")
    portable_root = run_root / "code/portable_source"
    observed_portable = {
        path.relative_to(portable_root).as_posix(): sha256_file(path)
        for path in sorted(portable_root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    if observed_portable != prereg["portable_source_sha256"]:
        raise ValueError("Frozen patched MonoDGP source differs from preregistration")
    return checkpoint, prereg, run_root


def read_tokens(path: Path) -> tuple[str, ...]:
    """Read an explicit inference population without accepting labels."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, dict):
        if set(value) != {"sample_tokens"}:
            raise ValueError("Token JSON mapping must contain only sample_tokens")
        value = value["sample_tokens"]
    if not isinstance(value, list):
        raise ValueError("Token JSON must be an array or sample_tokens mapping")
    result = tuple(value)
    if not result or any(not isinstance(item, str) or not item for item in result):
        raise ValueError("Inference tokens must be nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError("Inference tokens must be unique")
    return result


def assert_fresh_runtime() -> None:
    """Require a fresh process so verified modules cannot be shadowed by imports."""
    contaminated = sorted(
        name
        for name in sys.modules
        if name == "smoke_monodgp_portable"
        or name == "lib"
        or name.startswith("lib.")
        or name == "e_jepa_ttc"
        or name.startswith("e_jepa_ttc.")
    )
    if contaminated:
        raise RuntimeError(
            "Campaign inference requires a fresh subprocess; preloaded runtime modules: "
            f"{contaminated[:5]}"
        )


def verify_input_member(
    token: str,
    observed_sha256: str,
    checkpoint_ledger: Mapping[str, str],
    receipt: dict[str, str],
) -> None:
    """Reject changed known RGB bytes and record every replayed member."""
    expected = checkpoint_ledger.get(token)
    if expected is not None and expected != observed_sha256:
        raise ValueError(f"RGB member bytes differ from checkpoint ledger: {token}")
    receipt[token] = observed_sha256


def infer(
    checkpoint_path: Path,
    data_root: Path,
    tokens_path: Path,
    *,
    diagnostic_frames: list[dict[str, Any]] | None = None,
) -> dict[str, object]:
    """Execute label-free inference with strict model and source restoration."""
    checkpoint, prereg, run_root = verify_checkpoint_runtime(checkpoint_path)
    if not torch.cuda.is_available():
        raise RuntimeError("Campaign inference requires CUDA")
    if sha256_file(data_root / "data/train.parquet") != prereg["dataset_media_sha256"]:
        raise ValueError("Input media/calibration manifest differs from training provenance")
    assert_fresh_runtime()
    sys.path[:0] = [str(run_root / "code/src"), str(run_root / "code/scripts")]
    from smoke_monodgp_portable import build_detector, require_finite

    from e_jepa_ttc.detection.diagnostic_export import capture_exposed_outputs
    from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset
    from e_jepa_ttc.detection.monodgp_eap_adapter import decode_eap, prepare_sample

    class LabelFreeInputs(EAPRGBDataset):
        def _frame_labels(self, row: Mapping[str, Any]) -> list[dict[str, Any]]:
            return []

    config = prereg["config"]
    model_config = config["model"]
    torch.set_float32_matmul_precision(config["optimization"]["matmul_precision"])
    model, _, effective_model = build_detector(
        run_root / "code/portable_source",
        model_config["group_num"],
        predict_velocity=model_config["predict_velocity"],
        velocity_loss_coef=model_config["velocity_loss_coef"],
    )
    if effective_model != prereg["effective_model"]:
        raise ValueError("Restored model construction differs from training")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    predictions = {}
    decode_counts = {}
    input_member_sha256: dict[str, str] = {}
    checkpoint_ledger = checkpoint.get("data_member_sha256", {})
    if not isinstance(checkpoint_ledger, Mapping):
        raise ValueError("Checkpoint RGB member hash ledger is malformed")
    timings = []
    tokens = read_tokens(tokens_path)
    with LabelFreeInputs(data_root, tokens) as dataset, torch.no_grad():
        for index in range(len(dataset)):
            raw = dataset[index]
            if raw.annotations:
                raise AssertionError("Label-free campaign loader returned annotations")
            verify_input_member(
                raw.sample_token,
                raw.member_sha256,
                checkpoint_ledger,
                input_member_sha256,
            )
            prepared = prepare_sample(
                raw,
                width=model_config["width"],
                height=model_config["height"],
                include_velocity=model_config["predict_velocity"],
            )
            images = prepared.image.unsqueeze(0).cuda()
            calibration = prepared.calibration.unsqueeze(0).cuda()
            sizes = prepared.image_size.unsqueeze(0).cuda()
            torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = model(images, calibration, None, sizes)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
            require_finite(outputs)
            if diagnostic_frames is not None:
                diagnostic_frames.append(
                    capture_exposed_outputs(outputs, sample_token=raw.sample_token, batch_index=0)
                )
            predictions[raw.sample_token], decode_counts[raw.sample_token] = decode_eap(
                outputs,
                0,
                prepared.calibration,
                raw.event_from_ego,
                prepared.image_size,
            )
    return {
        "schema": "eap_detection_campaign_inference_v1",
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "runtime_provenance_sha256": checkpoint["runtime_provenance_sha256"],
        "sample_count": len(tokens),
        "targets_argument": None,
        "annotations_returned": 0,
        "input_member_sha256": input_member_sha256,
        "predictions": predictions,
        "decode_counts": decode_counts,
        "inference_seconds_median": statistics.median(timings),
        "inference_seconds_total": sum(timings),
    }


def validate_output_paths(output: Path, diagnostic_sidecar: Path | None) -> None:
    """Reject overwrite and path aliasing before executing expensive inference."""
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if diagnostic_sidecar is None:
        return
    if diagnostic_sidecar.exists():
        raise FileExistsError(f"Refusing to overwrite {diagnostic_sidecar}")
    if output.resolve(strict=False) == diagnostic_sidecar.resolve(strict=False):
        raise ValueError("Prediction output and diagnostic sidecar must use distinct paths")


def main() -> int:
    """CLI for strict, label-free campaign replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-sidecar",
        type=Path,
        help="Optional new JSON path for exposed query tensors; predictions are unchanged",
    )
    args = parser.parse_args()
    validate_output_paths(args.output, args.diagnostic_sidecar)
    diagnostic_frames: list[dict[str, Any]] | None = (
        [] if args.diagnostic_sidecar is not None else None
    )
    result = infer(
        args.checkpoint,
        args.data_root,
        args.tokens,
        diagnostic_frames=diagnostic_frames,
    )
    prediction_text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    prediction_sha256 = hashlib.sha256(prediction_text.encode()).hexdigest()
    sidecar_text: str | None = None
    if args.diagnostic_sidecar is not None:
        if diagnostic_frames is None:
            raise AssertionError("Diagnostic frame collector was not initialized")
        from e_jepa_ttc.detection.diagnostic_export import build_sidecar, validate_json_finite

        run_root = args.checkpoint.parent.parent
        observer_path = run_root / "code/src/e_jepa_ttc/detection/diagnostic_export.py"
        sidecar = build_sidecar(
            diagnostic_frames,
            checkpoint_sha256=sha256_file(args.checkpoint),
            prediction_output_sha256=prediction_sha256,
            observer_path=observer_path,
        )
        validate_json_finite(sidecar)
        sidecar_text = json.dumps(sidecar, indent=2, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(prediction_text, encoding="utf-8")
    if args.diagnostic_sidecar is not None and sidecar_text is not None:
        args.diagnostic_sidecar.parent.mkdir(parents=True, exist_ok=True)
        args.diagnostic_sidecar.write_text(sidecar_text, encoding="utf-8")
    print(json.dumps({"sample_count": result["sample_count"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
