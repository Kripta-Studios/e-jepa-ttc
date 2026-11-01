"""Prepare an eAP test submission from the frozen Dfull detector and TRAIN ranker.

No training, label access, score threshold tuning, or benchmark upload is performed.
The new media admission is explicit; historical runtime checks remain unchanged.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
import subprocess
import sys
import time
import zipfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from scripts.infer_detection_release import (
        digest,
        ranking_worker,
        verify_release,
        write_json_bytes,
    )
except ModuleNotFoundError:
    from infer_detection_release import digest, ranking_worker, verify_release, write_json_bytes

CLASSES = (
    "car",
    "bicycle",
    "truck",
    "pedestrian",
    "motorcycle",
    "bus",
    "tricycle",
    "other_vehicle",
)
FIELDS = ("class_name", "translation_m", "size_lwh_m", "yaw_rad", "velocity_mps", "score")
PARENT_ADAPTER_SHA256 = "6574b4814c57bf2265996fdcb9eed424e4228e96028970294ed0bcbecfd8c67e"
RESUME_BINDINGS = (
    "split",
    "sample_tokens",
    "media_sha256",
    "template_sha256",
    "release_manifest_sha256",
    "release_wrapper_sha256",
    "chunk_size",
    "batch_size",
    "prefetch_frames",
    "score_policy",
    "targets_read",
    "optimizer_updates",
    "rgb_shards",
)


def validate_resume_parent(parent: dict[str, Any], current: dict[str, Any]) -> None:
    """Admit unchanged completed v1 chunks under an explicit recovery receipt."""
    if parent.get("adapter_sha256") != PARENT_ADAPTER_SHA256:
        raise ValueError("Unsupported parent adapter for this recovery")
    for name in RESUME_BINDINGS:
        if name not in parent or parent[name] != current.get(name):
            raise ValueError(f"Recovery would change the parent contract: {name}")


def confirm_unavailable_image(raw: bytes, failure: dict[str, Any]) -> None:
    """Abstain only on exact pre-audited bytes that still fail strict PNG decoding."""
    from PIL import Image

    if hashlib.sha256(raw).hexdigest() != failure.get("sha256"):
        raise ValueError("Unavailable image bytes differ from the audit")
    if failure.get("error") != "image file is truncated":
        raise ValueError("Only verified truncated images are admitted for abstention")
    try:
        with Image.open(io.BytesIO(raw)) as decoded:
            decoded.load()
    except OSError as error:
        if str(error) != failure["error"]:
            raise ValueError("Image failure differs from the audit") from error
    else:
        raise ValueError("Audited unavailable image now decodes successfully")


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys rather than silently discarding a population."""
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_template(path: Path) -> list[str]:
    """Inspect the supplied official ZIP without extracting or trusting paths."""
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["submission.json"]:
            raise ValueError("Template must contain exactly submission.json at root")
        if archive.getinfo("submission.json").file_size > 20_000_000:
            raise ValueError("Unexpectedly large submission template")
        value = json.loads(archive.read("submission.json"), object_pairs_hook=unique_object)
    if value.get("meta", {}).get("format") != "eap_native_detection_result":
        raise ValueError("Unsupported official submission format")
    results = value.get("results")
    if not isinstance(results, dict) or not results:
        raise ValueError("Template population is missing")
    if any(not isinstance(k, str) or not k or v != [] for k, v in results.items()):
        raise ValueError("Expected label-free empty-list template entries")
    return list(results)


def validate_media_rows(rows: list[dict[str, Any]], tokens: list[str], split: str) -> None:
    """Require exact unique coverage and refuse any test label paths."""
    observed = [r["sample_token"] for r in rows]
    if not tokens or len(set(tokens)) != len(tokens) or len(set(observed)) != len(observed):
        raise ValueError("Missing or duplicate population")
    if set(tokens) != set(observed):
        raise ValueError("Media population differs from requested tokens")
    if any(r["split"] != split for r in rows):
        raise ValueError("Media split differs from explicit admission")
    if split == "test" and any(r.get("labels_path") is not None for r in rows):
        raise ValueError("Test metadata must not refer to labels")


def native_box(row: dict[str, Any]) -> dict[str, Any]:
    """Validate and project one prediction into the official six-field schema."""
    if row.get("class_name") not in CLASSES:
        raise ValueError("Unsupported prediction class")
    out = {k: row[k] for k in FIELDS}
    for name, size in (("translation_m", 3), ("size_lwh_m", 3), ("velocity_mps", 2)):
        value = out[name]
        if not isinstance(value, list) or len(value) != size:
            raise ValueError(f"Invalid {name} vector")
        if any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            for x in value
        ):
            raise ValueError(f"Nonfinite or nonnumeric {name}")
    for name in ("score", "yaw_rad"):
        value = out[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"Nonfinite or nonnumeric {name}")
    if any(x <= 0 for x in out["size_lwh_m"]) or not 0 <= out["score"] <= 1:
        raise ValueError("Invalid dimensions or confidence")
    return out


def atomic_json(path: Path, value: object) -> None:
    """Replace progress only after the complete new JSON has reached disk."""
    temporary = path.with_suffix(".tmp")
    write_json_bytes(temporary, value)
    temporary.replace(path)


def admit(args: argparse.Namespace) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Bind the frozen release, external media manifest, template and this adapter."""
    import pyarrow.parquet as pq

    manifest = verify_release(args.release)
    media = args.data_root / f"data/{args.split}.parquet"
    rows = pq.read_table(media).to_pylist()
    if args.split == "test":
        if args.template is None or args.tokens is not None:
            raise ValueError(
                "Test requires the full official template; token subsets are forbidden"
            )
        tokens = read_template(args.template)
    else:
        if args.tokens is None:
            raise ValueError("Train is only for explicit label-free parity tokens")
        tokens = json.loads(args.tokens.read_text(encoding="utf-8"))["sample_tokens"]
        wanted = set(tokens)
        rows = [r for r in rows if r["sample_token"] in wanted]
    validate_media_rows(rows, tokens, args.split)
    shards = sorted({r["rgb_shard_path"] for r in rows})
    for relative in shards:
        path = (args.data_root / relative).resolve()
        if not path.is_relative_to(args.data_root.resolve()) or not path.is_file():
            raise ValueError(f"Missing or unsafe RGB shard: {relative}")
    plan = {
        "schema": "phase_eap_benchmark_admission_v1",
        "split": args.split,
        "sample_tokens": tokens,
        "media_sha256": digest(media),
        "template_sha256": digest(args.template) if args.template else None,
        "release_manifest_sha256": digest(args.release / "MANIFEST.json"),
        "adapter_sha256": digest(Path(__file__)),
        "release_wrapper_sha256": digest(Path(verify_release.__code__.co_filename)),
        "chunk_size": args.chunk_size,
        "batch_size": 1,
        "prefetch_frames": 1,
        "score_policy": (
            "fixed TRAIN ranker, all original decoded proposals, no additional NMS or threshold"
        ),
        "targets_read": False,
        "optimizer_updates": 0,
        "rgb_shards": {
            s: {"bytes": (args.data_root / s).stat().st_size, "sha256": digest(args.data_root / s)}
            for s in shards
        },
    }
    failures = {}
    if args.decode_failures is not None:
        audit = json.loads(args.decode_failures.read_text(encoding="utf-8"))
        for failure in audit["errors"]:
            token = failure["token"]
            if token not in tokens or token in failures:
                raise ValueError("Unavailable-image audit has unexpected or duplicate tokens")
            row = next(r for r in rows if r["sample_token"] == token)
            if (
                failure["shard"] != row["rgb_shard_path"]
                or failure["member"] != row["rgb_member_path"]
            ):
                raise ValueError("Unavailable-image identity differs from test metadata")
            failures[token] = failure
    plan["unavailable_inputs"] = failures
    plan["unavailable_input_policy"] = "verified truncated image -> explicit empty predictions"
    if args.resume_from is not None:
        if args.resume_from.resolve() == args.output.resolve():
            raise ValueError("Recovery must preserve the original run directory")
        parent_path = args.resume_from / "ADMISSION.json"
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        validate_resume_parent(parent, plan)
        plan["parent_admission_sha256"] = digest(parent_path)
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "ADMISSION.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != plan:
            raise ValueError("Resume admission differs; use a new output directory")
    else:
        write_json_bytes(path, plan)
    return manifest, tokens, {r["sample_token"]: r for r in rows}


def seal_submission(output: Path, folders: list[Path], tokens: list[str]) -> None:
    """Stream exact complete coverage to a ZIP with one validated root member."""
    destination = output / "submission.json"
    partial = output / "submission.json.partial"
    seen: set[str] = set()
    boxes = 0
    empty = 0
    with partial.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write('{"meta":{"format":"eap_native_detection_result"},"results":{')
        for folder in folders:
            receipt = json.loads((folder / "COMPLETE.json").read_text())
            for name, expected in receipt["files"].items():
                if digest(folder / name) != expected:
                    raise ValueError("Changed completed inference chunk")
            value = json.loads((folder / "predictions.json").read_text(encoding="utf-8"))
            for token, predictions in value["predictions"].items():
                if token in seen or token != tokens[len(seen)]:
                    raise ValueError("Duplicate, reordered or unexpected submission token")
                if len(predictions) > 50:
                    raise ValueError("Proposal count exceeds the frozen decoder")
                native = [native_box(p) for p in predictions]
                if seen:
                    stream.write(",")
                stream.write(
                    json.dumps(token)
                    + ":"
                    + json.dumps(native, separators=(",", ":"), allow_nan=False)
                )
                seen.add(token)
                boxes += len(native)
                empty += not native
        stream.write("}}\n")
    if seen != set(tokens):
        raise ValueError("Incomplete submission population")
    partial.replace(destination)
    zip_path = output / "submission.zip"
    with zipfile.ZipFile(
        zip_path.with_suffix(".zip.partial"), "w", zipfile.ZIP_DEFLATED, compresslevel=6
    ) as archive:
        archive.write(destination, "submission.json")
    zip_path.with_suffix(".zip.partial").replace(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        if archive.namelist() != ["submission.json"]:
            raise ValueError("Incorrect ZIP layout")
        with archive.open("submission.json") as member:
            payload_hash = hashlib.sha256()
            for block in iter(lambda: member.read(1024 * 1024), b""):
                payload_hash.update(block)
            observed = payload_hash.hexdigest()
        if observed != digest(destination):
            raise ValueError("ZIP payload differs from validated JSON")
    admission = json.loads((output / "ADMISSION.json").read_text())
    split = admission["split"]
    write_json_bytes(
        output / "SUBMISSION_RECEIPT.json",
        {
            "status": "READY_TO_UPLOAD_NOT_SUBMITTED"
            if split == "test"
            else "PARITY_ONLY_NOT_FOR_SUBMISSION",
            "split": split,
            "frames": len(seen),
            "boxes": boxes,
            "empty_frames": empty,
            "input_unavailable_frames": len(admission.get("unavailable_inputs", {})),
            "submission_json_sha256": observed,
            "submission_zip_sha256": digest(zip_path),
            "zip_bytes": zip_path.stat().st_size,
            "admission_sha256": digest(output / "ADMISSION.json"),
            "labels_read": False,
            "official_score": None,
        },
    )


def run(args: argparse.Namespace) -> None:
    """Restore once, infer bounded chunks, rank in fresh processes, and seal output."""
    manifest, tokens, media_rows = admit(args)
    admission_sha = digest(args.output / "ADMISSION.json")
    admission = json.loads((args.output / "ADMISSION.json").read_text(encoding="utf-8"))
    unavailable_inputs = admission["unavailable_inputs"]
    checkpoint_path = args.release / manifest["checkpoint"]
    loader_path = checkpoint_path.parent.parent / "code/scripts/infer_detection_campaign.py"
    spec = importlib.util.spec_from_file_location("_frozen_benchmark_verifier", loader_path)
    if spec is None or spec.loader is None:
        raise ImportError("Frozen loader unavailable")
    loader = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loader
    spec.loader.exec_module(loader)
    checkpoint, prereg, run_root = loader.verify_checkpoint_runtime(checkpoint_path)
    if (
        args.split == "train"
        and digest(args.data_root / "data/train.parquet") != prereg["dataset_media_sha256"]
    ):
        raise ValueError("Parity media differs from historical provenance")
    loader.assert_fresh_runtime()
    sys.path[:0] = [str(run_root / "code/src"), str(run_root / "code/scripts")]
    import torch
    from smoke_monodgp_portable import build_detector, require_finite

    from e_jepa_ttc.detection.diagnostic_export import build_sidecar, capture_exposed_outputs
    from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset
    from e_jepa_ttc.detection.monodgp_eap_adapter import decode_eap, prepare_sample

    class LabelFreeMedia(EAPRGBDataset):
        def __init__(self, part: list[str]) -> None:
            # New split admission only. Image decoding and sample construction are inherited
            # byte-for-byte from the verified historical reader; no relabelled manifest.
            self.root = args.data_root.resolve()
            self.sample_tokens = tuple(part)
            self.rows = {t: media_rows[t] for t in part}
            self.max_open_shards = 2
            self.max_label_sequences = 1
            self._shards = OrderedDict()
            self._labels = OrderedDict()

        def _frame_labels(self, row: Any) -> list[dict[str, Any]]:
            return []

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(args.cpu_threads)
    config = prereg["config"]
    mc = config["model"]
    torch.set_float32_matmul_precision(config["optimization"]["matmul_precision"])
    model, _, effective = build_detector(
        run_root / "code/portable_source",
        mc["group_num"],
        predict_velocity=mc["predict_velocity"],
        velocity_loss_coef=mc["velocity_loss_coef"],
    )
    if effective != prereg["effective_model"]:
        raise ValueError("Restored model configuration differs")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    ledger = checkpoint.get("data_member_sha256", {})
    provenance = checkpoint["runtime_provenance_sha256"]
    del checkpoint
    gc.collect()
    folders: list[Path] = []
    started = time.perf_counter()
    processed = 0

    def status(done: int, state: str) -> None:
        value = {
            "state": state,
            "updated_at_utc": datetime.now(UTC).isoformat(),
            "completed_frames": done,
            "total_frames": len(tokens),
            "session_processed_frames": processed,
            "session_elapsed_s": time.perf_counter() - started,
            "admission_sha256": admission_sha,
        }
        atomic_json(args.output / "STATUS.json", value)
        with (args.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value) + "\n")

    status(0, "INFERENCE")
    reused = []
    for offset in range(0, len(tokens), args.chunk_size):
        part = tokens[offset : offset + args.chunk_size]
        folder = args.output / "chunks" / f"{offset:06d}"
        expected_admission = admission_sha
        if args.resume_from is not None and not (folder / "COMPLETE.json").exists():
            parent_folder = args.resume_from / "chunks" / f"{offset:06d}"
            if (parent_folder / "COMPLETE.json").exists():
                if any(t in unavailable_inputs for t in part):
                    raise ValueError(
                        "An unavailable image appears in previously completed inference"
                    )
                folder = parent_folder
                expected_admission = admission["parent_admission_sha256"]
                reused.append(
                    {
                        "offset": offset,
                        "frames": len(part),
                        "receipt_sha256": digest(folder / "COMPLETE.json"),
                    }
                )
        folders.append(folder)
        if (folder / "COMPLETE.json").exists():
            receipt = json.loads((folder / "COMPLETE.json").read_text())
            if receipt["admission_sha256"] != expected_admission or receipt["tokens"] != part:
                raise ValueError("Completed chunk belongs to another admission")
            for name, expected in receipt["files"].items():
                if digest(folder / name) != expected:
                    raise ValueError("Changed completed chunk")
            status(offset + len(part), "INFERENCE")
            continue
        atomic_json(args.output / "REUSED_CHUNKS.json", reused)
        folder.mkdir(parents=True, exist_ok=True)
        predictions: dict[str, Any] = {}
        frames: list[dict[str, Any]] = []
        members: dict[str, str] = {}
        failures: dict[str, Any] = {}
        with (
            LabelFreeMedia(part) as dataset,
            torch.no_grad(),
            ThreadPoolExecutor(max_workers=1) as pool,
        ):

            def prepare(index: int) -> tuple[Any, Any]:
                token = dataset.sample_tokens[index]
                if token in unavailable_inputs:
                    failure = unavailable_inputs[token]
                    shard = dataset._shard(failure["shard"])
                    stream = shard.extractfile(failure["member"])
                    if stream is None:
                        raise ValueError("Audited unavailable image member disappeared")
                    with stream:
                        confirm_unavailable_image(stream.read(), failure)
                    return None, failure
                raw = dataset[index]
                return raw, prepare_sample(
                    raw,
                    width=mc["width"],
                    height=mc["height"],
                    include_velocity=mc["predict_velocity"],
                )

            pending = pool.submit(prepare, 0)
            for index in range(len(dataset)):
                raw, prepared = pending.result()
                if index + 1 < len(dataset):
                    pending = pool.submit(prepare, index + 1)
                if raw is None:
                    failures[part[index]] = prepared
                    processed += 1
                    continue
                if raw.annotations:
                    raise ValueError("Label-free adapter returned annotations")
                loader.verify_input_member(raw.sample_token, raw.member_sha256, ledger, members)
                outputs = model(
                    prepared.image.unsqueeze(0).cuda(),
                    prepared.calibration.unsqueeze(0).cuda(),
                    None,
                    prepared.image_size.unsqueeze(0).cuda(),
                )
                require_finite(outputs)
                frames.append(
                    capture_exposed_outputs(outputs, sample_token=raw.sample_token, batch_index=0)
                )
                predictions[raw.sample_token], _ = decode_eap(
                    outputs, 0, prepared.calibration, raw.event_from_ego, prepared.image_size
                )
                processed += 1
                if processed % 32 == 0:
                    status(offset, "INFERENCE")
        sha = write_json_bytes(
            folder / "baseline.json",
            {
                "schema": "phase_eap_benchmark_inference_v1",
                "checkpoint_sha256": manifest["files"][manifest["checkpoint"]],
                "runtime_provenance_sha256": provenance,
                "media_admission_sha256": admission_sha,
                "sample_count": len(part),
                "targets_argument": None,
                "annotations_returned": 0,
                "input_member_sha256": members,
                "predictions": predictions,
            },
        )
        sidecar = build_sidecar(
            frames,
            checkpoint_sha256=manifest["files"][manifest["checkpoint"]],
            prediction_output_sha256=sha,
            observer_path=run_root / "code/src/e_jepa_ttc/detection/diagnostic_export.py",
        )
        write_json_bytes(folder / "diagnostics.json", sidecar)
        del frames, sidecar, predictions
        with (folder / "ranking.log").open("w", encoding="utf-8") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "rank",
                    "--release",
                    str(args.release),
                    "--output",
                    str(folder),
                ],
                check=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if failures:
            ranked_path = folder / "predictions.json"
            ranked = json.loads(ranked_path.read_text(encoding="utf-8"))
            ranked["observed_sample_count"] = ranked["sample_count"]
            ranked["sample_count"] = len(part)
            ranked["unavailable_inputs"] = failures
            ranked["predictions"] = {
                token: [] if token in failures else ranked["predictions"][token] for token in part
            }
            write_json_bytes(ranked_path, ranked)
        write_json_bytes(folder / "input_failures.json", failures)
        write_json_bytes(
            folder / "COMPLETE.json",
            {
                "admission_sha256": admission_sha,
                "tokens": part,
                "files": {
                    name: digest(folder / name)
                    for name in (
                        "baseline.json",
                        "diagnostics.json",
                        "predictions.json",
                        "input_failures.json",
                    )
                },
            },
        )
        gc.collect()
        status(offset + len(part), "INFERENCE")
    status(len(tokens), "PACKAGING")
    seal_submission(args.output, folders, tokens)
    status(len(tokens), "COMPLETE")


def main() -> None:
    """Execute a full test population or an explicitly scoped TRAIN parity check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "rank"))
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--tokens", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--decode-failures", type=Path)
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    args.release = args.release.resolve()
    args.output = args.output.resolve()
    if args.output.is_relative_to(args.release):
        parser.error("Output must be outside the frozen release")
    if args.command == "rank":
        ranking_worker(args, verify_release(args.release))
        return
    if args.data_root is None or not 1 <= args.chunk_size <= 512 or not 1 <= args.cpu_threads <= 16:
        parser.error("Require data-root, chunk-size 1..512 and cpu-threads 1..16")
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    try:
        run(args)
    except Exception as error:
        status_path = args.output / "STATUS.json"
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status.update(
                state="FAILED",
                error_type=type(error).__name__,
                error=str(error),
                updated_at_utc=datetime.now(UTC).isoformat(),
            )
            atomic_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
