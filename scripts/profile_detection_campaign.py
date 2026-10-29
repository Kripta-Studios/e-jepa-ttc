"""Profile the exact portable detector checkpoint variant on real labeled frames.

This command is deliberately separate from training.  It restores the frozen
runtime sources, velocity head, optimizer, batch size, and image geometry before
measuring preprocessing, transfer, forward, criterion, backward, and update.
"""

# ruff: noqa: E402 -- direct execution from an uninstalled worktree.

from __future__ import annotations

import argparse
import inspect
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import torch
from infer_monodgp_eap import load_runtime_modules, select_runtime_sources, sha256
from profile_monodgp_precision import memory_snapshot


def _tokens(manifest: Path, role: str) -> list[str]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    section = payload.get(role)
    if section is None and isinstance(payload.get("roles"), dict):
        section = payload["roles"].get(role)
    if not isinstance(section, dict) or not isinstance(section.get("sample_tokens"), list):
        raise ValueError(f"Manifest has no sample_tokens for role {role!r}")
    result = section["sample_tokens"]
    if not result or any(not isinstance(x, str) or not x for x in result):
        raise ValueError("Manifest sample tokens must be nonempty strings")
    if len(result) != len(set(result)):
        raise ValueError("Manifest sample tokens must be unique")
    return result


def _validate_manifest_admission(
    manifest: Path, candidate_split: Path, data_root: Path, role: str
) -> tuple[list[str], dict[str, str]]:
    """Bind the profile population to exact public-train metadata and split bytes."""
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_split.read_text(encoding="utf-8"))
    if payload.get("source_split") != "train" or candidate.get("source_split") != "train":
        raise ValueError("Detection profiling admits only the public train source split")
    revision = payload.get("dataset_revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Manifest dataset_revision must be a full lowercase commit")
    if candidate.get("dataset_revision") != revision:
        raise ValueError("Manifest and candidate split revisions differ")
    expected_media = payload.get("source_media_sha256")
    expected_split = payload.get("candidate_split_sha256")
    if not isinstance(expected_media, str) or re.fullmatch(r"[0-9a-f]{64}", expected_media) is None:
        raise ValueError("Manifest requires source_media_sha256")
    if not isinstance(expected_split, str) or re.fullmatch(r"[0-9a-f]{64}", expected_split) is None:
        raise ValueError("Manifest requires candidate_split_sha256")
    if sha256(data_root / "data/train.parquet") != expected_media:
        raise ValueError("Canonical train media metadata differs from the manifest")
    if sha256(candidate_split) != expected_split:
        raise ValueError("Candidate split bytes differ from the manifest")
    tokens = _tokens(manifest, role)
    section = payload.get(role)
    if section is None and isinstance(payload.get("roles"), dict):
        section = payload["roles"].get(role)
    assert isinstance(section, dict)
    sequences = section.get("sequences", section.get("sequence_ids"))
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("Manifest role requires declared sequences")
    token_sequences = {token.split(":", 1)[0] for token in tokens if token.count(":") == 1}
    if token_sequences != set(sequences):
        raise ValueError("Manifest token prefixes differ from declared role sequences")
    assignments = candidate.get("assignments")
    if not isinstance(assignments, dict) or role not in assignments:
        raise ValueError("Candidate split does not declare the requested role")
    if not set(sequences).issubset(set(assignments[role])):
        raise ValueError("Profile sequences do not belong to the candidate role")
    return tokens, {
        "manifest_sha256": sha256(manifest),
        "candidate_split_sha256": expected_split,
        "source_media_sha256": expected_media,
        "dataset_revision": revision,
    }


def _weighted_loss(losses: dict[str, torch.Tensor], criterion: Any) -> torch.Tensor:
    terms: list[torch.Tensor] = [
        value * criterion.weight_dict[name]
        for name, value in losses.items()
        if name in criterion.weight_dict
    ]
    if not terms:
        raise ValueError("Criterion produced no weighted loss terms")
    result = torch.stack(terms).sum()
    if result.ndim != 0 or not torch.isfinite(result).item() or result.item() <= 0:
        raise ValueError("Profile requires one positive finite scalar training loss")
    return result


def _collate(prepared: list[Any]) -> tuple[Any, Any, list[dict[str, Any]], Any]:
    return (
        torch.stack([row.image for row in prepared]).contiguous().cuda(),
        torch.stack([row.calibration for row in prepared]).cuda(),
        [{name: value.cuda() for name, value in row.target.items()} for row in prepared],
        torch.stack([row.image_size for row in prepared]).cuda(),
    )


def _gpu_memory() -> dict[str, int]:
    free, total = torch.cuda.mem_get_info()
    return {
        "torch_allocated_bytes": torch.cuda.memory_allocated(),
        "torch_reserved_bytes": torch.cuda.memory_reserved(),
        "torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "device_free_bytes": free,
        "device_total_bytes": total,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-split", type=Path, required=True)
    parser.add_argument("--role", default="train")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--portable-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measured", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new profile output; previous evidence is preserved")
    if min(args.warmup, args.measured, args.threads) < 0 or args.measured < 1 or args.threads < 1:
        parser.error("Need nonnegative warmup and positive measured/threads")

    manifest_tokens, admission = _validate_manifest_admission(
        args.manifest, args.candidate_split, args.data_root, args.role
    )
    prereg_path = args.checkpoint.parent / "preregistration.json"
    prereg = json.loads(prereg_path.read_text(encoding="utf-8"))
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if payload["preregistration_sha256"] != sha256(prereg_path):
        raise ValueError("Checkpoint preregistration hash differs")
    if payload["variant"] != prereg["variant"]:
        raise ValueError("Checkpoint variant differs from preregistration")
    predict_velocity = bool(payload["config"].get("predict_velocity", False))
    if not predict_velocity:
        raise ValueError("Campaign profiler requires the pilot velocity variant")
    sources = select_runtime_sources(
        prereg["code_sha256"], args.checkpoint.parent, require_velocity=True
    )
    modules = load_runtime_modules(sources)
    helper = modules["smoke_monodgp_portable"]
    adapter = modules["e_jepa_ttc.detection.monodgp_eap_adapter"]
    data_module = modules["e_jepa_ttc.detection.eap_rgb_data"]
    build_kwargs = {
        key: payload["config"][key]
        for key in ("predict_velocity", "velocity_loss_coef")
        if key in payload["config"]
    }
    for key in build_kwargs:
        if key not in inspect.signature(helper.build_detector).parameters:
            raise ValueError(f"Frozen builder cannot restore checkpoint option {key}")
    prepare_kwargs = {"predict_velocity": True}
    source = helper.prepare_source(
        args.source_root,
        args.portable_root,
        args.output.with_suffix(".upstream.patch"),
        **prepare_kwargs,
    )
    if (
        source["variant"] != payload["variant"]
        or source["patch_sha256"] != prereg["source"]["patch_sha256"]
    ):
        raise ValueError("Prepared portable source differs from the checkpoint source")
    model, criterion, config = helper.build_detector(
        args.portable_root, payload["config"]["group_num"], **build_kwargs
    )
    if config != payload["config"]:
        raise ValueError("Effective model config differs from checkpoint config")
    model.load_state_dict(payload["model"], strict=True)
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision(str(payload.get("matmul_precision", "highest")))
    torch.manual_seed(int(payload["seed"]))
    batch_size = args.batch_size or int(payload["batch_size"])
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    tokens = manifest_tokens[:batch_size]
    if len(tokens) != batch_size:
        raise ValueError("Manifest role is smaller than the requested batch")

    preprocess_started = time.perf_counter()
    with data_module.EAPRGBDataset(args.data_root, tokens) as dataset:
        prepared = [
            adapter.prepare_sample(
                dataset[index],
                width=int(prereg["image_hw"][1]),
                height=int(prereg["image_hw"][0]),
                include_velocity=True,
            )
            for index in range(batch_size)
        ]
    preprocess_seconds = time.perf_counter() - preprocess_started
    model.cuda().train()
    criterion.cuda().train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
    optimizer.load_state_dict(payload["optimizer"])
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.warmup + args.measured):
        phases: dict[str, float] = {}

        def measured(name: str, function: Any, phase_rows: dict[str, float] = phases) -> Any:
            torch.cuda.synchronize()
            started = time.perf_counter()
            value = function()
            torch.cuda.synchronize()
            phase_rows[name] = time.perf_counter() - started
            return value

        images, calibration, targets, sizes = measured("transfer", lambda: _collate(prepared))
        measured("zero_grad", lambda: optimizer.zero_grad(set_to_none=True))
        outputs = measured(
            "forward",
            lambda i=images, c=calibration, t=targets, s=sizes: model(i, c, t, s),
        )
        measured(
            "output_finiteness", lambda output_batch=outputs: helper.require_finite(output_batch)
        )
        losses = measured(
            "criterion",
            lambda output_batch=outputs, target_batch=targets: criterion(
                output_batch, target_batch
            ),
        )
        measured("loss_finiteness", lambda loss_values=losses: helper.require_finite(loss_values))
        loss = measured(
            "weighted_loss_sum",
            lambda loss_values=losses: _weighted_loss(loss_values, criterion),
        )
        measured("backward", loss.backward)
        norm = measured(
            "clip_grad",
            lambda: torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            ),
        )
        measured("optimizer", optimizer.step)
        if not math.isfinite(float(norm)):
            raise ValueError("Profile produced a non-finite gradient norm")
        rows.append(
            {
                "step": step,
                "warmup": step < args.warmup,
                "loss": float(loss.detach()),
                "gradient_norm_before_clip": float(norm),
                "phases_seconds": phases,
                "memory": {**memory_snapshot(), **_gpu_memory()},
            }
        )
        del images, calibration, targets, sizes, outputs, losses, loss, norm

    measured_rows = [row for row in rows if not row["warmup"]]
    report = {
        "schema": "monodgp_detection_campaign_profile_v1",
        "scope": "synchronized stage wall times on exact velocity checkpoint variant",
        "checkpoint_sha256": sha256(args.checkpoint),
        "preregistration_sha256": sha256(prereg_path),
        "profiler_sha256": sha256(Path(__file__)),
        "manifest_sha256": sha256(args.manifest),
        "admission": admission,
        "variant": payload["variant"],
        "config": config,
        "checkpoint_load": "PASS_STRICT",
        "optimizer_load": "PASS_CHECKPOINT_STATE",
        "source": source,
        "runtime_sources": sources,
        "sample_tokens": tokens,
        "batch_size": batch_size,
        "predict_velocity": predict_velocity,
        "preprocess_batch_seconds": preprocess_seconds,
        "warmup_steps": args.warmup,
        "measured_steps": args.measured,
        "rows": rows,
        "median_phase_seconds": {
            name: statistics.median(row["phases_seconds"][name] for row in measured_rows)
            for name in measured_rows[0]["phases_seconds"]
        },
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
