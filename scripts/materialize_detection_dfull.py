"""Freeze a Dfull producer with lossless cache and explicit model-only initialization."""

# ruff: noqa: E501 -- frozen producer source anchors must remain exact.

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT_SHA256 = "e8c7291de32f39e7326857c0722bb04068f81f605f0ab22ebdb5240f317519db"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rewrite(text: str, train_count: int) -> str:
    """Apply explicit frozen-source anchors, preserving detector/evaluation algorithms."""
    replacements = {
        '    "train_capacity": 4096,': f'    "train_capacity": {train_count},\n'
        '    "training_cache": "lossless_uint8_lut_and_packed_binary_region",\n'
        '    "compact_gpu_image_budget": 28 * 2**30,\n'
        '    "compact_cpu_image_budget": 30 * 2**30,\n'
        '    "legacy_maximum_image_bytes_applies_to": "evaluation fallback",',
        "from e_jepa_ttc.detection.resident_input_cache import ResidentInputCache": "from e_jepa_ttc.detection.compact_training_cache import CompactTrainingCache, peek_permutation_batch\n"
        "from e_jepa_ttc.detection.model_warm_start import load_model_warm_start\n"
        "from e_jepa_ttc.detection.resident_input_cache import ResidentInputCache",
        "    def __init__(self, dataset, config, member_sha256, *, label_free_model_input=False):": "    def __new__(cls, dataset, config, member_sha256, *, label_free_model_input=False):\n"
        "        if not label_free_model_input:\n"
        "            return CompactTrainingCache(dataset, config, member_sha256,\n"
        "                prepare=lambda raw: prepare_sample(raw, width=config.model.width,\n"
        "                    height=config.model.height, include_velocity=config.model.predict_velocity))\n"
        "        return super().__new__(cls)\n\n"
        "    def __init__(self, dataset, config, member_sha256, *, label_free_model_input=False):",
        "    owned_paths = [": "    owned_paths = [\n"
        "        ROOT / 'WARM_START.json',\n"
        "        ROOT / 'src/e_jepa_ttc/detection/compact_training_cache.py',\n"
        "        ROOT / 'src/e_jepa_ttc/detection/model_warm_start.py',",
        "        initialization = load_imagenet_resnet50_backbone(\n"
        "            model, Path(config.model.backbone_weights), config.model.backbone_weights_sha256\n"
        "        )": "        initialization = load_model_warm_start(\n"
        "            model, ROOT / 'WARM_START.json', canonical_sha256(effective_model)\n"
        "        )",
        "            images, calibrations, targets, sizes = batch_to_cuda(rows, config, training=True)": "            images, calibrations, targets, sizes = batch_to_cuda(rows, config, training=True)\n"
        "            if update < config.optimization.updates:\n"
        "                train_cache.prefetch(peek_permutation_batch(sampler, config.optimization.batch_size))",
        "    train_dataset.close()": "    train_cache.close()\n    train_dataset.close()",
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError(f"Unexpected frozen parent anchor: {old}")
        text = text.replace(old, new, 1)
    compile(text, "dfull_runner.py", "exec")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "parent-runner", "destination", "modules", "warm-start"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--sampling-plan", type=Path)
    args = parser.parse_args()
    if digest(args.parent_runner) != PARENT_SHA256:
        raise ValueError("Parent runner differs from admitted R2 producer")
    if args.destination.exists():
        raise FileExistsError(args.destination)
    runner = rewrite(args.parent_runner.read_text(), args.train_count)
    if args.sampling_plan is not None:
        from materialize_detection_rare_sampling import rewrite as rare_rewrite

        runner = rare_rewrite(runner, digest(args.sampling_plan))
    shutil.copytree(args.source, args.destination)
    for name in ("sampling_plan.json", "RARE_SAMPLING_RECEIPT.json"):
        (args.destination / name).unlink(missing_ok=True)
    (args.destination / "scripts/train_detection_campaign.py").write_text(
        runner, encoding="utf-8", newline="\n"
    )
    for name in ("compact_training_cache.py", "model_warm_start.py"):
        shutil.copy2(args.modules / name, args.destination / "src/e_jepa_ttc/detection" / name)
    shutil.copy2(args.warm_start, args.destination / "WARM_START.json")
    if args.sampling_plan is not None:
        shutil.copy2(args.sampling_plan, args.destination / "sampling_plan.json")
    receipt = {
        "schema": "phase_dfull_producer_v1",
        "parent_runner_sha256": PARENT_SHA256,
        "runner_sha256": digest(args.destination / "scripts/train_detection_campaign.py"),
        "module_sha256": {
            name: digest(args.modules / name)
            for name in ("compact_training_cache.py", "model_warm_start.py")
        },
        "warm_start_sha256": digest(args.warm_start),
        "training_sampling": (
            "registered rare schedule" if args.sampling_plan else "resumable uniform permutation"
        ),
        "sampling_plan_sha256": digest(args.sampling_plan) if args.sampling_plan else None,
        "detector_loss_decoder_evaluation": "unchanged parent algorithms",
    }
    (args.destination / "DFULL_PRODUCER_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
