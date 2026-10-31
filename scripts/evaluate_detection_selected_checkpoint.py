"""Evaluate a selected checkpoint using its verified frozen producer in a fresh process."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import torch


def module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = args.checkpoint.parent.parent
    loader = module(root / "code/scripts/infer_detection_campaign.py", "frozen_loader")
    checkpoint, prereg, verified_root = loader.verify_checkpoint_runtime(args.checkpoint)
    loader.assert_fresh_runtime()
    assert root.resolve() == verified_root.resolve()
    sys.path[:0] = [str(root / "code/src"), str(root / "code/scripts")]
    runner = module(root / "code/scripts/train_detection_campaign.py", "frozen_evaluation_runner")
    summary = json.loads((root / "summary.json").read_text())
    config = runner.load_campaign_config(Path(summary["config"]))
    if config.sha256 != checkpoint["config_sha256"]:
        raise ValueError("Selected checkpoint/config identity differs")
    if (
        loader.sha256_file(Path(config.data.root) / "data/train.parquet")
        != prereg["dataset_media_sha256"]
    ):
        raise ValueError("Dataset metadata differs from the admitted producer")
    torch.set_num_threads(config.resources.num_cpu_threads)
    torch.set_float32_matmul_precision(config.optimization.matmul_precision)
    model, _, settings = runner.build_detector(
        root / "code/portable_source",
        config.model.group_num,
        predict_velocity=config.model.predict_velocity,
        velocity_loss_coef=config.model.velocity_loss_coef,
    )
    if settings != prereg["effective_model"]:
        raise ValueError("Selected model architecture differs")
    model.load_state_dict(checkpoint["model"], strict=True)
    protocol = runner.load_protocol(config.evaluation.protocol)
    tokens_path = Path(config.data.full_selection_tokens)
    if runner.sha256_file(tokens_path) != config.data.expected_full_selection_sha256:
        raise ValueError("Full selection population differs")
    tokens = json.loads(tokens_path.read_text())["sample_tokens"]
    with runner.open_campaign_dataset(config.data.root, tokens, config) as dataset:
        cache = runner.PreparedCache(
            dataset, config, checkpoint["data_member_sha256"], label_free_model_input=True
        )
        result = runner.evaluate(model, cache, protocol, config, args.output)
    receipt = {
        "status": "COMPLETE",
        "checkpoint_sha256": loader.sha256_file(args.checkpoint),
        "checkpoint_update": checkpoint["update"],
        "evaluation_sha256": loader.sha256_file(args.output),
        "sample_count": result["sample_count"],
    }
    args.output.with_suffix(".receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
