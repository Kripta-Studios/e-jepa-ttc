"""Create a separate producer with bounded index and exact GPU image residency."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def materialize(source: Path, destination: Path, module: Path) -> None:
    """Preserve the parent producer and bind every changed source in a receipt."""
    if destination.exists():
        raise FileExistsError(destination)
    original = source / "scripts/train_detection_campaign.py"
    text = original.read_text()
    start, end = text.index("class PreparedCache:"), text.index("def prepare_model_input(")
    replacement = '''PERFORMANCE_PROFILE = {
    "schema": "phase_ttc_exact_input_residency_v1",
    "max_open_shards": 128,
    "train_capacity": 4096,
    "evaluation_capacity": 8,
    "training_image_device": "cuda",
    "maximum_image_bytes": 13 * 2**30,
    "sampling_precision_batch_loss_changed": False,
}


class PreparedCache(ResidentInputCache):
    """Use the original preparation and preserve evaluation's annotation-empty input."""

    def __init__(self, dataset, config, member_sha256, *, label_free_model_input=False):
        def prepare(raw, label_free):
            if label_free:
                return prepare_model_input(raw, config)
            return prepare_sample(raw, width=config.model.width, height=config.model.height,
                                  include_velocity=config.model.predict_velocity)
        super().__init__(dataset, config, member_sha256, prepare=prepare,
                         label_free_model_input=label_free_model_input,
                         train_capacity=PERFORMANCE_PROFILE["train_capacity"],
                         evaluation_capacity=PERFORMANCE_PROFILE["evaluation_capacity"],
                         device=PERFORMANCE_PROFILE["training_image_device"],
                         maximum_image_bytes=PERFORMANCE_PROFILE["maximum_image_bytes"])


'''
    text = text[:start] + replacement + text[end:]
    substitutions = {
        "from e_jepa_ttc.detection.backbone_initialization import": (
            "from e_jepa_ttc.detection.resident_input_cache import ResidentInputCache\n"
            "from e_jepa_ttc.detection.backbone_initialization import"
        ),
        'return EAPRGBDataset(root, tokens, max_label_sequences=settings["max_label_sequences"])': (
            "return EAPRGBDataset(root, tokens, "
            'max_open_shards=PERFORMANCE_PROFILE["max_open_shards"], '
            'max_label_sequences=settings["max_label_sequences"])'
        ),
        '        "git_commit": subprocess.check_output(': (
            '        "input_performance_profile": PERFORMANCE_PROFILE,\n'
            '        "git_commit": subprocess.check_output('
        ),
        "    owned_paths = [": (
            "    owned_paths = [\n"
            '        ROOT / "src/e_jepa_ttc/detection/resident_input_cache.py",'
        ),
        '                "preprocessing_seconds": preprocessing_times[-1],': (
            '                "input_cache": train_cache.statistics(),\n'
            '                "preprocessing_seconds": preprocessing_times[-1],'
        ),
        "        stage_image_exposures=": (
            "        input_residency=train_cache.statistics(),\n        stage_image_exposures="
        ),
    }
    for old, new in substitutions.items():
        if text.count(old) != 1:
            raise ValueError(f"Parent source anchor changed: {old}")
        text = text.replace(old, new, 1)
    compile(text, str(destination / "scripts/train_detection_campaign.py"), "exec")
    shutil.copytree(source, destination)
    target = destination / "scripts/train_detection_campaign.py"
    target.write_text(text, encoding="utf-8", newline="\n")
    target_module = destination / "src/e_jepa_ttc/detection/resident_input_cache.py"
    shutil.copy2(module, target_module)
    receipt = {
        "status": "MATERIALIZED_REQUIRES_GPU_ADMISSION",
        "parent_producer": str(source),
        "parent_runner_sha256": digest(original),
        "runner_sha256": digest(target),
        "resident_cache_sha256": digest(target_module),
        "changed_existing_files": ["scripts/train_detection_campaign.py"],
        "added_files": ["src/e_jepa_ttc/detection/resident_input_cache.py"],
        "precision_batch_sampler_model_loss_unchanged": True,
    }
    (destination / "INPUT_OPTIMIZATION_RECEIPT.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.source, args.destination, args.module)
