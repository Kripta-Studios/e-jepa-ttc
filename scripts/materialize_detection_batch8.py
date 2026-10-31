"""Freeze an explicitly exposure-mapped batch-eight continuation producer."""

# ruff: noqa: E501 -- exact frozen-source replacement anchors.

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

PARENT_SHA256 = "2a3effc4f92f105b42560d38547ef9fc5fab85fd7ca53cde375fc945b7bd761b"


def rewrite(text: str) -> str:
    """Change batch scheduling and migration only; preserve detector and loss algorithms."""
    replacements = {
        "def write_json(path: Path, value: object) -> None:": "from e_jepa_ttc.detection.batch_transition import (\n"
        "    ExposureLRScheduler, batch_size_at, exposure_units, load_transition_checkpoint\n"
        ")\n\n"
        "BATCH_RECEIPT = json.loads((ROOT / 'BATCH_TRANSITION.json').read_text())\n"
        "BATCH_PLAN = BATCH_RECEIPT['plan']\n\n\n"
        "def write_json(path: Path, value: object) -> None:",
        "    scheduler = UpdateLRScheduler(\n        optimizer,": "    scheduler = ExposureLRScheduler(\n        optimizer, plan=BATCH_PLAN,",
        "            load_campaign_checkpoint(\n                resume,": "            load_transition_checkpoint(\n                resume, receipt=BATCH_RECEIPT,",
        "    owned_paths = [": "    owned_paths = [\n        ROOT / 'BATCH_TRANSITION.json',\n"
        "        ROOT / 'src/e_jepa_ttc/detection/batch_transition.py',",
        "            indices = sampler.next_batch(config.optimization.batch_size)": "            indices = sampler.next_batch(batch_size_at(BATCH_PLAN, update))",
        "peek_permutation_batch(sampler, config.optimization.batch_size)": "peek_permutation_batch(sampler, batch_size_at(BATCH_PLAN, update + 1))",
        '            record = {\n                "update": update,': '            record = {\n                "update": update,\n'
        '                "batch_size": len(indices),\n'
        '                "original_schedule_update": exposure_units(BATCH_PLAN, update),\n'
        '                "cumulative_image_exposures": 4 * exposure_units(BATCH_PLAN, update),',
        "        stage_image_exposures=(final_update - start_update) * config.optimization.batch_size,": "        stage_image_exposures=4 * (exposure_units(BATCH_PLAN, final_update)\n"
        "                                   - exposure_units(BATCH_PLAN, start_update)),",
        "        cumulative_image_exposures=final_update * config.optimization.batch_size,": "        cumulative_image_exposures=4 * exposure_units(BATCH_PLAN, final_update),\n"
        "        batch_transition=BATCH_RECEIPT,",
    }
    for old, new in replacements.items():
        if text.count(old) != 1:
            raise ValueError(f"Batch transition source anchor differs: {old}")
        text = text.replace(old, new, 1)
    compile(text, "batch8_producer.py", "exec")
    return text


def main() -> None:
    """Materialize a new tree without modifying the historical producer."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "destination", "module", "receipt"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    parent = args.source / "scripts/train_detection_campaign.py"
    if hashlib.sha256(parent.read_bytes()).hexdigest() != PARENT_SHA256:
        raise ValueError("Unexpected Dfull parent producer")
    if args.destination.exists():
        raise FileExistsError(args.destination)
    text = rewrite(parent.read_text())
    shutil.copytree(args.source, args.destination)
    (args.destination / "scripts/train_detection_campaign.py").write_text(
        text, encoding="utf-8", newline="\n"
    )
    shutil.copy2(args.module, args.destination / "src/e_jepa_ttc/detection/batch_transition.py")
    shutil.copy2(args.receipt, args.destination / "BATCH_TRANSITION.json")
    hashes = {
        str(p.relative_to(args.destination)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            args.destination / "scripts/train_detection_campaign.py",
            args.destination / "src/e_jepa_ttc/detection/batch_transition.py",
            args.destination / "BATCH_TRANSITION.json",
        )
    }
    receipt = {
        "parent_runner_sha256": PARENT_SHA256,
        "files": hashes,
        "precision": "unchanged FP32 highest",
        "scope": "batch/LR exposure mapping",
    }
    (args.destination / "BATCH8_PRODUCER_RECEIPT.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
