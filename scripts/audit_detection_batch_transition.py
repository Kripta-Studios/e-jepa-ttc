"""Verify consumed tokens, exposure clock and learning rates after a batch transition."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    """Audit a committed metric prefix without modifying the running trainer."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("receipt", "run", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.run / "code/src"))
    import torch

    from e_jepa_ttc.detection.batch_transition import batch_size_at, exposure_units
    from e_jepa_ttc.detection.campaign_training import (
        ResumablePermutationSampler,
        UpdateLRScheduler,
        load_campaign_config,
    )

    receipt = json.loads(args.receipt.read_text())
    plan = receipt["plan"]
    source = torch.load(
        receipt["source_checkpoint"], map_location="cpu", weights_only=True, mmap=True
    )
    config = load_campaign_config(Path(receipt["source_config"]))
    tokens = json.loads(Path(config.data.manifest).read_text())["roles"]["train"]["sample_tokens"]
    sampler = ResumablePermutationSampler(len(tokens), config.seed)
    sampler.load_state_dict(source["sampler"])
    optimizer = torch.optim.AdamW(
        [
            {"params": [torch.nn.Parameter(torch.ones(1))], "lr": lr}
            for lr in source["scheduler"]["base_lrs"]
        ]
    )
    scheduler = UpdateLRScheduler(
        optimizer,
        total_updates=config.optimization.updates,
        kind=config.optimization.scheduler,
        warmup_fraction=config.optimization.warmup_fraction,
        minimum_ratio=config.optimization.minimum_lr_ratio,
    )
    data = (args.run / "metrics.jsonl").read_text()
    lines = data.splitlines()
    if not data.endswith("\n"):
        lines = lines[:-1]
    rows = [json.loads(line) for line in lines]
    if not rows:
        raise ValueError("No completed new optimizer steps")
    consumed = plan["source_update"] * 4
    for expected_update, row in enumerate(rows, plan["source_update"] + 1):
        size = batch_size_at(plan, expected_update)
        expected = [tokens[i] for i in sampler.next_batch(size)]
        consumed += size
        if row["update"] != expected_update or row["sample_tokens"] != expected:
            raise ValueError("Post-transition sampler order differs")
        if row["batch_size"] != size or row["cumulative_image_exposures"] != consumed:
            raise ValueError("Post-transition image budget differs")
        if row["original_schedule_update"] != exposure_units(plan, expected_update):
            raise ValueError("LR exposure clock differs")
        if row["learning_rates"] != scheduler.apply(exposure_units(plan, expected_update)):
            raise ValueError("Executed learning rates differ from the original exposure schedule")
    recent = rows[-200:]
    seconds = sum(r["preprocessing_seconds"] + r["forward_backward_update_seconds"] for r in recent)
    result = {
        "status": "PASS",
        "observed_utc": datetime.now(UTC).isoformat(),
        "source_update": plan["source_update"],
        "verified_new_updates": len(rows),
        "last_update": rows[-1]["update"],
        "cumulative_exposures": consumed,
        "total_exposures": plan["total_exposures"],
        "progress_fraction": consumed / plan["total_exposures"],
        "ordered_tokens_exact": True,
        "learning_rates_by_exposure_exact": True,
        "recent_window_steps": len(recent),
        "recent_stage_images_per_second": sum(r["batch_size"] for r in recent) / seconds,
        "recent_prepare_median_seconds": statistics.median(
            r["preprocessing_seconds"] for r in recent
        ),
        "recent_compute_median_seconds": statistics.median(
            r["forward_backward_update_seconds"] for r in recent
        ),
        "timing_scope": "Stage times only; excludes checkpoint, evaluation and launch overhead.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
