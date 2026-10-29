"""Locate sustained real-frame step costs on a trained portable detector checkpoint."""

# ruff: noqa: E402, B023 -- direct execution; timing callbacks run synchronously in their iteration.

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import torch
from profile_monodgp_precision import memory_snapshot
from smoke_monodgp_portable import build_detector, require_finite, sha256
from train_monodgp_portable import inputs_on_gpu, write_json

from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset
from e_jepa_ttc.detection.monodgp_eap_adapter import prepare_sample


def main() -> int:
    """Synchronize each named phase; preserve the overhead in the reported scope."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fresh-optimizer", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--portable-root", type=Path, default=ROOT / "output/detector_sources/MonoDGP-portable"
    )
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(7)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model, criterion, config = build_detector(args.portable_root, payload["config"]["group_num"])
    model.load_state_dict(payload["model"], strict=True)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=2e-4)
    if not args.fresh_optimizer:
        optimizer.load_state_dict(payload["optimizer"])
    del payload
    tokens = json.loads(args.manifest.read_text())["overfit16"]["sample_tokens"]
    chosen = [tokens[index] for index in (0, 7, 10)]
    with EAPRGBDataset(args.data_root, chosen) as dataset:
        prepared = [prepare_sample(dataset[i], width=640, height=384) for i in range(3)]
    report = {
        "scope": "synchronized_phase_wall_times_not_pure_kernel_time",
        "rows": [],
        "checkpoint_sha256": sha256(args.checkpoint),
        "script_sha256": sha256(Path(__file__)),
        "source_receipt_sha256": sha256(args.portable_root / ".portable_source.json"),
        "fresh_optimizer": args.fresh_optimizer,
        "threads": args.threads,
        "sample_tokens": chosen,
        "config": config,
    }
    model.train()
    criterion.train()
    for step in range(6):
        phases = {}

        def measured(name: str, function: Any) -> Any:
            torch.cuda.synchronize()
            start = time.perf_counter()
            value = function()
            torch.cuda.synchronize()
            phases[name] = time.perf_counter() - start
            return value

        images, calibration, targets, sizes = measured(
            "transfer", lambda: inputs_on_gpu(prepared[step % 3])
        )
        measured("zero_grad", lambda: optimizer.zero_grad(set_to_none=True))
        outputs = measured("forward", lambda: model(images, calibration, targets, sizes))
        measured("output_finiteness", lambda value=outputs: require_finite(value))
        losses = measured("criterion", lambda value=outputs: criterion(value, targets))
        measured("loss_finiteness", lambda value=losses: require_finite(value))
        loss = measured(
            "weighted_loss_sum",
            lambda values=losses: sum(
                value * criterion.weight_dict[key]
                for key, value in values.items()
                if key in criterion.weight_dict
            ),
        )
        measured("backward", loss.backward)
        norm = measured(
            "clip_grad",
            lambda: torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, error_if_nonfinite=True
            ),
        )
        measured("optimizer", optimizer.step)
        row = {
            "step": step,
            "warmup": step < 3,
            "phases_seconds": phases,
            "total_seconds": sum(phases.values()),
            "gradient_norm": float(norm),
            "loss": float(loss.detach()),
            "memory": memory_snapshot(),
        }
        report["rows"].append(row)
        print(json.dumps(row), flush=True)
        del outputs, losses, loss
    report["median_measured_phase_seconds"] = {
        name: statistics.median(row["phases_seconds"][name] for row in report["rows"][3:])
        for name in report["rows"][0]["phases_seconds"]
    }
    write_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
