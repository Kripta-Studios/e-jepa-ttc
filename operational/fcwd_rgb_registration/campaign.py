"""Resume registration, frozen-model inference, then post-seal FCWD scoring."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch

from operational.rgb_port.accounting import atomic_write_json

from .prepare import build
from .score import score
from .transfer import run


def main() -> None:
    """Retry transient storage failures from immutable committed fragments."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "manifest", "calibration", "output", "comparator"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            atomic_write_json(args.output / "STATE.json", {"status": "REGISTERING"})
            build(args.manifest, args.calibration, args.output / "registration")
            atomic_write_json(args.output / "STATE.json", {"status": "PREDICTING"})
            run(
                args.run,
                args.manifest,
                args.output / "registration",
                args.output / "predictions",
                args.device,
            )
            atomic_write_json(args.output / "STATE.json", {"status": "SCORING"})
            score(args.output / "predictions", args.comparator)
            atomic_write_json(args.output / "STATE.json", {"status": "COMPLETE"})
            break
        except OSError as exc:
            # Shape, provenance and scientific-contract errors are not retried.
            print(f"Storage unavailable; retrying from saved fragments in 30s: {exc}", flush=True)
            time.sleep(30)


if __name__ == "__main__":
    main()
