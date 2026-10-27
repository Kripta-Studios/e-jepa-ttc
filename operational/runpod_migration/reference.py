"""Record a tiny CPU-only input reference; never instantiate a model or optimizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .inventory import digest, read


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--rotation", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--modality", choices=("rgb", "event"), default="rgb")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import psutil

    if psutil.virtual_memory().available < 10 * 1024**3:
        raise RuntimeError(
            "Reference preparation needs 10GiB host headroom; defer, do not disturb training"
        )
    from operational.final_refit.fold_sources import load_fold_event_source, load_fold_rgb_source

    from .runner import batch_identity

    cfg = read(args.run / "SCIENTIFIC_CONFIG.json")
    source = (
        load_fold_rgb_source(
            Path(cfg["rgb_source"]), Path(cfg["teacher"]), Path(cfg["fold_contract"]), args.rotation
        )
        if args.modality == "rgb"
        else load_fold_event_source(
            Path(cfg["fold_contract"]), args.rotation, Path(cfg["event_root"])
        )
    )
    try:
        report = {
            "schema": "ejepa_cpu_input_reference_v1",
            "rotation": args.rotation,
            "optimizer_updates": 0,
            "model_forward_calls": 0,
            "authority_sha256": digest(args.run / "SCIENTIFIC_CONFIG.json"),
            "batch_identity": batch_identity(source),
        }
    finally:
        source.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "rotation": args.rotation,
                "population": report["batch_identity"]["population"],
            }
        )
    )


if __name__ == "__main__":
    main()
