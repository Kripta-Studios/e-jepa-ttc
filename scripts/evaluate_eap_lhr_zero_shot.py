"""Zero-shot evaluate the unchanged eAP-trained LHR object-JEPA TTC head."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.training.eap_lhr_jepa_ttc import evaluate_eap_lhr_zero_shot  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["validation"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = evaluate_eap_lhr_zero_shot(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        splits=tuple(args.splits),
        output_path=args.output,
        batch_size=args.batch_size,
        num_workers=args.workers,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
