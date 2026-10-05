"""Run a queued stage under the source-verified disk amendment and user time budget."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from . import common
from .common import ROOT, Campaign
from .deadline import permits
from .expanded_disk import install
from .garl_train_decode import verify_admission


def main() -> int:
    """Preserve the target's CLI/recipe, adding only admitted resource and time checks."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--target", required=True)
    args, forwarded = parser.parse_known_args()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/campaign/efficient_context_v1.json"
    )
    config, _ = config_parser.parse_known_args(forwarded)
    c = Campaign(config.protocol)
    verify_admission(c)
    install(c)
    if not permits(c):
        return 3
    original = common.Campaign

    class DeadlineCampaign(original):
        def check(self) -> bool:
            return super().check() and permits(self)

    common.Campaign = DeadlineCampaign
    sys.argv = [args.target, *forwarded]
    try:
        runpy.run_module(args.target, run_name="__main__")
    finally:
        common.Campaign = original
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
