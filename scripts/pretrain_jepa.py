"""Script wrapper for JEPA-style self-supervised pretraining."""

from __future__ import annotations

import sys

from e_jepa_ttc.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["pretrain", "jepa", *sys.argv[1:]]))
