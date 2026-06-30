"""Script wrapper for supervised TinyCNN training."""

from __future__ import annotations

import sys

from e_jepa_ttc.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", "tiny-cnn", *sys.argv[1:]]))