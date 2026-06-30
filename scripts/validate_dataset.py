"""Validate a dataset manifest."""

from __future__ import annotations

import sys

from e_jepa_ttc.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["data", "validate", *sys.argv[1:]]))
