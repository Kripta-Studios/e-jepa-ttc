"""Script wrapper for voxel-cache generation."""

from __future__ import annotations

import sys

from e_jepa_ttc.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["cache", "voxel", *sys.argv[1:]]))