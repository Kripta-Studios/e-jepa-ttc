"""Run the implemented trivial TTC baseline."""

from __future__ import annotations

import sys

from e_jepa_ttc.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["baseline", "trivial", *sys.argv[1:]]))
