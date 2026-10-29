"""Aggregate saved evaluation JSON files without inventing missing metrics."""

from __future__ import annotations

import sys

from aggregate_results import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
