"""Explicitly validate and bind the known legacy synthetic confirmation without inference."""

import argparse
from pathlib import Path

from operational.phase_event_expansion.validation import bind_confirmation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirmation", type=Path, required=True)
    args = parser.parse_args()
    print(bind_confirmation(args.confirmation))


if __name__ == "__main__":
    main()
