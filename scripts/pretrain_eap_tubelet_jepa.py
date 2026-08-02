"""Guard the not-yet-implemented high-resolution Tubelet JEPA pretrainer.

The legacy eAP pretrainer produces a pooled encoder with an incompatible state
dict.  Treating it as Tubelet pretraining would silently invalidate the matched
scratch-versus-JEPA experiment, so this entry point fails before opening data.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    """Reject accidental use until a matched dense-token SSL trainer exists."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eap-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.parse_args(argv)
    parser.error(
        "High-resolution Tubelet JEPA pretraining is not implemented. "
        "scripts/pretrain_eap_jepa.py trains the legacy pooled encoder and its "
        "checkpoint must not be relabelled or transferred as a Tubelet result."
    )


if __name__ == "__main__":
    raise SystemExit(main())
