"""Run the existing inference path with an admitted metadata serialization repair."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from operational.rgb_port import infer_experts
from operational.rgb_port_streaming import infer as parent

from .contracts import validate
from .signature import observation_signature


def main() -> int:
    """Keep native model execution and all repeated-observation consistency checks."""
    if "--help" not in sys.argv:
        run = Path(sys.argv[sys.argv.index("--run") + 1]).resolve(strict=True)
        validate(run)
    with patch.object(infer_experts, "_observation_signature", observation_signature):
        return parent.main()


if __name__ == "__main__":
    raise SystemExit(main())
