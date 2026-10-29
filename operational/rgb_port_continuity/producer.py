"""Keep all admitted producer wrappers and add the outermost lifecycle guard."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

from operational.rgb_port import train_producers as training
from operational.rgb_port.accounting import atomic_write_json, sha256_file
from operational.rgb_port_io_recovery import producer as io_producer

from .checkpoint import installed
from .contracts import NAME, validate


def main(argv: Sequence[str] | None = None) -> int:
    """Enter the guard inside the frozen fit, after every wrapper is installed."""
    values = list(sys.argv[1:] if argv is None else argv)
    if "--help" in values:
        return io_producer.main(values)
    run = Path(values[values.index("--run") + 1]).resolve(strict=True)
    fit_id = values[values.index("--fit-id") + 1]
    validate(run)
    atomic_write_json(
        run / "fits" / fit_id / "CONTINUITY_RUNTIME.json",
        {
            "schema": "rgb_port_continuity_runtime_v1",
            "fit_id": fit_id,
            "freeze_sha256": sha256_file(run / NAME),
            "objective_changed": False,
            "geometry_precision": "bf16_unchanged",
        },
    )
    original_fit = training.fit_producer

    def fit(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        with installed():
            return original_fit(*args, **kwargs)

    with patch.object(training, "fit_producer", fit):
        return io_producer.main(values)


if __name__ == "__main__":
    raise SystemExit(main())
