"""Carry the already-admitted H/V mask contract into transfer and cost routes."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from e_jepa_ttc.rgb_port.features import ProducerObservation
from operational.rgb_port_fusion_fix import runtime as parent
from operational.rgb_port_streaming.contracts import validate_observation

from .contracts import validate


@contextmanager
def observation_contract() -> Iterator[None]:
    """Retain native prediction masks and all clock/support/finite-value guards."""
    with patch.object(ProducerObservation, "validate", validate_observation):
        yield


def main() -> None:
    """Keep the fusion repair and run the original CLI under the H/V validator."""
    run = Path(sys.argv[1]).resolve(strict=True)
    if sys.argv[2] not in {"operational.rgb_port.transfer", "operational.rgb_port.cost_routes"}:
        raise ValueError("External mask repair only admits transfer and costs")
    validate(run)
    with observation_contract():
        parent.main()


if __name__ == "__main__":
    main()
