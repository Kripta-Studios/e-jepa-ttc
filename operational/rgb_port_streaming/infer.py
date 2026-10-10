"""Run native frozen feature extraction with its current-TTC mask contract."""

from unittest.mock import patch

from e_jepa_ttc.rgb_port.features import ProducerObservation
from operational.rgb_port import infer_experts

from .contracts import validate_observation


def main() -> int:
    with patch.object(ProducerObservation, "validate", validate_observation):
        return infer_experts.main()


if __name__ == "__main__":
    raise SystemExit(main())
