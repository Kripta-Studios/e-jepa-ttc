"""Admit the registered PhaseOnlyAnchor for descriptive external evaluation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from operational.multimodal_validation.common import read, sha, write

from . import external_evaluate_v2 as evaluator
from .common import OUT
from .phase_only_anchor import PhaseOnlyAnchor


def main() -> None:
    value = dict(
        study="PHASE_ONLY_REPLICATION",
        seeds=[13, 23],
        model="PhaseOnlyAnchor",
        source_sha256=sha(__file__),
        evaluator_source_sha256=sha(Path(evaluator.__file__)),
        scope="Admit registered small model and fit identity; metrics unchanged",
    )
    path = OUT / "studies/PHASE_ONLY_REPLICATION/EXTERNAL_EXTENSION.json"
    if path.exists():
        assert read(path) == value
    else:
        write(path, value)
    with (
        patch.dict(evaluator.STUDIES, {"PHASE_ONLY_REPLICATION": (13, 23)}),
        patch.object(evaluator, "PhaseUnitAnchor", PhaseOnlyAnchor),
    ):
        evaluator.main()


if __name__ == "__main__":
    main()
