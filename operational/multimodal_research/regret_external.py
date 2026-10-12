"""Admit the unchanged PhaseUnitAnchor architecture under the RGB_REGRET fit identity."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from operational.multimodal_validation.common import read, sha, write

from . import external_evaluate_v2 as evaluator
from .common import OUT


def main() -> None:
    value = dict(
        study="RGB_REGRET",
        seeds=[7],
        model="PhaseUnitAnchor",
        source_sha256=sha(__file__),
        evaluator_source_sha256=sha(Path(evaluator.__file__)),
        scope="Add registered fit identity to existing external evaluator; metrics unchanged",
    )
    path = OUT / "studies/RGB_REGRET/EXTERNAL_EXTENSION.json"
    if path.exists():
        assert read(path) == value
    else:
        write(path, value)
    with patch.dict(evaluator.STUDIES, {"RGB_REGRET": (7,)}):
        evaluator.main()


if __name__ == "__main__":
    main()
