"""Keep static diagnostics comparable without labelling moved lines as regressions."""

import importlib.util
from pathlib import Path

import pytest


def module():
    path = Path(__file__).resolve().parents[2] / "scripts/audit_simplex_t_static_failure_ids.py"
    spec = importlib.util.spec_from_file_location("simplex_static_ids", path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_ids_normalize_only_the_root(tmp_path):
    roots = [tmp_path / "baseline", tmp_path / "current"]
    outputs = []
    for root in roots:
        root.mkdir()
        path = root / "model.py"
        path.write_text("# fixture", encoding="utf-8")
        item = dict(
            filename=str(path),
            code="F401",
            message="unused import",
            location={"row": 1, "column": 2},
        )
        outputs.append(module().failure_ids([item], root))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("problem", ["escape", "duplicate"])
def test_invalid_diagnostic_identity_fails(tmp_path, problem):
    root = tmp_path / "code"
    root.mkdir()
    path = (tmp_path if problem == "escape" else root) / "model.py"
    path.write_text("# fixture", encoding="utf-8")
    item = dict(filename=str(path), code="F401", message="unused", location={"row": 1, "column": 1})
    with pytest.raises(ValueError):
        module().failure_ids([item, item] if problem == "duplicate" else [item], root)
