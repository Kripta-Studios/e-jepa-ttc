"""CLI compilation wiring; compiler is mocked, no scientific fits are authorized."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def entry():
    path = Path(__file__).resolve().parents[2] / "scripts/run_simplex_t_companion.py"
    spec = importlib.util.spec_from_file_location("simplex_t_cli_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("pool", ["D0", "D1"])
def test_prepare_compiles_requested_fold_without_scientific_authorization(
    entry, tmp_path, monkeypatch, capsys, pool
):
    paths = tmp_path / "local.json"
    paths.write_text(json.dumps({"worktree": str(tmp_path)}), encoding="utf-8")
    output = tmp_path / "outer1"
    calls = []
    monkeypatch.setattr(entry, "compile_fold", lambda *args, **kw: calls.append((args, kw)))
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "prepare",
            "--compile-fold",
            "1",
            "--compile-pool",
            pool,
            "--local-paths",
            str(paths),
            "--output",
            str(output),
        ],
    )
    assert entry.main() == 0
    temporal = tmp_path / "artifacts/simplex_t/T1"
    prefix = "expansion_" if pool == "D1" else ""
    assert calls == [
        (
            (
                temporal / f"{prefix}context_features_fp32",
                temporal / f"{prefix}query_context_index",
                temporal / f"{prefix}query_context_dedup",
                output,
                1,
            ),
            {"pool": pool},
        )
    ]
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "COMPLETE_FOLD_CACHE_NOT_SCIENTIFIC_FREEZE"
    assert result["optimizer_updates"] == 0


@pytest.mark.parametrize(
    "args",
    [
        ["run", "--compile-fold", "0", "--output", "unused"],
        ["prepare", "--compile-fold", "0"],
        ["prepare", "--compile-fold", "0", "--output", "unused", "--resume"],
        ["prepare", "--compile-fold", "3", "--output", "unused"],
        ["prepare", "--compile-pool", "D1", "--output", "unused"],
    ],
)
def test_invalid_compile_arguments_fail_before_local_path_access(entry, monkeypatch, args):
    monkeypatch.setattr("sys.argv", ["runner", *args, "--local-paths", "does-not-exist"])
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2
