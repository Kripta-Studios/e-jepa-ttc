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


@pytest.mark.parametrize("outcome,code", [(True, 0), (False, 10), ("pause", 3)])
def test_compiled_verify_dispatch_never_compiles(entry, tmp_path, monkeypatch, outcome, code):
    paths = tmp_path / "local.json"
    paths.write_text(json.dumps({"worktree": str(tmp_path)}))
    output = tmp_path / "output"

    def verify(*args, **kwargs):
        assert args == (
            output,
            tmp_path / "artifacts/simplex_t/T1/context_features_fp32",
            tmp_path / "artifacts/simplex_t/T1/query_context_index",
            tmp_path / "artifacts/simplex_t/T1/query_context_dedup",
            0,
        )
        assert kwargs["pool"] == "D0"
        if outcome == "pause":
            raise InterruptedError("RESOURCE_PAUSE: fixture")
        return outcome

    monkeypatch.setattr(entry, "verify_compiled_fold", verify)
    monkeypatch.setattr(entry, "compile_fold", lambda *a, **k: pytest.fail("compiled in verify"))
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "prepare",
            "--compile-fold",
            "0",
            "--verify-only",
            "--local-paths",
            str(paths),
            "--output",
            str(output),
            "--other-reserved-bytes",
            "0",
        ],
    )
    assert entry.main() == code
    assert not output.exists()


@pytest.mark.parametrize("pool", ["D0", "D1", "DENSE_OLD"])
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
            "--other-reserved-bytes",
            "20000000000",
        ],
    )
    assert entry.main() == 0
    temporal = tmp_path / "artifacts/simplex_t/T1"
    prefix = {"D0": "", "D1": "expansion_", "DENSE_OLD": "dense_"}[pool]
    assert calls == [
        (
            (
                temporal / f"{prefix}context_features_fp32",
                temporal / f"{prefix}query_context_index",
                temporal / f"{prefix}query_context_dedup",
                output,
                1,
            ),
            {"pool": pool, "other_reserved_bytes": 20_000_000_000},
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
        ["prepare", "--compile-fold", "0", "--output", "unused"],
    ],
)
def test_invalid_compile_arguments_fail_before_local_path_access(entry, monkeypatch, args):
    monkeypatch.setattr("sys.argv", ["runner", *args, "--local-paths", "does-not-exist"])
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2


def test_dense_prepare_passes_explicit_d0_catalog(entry, tmp_path, monkeypatch):
    paths = tmp_path / "local.json"
    paths.write_text(json.dumps({"worktree": str(tmp_path)}), encoding="utf-8")
    catalog, received, compilations = object(), [], []

    def construct(**kwargs):
        received.append(kwargs)
        return catalog

    monkeypatch.setattr(entry, "D0ReuseCatalog", construct)
    monkeypatch.setattr(entry, "compile_fold", lambda *args, **kw: compilations.append(kw))
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "prepare",
            "--compile-fold",
            "0",
            "--compile-pool",
            "DENSE_OLD",
            "--local-paths",
            str(paths),
            "--output",
            str(tmp_path / "output"),
            "--reuse-d0-compiled",
            str(tmp_path / "original"),
            "--reuse-d0-compiled-sha256",
            "a" * 64,
            "--other-reserved-bytes",
            "20000000000",
        ],
    )
    assert entry.main() == 0
    assert received[0]["compiled_sha256"] == "a" * 64
    assert received[0]["outer"] == 0
    assert compilations == [
        {"pool": "DENSE_OLD", "reuse": catalog, "other_reserved_bytes": 20_000_000_000}
    ]


def test_source_configuration_inspection_is_not_run(entry, monkeypatch):
    calls = []
    monkeypatch.setattr(
        entry, "inspect_source_configuration", lambda *args: calls.append(args) or {}
    )
    args = [
        "runner",
        "prepare",
        "--local-paths",
        "local.json",
        "--source-config",
        "sources.json",
        "--source-config-sha256",
        "a" * 64,
        "--output",
        "inspection.json",
    ]
    monkeypatch.setattr("sys.argv", args)
    assert entry.main() == 0
    assert calls == [(Path("local.json"), Path("sources.json"), "a" * 64, Path("inspection.json"))]
    args[1] = "run"
    with pytest.raises(SystemExit):
        entry.main()
    assert len(calls) == 1


def test_source_identities_pass_resume_and_reservations(entry, monkeypatch):
    calls = []
    monkeypatch.setattr(
        entry,
        "prepare_configured_sources",
        lambda *args, **kwargs: calls.append((args, kwargs)) or {"status": "PAUSED_RESOURCE"},
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "prepare",
            "--source-identities",
            "--local-paths",
            "local.json",
            "--source-config",
            "sources.json",
            "--source-config-sha256",
            "a" * 64,
            "--output",
            "prepared",
            "--other-reserved-bytes",
            "20000000000",
            "--resume",
        ],
    )
    assert entry.main() == 3
    assert calls[0][1] == {"other_reserved_bytes": 20_000_000_000, "resume": True}


def test_identity_preparation_requires_explicit_reservations(entry, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "prepare",
            "--source-identities",
            "--local-paths",
            "missing",
            "--source-config",
            "missing",
            "--source-config-sha256",
            "a" * 64,
            "--output",
            "unused",
        ],
    )
    with pytest.raises(SystemExit) as error:
        entry.main()
    assert error.value.code == 2


@pytest.mark.parametrize(
    "outcome",
    ["pause_exception", "pause_runtime", "lineage", "io", "unexpected", "complete"],
)
def test_source_preparation_only_resource_pauses_are_retryable(entry, monkeypatch, outcome):
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "prepare",
            "--source-identities",
            "--local-paths",
            "local.json",
            "--source-config",
            "sources.json",
            "--source-config-sha256",
            "a" * 64,
            "--output",
            "prepared",
            "--other-reserved-bytes",
            "20000000000",
        ],
    )

    def prepare(*args, **kwargs):
        if outcome == "pause_exception":
            raise InterruptedError("PAUSED_RESOURCE: before source loading")
        if outcome == "pause_runtime":
            raise RuntimeError("RESOURCE_PAUSE: available host RAM")
        if outcome == "lineage":
            raise RuntimeError("producer ancestry differs")
        if outcome == "io":
            raise InterruptedError("unclassified interrupted I/O")
        return {
            "status": "SOURCE_IDENTITIES_COMPLETE_NOT_SCIENTIFIC_FREEZE"
            if outcome == "complete"
            else "PREPARING"
        }

    monkeypatch.setattr(entry, "prepare_configured_sources", prepare)
    if outcome in {"lineage", "io", "unexpected"}:
        with pytest.raises((RuntimeError, InterruptedError, ValueError)):
            entry.main()
    else:
        assert entry.main() == (0 if outcome == "complete" else 3)


def test_source_verifier_incomplete_routes_to_orchestrator_without_resume(entry, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "prepare",
            "--source-identities",
            "--verify-only",
            "--local-paths",
            "local.json",
            "--source-config",
            "sources.json",
            "--source-config-sha256",
            "a" * 64,
            "--output",
            "prepared",
            "--other-reserved-bytes",
            "20000000000",
        ],
    )

    def verify(*args, **kwargs):
        assert kwargs["verify_only"] is True and kwargs["resume"] is False
        return {"status": "SOURCE_IDENTITIES_INCOMPLETE"}

    monkeypatch.setattr(entry, "prepare_configured_sources", verify)
    assert entry.main() == 10
