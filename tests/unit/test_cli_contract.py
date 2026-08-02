from __future__ import annotations

from pathlib import Path

from e_jepa_ttc.cli import build_parser


def test_required_plan_cli_aliases_parse() -> None:
    parser = build_parser()
    cases = [
        ["train", "finetune", "--cache-manifest", "cache.json", "--output-dir", "out"],
        [
            "export",
            "onnx",
            "--cache-manifest",
            "cache.json",
            "--checkpoint",
            "checkpoint.pt",
            "--output-dir",
            "out",
        ],
        ["robustness", "--output", "robustness.json"],
        ["demo", "--output-dir", "demo"],
        ["report", "build", "--output-dir", "report"],
        ["data", "validate", "--config", "configs/data/evttc_starter.yaml"],
    ]
    for argv in cases:
        parsed = parser.parse_args(argv)
        assert callable(parsed.func)


def test_report_cli_defaults_are_repo_relative() -> None:
    parsed = build_parser().parse_args(["report", "build"])
    assert parsed.repo_root == Path(".")
    assert parsed.output_dir == Path("artifacts/tables/regenerable_report")
