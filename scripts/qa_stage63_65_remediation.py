"""Capture reproducible local QA logs without starting a scientific campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def code_identity(repo: Path) -> dict[str, object]:
    """Capture code identity before and after QA, including uncommitted work."""

    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", *args], cwd=repo)

    untracked = git("ls-files", "--others", "--exclude-standard", "-z").decode("utf-8").split("\0")
    return {
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "dirty": bool(git("status", "--porcelain").strip()),
        "diff_sha256": hashlib.sha256(git("diff", "HEAD", "--")).hexdigest(),
        "untracked": {
            name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
            for name in untracked
            if name
        },
    }


def classify_historical(xml_path: Path, repo: Path) -> dict[str, object]:
    """Record each failure's observed cause; never report an unexecuted test as passed."""
    tree = ET.parse(xml_path)
    failures = []
    for case in tree.iter("testcase"):
        failure = case.find("failure")
        error = case.find("error")
        if failure is None and error is None:
            continue
        node = failure if failure is not None else error
        assert node is not None
        name = case.attrib["name"]
        message = node.attrib.get("message", "")
        source = case.attrib["classname"].replace(".", "/") + ".py"
        category = "unclassified_requires_review"
        subject = source
        schema = re.search(r"(scientific_recovery_v9_eclock_\S+\.schema\.json)", message)
        if schema and name.startswith("test_schema_requires_scientific_metadata"):
            category = "preexisting_eclock_schema_contract_failure"
            subject = "schemas/" + schema.group(1)
        elif "FileNotFoundError" in message and source in {
            "tests/regression/test_scientific_recovery_v8_immutable.py",
            "tests/unit/test_freeze_scientific_recovery_v5_garl_grouped.py",
            "tests/unit/test_freeze_scientific_recovery_v8_configs.py",
            "tests/unit/test_scientific_recovery_v8_autopsy.py",
        }:
            category = "historical_artifact_not_present_in_new_worktree"
        elif "grouped-development protocol file SHA256 differs" in message:
            category = "historical_grouped_protocol_pin_mismatch"
        elif "A5 DINO manifest file hash differs from frozen source" in message:
            category = "historical_v8_teacher_pin_mismatch"
        elif "test_freeze_scientific_recovery_v8_configs" in source and "assert 2 == 0" in message:
            log = (xml_path.parent / "historical_suite.log").read_text(
                encoding="utf-8", errors="replace"
            )
            if "V8 freeze failed closed:" in log and "baselines\\\\manifest.json" in log:
                category = "historical_v8_freeze_missing_baseline_manifest"
        unchanged = (
            subprocess.run(
                ["git", "diff", "--quiet", "HEAD", "--", source, subject],
                cwd=repo,
                check=False,
            ).returncode
            == 0
        )
        failures.append(
            {
                "test": case.attrib["classname"] + "::" + name,
                "category": category,
                "observed_message": message,
                "traceback": node.text,
                "source": source,
                "subject": subject,
                "source_and_subject_unchanged_from_pre_remediation_head": unchanged,
                "result": "failed_not_waived_or_relabelled_passed",
            }
        )
    return {
        "suites": [item.attrib for item in tree.iter("testsuite")],
        "failures": failures,
        "unclassified_count": sum(
            item["category"] == "unclassified_requires_review" for item in failures
        ),
        "interpretation": "Classification is evidence, not automatic training authorization.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--historical-suite", action="store_true")
    parser.add_argument("--reference-code", type=Path)
    parser.add_argument(
        "--input-bindings",
        type=Path,
        help="Prospective complete input inventory for final authorization QA",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    initial_code = code_identity(root)
    input_receipt = None
    if args.input_bindings is not None:
        input_receipt = {
            "path": str(args.input_bindings.resolve()),
            "sha256": hashlib.sha256(args.input_bindings.read_bytes()).hexdigest(),
        }
    environment = {
        **os.environ,
        "PYTHONPATH": str(root / "src") + os.pathsep + str(root),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    if Path(sys.prefix).resolve().is_relative_to(root):
        environment["UV_PROJECT_ENVIRONMENT"] = sys.prefix
    if args.reference_code is not None:
        environment["PYTHONPATH"] += os.pathsep + str(args.reference_code.resolve())
    args.output_root.mkdir(parents=True, exist_ok=False)
    test_paths = sorted(
        str(path.relative_to(root)) for path in (root / "tests").glob("test_stage63*.py")
    )
    test_paths += [
        "tests/test_raw_event_binding.py",
        "tests/test_raw_time_residual.py",
        "tests/test_full_regret_router.py",
    ]
    type_config = json.loads((root / "pyright-stage63-65.json").read_text())
    commands = {
        "targeted_pytest": [
            sys.executable,
            "-m",
            "pytest",
            *test_paths,
            "-q",
            "--junitxml",
            str(args.output_root / "targeted.xml"),
            "--basetemp",
            str(args.output_root.resolve() / "targeted_tmp"),
        ],
        "ruff": [
            str(root / ".venv/Scripts/ruff.exe"),
            "check",
            *type_config["include"],
            *test_paths,
        ],
        "pyright": [
            str(root / ".venv/Scripts/pyright.exe"),
            "--project",
            "pyright-stage63-65.json",
        ],
        "ruff_format": [
            str(root / ".venv/Scripts/ruff.exe"),
            "format",
            "--check",
            *type_config["include"],
            *test_paths,
        ],
        "diff_check": ["git", "diff", "--check"],
        "pip_freeze": [sys.executable, "-m", "pip", "freeze"],
        "gpu_environment": ["nvidia-smi", "-q"],
        "powershell_ast": [
            "pwsh",
            "-NoProfile",
            "-Command",
            "$parseTokens = $null; $parseErrors = $null; "
            "[System.Management.Automation.Language.Parser]::ParseFile("
            "(Join-Path (Get-Location) 'scripts/RUN_STAGE63_65.ps1'), "
            "[ref]$parseTokens, [ref]$parseErrors) | Out-Null; "
            "if ($parseErrors.Count) { $parseErrors | ConvertTo-Json; exit 1 }; "
            "'PowerShell AST: zero parse errors'",
        ],
    }
    if args.historical_suite:
        commands["historical_suite"] = [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            "-q",
            "--tb=short",
            "--junitxml",
            str(args.output_root / "historical.xml"),
            "--basetemp",
            str(args.output_root.resolve() / "historical_tmp"),
        ]
    if args.reference_code is not None:
        commands["reference_kernels"] = [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            str(args.reference_code / "test_reference.py"),
            "-q",
            "-p",
            "no:cacheprovider",
            "--junitxml",
            str(args.output_root / "reference.xml"),
        ]
    if args.input_bindings is not None:
        bindings = json.loads(args.input_bindings.read_text(encoding="utf-8"))["input_bindings"]
        campaign_root = Path(bindings["stage63"]["path"]).parents[1]
        coherent_root = (
            Path(bindings["coherent_a5:outer0_final/manifest.json"]["path"]).parents[1]
            if "coherent_a5:outer0_final/manifest.json" in bindings
            else campaign_root / "unused_raw_state_for_cpu_fallback"
        )
        commands["real_input_contracts"] = [
            sys.executable,
            "-B",
            str(root / "scripts/verify_stage63_65_inputs.py"),
            "--output-root",
            str(campaign_root),
            "--coherent-a5-root",
            str(coherent_root),
        ]
    records = []
    for name, command in commands.items():
        began = time.time()
        print(f"QA started: {name}", flush=True)
        path = args.output_root / f"{name}.log"
        with path.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                command,
                cwd=root,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
                env=environment,
            )
        records.append(
            {
                "name": name,
                "command": command,
                "exit_code": result.returncode,
                "elapsed_seconds": time.time() - began,
                "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        (args.output_root / "QA_COMMANDS.json").write_text(
            json.dumps(
                {
                    "status": "engineering_evidence_not_training_authorization",
                    "code_before": initial_code,
                    "code_after": code_identity(root),
                    "python": sys.version,
                    "python_executable": sys.executable,
                    "input_receipt": input_receipt,
                    "commands": records,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"QA completed: {name}, exit={result.returncode}", flush=True)
    if args.historical_suite:
        classification = classify_historical(args.output_root / "historical.xml", root)
        (args.output_root / "HISTORICAL_SUITE_CLASSIFICATION.json").write_text(
            json.dumps(classification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
