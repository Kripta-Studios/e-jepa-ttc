"""Prepare immutable QA inputs or accept complete final-code QA evidence; never train."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.stage63_65 import sign_stage63_65_artifact  # noqa: E402
from e_jepa_ttc.artifacts.training_authorization import read_signed  # noqa: E402
from scripts.qa_stage63_65_remediation import classify_historical, code_identity  # noqa: E402
from scripts.run_scientific_recovery_v9_stage63_65 import collect_lock_inputs  # noqa: E402


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(args: argparse.Namespace) -> dict[str, object]:
    """Require clean unchanged code, exact inputs and reviewed historical failure evidence."""
    code = code_identity(ROOT)
    if code["dirty"]:
        raise ValueError("final QA must run against the clean final training commit")
    bindings = collect_lock_inputs(args, ROOT)
    qa_root = args.output_root / "qa"
    qa_root.mkdir(parents=True, exist_ok=True)
    input_path = qa_root / "QA_INPUT_BINDINGS.json"
    if args.phase == "prepare":
        result = sign_stage63_65_artifact(
            {
                "artifact_type": "stage63_65_qa_inputs_v2",
                "training_commit": code["commit"],
                "input_bindings": bindings,
            },
            evidence_type="prospective_qa_inputs",
        )
        destination = input_path
    else:
        if args.qa_run is None:
            raise ValueError("accept requires --qa-run with complete captured command evidence")
        inputs = read_signed(input_path)
        if inputs["training_commit"] != code["commit"] or inputs["input_bindings"] != bindings:
            raise ValueError("QA inputs changed since preparation")
        commands_path = args.qa_run / "QA_COMMANDS.json"
        captured = json.loads(commands_path.read_text(encoding="utf-8"))
        if captured["code_before"] != code or captured["code_after"] != code:
            raise ValueError("QA was not executed against the unchanged final code")
        if captured["python"] != sys.version or captured["python_executable"] != sys.executable:
            raise ValueError("QA environment differs from the accepting runtime")
        if captured["input_receipt"] != {
            "path": str(input_path.resolve()),
            "sha256": sha(input_path),
        }:
            raise ValueError("QA commands did not bind the prospective input inventory")
        commands = {item["name"]: item for item in captured["commands"]}
        required = {
            "targeted_pytest",
            "ruff",
            "pyright",
            "powershell_ast",
            "historical_suite",
            "reference_kernels",
            "ruff_format",
            "diff_check",
            "pip_freeze",
            "gpu_environment",
            "real_input_contracts",
        }
        if set(commands) != required or len(commands) != len(captured["commands"]):
            raise ValueError("final QA command inventory is incomplete")
        evidence = {
            "commands": {"path": str(commands_path.resolve()), "sha256": sha(commands_path)}
        }
        for name, item in commands.items():
            log = args.qa_run / f"{name}.log"
            if sha(log) != item["log_sha256"]:
                raise ValueError(f"QA log identity mismatch: {name}")
            if name != "historical_suite" and item["exit_code"] != 0:
                raise ValueError(f"required QA command failed: {name}")
            evidence[name] = {"path": str(log.resolve()), "sha256": sha(log)}
        classification_path = args.qa_run / "HISTORICAL_SUITE_CLASSIFICATION.json"
        classification = json.loads(classification_path.read_text(encoding="utf-8"))
        if classification != classify_historical(args.qa_run / "historical.xml", ROOT):
            raise ValueError("historical failure classification does not reproduce its evidence")
        if classification["unclassified_count"] or any(
            item["source_and_subject_unchanged_from_pre_remediation_head"] is not True
            for item in classification["failures"]
        ):
            raise ValueError("historical failures are not fully classified and source-checked")
        historical = ET.parse(args.qa_run / "historical.xml")
        actual_failures = sum(
            t.find("failure") is not None or t.find("error") is not None
            for t in historical.iter("testcase")
        )
        if actual_failures != len(classification["failures"]):
            raise ValueError("historical classification omits failed tests")
        if args.baseline_qa is None:
            raise ValueError("final acceptance requires the executed pre-remediation baseline")
        baseline_record = json.loads(
            (args.baseline_qa / "BASELINE_QA.json").read_text(encoding="utf-8")
        )
        if (
            baseline_record["python"] != sys.version
            or baseline_record["python_executable"] != sys.executable
            or baseline_record["source_commit"] != "a9c2d6e8689e24c4eed4acac7aa9b6eede7f8a84"
            or sha(args.baseline_qa / "baseline.log") != baseline_record["log_sha256"]
        ):
            raise ValueError("historical baseline environment/source/log identity mismatch")
        baseline = ET.parse(args.baseline_qa / "baseline.xml")
        baseline_failures = {
            t.attrib["classname"] + "::" + t.attrib["name"]
            for t in baseline.iter("testcase")
            if t.find("failure") is not None or t.find("error") is not None
        }
        if not {item["test"] for item in classification["failures"]} <= baseline_failures:
            raise ValueError(
                "current historical suite has failures absent from the executed baseline"
            )
        for name in ("BASELINE_QA.json", "baseline.xml", "baseline.log"):
            path = args.baseline_qa / name
            evidence["baseline:" + name] = {"path": str(path.resolve()), "sha256": sha(path)}
        targeted = ET.parse(args.qa_run / "targeted.xml")
        cuda = [
            t
            for t in targeted.iter("testcase")
            if t.attrib["name"]
            == "test_real_cuda_model_resume_matches_full_optimizer_rng_and_schedule"
        ]
        if len(cuda) != 1 or list(cuda[0]):
            raise ValueError("real-model CUDA resume QA must execute and pass, not skip")
        for name in (
            "historical.xml",
            "targeted.xml",
            "reference.xml",
            "HISTORICAL_SUITE_CLASSIFICATION.json",
        ):
            path = args.qa_run / name
            evidence[name] = {"path": str(path.resolve()), "sha256": sha(path)}
        result = sign_stage63_65_artifact(
            {
                "artifact_type": "stage63_65_final_qa_manifest_v2",
                "status": "passed",
                "training_commit": code["commit"],
                "input_bindings": bindings,
                "historical_suite_all_passed": actual_failures == 0,
                "historical_failures_classified_not_waived": classification["failures"],
                "evidence": evidence,
                "authorization_scope": "corrected_stage63_65_only_not_historical_campaigns",
            },
            evidence_type="final_code_and_input_qa",
        )
        destination = qa_root / "QA_MANIFEST.json"
    if destination.exists():
        raise FileExistsError(f"preserve the existing QA receipt: {destination}")
    destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "accept"))
    for name in (
        "output-root",
        "reference-root",
        "stage61-worktree",
        "handoff-root",
        "frozen-teacher-audit",
        "coherent-a5-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--qa-run", type=Path)
    parser.add_argument("--baseline-qa", type=Path)
    print(json.dumps({"status": run(parser.parse_args()).get("status", "prepared")}))


if __name__ == "__main__":
    main()
