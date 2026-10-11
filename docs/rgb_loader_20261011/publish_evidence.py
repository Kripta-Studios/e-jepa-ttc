"""Export compact public evidence; stdlib only, no datasets or model imports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def relative_source(value: str) -> str:
    normalized = value.replace("\\", "/")
    for marker in ("e-jepa-ttc-v13-rgb-port/", "site-packages/", "/Lib/"):
        if marker in normalized:
            return normalized.split(marker, 1)[1]
    return Path(normalized).name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    root = args.evidence
    destination = Path(__file__).resolve().parent

    def read(name: str) -> Any:  # noqa: ANN401 -- heterogeneous JSON evidence schemas
        return json.loads((root / name).read_text(encoding="utf-8"))

    raw = read("cpu_abba/RESULT.json")
    trials = []
    for trial in raw["trials"]:
        trials.append(
            {
                key: value
                for key, value in trial.items()
                if key not in {"source_identity", "batch_details", "np_load_calls_by_path"}
            }
        )
        trials[-1]["source_identity_record_sha256"] = object_digest(trial["source_identity"])
        trials[-1]["batch_details_record_sha256"] = object_digest(trial["batch_details"])
        trials[-1]["batch_part_shapes"] = [
            [{"frames": part["frames"], "rows": part["rows"]} for part in item["parts"]]
            for item in trial["batch_details"]
        ]
    compact = {
        key: value
        for key, value in raw.items()
        if key not in {"host", "checkpoint", "indices", "trials"}
    }
    compact["trials"] = trials
    compact["raw_evidence_sha256"] = digest(root / "cpu_abba/RESULT.json")
    compact["source_omissions"] = "Absolute paths, host, row tokens and raw sample indices."

    guard_raw = read("GUARD_EXPERIMENT.json")
    guard = {
        "schema": guard_raw["schema"],
        "created_utc": guard_raw["created_utc"],
        "source_sha256": guard_raw["source_sha256"],
        "psutil_version": guard_raw["psutil_version"],
        "conditions": guard_raw["conditions"],
        "samples": [
            {
                "round": item["round"],
                "variant": item["variant"],
                "elapsed_s": item["elapsed_s"],
                "allowed": item["allowed"],
                "project_process_count": len(item["snapshot"]["project_processes"]),
                "project_tree_rss_bytes": item["snapshot"]["project_tree_rss_bytes"],
            }
            for item in guard_raw["samples"]
        ],
        "summary": {
            key: value for key, value in guard_raw["summary"].items() if key != "membership"
        },
        "raw_evidence_sha256": digest(root / "GUARD_EXPERIMENT.json"),
    }
    stacks_raw = read("BASELINE_STACK_SUMMARY.json")
    stacks = []
    for thread in stacks_raw:
        stacks.append(
            {
                "thread": "main" if "MainThread" in thread["name"] else "worker",
                "samples": thread["samples"],
            }
        )
        for section in ("top_inclusive", "top_leaf"):
            stacks[-1][section] = [
                {**frame, "file": relative_source(frame["file"])} for frame in thread[section]
            ]
    diagnostics = {
        "raw_stack_summary_sha256": digest(root / "BASELINE_STACK_SUMMARY.json"),
        "raw_stack_profile_sha256": digest(root / "BASELINE_STACKS.json"),
        "threads": stacks,
        "baseline_progress": [
            {
                "time": sample["time"],
                "updates": sample["updates"],
                "private_bytes": sample["private_bytes"],
                "rss_bytes": sample["rss_bytes"],
            }
            for sample in read("BASELINE_TELEMETRY.json")
        ],
        "raw_telemetry_sha256": digest(root / "BASELINE_TELEMETRY.json"),
        "inclusive_percentages_overlap": True,
        "sampling_profile_is_not_gpu_kernel_timing": True,
    }
    qa = read("QA_RECEIPT.json")
    qa["raw_receipt_sha256"] = digest(root / "QA_RECEIPT.json")
    suites = ElementTree.parse(root / "FINAL_QA_JUNIT.xml").getroot()
    qa["test_cases"] = [
        {
            "module": case.attrib["classname"],
            "name": case.attrib["name"],
            "seconds": float(case.attrib["time"]),
        }
        for case in suites.iter("testcase")
    ]
    for name, value in (
        ("CPU_ABBA.json", compact),
        ("GUARD_EXPERIMENT.json", guard),
        ("DIAGNOSTICS.json", diagnostics),
        ("QA.json", qa),
    ):
        (destination / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
