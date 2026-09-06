"""Reconcile actual test outcomes and bind a pre-fit QA acceptance receipt."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import platform
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import psutil
import torch

from e_jepa_ttc.artifacts.risk_geometry_v10 import atomic_json, binding


def results(path: Path) -> dict:
    root = ET.parse(path).getroot()
    failures = {}
    cases = list(root.iter("testcase"))
    for case in cases:
        failure = case.find("failure")
        if failure is None:
            failure = case.find("error")
        if failure is not None:
            failures[case.attrib["classname"] + "::" + case.attrib["name"]] = failure.attrib.get(
                "message", ""
            )
    return dict(
        cases=len(cases),
        failures=failures,
        skipped=sum(c.find("skipped") is not None for c in cases),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    root = parser.parse_args().output_root
    baseline = results(root / "baseline_corrected.xml")
    integrated = results(root / "integrated_accepted.xml")
    current = results(root / "full_suite_new.xml")
    reference = results(root / "reference_tests.xml")
    new = sorted(set(current["failures"]) - set(baseline["failures"]))
    if (
        new
        or integrated["failures"]
        or integrated["skipped"]
        or reference["failures"]
        or integrated["cases"] < 100
    ):
        raise ValueError("QA has new/relevant failures or incomplete real smoke")
    if (
        "All checks passed" not in (root / "ruff_final.log").read_text()
        or "0 errors" not in (root / "pyright_final.log").read_text()
    ):
        raise ValueError("lint/type QA incomplete")
    paths = [
        root / p
        for p in (
            "baseline_corrected.xml",
            "integrated_accepted.xml",
            "full_suite_new.xml",
            "reference_tests.xml",
            "ruff_final.log",
            "pyright_final.log",
            "historical_posthoc/BOOTSTRAP_REPOSITORY_PARITY.json",
        )
    ]
    atomic_json(
        root / "QA_ACCEPTANCE.json",
        dict(
            passed=True,
            baseline=baseline,
            integrated=integrated,
            current=current,
            reference=reference,
            new_failure_ids=new,
            files=[binding(p) for p in paths],
            inherited_failure_scope=(
                "Unmodified historical fixture/config/release tests; exact "
                "failure IDs and reasons retained. No new module is waived."
            ),
        ),
    )
    atomic_json(
        root / "ENVIRONMENT.json",
        dict(
            executable=sys.executable,
            python=platform.python_version(),
            torch=torch.__version__,
            numpy=np.__version__,
            platform=platform.platform(),
            processor=platform.processor(),
            cpus=psutil.cpu_count(),
            available_ram=psutil.virtual_memory().available,
            cuda_runtime=torch.version.cuda,
            device="cpu",
            dtype="float64",
            torch_threads=1,
        ),
    )
    print(
        f"QA accepted: {integrated['cases']} integrated cases, "
        f"{reference['cases']} reference cases, {len(new)} new full-suite failures"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
