"""Produce a release gate from test XML, static logs and bound artifact checks."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt
from operational.phase_campaign.data import read
from operational.phase_robust.bundle import verified_upstream_paths
from operational.phase_startup.runtime import load_bundle_router


def test_result(path: Path) -> dict:
    """Read actual testcase elements separately from pytest subtest accounting."""
    tree = ET.parse(path)
    cases = list(tree.iter("testcase"))
    if not cases or any(
        c.find("failure") is not None or c.find("error") is not None for c in cases
    ):
        raise ValueError(f"Missing or failed test cases: {path}")
    skipped = sum(c.find("skipped") is not None for c in cases)
    suites = list(tree.iter("testsuite"))
    if any(int(s.attrib.get("failures", 0)) or int(s.attrib.get("errors", 0)) for s in suites):
        raise ValueError(f"Suite-level failures or errors: {path}")
    return {
        "passed": len(cases) - skipped,
        "skipped": skipped,
        "xml_reported_tests": sum(int(s.attrib["tests"]) for s in suites),
        "seconds": sum(float(s.attrib["time"]) for s in suites),
        "sha256": file_sha(path),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root
    full = test_result(root.parent / "phase_startup_full.xml")
    focused = test_result(root.parent / "phase_startup_focused_final.xml")
    if focused["passed"] < 11:
        raise ValueError("Expected all final focused regressions")
    for name, marker in (("ruff", "All checks passed!"), ("pyright", "0 errors, 0 warnings")):
        if marker not in (root.parent / f"phase_startup_{name}_final.log").read_text(
            encoding="utf-8"
        ):
            raise ValueError(f"Static verification incomplete: {name}")
    bundle = root / "FINAL_MODEL"
    verified_upstream_paths(bundle / "upstream")
    for variant in ("evaluated", "all_data_refit"):
        load_bundle_router(bundle, variant)
    binding_sha = file_sha(bundle / "ROUTER_BINDING.json")
    for name in ("NATIVE_REPLAY_FINAL.json", "SLIDER_STARTUP_REPLAY_FINAL.json"):
        receipt = read(root / name)
        if receipt["binding_sha256"] != binding_sha or receipt["labels_read"]:
            raise ValueError("Fresh target-blind replay does not bind delivered implementation")
    fit_paths = [
        path
        for name in ("training", "training_rate", "all_data_refit")
        for path in (root / name).rglob("COMPLETE.json")
    ]
    times = [read(path)["seconds"] for path in fit_paths]
    cache = root / "training_rate/POPULATION.npz"
    with np.load(cache) as arrays:
        array_bytes = sum(arrays[k].nbytes for k in arrays.files)
    performance = {
        "research_fits": len(times),
        "summed_optimizer_seconds": float(np.sum(times)),
        "optimizer_seconds_median": float(np.median(times)),
        "optimizer_seconds_p95": float(np.quantile(times, 0.95)),
        "cache_compressed_bytes": cache.stat().st_size,
        "uncompressed_array_bytes": array_bytes,
        "scope": (
            "optimizer timers exclude setup/I/O/evaluation; "
            "cached CPU heads, not end-to-end latency"
        ),
    }
    atomic_receipt(
        root / "FINAL_CHECKS.json",
        {
            "status": "passed",
            "full_suite": full,
            "focused_final": focused,
            "suite_scope": (
                "full suite plus final focused regressions after interval/report review corrections"
            ),
            "ruff": "passed",
            "pyright": "0errors0warnings",
            "review": {"critical": 0, "important": 0, "minor": 2, "both_minor_fixed": True},
            "bound_upstream_verified": True,
            "bound_variants_verified": True,
            "binding_sha256": binding_sha,
            "performance": performance,
            "export": read(root / "report/EXPORT.json"),
            "reproduction": read(root / "verification/REPRODUCIBILITY.json"),
            "artifact_hashes": {
                name: file_sha(root / name)
                for name in (
                    "report/RESULTS.json",
                    "report/REPORT.md",
                    "report/metrics.csv",
                    "AUDIT.json",
                    "training/FINAL_FREEZE.json",
                    "training_rate/FINAL_FREEZE.json",
                    "all_data_refit/FINAL_FREEZE.json",
                    "validation_v2/VALIDATION.json",
                    "raw_robustness/REPORT.json",
                    "NATIVE_REPLAY_FINAL.json",
                    "SLIDER_STARTUP_REPLAY_FINAL.json",
                )
            },
        },
    )
    print(
        f"COMPLETE verification: {full['passed']} full passed; {focused['passed']} final focused",
        flush=True,
    )


if __name__ == "__main__":
    main()
