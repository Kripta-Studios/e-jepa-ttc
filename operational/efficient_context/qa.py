"""Persist actual focused QA exit codes before admitting scientific updates."""

import argparse
import os
import subprocess
import sys

from .common import ROOT, Campaign, atomic_bytes, atomic_json


def main() -> int:
    """Run optimizer-free contract tests, static checks and useful CLI help."""
    c = Campaign(ROOT / "configs/campaign/efficient_context_v1.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="final")
    args = parser.parse_args()
    destination = c.out / "TEST_RESULTS" / args.stage
    commands = {
        "pytest": ["-m", "pytest", "tests/unit/test_efficient_context_contracts.py", "-q"],
        "ruff": [
            "-m",
            "ruff",
            "check",
            "src/e_jepa_ttc/efficient_context",
            "operational/efficient_context",
            "tests/unit/test_efficient_context_contracts.py",
        ],
        "format": [
            "-m",
            "ruff",
            "format",
            "--check",
            "src/e_jepa_ttc/efficient_context",
            "operational/efficient_context",
            "tests/unit/test_efficient_context_contracts.py",
        ],
        "pyright": [
            "-m",
            "pyright",
            "--project",
            "operational/efficient_context/pyrightconfig.json",
        ],
        "help": ["-m", "operational.efficient_context.run", "--help"],
        "queue_help": ["-m", "operational.efficient_context.queue", "--help"],
    }
    env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=str(ROOT / "src"))
    results = {}
    for name, args in commands.items():
        run = subprocess.run(
            [sys.executable, *args],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        atomic_bytes(destination / f"{name}.txt", run.stdout)
        results[name] = {"returncode": run.returncode, "command": [sys.executable, *args]}
        print(name, run.returncode, flush=True)
    passed = all(v["returncode"] == 0 for v in results.values())
    atomic_json(
        destination / "QA.json",
        {"status": "PASSED" if passed else "FAILED", "results": results, "optimizer_updates": 0},
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
