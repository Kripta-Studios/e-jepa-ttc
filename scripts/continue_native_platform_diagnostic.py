"""Execute the preregistered local-platform diagnostic after retaining historical parity failure."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from operational.release_research.protocol import ResearchRun, sha256, write_json
from scripts.audit_phase_native_source_closure import DEPENDENCIES
from scripts.release_native_platform_pair import PLAN
from scripts.release_research import read, sources, verify_complete
from scripts.report_phase_canonical_population import verified_inputs


def validate_profile(parity: dict) -> None:
    """Require both fixed profile outputs, retaining rather than excusing internal mismatches."""
    if len(parity) != 2:
        raise ValueError("Require both fixed profile queries")
    for row in parity.values():
        fields = [v for k, v in row["fields"].items() if k.endswith("output_ttc")]
        if not fields or not all(v["passed"] for v in fields):
            raise ValueError("Original profile TTC exceeds its existing tolerance")


def admit(continuation: Path, backend: Path) -> Path:
    """Admit a distinct scoped experiment without changing or approving the historical result."""
    for folder in (continuation, backend):
        verify_complete(folder)
    if (
        read(continuation / "DECISION.json")["outcome"]
        != "native_cuda_profile_parity_failed_candidate_not_run"
        or read(backend / "RESULTS.json")["local_original_repeatability"] != "EXACT"
        or read(backend / "FREEZE.json")["inputs"]["continuation"]
        != sha256(continuation / "COMPLETE.json")
    ):
        raise ValueError("Require completed failed historical route and exact local repeatability")
    parity = read(continuation / "native_profile_parity/ORIGINAL_PARITY.json")
    validate_profile(parity)
    _, population = verified_inputs(
        Path("output/phase_timewarp/population2608_v1"), continuation / "population_evaluation"
    )
    if not population["routes"]["F0"]["eligible_for_eap_validation"]:
        raise ValueError("Population F0 did not qualify")
    return Path(
        read(continuation / "native_profile_cuda/FREEZE.json")["input_paths"]["public_inputs"]
    )


def continue_work(continuation: Path, backend: Path, output: Path) -> None:
    public = admit(continuation, backend)
    files = {
        **sources(),
        "preregistration": PLAN,
        "continuation": continuation / "COMPLETE.json",
        "backend": backend / "COMPLETE.json",
        "public_inputs": public,
    }
    for name in (
        *DEPENDENCIES,
        "scripts/continue_native_platform_diagnostic.py",
        "scripts/release_native_platform_pair.py",
        "scripts/release_native_backend_diagnosis.py",
        "scripts/release_native_pair_capture.py",
        "scripts/release_native_pair_replay.py",
        "scripts/final_train40_transfer.py",
        "scripts/report_phase_canonical_population.py",
        "scripts/runpod_test12_staged.py",
        "operational/phase_timewarp/compact_capture.py",
        "operational/phase_timewarp/capture.py",
        "operational/phase_timewarp/canonical_pair.py",
        "operational/phase_timewarp/screen.py",
        "operational/phase_observation_cache/replay.py",
        "operational/phase_robust/bundle.py",
    ):
        files["source:" + name] = Path(name)
    run = ResearchRun(
        output,
        files,
        {
            "experiment_id": "native-platform-diagnostic-continuation-v1",
            "scope": "local sensitivity; historical failure retained; no accuracy/promotion claim",
            "targets_read": False,
            "optimizer_updates": 0,
            "seed": 7,
        },
    )
    environment = {
        **os.environ,
        "PYTHONUTF8": "1",
        "PYTHONPATH": "src;.",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    commands = []

    def execute(name: str, arguments: list[str]) -> None:
        for key, path in files.items():
            if sha256(path) != run.hashes[key]:
                raise ValueError("Frozen diagnostic source/input changed: " + key)
        write_json(output / "STATUS.json", {"stage": name})
        command = [sys.executable, "-u", *arguments]
        commands.append(command)
        write_json(output / "COMMANDS.json", commands)
        print(name, flush=True)
        with (output / (name + ".log")).open("w", encoding="utf-8") as log:
            subprocess.run(
                command, stdout=log, stderr=subprocess.STDOUT, env=environment, check=True
            )

    try:
        capture, original, paired = (
            output / name for name in ("capture", "original_replay", "paired_replay")
        )
        execute(
            "capture",
            [
                "scripts/release_native_pair_capture.py",
                "--inputs",
                str(public),
                "--device",
                "cuda",
                "--population-evaluation",
                str(continuation / "population_evaluation"),
                "--output",
                str(capture),
            ],
        )
        execute(
            "original_replay",
            [
                "scripts/release_native_pair_replay.py",
                "--capture",
                str(capture),
                "--output",
                str(original),
            ],
        )
        execute(
            "paired_replay",
            [
                "scripts/release_native_platform_pair.py",
                "--capture",
                str(capture),
                "--original-replay",
                str(original),
                "--output",
                str(paired),
            ],
        )
        result = read(paired / "RESULTS.json")
        write_json(
            output / "DECISION.json",
            {
                "outcome": "local_sensitivity_complete",
                "queries": result["queries"],
                "historical_gate_reclassified": False,
                "native_accuracy_evaluated": False,
                "promotion_admitted": False,
                "release_modified": False,
            },
        )
        write_json(output / "STATUS.json", {"stage": "finished"})
        run.finish()
    except Exception as error:
        write_json(output / "STATUS.json", {"stage": "failed", "error": repr(error)})
        run.fail(error)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path("output/release_research_20261031")
    parser.add_argument("--continuation", type=Path, default=root / "continuation_v1")
    parser.add_argument("--backend", type=Path, default=root / "native_backend_diagnosis_v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    continue_work(args.continuation, args.backend, args.output)


if __name__ == "__main__":
    main()
