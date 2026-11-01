"""Recover complete interrupted capture fragments, then finish the registered studies."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from operational.final_refit.test12 import FinalV16Train40Predictor
from operational.release_research.protocol import ResearchRun, sha256, write_json
from scripts import continue_native_platform_diagnostic as coordinator
from scripts import release_native_pair_capture as capture_module
from scripts.release_research import read

PLAN = Path("docs/release_research_implementation_20261031/NATIVE_INTERRUPTION_RECOVERY.md")
ROOT = Path("output/release_research_20261031")


def inventory(prior: Path, destination: Path) -> dict:
    """Seal surviving fragment bytes without pretending a former COMPLETE existed."""
    if (
        destination.exists()
        or (prior / "COMPLETE.json").exists()
        or (prior / "FAILED.json").exists()
    ):
        raise ValueError("Recovery requires interrupted inputs and a fresh inventory")
    freeze = read(prior / "FREEZE.json")
    for name, path in freeze["input_paths"].items():
        if sha256(Path(path)) != freeze["inputs"][name]:
            raise ValueError("Interrupted capture source/input changed: " + name)
    data = read(Path(freeze["input_paths"]["public_inputs"]))
    groups, records, incomplete = defaultdict(list), {}, []
    physical = set(data["physical_aliases"].values())
    queries = [q for q in data["queries"] if q["sample_token"] in physical]
    for modality in ("event", "rgb"):
        sequences = {}
        for query in queries:
            for identity in query[modality + "_history"]:
                if identity in sequences and sequences[identity] != query["sequence_id"]:
                    raise ValueError("Observation shared across different sequences")
                sequences[identity] = query["sequence_id"]
        identities = list(dict.fromkeys(i for q in queries for i in q[modality + "_history"]))
        for identity in identities:
            key = hashlib.sha256((modality + ":" + identity).encode()).hexdigest()[:24]
            folder = prior / "observations" / key
            if not folder.exists():
                continue
            calls, arrays = folder / "CALLS.json", folder / "PAIR_INPUTS.npz"
            if not calls.exists() or not arrays.exists():
                incomplete.append(str(folder))
                continue
            item = data[modality + "_observations"][identity]
            try:
                metadata = read(calls)
            except (json.JSONDecodeError, UnicodeError):
                incomplete.append(str(folder))
                continue
            if metadata["source"] != item or len(metadata["calls"]) != 1:
                raise ValueError("Surviving fragment metadata differs")
            call = metadata["calls"][0]
            if (
                call["status"] != "accepted"
                or call["prefix"] != {"event": "E", "rgb": "R"}[modality]
                or call["sensor_shape"][0] != 1
                or call["observations"][0]["observation_id"] != identity
            ):
                raise ValueError("Surviving fragment identity differs")
            pair = "PAIR_E_MATCHED" if modality == "event" else "PAIR_R"
            expected = {
                "call0.delta",
                "call0.raw17",
                "call0.expert3",
                f"call0.{pair}.input133",
                f"call0.{pair}.raw_phase",
            }
            with np.load(arrays, allow_pickle=False) as archive:
                if set(archive.files) != expected or any(
                    not np.isfinite(archive[k]).all() for k in expected
                ):
                    raise ValueError("Incomplete/nonfinite fragment arrays")
            token = modality + ":" + identity
            records[token] = {
                "folder": str(folder),
                "calls_sha256": sha256(calls),
                "pair_sha256": sha256(arrays),
            }
            groups[(modality, sequences[identity])].append(token)
    sentinels = sorted({token for group in groups.values() for token in (group[0], group[-1])})
    result = {
        "prior": str(prior),
        "prior_freeze_sha256": sha256(prior / "FREEZE.json"),
        "records": records,
        "sentinels": sentinels,
        "incomplete_recompute": incomplete,
        "previous_complete_existed": False,
        "hashes_first_sealed_at_recovery": True,
    }
    write_json(destination, result)
    return result


def recovered_capture(prior: Path, inventory_path: Path, output: Path) -> None:
    """Replay fixed sentinels first and then resume the unchanged capture loop."""
    frozen = read(prior / "FREEZE.json")
    receipt = read(inventory_path)
    if receipt["prior_freeze_sha256"] != sha256(prior / "FREEZE.json"):
        raise ValueError("Prior freeze changed")
    for name, path in frozen["input_paths"].items():
        if sha256(Path(path)) != frozen["inputs"][name]:
            raise ValueError("Original source/input changed: " + name)
    # The original capture creates the new ResearchRun and performs every usual admission.
    original_run = capture_module.ResearchRun
    extra = {
        "recovery_inventory": inventory_path,
        "interrupted_freeze": prior / "FREEZE.json",
        "recovery_protocol": PLAN,
        "source:scripts/resume_release_native_research.py": Path(__file__),
    }
    for key, item in receipt["records"].items():
        folder = Path(item["folder"])
        for name, field in (("CALLS.json", "calls_sha256"), ("PAIR_INPUTS.npz", "pair_sha256")):
            path = folder / name
            if sha256(path) != item[field]:
                raise ValueError("Recovery fragment changed")
            extra["recovered:" + key + ":" + name] = path
    checked = []
    active = {}
    original_compact = capture_module.capture_compact

    @contextmanager
    def compact(models: dict) -> Iterator[tuple[list, dict]]:
        with original_compact(models) as (calls, arrays):
            active.update(calls=calls, arrays=arrays, models=models)
            yield calls, arrays

    originals = {
        m: getattr(FinalV16Train40Predictor, "_" + m + "_observation") for m in ("event", "rgb")
    }

    def load(token: str) -> tuple[list, dict]:
        folder = Path(receipt["records"][token]["folder"])
        calls = read(folder / "CALLS.json")["calls"]
        with np.load(folder / "PAIR_INPUTS.npz", allow_pickle=False) as archive:
            arrays = {k: archive[k].copy() for k in archive.files}
        return calls, arrays

    admitted = False

    def method(modality: str) -> Callable:
        def observation(self: FinalV16Train40Predictor, identity: str) -> tuple:
            nonlocal admitted
            if not admitted:
                # Before any reuse: all first/last sentinels from all recovered groups.
                for token in receipt["sentinels"]:
                    mod, source_identity = token.split(":", 1)
                    _, expected = load(token)
                    with original_compact(self.models) as (_, arrays):
                        originals[mod](self, source_identity)
                    if set(arrays) != set(expected):
                        raise ValueError("Recovery sentinel field inventory differs")
                    for key in arrays:
                        np.testing.assert_array_equal(arrays[key], expected[key])
                    checked.append(token)
                # Remove sentinel cache so new observation capture retains its original semantics.
                self.event_values.clear()
                self.rgb_values.clear()
                active["calls"].clear()
                active["arrays"].clear()
                admitted = True
                write_json(output / "RECOVERY_SENTINELS.json", {"exact": checked})
                print(f"Recovery sentinels exact: {len(checked)}", flush=True)
            token = modality + ":" + identity
            if token not in receipt["records"]:
                return originals[modality](self, identity)
            calls, arrays = load(token)
            active["calls"].extend(copy.deepcopy(calls))
            active["arrays"].update(arrays)
            return arrays["call0.raw17"][0], arrays["call0.expert3"][0]

        return observation

    def run_factory(destination: Path, inputs: dict, config: dict) -> ResearchRun:
        return original_run(
            destination,
            {**inputs, **extra},
            {
                **config,
                "recovered_fragments": len(receipt["records"]),
                "recovery_times_are_not_inference": True,
            },
        )

    with (
        patch.object(capture_module, "ResearchRun", run_factory),
        patch.object(capture_module, "capture_compact", compact),
        patch.object(FinalV16Train40Predictor, "_event_observation", method("event")),
        patch.object(FinalV16Train40Predictor, "_rgb_observation", method("rgb")),
    ):
        capture_module.capture(
            Path(frozen["input_paths"]["public_inputs"]),
            Path(frozen["input_paths"]["media_complete"]).parent,
            Path(frozen["input_paths"]["upstream_bundle"]).parent,
            output,
            "cuda",
            None,
            Path(frozen["input_paths"]["population_report"]).parent,
        )


def run_all(prior: Path, output: Path, inventory_path: Path) -> None:
    """Run recovery, both complete replays and both latency boundaries sequentially."""
    inventory(prior, inventory_path)
    original_subprocess = subprocess.run
    original_run = coordinator.ResearchRun

    def replace_capture(command: list, **kwargs: Any) -> subprocess.CompletedProcess:  # noqa: ANN401 -- forward subprocess.run options unchanged
        if "scripts/release_native_pair_capture.py" in command:
            destination = command[command.index("--output") + 1]
            command = [
                sys.executable,
                "-u",
                str(Path(__file__)),
                "capture",
                "--prior",
                str(prior),
                "--inventory",
                str(inventory_path),
                "--output",
                destination,
            ]
        return original_subprocess(command, **kwargs)

    def run_factory(destination: Path, inputs: dict, config: dict) -> ResearchRun:
        return original_run(
            destination,
            {
                **inputs,
                "recovery_inventory": inventory_path,
                "recovery_protocol": PLAN,
                "source:scripts/resume_release_native_research.py": Path(__file__),
            },
            config,
        )

    with (
        patch.object(coordinator, "ResearchRun", run_factory),
        patch.object(coordinator.subprocess, "run", replace_capture),
    ):
        coordinator.continue_work(
            ROOT / "continuation_v1", ROOT / "native_backend_diagnosis_v1", output
        )
    finish = output.parent / "final_reporting_v2"
    reporting = ResearchRun(
        finish,
        {
            "native_freeze": output / "FREEZE.json",
            "native_complete": output / "COMPLETE.json",
            "source:scripts/resume_release_native_research.py": Path(__file__),
        },
        {"experiment_id": "recovered-research-reporting-v1", "targets_read": False},
    )
    report = Path("docs/release_research_implementation_20261031/campaign_results_v1")

    def execute(stage: str, arguments: list[str]) -> None:
        write_json(finish / "STATUS.json", {"stage": stage})
        with (finish / (stage + ".log")).open("w", encoding="utf-8") as log:
            original_subprocess(
                [sys.executable, "-u", *arguments], stdout=log, stderr=subprocess.STDOUT, check=True
            )

    try:
        execute(
            "latency",
            [
                "scripts/release_native_local_latency.py",
                "--parent",
                str(output),
                "--output",
                str(finish / "latency"),
            ],
        )
        execute(
            "report",
            [
                "scripts/report_release_research_campaign.py",
                "--native",
                str(output),
                "--latency",
                str(finish / "latency"),
                "--output",
                str(report),
            ],
        )
        write_json(finish / "STATUS.json", {"stage": "finished"})
        reporting.finish()
    except Exception as error:
        reporting.fail(error)
        raise
    public = read(output / "capture/FREEZE.json")["input_paths"]["public_inputs"]
    raw = ROOT / "native_raw_latency_v1"
    original_subprocess(
        [
            sys.executable,
            "-u",
            "scripts/release_native_raw_latency.py",
            "benchmark",
            "--public",
            public,
            "--parity",
            str(ROOT / "native_raw_parity_v1"),
            "--completed",
            str(finish),
            "--output",
            str(raw),
        ],
        check=True,
    )
    original_subprocess(
        [
            sys.executable,
            "-u",
            "scripts/report_release_research_raw_latency.py",
            "--campaign",
            str(report),
            "--parity",
            str(ROOT / "native_raw_parity_v1"),
            "--latency",
            str(raw),
            "--output",
            "docs/release_research_implementation_20261031/raw_latency_results_v1",
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("all", "capture"))
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ.update(
        PYTHONUTF8="1",
        PYTHONPATH="src;.",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    if args.mode == "all":
        run_all(args.prior, args.output, args.inventory)
    else:
        recovered_capture(args.prior, args.inventory, args.output)


if __name__ == "__main__":
    main()
