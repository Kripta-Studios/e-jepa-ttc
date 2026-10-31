"""Resume an admitted rare-sampling smoke, verify completion and compare against R2."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main() -> None:
    """Own a single training child, without retries or automatic extra seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "producer", "campaign", "baseline", "verifier", "root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume-run", type=Path)
    parser.add_argument("--output-name", default="r2_rare_seed7")
    args = parser.parse_args()
    os.chdir(args.project)
    sys.path.insert(0, str(args.producer / "src"))
    import torch

    from e_jepa_ttc.detection.campaign_training import load_campaign_config
    from e_jepa_ttc.detection.rare_sampling import load_schedule

    if Path(args.output_name).name != args.output_name or args.output_name in {".", ".."}:
        raise ValueError("Output name must be a single path component")
    owner = args.campaign / f"CONTROLLER_OWNER_{args.output_name}.json"
    with owner.open("x") as stream:
        json.dump({"pid": os.getpid(), "started_at": datetime.now(UTC).isoformat()}, stream)
    state: dict[str, Any] = {
        "status": "ADMISSION",
        "controller_pid": os.getpid(),
        "pending": [],
        "deferred_seeds": [13, 23],
    }

    def status(**updates: Any) -> None:
        state.update(updates, updated_at=datetime.now(UTC).isoformat())
        write(args.campaign / "STATUS.json", state)
        write(
            args.root / "ACTIVE_CAMPAIGN.json",
            state | {"status_path": str(args.campaign / "STATUS.json")},
        )

    try:
        config = load_campaign_config(args.campaign / "config.yaml")
        baseline_config = read(args.baseline)["runs"]["r2"]["config"]
        baseline_config.pop("sha256", None)
        actual_config = json.loads(json.dumps(asdict(config)))
        for value in (baseline_config, actual_config):
            value.pop("experiment_id")
            value["model"].pop("portable_root")
        if actual_config != baseline_config:
            raise ValueError("Candidate changes an undeclared R2 configuration field")
        manifest = read(Path(config.data.manifest))
        producer_receipt = read(args.producer / "RARE_SAMPLING_RECEIPT.json")
        sampler, sampling_receipt = load_schedule(
            args.producer / "sampling_plan.json",
            expected_sha256=producer_receipt["plan_sha256"],
            tokens=manifest["roles"]["train"]["sample_tokens"],
            seed=config.seed,
            updates=config.optimization.updates,
            batch_size=config.optimization.batch_size,
            manifest_sha256=config.data.expected_manifest_sha256,
        )
        smoke = args.resume_run or args.campaign / "smoke8"
        checkpoint = smoke / "checkpoints/last.pt"
        sha = digest(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        start_update = payload["update"]
        if args.resume_run is None:
            summary = read(smoke / "summary.json")
            if (
                summary["status"] != "BUDGET_STOPPED"
                or summary["final_update"] != 8
                or summary["config_sha256"] != config.sha256
                or summary["last_checkpoint_sha256"] != sha
                or start_update != 8
            ):
                raise ValueError("Smoke completion or identity failed")
        else:
            registered = read(smoke / "preregistration.json")
            if (
                payload["config_sha256"] != config.sha256
                or payload["runtime_provenance"] != registered["runtime_provenance"]
                or type(start_update) is not int
                or not 0 < start_update < config.optimization.updates
            ):
                raise ValueError(
                    "Interrupted checkpoint identity differs from its registered producer"
                )
            runtime = read(smoke / "runtime_preregistration.json")
            for relative, expected_sha in runtime["code_sha256"].items():
                source = (
                    smoke / "code" / relative
                    if relative.startswith("data/admission/")
                    else args.producer / relative
                )
                if digest(source) != expected_sha:
                    raise ValueError(f"Interrupted producer source differs: {relative}")
        sampler.load_state_dict(payload["sampler"])
        if sampler.cursor != start_update * config.optimization.batch_size:
            raise ValueError("Resume cursor differs from completed updates")
        rows = [json.loads(line) for line in (smoke / "metrics.jsonl").read_text().splitlines()]
        expected_tokens = manifest["roles"]["train"]["sample_tokens"]
        schedule = read(args.producer / "sampling_plan.json")["indices"]
        committed_rows = [row for row in rows if row["update"] <= start_update]
        registered_start = read(smoke / "preregistration.json")["start_update"]
        if [row["update"] for row in committed_rows] != list(
            range(registered_start + 1, start_update + 1)
        ):
            raise ValueError("Committed training ledger is incomplete")
        for row in committed_rows:
            update = row["update"]
            expected = [expected_tokens[i] for i in schedule[(update - 1) * 4 : update * 4]]
            if row["sample_tokens"] != expected:
                raise ValueError("Run did not execute the registered calendar")
        write(
            args.campaign / f"RESUME_ADMISSION_{args.output_name}.json",
            {
                "status": "PASS",
                "update": start_update,
                "last_logged_update": rows[-1]["update"],
                "discarded_uncheckpointed_updates": rows[-1]["update"] - start_update,
                "checkpoint_sha256": sha,
                "sampling": sampling_receipt,
                "strict_resume_next_update": start_update + 1,
                "next_tokens": [expected_tokens[i] for i in sampler.next_batch(4)],
            },
        )
        del payload
        output = args.campaign / args.output_name
        command = [
            sys.executable,
            str(args.producer / "scripts/train_detection_campaign.py"),
            "--config",
            str(args.campaign / "config.yaml"),
            "--output",
            str(output),
            "--resume",
            str(checkpoint),
        ]
        with (args.campaign / f"training_{args.output_name}.log").open("x") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            status(
                status="RUNNING",
                child_pid=child.pid,
                output=str(output),
                start_update=start_update,
                target_update=config.optimization.updates,
                sampling=sampling_receipt,
            )
            code = child.wait()
        if code:
            raise RuntimeError(f"Training process exited {code}; see training.log")
        status(status="VERIFYING", child_exit_code=code)
        spec = importlib.util.spec_from_file_location("frozen_verifier", args.verifier)
        assert spec is not None and spec.loader is not None
        helper: Any = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        completion = helper.verify_summary(
            output,
            config,
            expected_status="COMPLETED",
            expected_final_update=8192,
            expected_start_update=start_update,
            resume_sha256=sha,
        )
        compact = {}
        original = helper._load_object

        def capture(path: Path) -> dict[str, Any]:
            value = original(path)
            if "metrics_v2" in value:
                metrics = value["metrics_v2"]
                compact[str(path)] = {
                    "AP": metrics["dev_AP101_supported_classes"],
                    "sample_count": metrics["sample_count"],
                    "per_class": {
                        k: v["AP101_mean_thresholds"] for k, v in metrics["per_class"].items()
                    },
                }
            return value

        helper._load_object = capture
        evaluations = helper.verify_full_evaluations(output, completion["summary"])
        baseline_record = read(args.baseline)["runs"]["r2"]["full"]["8192"]
        previous = helper.verify_evaluation(
            Path(baseline_record["path"]),
            protocol_sha256=baseline_record["protocol_sha256"],
            expected_sample_count=baseline_record["sample_count"],
        )
        if (
            previous["sha256"] != baseline_record["sha256"]
            or previous["ground_truth_sha256_by_token"]
            != evaluations["8192"]["ground_truth_sha256_by_token"]
        ):
            raise ValueError("Baseline identity or comparison population differs")
        before = compact[str(baseline_record["path"])]
        after = compact[str(evaluations["8192"]["path"])]
        rare_classes = {"bicycle", "other_vehicle"}
        delta = {k: after["per_class"][k] - v for k, v in before["per_class"].items()}
        checks = {
            "macro_improves": after["AP"] > before["AP"],
            "target_average_improves": sum(delta[k] for k in rare_classes) > 0,
            "no_other_class_drops_over_001": all(
                v >= -0.01 for k, v in delta.items() if k not in rare_classes
            ),
        }
        comparison = {
            "schema": "phase_rare_sampling_comparison_v1",
            "baseline": before,
            "candidate": after,
            "class_deltas": delta,
            "acceptance_checks": checks,
            "screening_pass": all(checks.values()),
            "same_population": True,
            "completion": completion,
            "evaluations": {
                k: {field: v for field, v in row.items() if field != "ground_truth_sha256_by_token"}
                for k, row in evaluations.items()
            },
            "scope": "Single seed, development only, not official eAP or SOTA.",
        }
        write(args.campaign / "COMPARISON.json", comparison)
        status(
            status="COMPLETED_VERIFIED",
            screening_pass=comparison["screening_pass"],
            next_action="Review diagnostics; extra seeds remain deferred.",
        )
    except Exception as error:
        status(status="FAILED", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
