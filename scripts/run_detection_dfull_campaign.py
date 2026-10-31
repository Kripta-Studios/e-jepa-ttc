"""Own one bounded Dfull run, verify it, and save its selected full-evaluation result."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path: Path, value: object) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "producer", "campaign", "verifier", "root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--batch-transition", type=Path)
    args = parser.parse_args()
    os.chdir(args.project)
    sys.path.insert(0, str(args.producer / "src"))
    import torch

    from e_jepa_ttc.detection.campaign_training import (
        ResumablePermutationSampler,
        load_campaign_config,
    )

    transition = read(args.batch_transition) if args.batch_transition else None
    plan = transition["plan"] if transition else None
    owner_name = "CONTROLLER_OWNER_BATCH8.json" if transition else "CONTROLLER_OWNER.json"
    config_name = "config_batch8.yaml" if transition else "config_v2.yaml"
    output_name = "r2_dfull_seed7_batch8_v1" if transition else "r2_dfull_seed7_v2"
    log_name = "training_batch8.log" if transition else "training_v2.log"
    start_update = plan["source_update"] if plan else 0
    with (args.campaign / owner_name).open("x") as stream:
        json.dump({"pid": os.getpid(), "started_at": datetime.now(UTC).isoformat()}, stream)
    state: dict[str, Any] = {"status": "ADMISSION", "controller_pid": os.getpid(), "pending": []}

    def status(**updates: Any) -> None:
        state.update(updates, updated_at=datetime.now(UTC).isoformat())
        write(args.campaign / "STATUS.json", state)
        write(
            args.root / "ACTIVE_CAMPAIGN.json",
            state | {"status_path": str(args.campaign / "STATUS.json")},
        )

    try:
        config = load_campaign_config(args.campaign / config_name)
        tokens = read(Path(config.data.manifest))["roles"]["train"]["sample_tokens"]
        exposures = (
            plan["total_exposures"]
            if plan
            else (config.optimization.updates * config.optimization.batch_size)
        )
        if exposures != 4 * len(tokens):
            raise ValueError("Dfull budget must be exactly four complete uniform passes")
        first_full_pass = (
            len(tokens) + config.optimization.batch_size - 1
        ) // config.optimization.batch_size
        if plan:
            assert transition is not None
            first_full_pass = plan["selection_update_map"][str((len(tokens) + 3) // 4)]
            if (
                digest(Path(transition["source_checkpoint"]))
                != transition["source_checkpoint_sha256"]
            ):
                raise ValueError("Transition source checkpoint differs")
            from e_jepa_ttc.detection.batch_transition import batch_size_at
        if min(config.evaluation.updates) < first_full_pass:
            raise ValueError(
                "Selected release candidates must have consumed the full train population"
            )
        parity = read(args.campaign / "REAL_SAMPLE_CACHE_PREFETCH_PARITY.json")
        if parity["status"] != "PASS" or not parity["cpu_and_cuda_rng_unchanged"]:
            raise ValueError("Prefetched compact input parity has not passed")
        profile = read(args.campaign / "smoke64/summary.json")
        if profile["status"] != "BUDGET_STOPPED" or profile["final_update"] != 64:
            raise ValueError("Prior full-model memory/initialization profile did not finish")
        if (
            profile["initialization"]["checkpoint_sha256"]
            != read(args.producer / "WARM_START.json")["checkpoint_sha256"]
        ):
            raise ValueError("Profile warm-start identity differs")
        output = args.campaign / output_name
        command = [
            sys.executable,
            str(args.producer / "scripts/train_detection_campaign.py"),
            "--config",
            str(args.campaign / config_name),
            "--output",
            str(output),
        ]
        if transition:
            command += ["--resume", transition["source_checkpoint"]]
        with (args.campaign / log_name).open("x") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            status(
                status="RUNNING",
                child_pid=child.pid,
                output=str(output),
                target_update=config.optimization.updates,
                unique_train_frames=len(tokens),
                exposures=4 * len(tokens),
                effective_passes=4,
                batch_size=config.optimization.batch_size,
                start_update=start_update,
            )
            code = child.wait()
        if code:
            raise RuntimeError(f"Dfull trainer exited {code}; see training_v2.log")
        status(status="VERIFYING", child_exit_code=code)
        summary = read(output / "summary.json")
        if summary["status"] != "COMPLETED":
            status(status="BUDGET_STOPPED", stop_reason=summary.get("stop_reason"))
            return
        sampler = ResumablePermutationSampler(len(tokens), config.seed)
        observed = set()
        updates = 0
        ledgers: list[tuple[Path, int | None]] = [(output / "metrics.jsonl", None)]
        if transition:
            ledgers.insert(0, (Path(transition["source_run"]) / "metrics.jsonl", start_update))
        consumed_exposures = 0
        for ledger, limit in ledgers:
            with ledger.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if limit is not None and row["update"] > limit:
                        break
                    updates += 1
                    size = batch_size_at(plan, updates) if plan else config.optimization.batch_size
                    expected = [tokens[i] for i in sampler.next_batch(size)]
                    if row["update"] != updates or row["sample_tokens"] != expected:
                        raise ValueError(
                            "Executed Dfull ledger differs from registered permutation"
                        )
                    observed.update(expected)
                    consumed_exposures += size
        if updates != config.optimization.updates or observed != set(tokens):
            raise ValueError("Dfull training coverage/budget is incomplete")
        if consumed_exposures != exposures:
            raise ValueError("Dfull exposure budget differs")
        spec = importlib.util.spec_from_file_location("frozen_verifier", args.verifier)
        assert spec is not None and spec.loader is not None
        helper: Any = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        completion = helper.verify_summary(
            output,
            config,
            expected_status="COMPLETED",
            expected_final_update=config.optimization.updates,
            expected_start_update=start_update,
            resume_sha256=transition["source_checkpoint_sha256"] if transition else None,
        )
        evaluations = {}
        compact = {}
        original = helper._load_object

        def capture(path: Path) -> dict[str, Any]:
            value = original(path)
            if "metrics_v2" in value:
                m = value["metrics_v2"]
                compact[str(path)] = {
                    "AP": m["dev_AP101_supported_classes"],
                    "sample_count": m["sample_count"],
                    "per_class_AP": {
                        k: v["AP101_mean_thresholds"] for k, v in m["per_class"].items()
                    },
                }
            return value

        helper._load_object = capture
        for update in config.evaluation.full_selection_updates:
            path = output / f"evaluation/full_selection_update_{update:06d}.json"
            record = helper.verify_evaluation(
                path,
                protocol_sha256=summary["evaluation_protocol_sha256"],
                expected_sample_count=summary["full_selection_count"],
            )
            record.pop("ground_truth_sha256_by_token")
            evaluations[str(update)] = record | {"metrics": compact[str(path)]}
            gc.collect()
        selected = output / "checkpoints/selected.pt"
        payload = torch.load(selected, map_location="cpu", weights_only=True, mmap=True)
        selected_update = payload["update"]
        if selected_update < first_full_pass:
            raise ValueError("Selected checkpoint has not seen all Dfull train frames")
        if selected_update == config.optimization.updates:
            last = torch.load(
                output / "checkpoints/last.pt", map_location="cpu", weights_only=True, mmap=True
            )
            if any(
                not torch.equal(value, last["model"][key])
                for key, value in payload["model"].items()
            ):
                raise ValueError("Final and selected model tensors differ at the same update")
            selected_evaluation = evaluations[str(config.optimization.updates)]
            del last, payload
        else:
            del payload
            status(status="EVALUATING_SELECTED", selected_update=selected_update)
            path = output / "evaluation/selected_checkpoint_full.json"
            command = [
                sys.executable,
                str(args.project / "scripts/evaluate_detection_selected_checkpoint.py"),
                "--checkpoint",
                str(selected),
                "--output",
                str(path),
            ]
            with (args.campaign / "selected_evaluation.log").open("x") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            record = helper.verify_evaluation(
                path,
                protocol_sha256=summary["evaluation_protocol_sha256"],
                expected_sample_count=summary["full_selection_count"],
            )
            record.pop("ground_truth_sha256_by_token")
            selected_evaluation = record | {"metrics": compact[str(path)]}
        release = {
            "schema": "phase_dfull_final_model_v1",
            "status": "VERIFIED",
            "checkpoint": str(selected),
            "checkpoint_sha256": digest(selected),
            "checkpoint_update": selected_update,
            "unique_train_frames": len(tokens),
            "run_updates": updates,
            "run_exposures": consumed_exposures,
            "batch_transition": transition,
            "selected_evaluation": selected_evaluation,
            "full_evaluations": evaluations,
            "completion": completion,
            "ledger_sha256": digest(output / "metrics.jsonl"),
            "scope": (
                "Dfull training candidate; development selection, "
                "not official eAP/SOTA or validated production TTC."
            ),
        }
        write(args.campaign / "FINAL_MODEL.json", release)
        status(
            status="COMPLETED_VERIFIED",
            selected_checkpoint=str(selected),
            selected_update=selected_update,
            AP=selected_evaluation["metrics"]["AP"],
            next_action="Download/verify final artifacts; no further training queued.",
        )
    except Exception as error:
        status(status="FAILED", error_type=type(error).__name__, error=str(error))
        raise


if __name__ == "__main__":
    main()
