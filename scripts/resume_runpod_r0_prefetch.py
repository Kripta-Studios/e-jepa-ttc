"""Complete pending evaluation from migrated R0 state, then resume the RGB queue."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path: Path, value: Any) -> None:  # noqa: ANN401
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("evaluate", "run"))
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    project = root / "workspace/phase-ttc-s0-evaluation"
    os.chdir(project)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    producer_root = project / "output/cloud_evaluation_prefetch_v1/producer"
    producer = producer_root / "scripts/train_detection_campaign.py"
    sys.path[:0] = [str(producer_root / "src"), str(producer_root / "scripts")]
    runner = load("admitted_r0_evaluator", producer)
    helper = load(
        "resumed_queue_verifier",
        project
        / (
            "output/detection_campaign/rgb_screen_v5_seed7_v1/automatic_queue_v1/"
            "scripts/run_detection_campaign_queue.py"
        ),
    )
    migration_root = root / "evaluation_optimization_v1/r0_migration"
    migration = json.loads((migration_root / "MIGRATION.json").read_text())
    admission = json.loads(
        (root / "evaluation_optimization_v1/deployment_admission.json").read_text()
    )
    smoke = json.loads((migration_root / "RESUME_ADMISSION.json").read_text())
    if (
        migration["status"] != "PASS_EXACT_STATE_PROVENANCE_ONLY"
        or smoke["status"] != "PASS_STRICT_RESUME_4096_TO_4097"
        or smoke["checkpoint_sha256"] != migration["checkpoint_sha256"]
    ):
        raise ValueError("Exact-state migration and strict resume must pass")
    checkpoint = Path(migration["checkpoint"])
    if (
        sha(checkpoint) != migration["checkpoint_sha256"]
        or sha(producer) != admission["runner_sha256"]
    ):
        raise ValueError("Checkpoint or producer identity changed")
    for relative, expected in admission["modules"].items():
        if sha(producer_root / relative) != expected:
            raise ValueError("Prefetch dependency changed")
    config_path = root / "cloud_configs_input_v1/r0_seed7.yaml"
    config = runner.load_campaign_config(config_path)
    runtime = json.loads((migration_root / "runtime_preregistration.json").read_text())
    for relative, expected in runtime["code_sha256"].items():
        source = (
            migration_root / "resume_smoke/code" / relative
            if relative.startswith("data/admission/")
            else producer_root / relative
        )
        if sha(source) != expected:
            raise ValueError(f"Executed source differs: {relative}")
    pending = migration_root / "pending_evaluation"
    evaluation = pending / "evaluation/full_selection_update_004096.json"
    if args.mode == "evaluate":
        import torch

        if pending.exists():
            raise FileExistsError(pending)
        pending.mkdir()
        if config.sha256 != runtime["config_sha256"]:
            raise ValueError("R0 configuration changed")
        runner.validate_manifest_against_metadata(runner.validate_campaign_manifest(config), config)
        tokens_path = Path(config.data.full_selection_tokens)
        if sha(tokens_path) != config.data.expected_full_selection_sha256:
            raise ValueError("Full evaluation population changed")
        tokens = json.loads(tokens_path.read_text())["sample_tokens"]
        portable = Path(config.model.portable_root)
        for relative, expected in runtime["portable_source_sha256"].items():
            if sha(portable / relative) != expected:
                raise ValueError("Portable model source changed")
        torch.set_num_threads(config.resources.num_cpu_threads)
        torch.set_float32_matmul_precision(config.optimization.matmul_precision)
        model, _, effective = runner.build_detector(
            portable,
            config.model.group_num,
            predict_velocity=config.model.predict_velocity,
            velocity_loss_coef=config.model.velocity_loss_coef,
        )
        if effective != runtime["effective_model"]:
            raise ValueError("Effective model differs")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if payload["runtime_provenance"] != migration["new_runtime_provenance"]:
            raise ValueError("Checkpoint runtime differs")
        model.load_state_dict(payload["model"], strict=True)
        member_hashes = payload["data_member_sha256"]
        del payload
        with runner.open_campaign_dataset(config.data.root, tokens, config) as dataset:
            result = runner.evaluate(
                model,
                runner.PreparedCache(dataset, config, member_hashes, label_free_model_input=True),
                runner.load_protocol(config.evaluation.protocol),
                config,
                evaluation,
            )
        write(
            pending / "RECEIPT.json",
            {
                "status": "COMPLETED_PENDING_EVALUATION",
                "update": 4096,
                "checkpoint_sha256": migration["checkpoint_sha256"],
                "evaluation_sha256": sha(evaluation),
                "sample_count": len(tokens),
                "AP": result["dev_AP101_supported_classes"],
                "runner_sha256": sha(producer),
            },
        )
        return

    queue = root / "cloud_runs_eval_v1"
    jobs = json.loads((queue / "admission.json").read_text())["jobs"]
    state = {
        "status": "EVALUATING_R0_UPDATE_4096",
        "active": ["r0_seed7"],
        "completed": {},
        "failures": {},
        "pending": [j["name"] for j in jobs],
        "concurrency": 1,
        "resume_update": 4096,
        "evaluation_progress": str(
            pending / "evaluation_progress/full_selection_update_004096.json"
        ),
    }
    write(queue / "OWNER.json", {"pid": os.getpid(), "jobs": 1, "mode": "resumed_r0"})
    write(queue / "status.json", state)
    write(
        root / "ACTIVE_CAMPAIGN.json",
        {
            "queue": str(queue),
            "status_file": str(queue / "status.json"),
            "optimization": "exact_evaluation_prefetch_v1",
            "jobs": 1,
        },
    )
    try:
        with (migration_root / "pending_evaluation.log").open("x") as log:
            child = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), "evaluate", "--root", str(root)],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            write(
                queue / "receipts/pending_r0_evaluation.json",
                {"pid": child.pid, "checkpoint_sha256": migration["checkpoint_sha256"]},
            )
            try:
                returncode = child.wait(timeout=config.resources.maximum_seconds)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
            if returncode:
                raise RuntimeError("Pending R0 evaluation failed")
        pending_receipt = json.loads((pending / "RECEIPT.json").read_text())
        if pending_receipt["evaluation_sha256"] != sha(evaluation):
            raise ValueError("Pending evaluation artifact changed")
        output = queue / "r0_seed7_resumed"
        state.update(status="RUNNING", active=["r0_seed7"], run_output=str(output))
        write(queue / "status.json", state)
        helper.run_child(
            name="r0_seed7_resumed",
            python=Path(sys.executable),
            producer=producer,
            config_path=config_path,
            config=config,
            output=output,
            queue_output=queue / "receipts",
            gpu="0",
            resume=checkpoint,
        )
        attached = output / "evaluation/full_selection_update_004096.json"
        shutil.copyfile(evaluation, attached)
        if sha(attached) != pending_receipt["evaluation_sha256"]:
            raise ValueError("Inherited evaluation copy differs")
        write(output / "inherited_evaluation_4096.json", pending_receipt)
        result = helper.verify_summary(
            output,
            config,
            expected_status="COMPLETED",
            expected_final_update=8192,
            expected_start_update=4096,
            resume_sha256=migration["checkpoint_sha256"],
        )
        population = helper.verify_full_evaluations(output, result["summary"])
        state["completed"]["r0_seed7"] = {
            "output": str(output),
            "summary_sha256": result["summary_sha256"],
            "metric": result["summary"]["best_selection_metric"],
        }
        for job in jobs:
            name, path = job["name"], Path(job["path"])
            if sha(path) != job["sha256"] or sha(producer) != admission["runner_sha256"]:
                raise ValueError("Queued input changed")
            config = helper.load_campaign_config(path)
            helper.resource_preflight(config, "0")
            state.update(active=[name], run_output=str(queue / name))
            state["pending"].remove(name)
            write(queue / "status.json", state)
            helper.run_child(
                name=name,
                python=Path(sys.executable),
                producer=producer,
                config_path=path,
                config=config,
                output=queue / name,
                queue_output=queue / "receipts",
                gpu="0",
            )
            result = helper.verify_summary(
                queue / name,
                config,
                expected_status="COMPLETED",
                expected_final_update=8192,
                expected_start_update=0,
                resume_sha256=None,
            )
            helper.verify_population_same(
                population, helper.verify_full_evaluations(queue / name, result["summary"])
            )
            state["completed"][name] = {
                "summary_sha256": result["summary_sha256"],
                "metric": result["summary"]["best_selection_metric"],
            }
        state.update(status="RGB_CONFIRMATIONS_COMPLETE", active=[])
        write(queue / "status.json", state)
    except BaseException as error:
        state["status"] = "FAILED_NEEDS_REVIEW"
        state["failures"]["controller"] = str(error)
        write(queue / "status.json", state)
        raise


if __name__ == "__main__":
    main()
