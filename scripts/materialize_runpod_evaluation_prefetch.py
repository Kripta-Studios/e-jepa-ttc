"""Freeze an evaluation-only scheduling revision; retain the parent producer."""

# ruff: noqa: E501 -- exact source anchors and generated source preserve parent text.

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rewrite_evaluation(text: str) -> str:
    """Move unchanged CPU preparation ahead of B1 inference, with bounded overlap."""
    start = text.index("def evaluate(\n")
    end = text.index("\ndef _metric(", start)
    function = text[start:end]
    truth = """build_evaluation_ground_truth(
                raw.annotations,
                intrinsic=raw.intrinsic,
                event_from_ego=raw.event_from_ego,
                image_wh=(raw.image.shape[1], raw.image.shape[0]),
                protocol=protocol,
            )"""
    old_loop = """    with torch.no_grad():
        for index in range(len(cache.dataset)):
            prepared, raw = cache.get(index)"""
    new_loop = """    evaluation_started = time.perf_counter()
    progress_path = output.parent.parent / "evaluation_progress" / output.name
    cpu_totals = {"load_prepare_cpu_seconds": 0.0, "ground_truth_cpu_seconds": 0.0}

    def report_progress(stage, completed):
        write_json(progress_path, {
            "stage": stage, "completed_frames": completed,
            "total_frames": len(cache.dataset),
            "elapsed_seconds": time.perf_counter() - evaluation_started,
            "inference_seconds_total": sum(timings), **cpu_totals,
        })

    def truth_builder(raw):
        return TRUTH_EXPRESSION

    def legacy_builder(raw):
        return legacy_ground_truth(raw, config) if config.evaluation.legacy_metrics else (None, None)

    report_progress("inference", 0)
    with torch.no_grad(), prefetch_evaluation_rows(cache, truth_builder, legacy_builder) as rows:
        for index, (prepared, raw, ground_truth, historical_result, cpu_times) in enumerate(rows):
            for key, value in cpu_times.items():
                cpu_totals[key] += value""".replace("TRUTH_EXPRESSION", truth)
    changes = {
        old_loop: new_loop,
        f"raw_truth[raw.sample_token] = {truth}": "raw_truth[raw.sample_token] = ground_truth",
        "historical, reason = legacy_ground_truth(raw, config)": "historical, reason = historical_result",
        "    result = evaluate_detections_v2(predictions, raw_truth, protocol=protocol)": """            if (index + 1) % 256 == 0:
                report_progress("inference", index + 1)
    report_progress("scoring", len(cache.dataset))
    result = evaluate_detections_v2(predictions, raw_truth, protocol=protocol)""",
        "    write_json(output, payload)": """    report_progress("serialization", len(cache.dataset))
    write_json(output, payload)
    report_progress("complete", len(cache.dataset))""",
    }
    for old, new in changes.items():
        if function.count(old) != 1:
            raise ValueError(f"Evaluation source anchor changed: {old[:100]}")
        function = function.replace(old, new, 1)
    return text[:start] + function + text[end:]


def materialize(source: Path, destination: Path, module_root: Path) -> None:
    """Create a hash-bound producer with unchanged model and metric algorithms."""
    if destination.exists():
        raise FileExistsError(destination)
    original = source / "scripts/train_detection_campaign.py"
    text = rewrite_evaluation(original.read_text())
    anchor = "from e_jepa_ttc.detection.resident_input_cache import ResidentInputCache"
    if text.count(anchor) != 1 or text.count("    owned_paths = [") != 1:
        raise ValueError("Expected the admitted input-residency producer")
    text = text.replace(
        anchor,
        "from e_jepa_ttc.detection.evaluation_prefetch import prefetch_evaluation_rows\n" + anchor,
    )
    text = text.replace(
        "    owned_paths = [",
        """    owned_paths = [
        ROOT / "src/e_jepa_ttc/detection/evaluation_prefetch.py",
        ROOT / "src/e_jepa_ttc/detection/bounded_prefetch.py",""",
    )
    text = text.replace(
        '"schema": "phase_ttc_exact_input_residency_v1",',
        '"schema": "phase_ttc_exact_input_residency_v1",\n    "evaluation_cpu_prefetch_capacity": 2,',
    )
    compile(text, str(destination / original.relative_to(source)), "exec")
    shutil.copytree(source, destination)
    target = destination / original.relative_to(source)
    target.write_text(text, encoding="utf-8", newline="\n")
    modules = {}
    for name in ("evaluation_prefetch.py", "bounded_prefetch.py"):
        relative = Path("src/e_jepa_ttc/detection") / name
        shutil.copy2(module_root / relative, destination / relative)
        modules[relative.as_posix()] = digest(destination / relative)
    receipt = {
        "status": "MATERIALIZED_REQUIRES_PARITY_AND_TIMING",
        "parent_runner_sha256": digest(original),
        "runner_sha256": digest(target),
        "modules": modules,
        "batch_size": 1,
        "model_and_scorer_unchanged": True,
    }
    (destination / "EVALUATION_PREFETCH_RECEIPT.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--module-root", type=Path, required=True)
    args = parser.parse_args()
    materialize(args.source, args.destination, args.module_root)
