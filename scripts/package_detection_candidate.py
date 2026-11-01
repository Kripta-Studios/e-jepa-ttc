"""Assemble a portable, hash-bound Dfull + ranking research candidate."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
from pathlib import Path

from infer_detection_release import digest, verify_release, write_json_bytes

RANKING_MODULES = (
    "__init__.py",
    "detection/__init__.py",
    "detection/quality_ranking.py",
    "detection/evaluation_ground_truth.py",
    "detection/evaluation_protocol.py",
    "detection/geometry_contract.py",
    "detection/development_metrics.py",
    "detection/evaluation_ap_fast.py",
)


def main() -> None:
    """Package supplied completed artifacts without changing the frozen producer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ranker", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1]
    checkpoint = args.checkpoint.resolve(strict=True)
    parent = checkpoint.parent.parent
    ranker = json.loads(args.ranker.read_text(encoding="utf-8"))
    checkpoint_sha = digest(checkpoint)
    if ranker["provenance"]["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("Ranker does not belong to this checkpoint")
    if ranker["provenance"]["dataset_role"] != "train":
        raise ValueError("Ranker must be fitted only on TRAIN")
    prereg = json.loads((parent / "runtime_preregistration.json").read_text(encoding="utf-8"))
    for relative, expected in prereg["code_sha256"].items():
        if digest(parent / "code" / relative) != expected:
            raise ValueError(f"Changed frozen runtime: {relative}")
    for relative, expected in prereg["portable_source_sha256"].items():
        if digest(parent / "code/portable_source" / relative) != expected:
            raise ValueError(f"Changed frozen portable source: {relative}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    run = output / "detector"
    (run / "checkpoints").mkdir(parents=True)
    shutil.copyfile(checkpoint, run / "checkpoints/selected.pt")
    shutil.copyfile(parent / "runtime_preregistration.json", run / "runtime_preregistration.json")
    shutil.copytree(
        parent / "code",
        run / "code",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        copy_function=shutil.copyfile,
    )
    shutil.copyfile(args.ranker, output / "ranker.json")
    for relative in RANKING_MODULES:
        target = output / "ranking_runtime/src/e_jepa_ttc" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "src/e_jepa_ttc" / relative, target)
    shutil.copyfile(source / "scripts/infer_detection_release.py", output / "infer.py")
    shutil.copyfile(source / "LICENSE", output / "LICENSE")
    shutil.copytree(args.evidence, output / "evidence", copy_function=shutil.copyfile)
    packages = [
        "torch",
        "torchvision",
        "numpy",
        "scipy",
        "pandas",
        "pyarrow",
        "Pillow",
        "PyYAML",
        "scikit-learn",
    ]
    versions = {name: importlib.metadata.version(name) for name in packages}
    write_json_bytes(output / "ENVIRONMENT.json", {"tested_packages": versions})
    (output / "requirements-tested.txt").write_text(
        "\n".join(f"{name}=={version}" for name, version in versions.items()) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        "# PHASE-TTC Dfull RGB + ranking — research candidate\n\n"
        "The detector and TRAIN-fitted ranker are frozen. No additional training is needed.\n"
        "This package provides eAP RGB/calibration inference.\n"
        "PHASE TTC weights are not included.\n\n"
        "## Run\n\n"
        "Use Python 3.11 and a compatible CUDA-enabled PyTorch installation. The exact tested\n"
        "package versions are in ENVIRONMENT.json and requirements-tested.txt. Install the\n"
        "dependencies in your detector environment, then run:\n\n"
        "```text\npython infer.py verify\n"
        "python infer.py infer --data-root DATA_ROOT --tokens tokens.json --output ../replay\n"
        "```\n\n"
        'tokens.json contains only `{"sample_tokens": ["sequence:frame"]}` or a token list.\n'
        "Use 1–128 frames per call; split larger populations. The label-free reader requires\n"
        "the original eAP train.parquet media/calibration manifest and RGB TAR members.\n"
        "Raw object labels are not read. CUDA is required. Output must be outside this package.\n"
        "The output directory must be new. Do not overwrite an earlier inference receipt.\n\n"
        "predictions.json contains the ranked 3D detections. baseline.json, diagnostics.json and\n"
        "RECEIPT.json preserve provenance. Class, 3D geometry, yaw and velocity are unchanged;\n"
        "only the ranking score changes. Scores are not calibrated collision probabilities.\n\n"
        "## Limits\n\n"
        "This is a single-seed development candidate, not an official SOTA or safety-certified\n"
        "production release. The development selection was exposed to model selection.\n"
        "See evidence for all class results, including the other_vehicle regression.\n"
        "Temporal and event screening did not pass the predefined promotion gate.\n"
        "Independent holdout confirmation, calibration, additional seeds, official scorer\n"
        "comparison and independent end-to-end TTC quality remain unestablished.\n"
        "Do not interpret this detector's AP as a TTC accuracy result.\n\n"
        "Project and upstream licenses are retained. Dataset files are not redistributed.\n",
        encoding="utf-8",
    )
    files = {
        path.relative_to(output).as_posix(): digest(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema": "phase_detection_release_v1",
        "release_id": "dfull-rgb-ranked-seed7-20261101",
        "status": "research_candidate",
        "checkpoint": "detector/checkpoints/selected.pt",
        "ranker": "ranker.json",
        "files": files,
        "claims": {"official_sota": False, "production_ready": False, "new_ttc_accuracy": False},
    }
    write_json_bytes(output / "MANIFEST.json", manifest)
    verify_release(output)
    print(
        json.dumps(
            {
                "status": "VERIFIED",
                "release": str(output),
                "files": len(files),
                "manifest_sha256": digest(output / "MANIFEST.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
