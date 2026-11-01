"""Reevaluate saved selection proposals under the versioned eAP reference profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.detection.benchmark_evaluation import (  # noqa: E402
    CLASSES,
    PROFILE,
    PROFILE_SHA256,
    BenchmarkEvaluator,
)


def sha(path: Path) -> str:
    """Hash input evidence without loading the whole file."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> Any:
    """Read UTF-8 JSON."""
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: Any) -> None:
    """Write finite, readable UTF-8 evidence."""
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def verify_shard_coverage(shards: Path) -> tuple[list[Path], dict[str, Any]]:
    """Reject missing shards against the historical full-selection receipt."""
    historical = read(shards.parent / "COMPARISON.json")
    if historical["schema"] != "dfull_ranking_full_confirmation_v1":
        raise ValueError("Expected historical full-selection receipt")
    folders = sorted(p.parent for p in shards.glob("*/COMPLETE.json"))
    count = sum(read(p / "COMPLETE.json")["frames"] for p in folders)
    if not folders or count != historical["frames"]:
        raise ValueError("Incomplete shard population")
    return folders, historical


def main() -> None:
    """Verify selection provenance, recover raw denominators and compare frozen scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    folders, historical = verify_shard_coverage(args.shards)
    sources = (Path(__file__), ROOT / "src/e_jepa_ttc/detection/benchmark_evaluation.py")
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    variants = {
        "baseline": "predictions.json",
        "ranked": "learned.json",
        "scalar_uncertainty": "scalar_uncertainty.json",
    }
    evaluators = {name: BenchmarkEvaluator() for name in variants}
    inputs = {}
    eligible: Counter[str] = Counter()
    ignored: Counter[str] = Counter()
    sequences: Counter[str] = Counter()
    labels: dict[str, str] = {}
    checkpoint = None
    for number, folder in enumerate(folders, 1):
        complete = read(folder / "COMPLETE.json")
        if complete["status"] != "VERIFIED":
            raise ValueError(f"Unverified shard: {folder}")
        verified = {}
        for filename in ["tokens.json", "gt.json", "gt_receipt.json", *variants.values()]:
            actual = sha(folder / filename)
            if actual != complete["files"][filename]:
                raise ValueError(f"Input SHA-256 mismatch: {folder / filename}")
            verified[filename] = actual
        verified["COMPLETE.json"] = sha(folder / "COMPLETE.json")
        inputs[folder.name] = verified
        receipt = read(folder / "gt_receipt.json")
        if (
            receipt["role"] != "selection"
            or receipt["panel"] != "selection_full"
            or receipt["test_labels_read"] is not False
            or receipt["ground_truth_sha256"] != verified["gt.json"]
        ):
            raise ValueError("Requires saved public selection_full ground truth")
        for label, digest in receipt["label_file_sha256"].items():
            if not label.startswith("data/train/") or labels.get(label, digest) != digest:
                raise ValueError("Public label provenance differs across shards")
            labels[label] = digest
        tokens = read(folder / "tokens.json")["sample_tokens"]
        if len(set(tokens)) != len(tokens) or len(tokens) != complete["frames"]:
            raise ValueError("Shard frame denominator mismatch")
        gt = read(folder / "gt.json")["ground_truth_by_token"]
        payloads = {name: read(folder / filename) for name, filename in variants.items()}
        if set(gt) != set(tokens):
            raise ValueError("Ground-truth token mismatch")
        for payload in payloads.values():
            if set(payload["predictions"]) != set(tokens):
                raise ValueError("Prediction token mismatch")
            checkpoint = checkpoint or payload["checkpoint_sha256"]
            if payload["checkpoint_sha256"] != checkpoint:
                raise ValueError("Checkpoint mismatch")
        for token in tokens:
            frame = gt[token]
            raw = sorted(frame["eligible"] + frame["ignored"], key=lambda b: b["source_index"])
            if [b["source_index"] for b in raw] != list(range(frame["raw_count"])):
                raise ValueError("Raw annotations missing or duplicated")
            eligible.update(b["class_name"] for b in frame["eligible"])
            ignored.update(b["class_name"] for b in frame["ignored"])
            sequences[token.split(":")[0]] += 1
            base = payloads["baseline"]["predictions"][token]
            for name, payload in payloads.items():
                predictions = payload["predictions"][token]

                def geometry_counts(boxes: list[dict[str, Any]]) -> Counter[str]:
                    return Counter(
                        json.dumps(
                            {
                                k: box[k]
                                for k in (
                                    "class_name",
                                    "translation_m",
                                    "size_lwh_m",
                                    "yaw_rad",
                                    "velocity_mps",
                                )
                            },
                            sort_keys=True,
                        )
                        for box in boxes
                    )

                if geometry_counts(predictions) != geometry_counts(base):
                    raise ValueError("Ranking changed proposal geometry or multiplicity")
                evaluators[name].add_frame(token, predictions, raw)
        elapsed = time.perf_counter() - start
        status = {
            "status": "matching",
            "shards": number,
            "total_shards": len(folders),
            "frames": sum(sequences.values()),
            "elapsed_seconds": elapsed,
        }
        write(args.output / "STATUS.json", status)
        print(json.dumps(status), flush=True)
    if sum(sequences.values()) != historical["frames"]:
        raise ValueError("Full-selection frame count differs from historical receipt")
    reports = {name: evaluator.summarize() for name, evaluator in evaluators.items()}
    if source_hashes != {str(p.relative_to(ROOT)): sha(p) for p in sources}:
        raise ValueError("Evaluation source changed during execution")
    for name, report in reports.items():
        write(args.output / f"{name}.json", report)
    comparison = {
        "schema": "phase_ttc_benchmark_local_comparison_v1",
        "profile": PROFILE,
        "profile_sha256": PROFILE_SHA256,
        "checkpoint_sha256": checkpoint,
        "sequence_frames": sequences,
        "samples": sum(sequences.values()),
        "historical_eligible_by_class": eligible,
        "historical_ignored_by_class": ignored,
        "public_label_sha256": labels,
        "input_sha256": inputs,
        "same_geometry_verified": True,
        "test_labels_read": False,
        "new_training": False,
        "new_inference": False,
        "variants": {
            name: {
                k: report[k]
                for k in (
                    "mAP",
                    "tp_errors",
                    "eAP_score",
                    "gt_boxes",
                    "pred_boxes",
                    "diagnostic_unclipped_envelope_AP101_same_population",
                )
            }
            for name, report in reports.items()
        },
        "ranked_minus_baseline_mAP": reports["ranked"]["mAP"] - reports["baseline"]["mAP"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "source_sha256": source_hashes,
        "numpy": np.__version__,
        "historical_comparison_sha256": sha(args.shards.parent / "COMPARISON.json"),
        "historical_comparison": historical,
        "elapsed_seconds": time.perf_counter() - start,
    }
    write(args.output / "COMPARISON.json", comparison)
    lines = [
        "# Evaluacion local alineada con eAP: seleccion completa",
        "",
        "Perfil observado v1; equivalencia con el scorer privado NO certificada.",
        "Solo etiquetas publicas de seleccion; pesos y scores congelados, sin entrenamiento.",
        "",
        "| Variante | mAP | ATE | ASE | AOE | eAP compuesto |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in reports.items():
        e = r["tp_errors"]
        lines.append(
            f"| {name} | {r['mAP']:.6f} | {e['trans_err']:.6f} | "
            f"{e['scale_err']:.6f} | {e['orient_err']:.6f} | {r['eAP_score']:.6f} |"
        )
    lines += [
        "",
        "| Clase | GT raw | Baseline AP | Ranked AP | Diferencia |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in CLASSES:
        b = reports["baseline"]["classes"][name]
        r = reports["ranked"]["classes"][name]
        lines.append(
            f"| {name} | {b['gt_count']} | {b['AP']:.6f} | {r['AP']:.6f} | "
            f"{r['AP'] - b['AP']:+.6f} |"
        )
    lines += [
        "",
        f"Frames: {sum(sequences.values())}; "
        f"GT raw: {sum(eligible.values()) + sum(ignored.values())}.",
        f"GT elegible historico: {sum(eligible.values())}; "
        f"antes ignorado: {sum(ignored.values())}.",
        "",
        "Comparacion pareada sobre las mismas propuestas y poblacion. No mide significancia",
        "entre semillas ni rendimiento oficial del baseline en test.",
        "El test no se usa para tuning.",
        "La metrica historica v2 conserva su definicion y sus resultados originales.",
        "Los empates siguen orden inverso de indice de entrada; GT equidistante usa el primero.",
        "Orden de entrada fijado por shards/tokens guardados y hashes de COMPARISON.json.",
        "",
        "Referencia de matching/interpolacion: "
        "https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/detection/algo.py",
        "Configuracion y aritmetica contrastadas con los agregados del organizador.",
    ]
    (args.output / "REPORT.md").write_bytes(("\n".join(lines) + "\n").encode())
    write(
        args.output / "STATUS.json",
        {
            "status": "COMPLETE",
            "frames": sum(sequences.values()),
            "elapsed_seconds": time.perf_counter() - start,
        },
    )
    write(
        args.output / "MANIFEST.json",
        {
            p.name: sha(p)
            for p in args.output.iterdir()
            if p.is_file() and p.name != "MANIFEST.json"
        },
    )
    print(json.dumps(comparison["variants"], indent=2), flush=True)


if __name__ == "__main__":
    main()
