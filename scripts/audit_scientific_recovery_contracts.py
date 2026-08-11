#!/usr/bin/env python
"""Audit causal/oracle-ROI and Garl protocol claims without opening any sealed test."""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.artifacts.hashing import sign_artifact
from e_jepa_ttc.models.causal_scale_ttc import CausalScaleTTC


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, text=True, capture_output=True).stdout.strip()


def _contains_all(text: str, needles: tuple[str, ...]) -> bool:
    return all(needle in text for needle in needles)


def audit(garl_repo: Path, output: Path) -> dict[str, Any]:
    e_materializer = ROOT / "src/e_jepa_ttc/data/garlttc_lhr_cache.py"
    e_model = ROOT / "src/e_jepa_ttc/models/causal_scale_ttc.py"
    g_dataset = garl_repo / "garl_ttc/datasets/ttc_dataset.py"
    for path in (e_materializer, e_model, g_dataset):
        if not path.is_file():
            raise FileNotFoundError(path)

    e_text = e_materializer.read_text(encoding="utf-8")
    g_text = g_dataset.read_text(encoding="utf-8")
    signature = inspect.signature(CausalScaleTTC.forward)
    forward_names = [name for name in signature.parameters if name != "self"]
    privileged = {"bbox", "box", "rgb", "ttc", "target", "dino", "teacher", "track_id"}
    forward_privileged = sorted(name for name in forward_names if any(p in name.lower() for p in privileged))

    # Static source contract: materializer explicitly passes only the two endpoint
    # indices to common_square_from_boxes. This is paired with dynamic unit tests.
    ejepa_roi_endpoint_only = _contains_all(
        e_text,
        (
            "common_square_from_boxes(",
            "(first_index, second_index)",
            "event_v4_common_square_xyxy",
        ),
    )
    garl_oracle_crop = (
        "['box']" in g_text or '["box"]' in g_text
    ) and ("crop" in g_text.lower() or "square" in g_text.lower())

    model_text = e_model.read_text(encoding="utf-8")
    legacy_has_next_neighbor = "padded[:, 2:]" in model_text
    causal_left_exists = 'mode == "causal_left"' in model_text

    result: dict[str, Any] = {
        "artifact_type": "scientific_recovery_contract_audit_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PASS" if (not forward_privileged and ejepa_roi_endpoint_only and garl_oracle_crop and causal_left_exists) else "FAIL",
        "ejepa": {
            "commit": _git(ROOT, "rev-parse", "HEAD"),
            "forward_parameters": forward_names,
            "privileged_forward_parameters": forward_privileged,
            "neural_forward_event_delta_only": not forward_privileged,
            "oracle_roi_preprocessing": True,
            "common_roi_endpoint_box_indices_only_static_source_audit": ejepa_roi_endpoint_only,
            "legacy_symmetric_smoothing_reads_next_endpoint": legacy_has_next_neighbor,
            "causal_left_mode_available": causal_left_exists,
        },
        "garl": {
            "repo": str(garl_repo.resolve()),
            "commit": _git(garl_repo, "rev-parse", "HEAD"),
            "dirty": bool(_git(garl_repo, "status", "--porcelain")),
            "oracle_box_crop_in_preprocessing_static_source_audit": garl_oracle_crop,
        },
        "claim_contract": {
            "matched_oracle_roi_comparison_allowed": bool(ejepa_roi_endpoint_only and garl_oracle_crop),
            "event_only_neural_forward_claim_allowed": not forward_privileged,
            "end_to_end_no_oracle_localization_claim_allowed": False,
            "legacy_strict_streaming_prefix_causal_claim_allowed": False,
            "causal_left_model_prefix_causality_requires_dynamic_test": True,
            "full_pipeline_strict_streaming_requires_non_oracle_causal_localizer": True,
            "honest_oracle_roi_claim": "event-only neural TTC under matched oracle-box/object-ROI preprocessing",
        },
        "sealed_sources": {
            "private_test_opened": False,
            "codabench_opened": False,
            "evttc_test_opened": False,
        },
        "sources": {
            "ejepa_materializer": {"path": str(e_materializer.relative_to(ROOT)), "sha256": _sha(e_materializer)},
            "ejepa_model": {"path": str(e_model.relative_to(ROOT)), "sha256": _sha(e_model)},
            "garl_dataset": {"path": str(g_dataset), "sha256": _sha(g_dataset)},
        },
    }
    sign_artifact(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--garl-repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    try:
        result = audit(args.garl_repo.resolve(), args.output.resolve())
    except Exception as exc:
        print(f"contract audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
