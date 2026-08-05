#!/usr/bin/env python
"""Fail-closed preflight for the exact Object Event TTC v4 patch base."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BASE_COMMIT = "282b6658aeb1ad3110b5d9241872f6c2662369c3"
EXPECTED_PATCHED_HASHES = {
    "src/e_jepa_ttc/data/garlttc_lhr_cache.py": "5cd40940bccc8986dd61744882665c60292751587fc34dfb301a12a6e4d2310b",
    "src/e_jepa_ttc/data/event_v4_geometry.py": "6538f8f7e8324a3b048fc254225c8b4aca5cc9f10ff4f678ae7d7e75027b5351",
    "src/e_jepa_ttc/data/object_event_v4.py": "8b20a0ed5b799641ed910f113d4f9b3bde4d1322053b5942e42f7173ad4f84cb",
    "src/e_jepa_ttc/models/object_event_v4.py": "6560673c38deb76f6f7fa26fd022c110a2e88aca6278e11b224835cbe25c5260",
    "src/e_jepa_ttc/training/object_event_v4.py": "6a2a7412aafa759467d46115249231ce08eef3825ae200630d1b5087f639f864",
    "scripts/train_e_jepa_object_event_v4.py": "25958e577b2b94070d58c699e3a44bbce4966f59f49d5f4cf05649c4f11048b3",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--allow-descendant", action="store_true")
    args = parser.parse_args()

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != EXPECTED_BASE_COMMIT and not args.allow_descendant:
        raise RuntimeError(
            f"Expected exact v4 patch base {EXPECTED_BASE_COMMIT}, got {head}. "
            "Use --allow-descendant only after git apply --check has succeeded."
        )

    hashes = {}
    for relative, expected in EXPECTED_PATCHED_HASHES.items():
        path = ROOT / relative
        actual = _sha256(path)
        hashes[relative] = actual
        if actual != expected:
            raise RuntimeError(f"V4 source drift for {relative}: {actual} != {expected}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    if not readme.startswith("# E-JEPA-TTC") or "Object Event TTC v4" not in readme:
        raise RuntimeError("Root README was not restored to the project README plus v4 notes")

    report: dict[str, object] = {
        "status": "passed",
        "head": head,
        "expected_base": EXPECTED_BASE_COMMIT,
        "patched_hashes": hashes,
        "manifest_checked": False,
    }
    if args.manifest is not None:
        manifest_path = args.manifest.resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {
            "event_v4_common_roi",
            "event_v4_boxes_xyxy",
            "event_v4_common_square_xyxy",
            "event_v4_precontext_valid",
            "garl_delta_t_s",
            "observable_motion",
        }
        missing = sorted(required - set(manifest.get("model_input_fields", [])))
        if missing:
            raise RuntimeError(f"V4 cache lacks required fields: {missing}")
        extension = manifest.get("object_lhr_extension")
        if not isinstance(extension, dict) or int(extension.get("version", 0)) < 3:
            raise RuntimeError("V4 cache extension version is missing")
        if extension.get("event_v4_common_roi_shape", [])[:2] != [3, 12]:
            raise RuntimeError("V4 cache does not expose [3,12,H,W] event tensors")
        if extension.get("event_v4_independent_endpoint_resize") is not False:
            raise RuntimeError("V4 cache still resizes endpoints independently")
        precontext = float(
            manifest.get("event_v4_precontext_valid_fraction", 0.0)
        )
        if precontext < 0.80:
            raise RuntimeError(
                f"V4 real-event precontext coverage is too low: {precontext:.6f}"
            )
        if manifest.get("uses_official_garl_ttc_labels") is not True:
            raise RuntimeError("V4 cache must use official GarlTTC labels")
        report.update(
            {
                "manifest_checked": True,
                "manifest": manifest_path.as_posix(),
                "split_counts": manifest.get("split_counts"),
                "precontext_fraction": precontext,
                "object_lhr_extension": extension,
            }
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
