"""Audit official GarlTTC cache provenance and leakage boundaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_jepa_ttc.data.garl_input_contract import validate_cache_manifest_input_schema  # noqa: E402
from e_jepa_ttc.data.garlttc_lhr_cache import (  # noqa: E402
    FORBIDDEN_MODEL_INPUT_KEYS,
    GarlTTCLHRCacheDataset,
)
from e_jepa_ttc.utils.io import read_structured, write_structured  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=128)
    args = parser.parse_args()

    manifest = read_structured(args.manifest)
    errors: list[str] = []
    if manifest.get("uses_official_garl_ttc_labels") is not True:
        errors.append("uses_official_garl_ttc_labels is not true")
    if manifest.get("uses_reconstructed_public_eap_ttc") is not False:
        errors.append("uses_reconstructed_public_eap_ttc is not false")
    if manifest.get("no_label_fallback") is not True:
        errors.append("no_label_fallback is not true")
    try:
        validate_cache_manifest_input_schema(manifest)
    except ValueError as exc:
        errors.append(f"invalid v4 input schema: {exc}")
    model_inputs = set(manifest.get("model_input_fields", []))
    leaked = sorted(model_inputs & FORBIDDEN_MODEL_INPUT_KEYS)
    if leaked:
        errors.append(f"privileged fields declared as model inputs: {leaked}")

    sample_checks = []
    for split in ("train", "validation"):
        dataset = GarlTTCLHRCacheDataset(args.manifest, splits=(split,))
        count = min(len(dataset), args.max_samples)
        for index in range(count):
            sample = dataset[index]
            if not str(sample.get("ttc_label_source", "")).startswith(
                "official_garlttc_annotations_train_parquet.frame_ttc[t2]"
            ):
                errors.append(f"{split}[{index}] has wrong TTC provenance")
                break
            for key in (
                "full_frame_events",
                "garl_event_roi",
                "garl_delta_t_s",
                "observable_motion",
            ):
                if key not in sample:
                    errors.append(f"{split}[{index}] missing model input {key}")
            for key in ("ttc_s", "garl_visible_heights_px", "geometry_v2_target"):
                if key not in sample:
                    errors.append(f"{split}[{index}] missing supervision {key}")
        sample_checks.append({"split": split, "checked": count, "total": len(dataset)})

    result = {
        "artifact_type": "garlttc_lhr_cache_audit_v2",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "sample_checks": sample_checks,
        "model_input_fields": sorted(model_inputs),
        "forbidden_model_input_fields": sorted(FORBIDDEN_MODEL_INPUT_KEYS),
        "manifest": str(args.manifest),
    }
    write_structured(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
