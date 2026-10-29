"""Admit native Garl producers against exact D1 token and exclusion contracts."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from .common import ROOT, Campaign, atomic_json, digest, npz, read


def admit(c: Campaign) -> dict:
    """Count source-style50-epoch updates and required TRAIN geometric labels."""
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    c.require_resources()
    audit = read(c.out / "GARL_SOURCE_AUDIT.json")
    if not audit["upstream"].get("verified"):
        raise ValueError("pinned native Garl source/config unavailable")
    historical = Path(c.launch["roots"]["historical"])
    ancestry_path = historical / "NESTED_ANCESTRY_AUDIT.json"
    ancestry = read(ancestry_path)
    pool_path = c.historical / "artifacts/simplex_t/T0/EXPANSION_POOL_PLAN.json"
    expansion_root = c.historical / "artifacts/simplex_t/T1/expansion_query_context_index"
    index_manifest = read(expansion_root / "INDEX_MANIFEST.json")
    index_path = expansion_root / "query_context_index.npz"
    if digest(index_path) != index_manifest["index_sha256"]:
        raise ValueError("D1 query mappings changed")
    with np.load(index_path, allow_pickle=False) as z:
        tokens = z["tokens"]
        sequences = z["sequences"]
        families = z["producer_family"]
    fits = []
    token_union = set()
    for fold in range(3):
        base = pd.read_csv(
            historical / f"tables/outer{fold}_inner_oof.csv",
            usecols=["sample_token", "sequence_id", "inner_fold"],
        )
        compiled = read(
            c.historical
            / f"artifacts/simplex_t/T1/compiled_expansion_context/outer{fold}/COMPILED.json"
        )
        selected = np.asarray(compiled["selected_query_ids"], dtype=np.int64)
        expansion = pd.DataFrame(
            {
                "sample_token": tokens[selected],
                "sequence_id": sequences[selected],
                "inner_fold": [
                    int(index_manifest["families"][int(i)]["role"].replace("inner", ""))
                    for i in families[fold, selected]
                ],
            }
        )
        outer = pd.concat([base, expansion], ignore_index=True)
        expected = c.freeze()["sources"][str(fold)]["population"]
        if len(outer) != expected or outer.sample_token.duplicated().any():
            raise ValueError("Garl D1 training universe differs from WIDE/head TRAIN")
        excluded = set(
            next(
                r["split_validation"]["dev_sequence_ids"]
                for r in ancestry["producers"]
                if r["outer_fold"] == fold and r["expert"] == "A5" and r["role"] == "outer_dev"
            )
        )
        if set(outer.sequence_id) & excluded:
            raise ValueError("Garl outer training includes outer holdout")
        for inner in (None, 0, 1, 2):
            training = outer if inner is None else outer.loc[outer.inner_fold != inner]
            held = (
                set() if inner is None else set(outer.loc[outer.inner_fold == inner, "sequence_id"])
            )
            if set(training.sequence_id) & held:
                raise ValueError("Garl sequence crosses inner partition")
            key = f"outer{fold}" if inner is None else f"outer{fold}_inner{inner}"
            ids = training.sample_token.to_numpy(dtype=str)
            npz(c.out / f"garl/admission/{key}_tokens.npz", tokens=ids)
            fits.append(
                {
                    "key": key,
                    "outer": fold,
                    "inner": inner,
                    "samples_before_native_filter": len(ids),
                    "train_sequences": sorted(set(training.sequence_id)),
                    "excluded_outer": sorted(excluded),
                    "excluded_inner": sorted(held),
                    "tokens_sha256": digest(c.out / f"garl/admission/{key}_tokens.npz"),
                }
            )
            token_union.update(ids)
    # Filter by allowed tokens at the parquet read boundary: protected TRAIN
    # sequence payloads are never materialized as an intermediate full table.
    label_path = (
        Path(c.local["garl_annotations_candidate"]).parents[1] / "annotations/train.parquet"
    )
    columns = ["sample_token", "sequence_id", "ttc", "frame_ttc", "box3d_h", "box3d_Fcam"]
    labels = pq.read_table(
        label_path,
        columns=columns,
        filters=[("sample_token", "in", sorted(token_union))],
        use_threads=False,
    ).to_pylist()
    label_tokens = [row["sample_token"] for row in labels]
    if len(set(label_tokens)) != len(label_tokens):
        raise ValueError("ambiguous Garl label mapping")
    allowed = {}
    missing = {}
    for row in labels:
        token = row["sample_token"]
        try:
            frame = np.asarray(row["frame_ttc"], dtype=np.float64)
            geometry = np.asarray([np.asarray(v, dtype=np.float64) for v in row["box3d_Fcam"]])
            height = float(row["box3d_h"])
            # Source-style training range: every selected frame [-10,10],
            # phase-domain support, geometric height and positive visible depths.
            native_range = bool(np.isfinite(frame).all() and ((frame >= -10) & (frame <= 10)).all())
            geom = bool(
                0 < height <= 10
                and geometry.ndim == 3
                and geometry.shape[-1] == 3
                and np.isfinite(geometry).all()
                and (geometry[:, :, 2].min(1) > 0).all()
            )
            allowed[token] = native_range and geom
            if native_range and not geom:
                missing[token] = "missing_or_invalid_source_visible_height_geometry"
        except (ValueError, TypeError):
            allowed[token] = False
            missing[token] = "unparseable_source_geometry_or_frame_ttc"
    for token in token_union - set(label_tokens):
        missing[token] = "missing_exact_token_label_join"
    for fit in fits:
        path = c.out / f"garl/admission/{fit['key']}_tokens.npz"
        with np.load(path, allow_pickle=False) as z:
            ids = z["tokens"]
        admitted = [str(t) for t in ids if allowed.get(str(t), False)]
        fit["samples_after_native_filter"] = len(admitted)
        fit["updates_50_epochs"] = 50 * math.ceil(len(admitted) / 128)
        fit["native_filtered_samples"] = len(ids) - len(admitted)
        npz(c.out / f"garl/admission/{fit['key']}_native_tokens.npz", tokens=np.asarray(admitted))
    total = sum(row["updates_50_epochs"] for row in fits)
    reasons = []
    if missing:
        reasons.append("MISSING_NATIVE_TRAIN_GEOMETRY_OR_LABEL_MAPPING")
    if total > c.policy["garl_producer_updates_total_max"]:
        reasons.append("FIXED_50_EPOCHS_EXCEED_PRODUCER_BUDGET")
    if any(v["samples_after_native_filter"] == 0 for v in fits):
        reasons.append("EMPTY_NATIVE_TRAIN_PARTITION")
    result = {
        "status": "BLOCKED_DEPENDENCY"
        if reasons
        else "DATA_BUDGET_ADMITTED_PENDING_MICROBATCH_PROFILE",
        "reasons": reasons,
        "fits": fits,
        "updates_exact_50_epochs": total,
        "producer_cap": 200000,
        "required_producers": 12,
        "missing_geometry_examples": list(missing.items())[:32],
        "missing_geometry_count": len(missing),
        "native_filter": "all frame TTC within[-10,10], valid geometric visible-height supervision",
        "geometry_used_in_forward": False,
        "pretrained_eap_checkpoint_loaded": False,
        "heads": "BLOCKED_BY_NATIVE_PRODUCERS" if reasons else "PENDING_INNER_OOF",
        "optimizer_updates": 0,
        "label_path": str(label_path),
        "label_sha256": digest(label_path),
        "ancestry_sha256": digest(ancestry_path),
        "pool_sha256": digest(pool_path),
    }
    atomic_json(c.out / "GARL_COMPARISON.json", result)
    admission_path = c.out / "garl/ADMISSION.json"
    if not admission_path.exists():
        atomic_json(admission_path, result)
    elif read(admission_path) != result:
        raise ValueError("preserve the native scientific admission; changed inputs need review")
    return result


def main() -> int:
    """Run bounded native-source admission independently of the WIDE writer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("admit",))
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/campaign/efficient_context_v1.json"
    )
    args = parser.parse_args()
    c = Campaign(args.protocol)
    result = admit(c)
    print(result["status"], result["updates_exact_50_epochs"], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
