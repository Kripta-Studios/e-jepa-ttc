"""Compare saved current observations across QA and H16 batch layouts, without inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.compiled_context import validate_block


def difference(left: np.ndarray, right: np.ndarray) -> dict:
    """Describe differences without selecting or applying a numerical tolerance."""
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("array schemas differ")
    finite = np.isfinite(left) & np.isfinite(right)
    delta = np.abs(left[finite].astype(np.float64) - right[finite].astype(np.float64))
    return {
        "changed_values": int(np.count_nonzero(left != right)),
        "finite_pattern_changes": int(np.count_nonzero(np.isfinite(left) != np.isfinite(right))),
        "max_abs_joint_finite": float(delta.max()) if delta.size else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    work = args.worktree.resolve(strict=True)
    if args.output.exists() or not args.output.resolve().is_relative_to(work):
        raise ValueError("new report inside companion required")
    base = work / "artifacts/simplex_t"
    cache, index_root = base / "T1/context_features_fp32", base / "T1/query_context_index"
    qa_root = base / "T0/coherent_fp32_extractor_receipt"
    dedup_root = base / "T1/query_context_dedup"
    pins = {
        qa_root / "QA.json": "5801fee3e03f967b471344a72bdbaa9830679a1e0d71539478fc578c5a1694de",
        cache / "IDENTITY.json": "00097c6e78bff173f9fe1497aae49da93f6ad7e86e678fe714da4269219955b1",
        index_root
        / "INDEX_MANIFEST.json": "93a4f62e5025c5046fc82fcb1428a428f8a8df869b34486b92d5c753b0f68a3b",
    }
    pins[dedup_root / "DEDUP_MANIFEST.json"] = (
        "8f2bbfbacbb762a39d6d329dbe3f135fd58f6ac454e7117962c3e84edb60c1a3"
    )
    for path, digest in pins.items():
        if sha256(path) != digest:
            raise ValueError("diagnostic input binding changed")
    manifest = json.loads((index_root / "INDEX_MANIFEST.json").read_text(encoding="utf-8"))
    if sha256(index_root / "query_context_index.npz") != manifest["index_sha256"]:
        raise ValueError("query index changed")
    with np.load(index_root / "query_context_index.npz", allow_pickle=False) as archive:
        index = {key: archive[key] for key in archive.files}
    positions = {str(token): row for row, token in enumerate(index["tokens"])}
    dedup_manifest = json.loads((dedup_root / "DEDUP_MANIFEST.json").read_text(encoding="utf-8"))
    histories = {}
    for outer, record in enumerate(dedup_manifest["outputs"]):
        path = dedup_root / f"outer{outer}.npz"
        if record["path"] != path.name or sha256(path) != record["sha256"]:
            raise ValueError("deduplicated observation identities changed")
        with np.load(path, allow_pickle=False) as archive:
            histories[outer] = archive["history"]
    qa = json.loads((qa_root / "QA.json").read_text(encoding="utf-8"))
    results, missing = [], []
    for family_record in qa["results"]:
        outer_name, role = family_record["family"].split("/")
        outer = int(outer_name[-1])
        family = 4 * outer + (3 if role == "outer_dev" else int(role[-1]))
        if family_record["checkpoint_sha256"] != manifest["families"][family]["experts"]:
            raise ValueError("current-point comparison changes producer family")
        old_path = qa_root / (family_record["family"].replace("/", "_") + ".npz")
        if sha256(old_path) != family_record["output_sha256"]:
            raise ValueError("coherent reference bytes changed")
        with np.load(old_path, allow_pickle=False) as old:
            for row, token in enumerate(old["tokens"]):
                query = positions[str(token)]
                if index["producer_family"][outer, query] != family:
                    raise ValueError("QA query assigned to another producer")
                stem = cache / f"family{family:02d}_query{query:05d}"
                receipt = stem.with_suffix(".json")
                if not receipt.exists():
                    missing.append(dict(token=str(token), family=family, query=query))
                    continue
                record = json.loads(receipt.read_text(encoding="utf-8"))
                payload = stem.with_suffix(".npz")
                if (
                    record["query"] != query
                    or record["family"] != family
                    or sha256(payload) != record["sha256"]
                ):
                    raise ValueError("completed H16 receipt or payload changed")
                with np.load(payload, allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in archive.files}
                mask = index["valid"][query]
                # Check IDs against the independent content index, plus times/ROI availability.
                validate_block(
                    arrays,
                    histories[outer][query, mask],
                    index["anchor_us"][query] - index["lag_us"][mask],
                    int(index["roi_available_us"][query]),
                )
                comparisons = {
                    key: difference(arrays[key][-1], old[key][row])
                    for key in ("features145", "expert_ttc", "known", "pair_features")
                }
                results.append(
                    dict(
                        token=str(token),
                        family=family,
                        query=query,
                        receipt_sha256=sha256(receipt),
                        payload_sha256=record["sha256"],
                        comparisons=comparisons,
                    )
                )
    write_new_json(
        args.output,
        {
            "status": "SAVED_CURRENT_LAYOUT_DIAGNOSTIC_NOT_H16_PARITY_PASS",
            "compared": len(results),
            "missing": missing,
            "results": results,
            "input_pins": {str(path.relative_to(work)): digest for path, digest in pins.items()},
            "expert_forwards": 0,
            "optimizer_updates": 0,
            "tolerance_selected": False,
            "script_sha256": sha256(Path(__file__)),
            "limitation": (
                "Different batch layouts; not independent H16 recomputation "
                "or a same-layout exact-parity claim."
            ),
        },
    )
    print(
        json.dumps(
            {
                "compared": len(results),
                "missing": len(missing),
                "queries_with_any_difference": sum(
                    any(c["changed_values"] for c in r["comparisons"].values()) for r in results
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
