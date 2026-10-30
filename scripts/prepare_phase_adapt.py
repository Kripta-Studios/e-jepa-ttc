"""Extract all compatible local EvTTC head states and verified TRAIN40 OOF states."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import torch
import yaml

from e_jepa_ttc.simplex_t.phase import ttc_to_phase
from operational.evttc_transfer_audit.common import canonical_sha, file_sha, write_receipt
from operational.evttc_transfer_audit.provenance import assert_sources_unchanged, snapshot_sources
from operational.phase_adapt import ROLE, compatible_target, validate_population
from scripts.final_train40_transfer import HEAD, FinalTransferPredictor, verify_candidate
from scripts.final_train40_transfer_inputs import sequence_population


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_npz(path: Path, values: dict[str, np.ndarray]) -> None:
    """Atomically commit a new feature shard, without replacing prior artifacts."""
    if path.exists():
        raise ValueError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def restore_train40(candidate: Path, output: Path) -> dict[str, np.ndarray]:
    """Validate archived OOF bytes without rewriting the original remote-path receipt."""
    preservation = candidate.parent / "LOCAL_FINAL_PRESERVATION_COMPLETE.json"
    entry = read(preservation)["groups"]["final_generated"]
    archive = Path(entry["archive"])
    if file_sha(archive) != entry["sha256"]:
        raise ValueError("TRAIN40 preservation archive hash mismatch")
    prefix = "workspace/ejepa_final_runs/final/cache/PHASE_ONLY_EVENT_S7_TRAIN40"
    with tarfile.open(archive) as source:
        payload = source.extractfile(prefix + ".npz").read()
        receipt = json.loads(source.extractfile(prefix + ".json").read())
    unsigned = {k: v for k, v in receipt.items() if k != "receipt_payload_sha256"}
    if (
        canonical_sha(unsigned) != receipt["receipt_payload_sha256"]
        or hashlib.sha256(payload).hexdigest() != receipt["output_sha256"]
        or receipt["role"] != "TRAIN"
        or receipt["status"] != "COMPLETE"
        or receipt["training_feature_domain"] != "FIVE_FOLD_PHYSICAL_OOF"
        or receipt["targets_used_for_feature_construction"] is not False
    ):
        raise ValueError("TRAIN40 OOF cache contract differs")
    with np.load(io.BytesIO(payload), allow_pickle=False) as stored:
        values = {k: stored[k] for k in stored.files}
    if len(values["features"]) != receipt["rows"] or set(values["role"].tolist()) != {"TRAIN"}:
        raise ValueError("TRAIN40 cache population differs")
    write_receipt(output / "TRAIN40_ORIGINAL_RECEIPT.json", receipt)
    write_receipt(
        output / "TRAIN40_RESTORE.json",
        {
            "archive_sha256": entry["sha256"],
            "preservation_sha256": file_sha(preservation),
            "member": prefix + ".npz",
            "npz_sha256": receipt["output_sha256"],
            "feature_domain": "original_five_fold_oof_not_deployment_features",
        },
    )
    return values


def prepare(args: argparse.Namespace) -> None:
    """Build resumable sensor-only shards before joining supervised local targets."""
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    paths, candidate = verify_candidate(args.candidate)
    sources = snapshot_sources(ROOT)
    for name in ("scripts/prepare_phase_adapt.py", "scripts/final_train40_transfer_inputs.py"):
        sources[name] = file_sha(ROOT / name)
    binding = {
        "candidate": candidate,
        "sources": sources,
        "manifest_sha256": file_sha(args.manifest),
        "role": ROLE,
        "policy": "all_native_annotated_anchors_bounded20ms_labels_v1",
    }
    write_receipt(args.output / "BINDING.json", binding)
    specs = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))["sequences"]
    rows, excluded, inventory = [], [], []
    for spec in sorted(specs, key=lambda item: item["sequence_id"]):
        result = sequence_population(spec, args.data_root, limit=None)
        rows.extend(result["rows"])
        excluded.extend(result["excluded"])
        inventory.append(result.get("inventory", {"sequence_id": spec["sequence_id"]}))
    for row in rows:
        row["strata"] = ["ALL_ELIGIBLE_NATIVE_ANCHORS"]
        row.pop("metadata_sha256", None)
        row["metadata_sha256"] = canonical_sha(row)
    write_receipt(
        args.output / "POPULATION.json",
        {
            "rows": rows,
            "inventory": inventory,
            "sensor_excluded": excluded,
            "role": ROLE,
            "independence": "not_certified",
        },
    )
    print(f"Inventory: {len(specs)} recordings, {len(rows)} eligible anchors", flush=True)
    if args.inventory_only:
        return
    source_values = restore_train40(args.candidate, args.output)
    predictor = FinalTransferPredictor(paths, args.device)
    captured: dict[str, np.ndarray] = {}

    def capture(_module: Any, inputs: Any) -> None:
        captured["features"] = inputs[0].detach().cpu().numpy().copy()
        captured["rgb_width_logits"] = inputs[1].detach().cpu().numpy().copy()

    hook = predictor.models[HEAD].register_forward_pre_hook(capture)
    features, widths, truths, groups, tokens = [], [], [], [], []
    ledger, label_hashes, tables = [], {}, {}
    start = time.perf_counter()
    try:
        for index, row in enumerate(rows):
            key = canonical_sha({"row": row, "binding": binding})
            destination = args.output / "shards" / f"{key}.npz"
            receipt_path = destination.with_suffix(".json")
            if receipt_path.exists():
                receipt = read(receipt_path)
                if receipt["key"] != key or (
                    receipt["feature_sha256"] is not None
                    and file_sha(destination) != receipt["feature_sha256"]
                ):
                    raise ValueError("Cached shard changed")
            else:
                captured.clear()
                sensor_row = {k: v for k, v in row.items() if k != "ttc_csv_path"}
                prediction = predictor.predict(sensor_row)
                if captured:
                    save_npz(destination, captured)
                receipt = {
                    "key": key,
                    "query_id": row["query_id"],
                    "feature_sha256": file_sha(destination) if captured else None,
                    "prediction": prediction,
                    "targets_read_during_inference": False,
                }
                write_receipt(receipt_path, receipt)
            label_path = Path(row["ttc_csv_path"])
            if str(label_path) not in tables:
                label_hashes[str(label_path)] = file_sha(label_path)
                tables[str(label_path)] = np.loadtxt(label_path, ndmin=2)
            target = compatible_target(tables[str(label_path)], row["anchor_us"])
            reason = "used" if target is not None else "unsupported_local_target"
            if receipt["feature_sha256"] is None:
                reason = "no_fusion_state"
            ledger.append(
                {
                    "query_id": row["query_id"],
                    "group": row["sequence_id"],
                    "target_ttc": target,
                    "status": reason,
                    "shard": destination.name,
                }
            )
            if reason == "used":
                with np.load(destination, allow_pickle=False) as shard:
                    features.append(shard["features"][0])
                    widths.append(shard["rgb_width_logits"][0])
                truths.append(target)
                groups.append(row["sequence_id"])
                tokens.append("evttc:" + row["query_id"])
            if index % 20 == 0 or index + 1 == len(rows):
                print(
                    f"Extracted {index + 1}/{len(rows)}, usable={len(truths)}, "
                    f"elapsed={time.perf_counter() - start:.1f}s",
                    flush=True,
                )
    finally:
        hook.remove()
    if not features:
        raise ValueError("No EvTTC fusion training states")
    n_source = len(source_values["features"])
    all_tokens = ["eap:" + str(t) for t in source_values["sample_token"]] + tokens
    validate_population(all_tokens, [ROLE] * len(all_tokens))
    merged = {
        "features": np.concatenate([source_values["features"], np.asarray(features)]),
        "rgb_width_logits": np.concatenate([source_values["rgb_width_logits"], np.asarray(widths)]),
        "target_phase": np.concatenate(
            [
                source_values["target_phase"],
                ttc_to_phase(torch.tensor(truths, dtype=torch.float32)).numpy(),
            ]
        ),
        "group_id": np.asarray(["eap:" + str(g) for g in source_values["group_id"]] + groups),
        "domain": np.asarray(["eap"] * n_source + ["evttc"] * len(features)),
        "sample_token": np.asarray(all_tokens),
        "role": np.asarray([ROLE] * len(all_tokens)),
    }
    for key in ("features", "rgb_width_logits", "target_phase"):
        if merged[key].dtype != np.float32 or not np.isfinite(merged[key]).all():
            raise ValueError(f"Nonfinite or non-FP32 {key}")
    assert_sources_unchanged(ROOT, sources)
    if verify_candidate(args.candidate)[1] != candidate:
        raise ValueError("Parent candidate changed")
    output = args.output / "training.npz"
    if not output.exists():
        save_npz(output, merged)
    else:
        with np.load(output, allow_pickle=False) as previous:
            if set(previous.files) != set(merged) or any(
                not np.array_equal(previous[k], v) for k, v in merged.items()
            ):
                raise ValueError("Existing merged cache differs")
    write_receipt(
        args.output / "TARGET_LEDGER.json", {"rows": ledger, "label_sha256": label_hashes}
    )
    write_receipt(
        args.output / "COMPLETE.json",
        {
            "status": "COMPLETE",
            "role": ROLE,
            "rows": len(all_tokens),
            "eap_rows": n_source,
            "evttc_rows": len(features),
            "evttc_sensor_eligible": len(rows),
            "evttc_recordings": len(set(groups)),
            "training_npz_sha256": file_sha(output),
            "binding_sha256": file_sha(args.output / "BINDING.json"),
            "population_sha256": file_sha(args.output / "POPULATION.json"),
            "targets_sha256": file_sha(args.output / "TARGET_LEDGER.json"),
            "train40_restore_sha256": file_sha(args.output / "TRAIN40_RESTORE.json"),
            "official_equivalence": "unverified",
            "independent_test": False,
        },
    )
    print(f"COMPLETE: {len(all_tokens)} training examples", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("candidate", "manifest", "data-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--inventory-only", action="store_true")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
