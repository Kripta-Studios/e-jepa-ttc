"""Resumable frozen FP32 extraction of the authorized query-context amendment."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.simplex_t_preflight import write_new_json
from e_jepa_ttc.data.eap import EAPEventReader
from e_jepa_ttc.evaluation.scientific_recovery_v8 import load_causal_scale_replay_checkpoint
from e_jepa_ttc.simplex_t.context_raw_union import encode_context_union
from e_jepa_ttc.simplex_t.coordination import shared_write_admission, verified_ack
from e_jepa_ttc.simplex_t.expert_features import extract_family
from e_jepa_ttc.simplex_t.lifecycle import ExclusiveLease, admitted
from e_jepa_ttc.training.stage61_pair_head import load_pair_head


def main() -> None:
    """Persist one fixed16-slot query block; never resume a partially written block."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--dedup", type=Path, required=True)
    parser.add_argument("--preprocessing-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-queries", type=int, required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    args = parser.parse_args()
    if args.max_new_queries < 1 or args.other_reserved_bytes < 0:
        raise ValueError("invalid execution slice or reservation")
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    coordination_path = Path("configs/experiment/simplex_t_coordination.json")
    coordination = json.loads(coordination_path.read_text(encoding="utf-8"))
    if not coordination["exclusive_inference_granted"]:
        raise ValueError("exclusive inference handoff not granted")
    ack = verified_ack(
        Path(paths["shared_coordination"]) / coordination["ack_filename"],
        coordination["ack_sha256"],
    )
    ancestry_ref = ack["producers"]["authoritative_historical_manifest"]
    ancestry = json.loads(Path(ancestry_ref["path"]).read_text(encoding="utf-8"))
    checkpoint_paths = {r["sha256"]: Path(r["path"]) for r in ancestry["input_bindings"].values()}
    manifest_path = args.index / "INDEX_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["ancestry"] != ancestry_ref or manifest["queries"] != 8192:
        raise ValueError("index lineage mismatch")
    array_path = args.index / "query_context_index.npz"
    if compute_file_hash(str(array_path)) != manifest["index_sha256"]:
        raise ValueError("index content mismatch")
    prep_sha = compute_file_hash(str(args.preprocessing_manifest))
    if prep_sha != "063980fdae5fda0b2836befc662fdd1cd5659bf06f10d9760dfc0d566fac8e39":
        raise ValueError("preprocessing changed")
    prep = json.loads(args.preprocessing_manifest.read_text(encoding="utf-8"))["config"]
    dedup_manifest = json.loads((args.dedup / "DEDUP_MANIFEST.json").read_text(encoding="utf-8"))
    if dedup_manifest["input_manifest_sha256"] != compute_file_hash(str(manifest_path)):
        raise ValueError("dedup input mismatch")
    with np.load(array_path, allow_pickle=False) as archive:
        index = {name: archive[name] for name in archive.files}
    args.output.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema": "simplex_t_query_context_fp32_cache_v1",
        "index_sha256": manifest["index_sha256"],
        "preprocessing_sha256": prep_sha,
        "extractor_sha256": compute_file_hash("src/e_jepa_ttc/simplex_t/expert_features.py"),
        "expert_phase_sha256": compute_file_hash("src/e_jepa_ttc/simplex_t/expert_phase.py"),
        "voxel_sha256": compute_file_hash("src/e_jepa_ttc/simplex_t/query_context_voxel.py"),
        "union_reader_sha256": compute_file_hash("src/e_jepa_ttc/simplex_t/context_raw_union.py"),
        "runner_sha256": compute_file_hash(__file__),
        "torch": str(torch.__version__),
        "batch_size": 16,
        "layout": "one query, chronological H16; absent slots zero, never exported",
        "precision": "FP32",
        "tf32": False,
        "optimizer_updates": 0,
    }
    identity_path = args.output / "IDENTITY.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise ValueError("cache runtime/source changed; no silent resume")
    else:
        write_new_json(identity_path, identity)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    started = time.perf_counter()
    completed = 0
    with ExclusiveLease(args.output.parent / "CURRENT_REPLAY.lock"):
        for family_id, family in enumerate(manifest["families"]):
            outer = family["outer_fold"]
            dedup_record = dedup_manifest["outputs"][outer]
            dedup_path = args.dedup / dedup_record["path"]
            if compute_file_hash(str(dedup_path)) != dedup_record["sha256"]:
                raise ValueError("dedup content changed")
            with np.load(dedup_path, allow_pickle=False) as archive:
                history = archive["history"]
            selected = np.flatnonzero(index["producer_family"][outer] == family_id)
            pending = []
            for qi in selected:
                stem = args.output / f"family{family_id:02d}_query{qi:05d}"
                receipt = stem.with_suffix(".json")
                if receipt.exists():
                    saved = json.loads(receipt.read_text(encoding="utf-8"))
                    if compute_file_hash(str(stem.with_suffix(".npz"))) != saved["sha256"]:
                        raise ValueError("completed cache block changed")
                else:
                    pending.append((int(qi), stem))
            if not pending:
                continue
            resources = admitted([args.output])
            if not resources["has_headroom"] or not shared_write_admission(
                resources["written_volume_free_bytes"][0], args.other_reserved_bytes + 2_000_000_000
            ):
                raise RuntimeError("RESOURCE_PAUSE before model load")
            hashes = family["experts"]
            for digest in hashes.values():
                if compute_file_hash(str(checkpoint_paths[digest])) != digest:
                    raise ValueError("checkpoint bytes changed")
            a5 = load_causal_scale_replay_checkpoint(checkpoint_paths[hashes["A5"]], device=device)
            c2f = load_causal_scale_replay_checkpoint(
                checkpoint_paths[hashes["C2F"]], device=device
            )
            pair = load_pair_head(checkpoint_paths[hashes["PAIR"]], device=device)
            for qi, stem in pending:
                resources = admitted([args.output])
                if not resources["has_headroom"]:
                    raise RuntimeError("RESOURCE_PAUSE at completed query boundary")
                sequence = str(index["sequences"][qi])
                raw_root = (Path(paths["eap_root"]) / "data/train").resolve()
                raw_path = (raw_root / sequence / "events.h5").resolve(strict=True)
                if not raw_path.is_relative_to(raw_root):
                    raise ValueError("raw path escapes TRAIN")
                mask = index["valid"][qi]
                windows = index["base_windows_us"][qi]
                with EAPEventReader(raw_path) as reader:
                    tensor = encode_context_union(
                        reader,
                        windows,
                        index["lag_us"],
                        mask,
                        tuple(index["square_xyxy"][qi]),
                        sequence_id=sequence,
                        roi_size=prep["roi_size"],
                        event_pixel_diff=prep["event_pixel_diff"],
                    )
                delta = torch.tensor(np.diff(windows[:, 1]) / 1e6, dtype=torch.float32)
                arrays = extract_family(
                    a5, c2f, pair, tensor.to(device), delta.repeat(16, 1).to(device)
                )
                output = stem.with_suffix(".npz")
                temporary = stem.with_suffix(".partial")
                if output.exists() or temporary.exists():
                    raise FileExistsError("unreceipted block retained for recovery audit")
                with temporary.open("xb") as stream:
                    np.savez_compressed(
                        stream,
                        **{k: v[mask] for k, v in arrays.items()},  # pyright: ignore[reportArgumentType]
                        observation_ids=history[qi, mask],
                        anchor_us=index["anchor_us"][qi] - index["lag_us"][mask],
                        available_us=np.full(int(mask.sum()), index["roi_available_us"][qi]),
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.rename(output)
                write_new_json(
                    stem.with_suffix(".json"),
                    {
                        "sha256": compute_file_hash(str(output)),
                        "query": qi,
                        "family": family_id,
                        "rows": int(mask.sum()),
                        "resources": admitted([args.output]),
                        "seconds_since_launch": time.perf_counter() - started,
                    },
                )
                completed += 1
                print(
                    json.dumps({"query": qi, "family": family_id, "new_blocks": completed}),
                    flush=True,
                )
                if completed >= args.max_new_queries:
                    return
            del a5, c2f, pair
            gc.collect()


if __name__ == "__main__":
    main()
