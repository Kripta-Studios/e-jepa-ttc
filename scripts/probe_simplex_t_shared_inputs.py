"""Real input-reuse QA against D0 blocks; not the mandatory fixed H16 cohort."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.compiled_context import validate_block
from e_jepa_ttc.simplex_t.coordination import shared_write_admission, verified_ack
from e_jepa_ttc.simplex_t.expanded_inference import expanded_inference_family
from e_jepa_ttc.simplex_t.h16_qa_plan import plan_h16_replay_qa, require_exact_h16_arrays
from e_jepa_ttc.simplex_t.lifecycle import ExclusiveLease, admitted
from e_jepa_ttc.simplex_t.prepared_query_input import PreparedQueryInput


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.other_reserved_bytes < 0 or args.output.exists():
        raise ValueError("new output and nonnegative outstanding reservations required")
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    work = Path(paths["worktree"]).resolve(strict=True)
    if not args.output.resolve().is_relative_to(work / "artifacts/simplex_t/T0"):
        raise ValueError("QA output must stay under companion T0")
    base = work / "artifacts/simplex_t/T1"
    expected_gpu = plan_h16_replay_qa(work)["expected_gpu_name"]
    ack_path = Path(paths["shared_coordination"]) / "SIMPLEX_T_STAGE70_ACK.json"
    ack_hash = "3e55ab3c6e9a57eecd862ad05e999627ea90957e58e329b2eb3652120e953318"
    ack = verified_ack(ack_path, ack_hash)
    ancestry_ref = ack["producers"]["authoritative_historical_manifest"]
    ancestry = Path(ancestry_ref["path"])
    index_path = base / "query_context_index/query_context_index.npz"
    manifest_path = base / "query_context_index/INDEX_MANIFEST.json"
    prep_path = (
        work.parent
        / "e-jepa-ttc/artifacts/cache/garl_object_event_common_roi_train8192_v1/manifest.json"
    )
    pins = {
        index_path: "0fe7d7bb597dc768073a2940795442417ab6dba0be6110cd356d3ff0b9f656bf",
        manifest_path: "93a4f62e5025c5046fc82fcb1428a428f8a8df869b34486b92d5c753b0f68a3b",
        prep_path: "063980fdae5fda0b2836befc662fdd1cd5659bf06f10d9760dfc0d566fac8e39",
        ancestry: ancestry_ref["sha256"],
        args.local_paths.resolve(): sha256(args.local_paths),
        base / "context_features_fp32/IDENTITY.json": (
            "00097c6e78bff173f9fe1497aae49da93f6ad7e86e678fe714da4269219955b1"
        ),
        base / "query_context_dedup/DEDUP_MANIFEST.json": (
            "8f2bbfbacbb762a39d6d329dbe3f135fd58f6ac454e7117962c3e84edb60c1a3"
        ),
    }
    numerical = json.loads((base / "context_features_fp32/IDENTITY.json").read_text("utf-8"))
    for name, field in (
        ("expert_features.py", "extractor_sha256"),
        ("expert_phase.py", "expert_phase_sha256"),
        ("query_context_voxel.py", "voxel_sha256"),
        ("context_raw_union.py", "union_reader_sha256"),
    ):
        pins[work / "src/e_jepa_ttc/simplex_t" / name] = numerical[field]
    for name in ("expanded_inference.py", "prepared_query_input.py", "cached_event_reader.py"):
        path = work / "src/e_jepa_ttc/simplex_t" / name
        pins[path] = sha256(path)
    pins[Path(__file__).resolve()] = sha256(Path(__file__))

    def validate() -> None:
        if any(sha256(path) != expected for path, expected in pins.items()):
            raise ValueError("input reuse QA pin changed")
        verified_ack(ack_path, ack_hash)

    validate()
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest["ancestry"] != ancestry_ref:
        raise ValueError("historical producer ancestry changed")
    with np.load(index_path, allow_pickle=False) as archive:
        index = dict(archive)
    dedup = json.loads((base / "query_context_dedup/DEDUP_MANIFEST.json").read_text("utf-8"))
    histories = []
    for outer, record in enumerate(dedup["outputs"]):
        path = base / f"query_context_dedup/outer{outer}.npz"
        pins[path] = record["sha256"]
        if sha256(path) != record["sha256"]:
            raise ValueError("history binding changed")
        with np.load(path, allow_pickle=False) as archive:
            histories.append(archive["history"])
    # Fixed input-index medians of the first six sequences, selected before scores.
    queries = (455, 1366, 2277, 3187, 4097, 5007)
    references = {}
    for query in queries:
        for outer in range(3):
            family = int(index["producer_family"][outer, query])
            stem = base / f"context_features_fp32/family{family:02d}_query{query:05d}"
            receipt = json.loads(stem.with_suffix(".json").read_text("utf-8"))
            if receipt["query"] != query or receipt["family"] != family:
                raise ValueError("reference family or query changed")
            pins[stem.with_suffix(".json")] = sha256(stem.with_suffix(".json"))
            pins[stem.with_suffix(".npz")] = receipt["sha256"]
            if sha256(stem.with_suffix(".npz")) != receipt["sha256"]:
                raise ValueError("reference output changed")
            with np.load(stem.with_suffix(".npz"), allow_pickle=False) as archive:
                arrays = dict(archive)
            mask = index["valid"][query]
            validate_block(
                arrays,
                histories[outer][query, mask],
                index["anchor_us"][query] - index["lag_us"][mask],
                int(index["roi_available_us"][query]),
            )
            references[query, family] = arrays
    if args.validate_only:
        print(json.dumps({"references_verified": len(references), "gpu_inference": False}))
        return
    rss_max = 0

    def resource_ok() -> None:
        nonlocal rss_max
        snapshot = admitted([work])
        rss_max = max(rss_max, snapshot["process_tree_rss_bytes"])
        if not snapshot["has_headroom"] or not shared_write_admission(
            snapshot["written_volume_free_bytes"][0], args.other_reserved_bytes + 67108864
        ):
            raise RuntimeError("RESOURCE_PAUSE: retain QA artifacts")

    with ExclusiveLease(base / "CURRENT_REPLAY.lock"), ExitStack() as residents:
        resource_ok()
        torch.set_num_threads(4)
        torch.set_num_interop_threads(2)
        torch.cuda.init()
        if torch.cuda.get_device_name(0) != expected_gpu:
            raise ValueError("same-device input reuse QA required")
        if str(torch.__version__) != numerical["torch"]:
            raise ValueError("Torch differs from D0")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats(0)
        checkpoints = {
            item["sha256"]: Path(item["path"])
            for item in json.loads(ancestry.read_text("utf-8"))["input_bindings"].values()
        }
        prep = json.loads(prep_path.read_text("utf-8"))["config"]
        shared = PreparedQueryInput()
        args.output.mkdir(parents=True)
        write_new_json(
            args.output / "CONTRACT.json",
            {
                "pins": {str(path): digest for path, digest in pins.items()},
                "queries": queries,
                "scope": "INPUT_REUSE_QA_NOT_MANDATORY_H16_ADMISSION",
                "optimizer_updates": 0,
            },
        )
        infer = {}
        for family in range(12):
            resource_ok()
            infer[family] = residents.enter_context(
                expanded_inference_family(
                    family,
                    families=manifest["families"],
                    checkpoint_paths=checkpoints,
                    index=index,
                    history=histories[family // 4],
                    raw_train_root=Path(paths["eap_root"]) / "data/train",
                    allowed_sequences=set(ack["interfaces"]["role_manifest"]["roles"]["original"]),
                    preprocessing=prep,
                    validate_prerequisites=validate,
                    producer_scope="original_qa",
                    prepared_inputs=shared,
                )
            )
        results = []
        for mode in ("baseline", "query_major"):
            shared.clear()
            preparations, hits = shared.preparations, shared.hits
            started = time.perf_counter()
            order = sorted(
                references, key=(lambda item: (item[1], item[0])) if mode == "baseline" else None
            )
            for query, family in order:
                resource_ok()
                if mode == "baseline":
                    shared.clear()
                observed = infer[family](query)
                with (args.output / f"{mode}_f{family}_q{query}.npz").open("xb") as stream:
                    np.savez_compressed(stream, **observed)
                require_exact_h16_arrays(references[query, family], observed)
            results.append(
                {
                    "mode": mode,
                    "seconds": time.perf_counter() - started,
                    "preparations": shared.preparations - preparations,
                    "hits": shared.hits - hits,
                }
            )
        write_new_json(
            args.output / "QA.json",
            {
                "status": "QUERY_MAJOR_INPUT_REUSE_EXACT_PASS",
                "results": results,
                "compared_blocks_per_pass": len(references),
                "sampled_rss_max_bytes": rss_max,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
                "gpu": torch.cuda.get_device_name(0),
                "optimizer_updates": 0,
                "scientific_admission": False,
                "limitation": "Baseline first; warm-cache order effect possible",
            },
        )


if __name__ == "__main__":
    main()
