"""Recompute the fixed original QA cohort in production H16 layout; zero fits."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

import numpy as np
import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t.coordination import shared_write_admission, verified_ack
from e_jepa_ttc.simplex_t.expanded_inference import expanded_inference_family
from e_jepa_ttc.simplex_t.h16_qa_execution import execute_h16_qa
from e_jepa_ttc.simplex_t.h16_qa_plan import plan_h16_replay_qa
from e_jepa_ttc.simplex_t.lifecycle import admitted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--preprocessing-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--other-reserved-bytes", type=int, required=True)
    args = parser.parse_args()
    if args.other_reserved_bytes < 0:
        raise ValueError("nonnegative outstanding output reservation required")
    paths_hash = sha256(args.local_paths)
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    work = Path(paths["worktree"]).resolve(strict=True)
    base = work / "artifacts/simplex_t"
    plan = plan_h16_replay_qa(work)
    identity = json.loads((base / "T1/context_features_fp32/IDENTITY.json").read_text("utf-8"))
    code_pins = {
        work / "src/e_jepa_ttc/simplex_t" / filename: identity[field]
        for filename, field in (
            ("expert_features.py", "extractor_sha256"),
            ("expert_phase.py", "expert_phase_sha256"),
            ("query_context_voxel.py", "voxel_sha256"),
            ("context_raw_union.py", "union_reader_sha256"),
        )
    }
    for filename in ("expanded_inference.py", "h16_qa_execution.py", "h16_qa_plan.py"):
        path = work / "src/e_jepa_ttc/simplex_t" / filename
        code_pins[path] = sha256(path)
    code_pins[Path(__file__).resolve()] = sha256(Path(__file__))
    ack_path = Path(paths["shared_coordination"]) / "SIMPLEX_T_STAGE70_ACK.json"
    ack_hash = "3e55ab3c6e9a57eecd862ad05e999627ea90957e58e329b2eb3652120e953318"
    ack = verified_ack(ack_path, ack_hash)
    ancestry_ref = ack["producers"]["authoritative_historical_manifest"]
    ancestry = json.loads(Path(ancestry_ref["path"]).read_text("utf-8"))
    checkpoints = {
        item["sha256"]: Path(item["path"]) for item in ancestry["input_bindings"].values()
    }
    manifest = json.loads((base / "T1/query_context_index/INDEX_MANIFEST.json").read_text("utf-8"))
    if manifest["ancestry"] != ancestry_ref:
        raise ValueError("original QA producer ancestry differs")
    with np.load(
        base / "T1/query_context_index/query_context_index.npz", allow_pickle=False
    ) as data:
        index = dict(data)
    histories = []
    dedup_root = base / "T1/query_context_dedup"
    dedup = json.loads((dedup_root / "DEDUP_MANIFEST.json").read_text("utf-8"))
    for outer, record in enumerate(dedup["outputs"]):
        path = dedup_root / f"outer{outer}.npz"
        if sha256(path) != record["sha256"]:
            raise ValueError("QA history index changed")
        with np.load(path, allow_pickle=False) as data:
            histories.append(data["history"])

    def validate() -> None:
        verified_ack(ack_path, ack_hash)
        if sha256(args.local_paths) != paths_hash:
            raise ValueError("local configuration changed")
        if sha256(args.preprocessing_manifest) != identity["preprocessing_sha256"]:
            raise ValueError("frozen preprocessing changed")
        for path, digest in code_pins.items():
            if sha256(path) != digest:
                raise ValueError("H16 QA code changed")
        if str(torch.__version__) != identity["torch"]:
            raise ValueError("H16 QA Torch runtime differs")

    validate()
    prep = json.loads(args.preprocessing_manifest.read_text("utf-8"))["config"]
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False

    def factory(family: int) -> AbstractContextManager[Callable[[int], dict[str, np.ndarray]]]:
        if torch.cuda.get_device_name(0) != plan["expected_gpu_name"]:
            raise ValueError("same-device H16 QA required")
        return expanded_inference_family(
            family,
            families=manifest["families"],
            checkpoint_paths=checkpoints,
            index=index,
            history=histories[family // 4],
            raw_train_root=Path(paths["eap_root"]) / "data/train",
            allowed_sequences=set(map(str, index["sequences"])),
            preprocessing=prep,
            validate_prerequisites=validate,
            producer_scope="original_qa",
        )

    def resource_ok() -> bool:
        snapshot = admitted([work])
        return snapshot["has_headroom"] and shared_write_admission(
            snapshot["written_volume_free_bytes"][0], args.other_reserved_bytes + 67_108_864
        )

    result = execute_h16_qa(
        work,
        args.output,
        family_factory=factory,
        validate_prerequisites=validate,
        resource_ok=resource_ok,
        execution_identity={
            "code": {str(path.relative_to(work)): digest for path, digest in code_pins.items()},
            "local_paths_sha256": paths_hash,
            "ack_sha256": ack_hash,
            "preprocessing_sha256": identity["preprocessing_sha256"],
            "device": plan["expected_gpu_name"],
            "torch": str(torch.__version__),
        },
    )
    print(json.dumps({"status": result["status"], "completed": result["completed"]}))
    if result["status"] != "SAME_LAYOUT_H16_EXACT_PASS":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
