"""Replay the 64-row TRAIN cohort; export differences without declaring parity."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from e_jepa_ttc.artifacts.hashing import compute_file_hash
from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.data.collision_clock_cache import (
    CollisionClockTrain8192Cache,
    load_canonical_supervision,
)
from e_jepa_ttc.data.stage61_pair_feature_cache import PairFeatureBatch
from e_jepa_ttc.evaluation.scientific_recovery_v8 import load_causal_scale_replay_checkpoint
from e_jepa_ttc.simplex_t.coordination import verified_ack
from e_jepa_ttc.simplex_t.current_inputs import load_current_inputs
from e_jepa_ttc.simplex_t.lifecycle import ExclusiveLease, admitted
from e_jepa_ttc.training.stage61_pair_head import load_pair_head


def main() -> None:
    """Run signed frozen producers sequentially, without any optimizer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recover-inputs-sha256")
    parser.add_argument("--confirmed-user-replay-handoff", action="store_true", required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.output.exists() and not args.recover_inputs_sha256:
        raise FileExistsError("preserve prior replay evidence")
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    config = json.loads(Path("configs/experiment/simplex_t_coordination.json").read_text())
    ack = verified_ack(
        Path(paths["shared_coordination"]) / config["ack_filename"], config["ack_sha256"]
    )
    ancestry_ref = ack["producers"]["authoritative_historical_manifest"]
    ancestry = json.loads(Path(ancestry_ref["path"]).read_text())
    root = Path(ancestry_ref["path"]).parent
    allowed = set(ack["interfaces"]["role_manifest"]["roles"]["original"])
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    if cohort["ancestry"] != ancestry_ref or cohort["unique_rows"] != 64:
        raise ValueError("cohort ancestry/count mismatch")
    selection = cohort["selection"]
    tokens = [token for values in selection.values() for token in values]
    if len(tokens) != 64 or len(set(tokens)) != 64:
        raise ValueError("cohort must contain64 unique identities")
    checkpoints = {
        record["sha256"]: Path(record["path"])
        for record in ancestry["input_bindings"].values()
        if Path(record["path"]).suffix == ".pt"
    }
    producers = {(p["outer_fold"], p["role"], p["expert"]): p for p in ancestry["producers"]}
    args.output.mkdir(parents=True, exist_ok=bool(args.recover_inputs_sha256))
    started = time.perf_counter()

    def check_resources() -> dict[str, Any]:
        state = admitted([args.output])
        if not state["has_headroom"]:
            raise RuntimeError(f"RESOURCE_PAUSE:{state}")
        return state

    with ExclusiveLease(args.output.parent / "CURRENT_REPLAY.lock"):
        initial = check_resources()
        reference = json.loads(
            Path("configs/protocol/scientific_recovery_v9_eclock_x0_reference.json").read_text()
        )
        protocol = json.loads(
            Path("configs/protocol/scientific_recovery_v9_eclock_x0.json").read_text()
        )
        if args.recover_inputs_sha256:
            if compute_file_hash(str(args.output / "inputs.pt")) != args.recover_inputs_sha256:
                raise ValueError("recovered input hash mismatch")
            saved = torch.load(args.output / "inputs.pt", map_location="cpu", weights_only=True)
            if saved["tokens"] != tokens:
                raise ValueError("recovered input token order mismatch")
            inputs, delta = saved["inputs"], saved["delta"]
        else:
            adapter = CollisionClockTrain8192Cache(
                args.reference_root / "artifacts/cache/garl_object_event_common_roi_train8192_v1",
                protocol,
                cache_mode="direct",
                canonical_supervision=load_canonical_supervision(reference, args.reference_root),
            )
            print("Verifying signed TRAIN shards; no model or optimizer running", flush=True)
            locators = {item.sample_token: item for item in adapter.verify_and_index()}
            if any(locators[token].sequence_id not in allowed for token in tokens):
                raise ValueError("replay token outside authorized original TRAIN groups")
            inputs, delta, _ = adapter._materialize([locators[token] for token in tokens])
            torch.save(
                {"tokens": tokens, "inputs": inputs, "delta": delta}, args.output / "inputs.pt"
            )
            del adapter
        write_new_json(
            args.output / "INPUTS_READY.json",
            {
                "sha256": compute_file_hash(str(args.output / "inputs.pt")),
                "cohort_sha256": sha256(args.cohort),
                "recovered_after_metadata_hasher_size_error": bool(args.recover_inputs_sha256),
            },
        )
        gc.collect()
        device = torch.device("cuda")
        results = []
        offset = 0
        for family, chosen in selection.items():
            check_resources()
            outer_name, role = family.split("/")
            outer = int(outer_name.removeprefix("outer"))
            table = load_current_inputs(
                root,
                outer,
                "outer_dev" if role == "outer_dev" else "inner_oof",
                ancestry_sha256=ancestry_ref["sha256"],
                allowed_sequences=allowed,
            )
            metadata = table["metadata"].reset_index().set_index("sample_token")
            rows = metadata.loc[chosen, "index"].to_numpy()
            if (
                role != "outer_dev"
                and not (
                    metadata.loc[chosen, "inner_fold"] == int(role.removeprefix("inner"))
                ).all()
            ):
                raise ValueError("cohort selected wrong inner producer")
            x = inputs[offset : offset + len(chosen)].to(device)
            dt = delta[offset : offset + len(chosen)].to(device)
            offset += len(chosen)
            predicted = []
            variances = []
            pair_features = None
            hashes = {}
            for expert in ("A5", "C2F", "PAIR"):
                digest = producers[(outer, role, expert)]["checkpoint_sha256"]
                path = checkpoints[digest]
                if sha256(path) != digest:
                    raise ValueError("checkpoint bytes changed")
                hashes[expert] = digest
                if expert == "PAIR":
                    if pair_features is None:
                        raise ValueError("A5 features required before the PAIR producer")
                    model = load_pair_head(path, device=device)
                    with torch.inference_mode():
                        prediction = model.predict_ttc(PairFeatureBatch(pair_features))
                else:
                    model = load_causal_scale_replay_checkpoint(path, device=device)
                    with torch.inference_mode():
                        output = model(x, dt, return_dense_features=True)
                        prediction = output.ttc_mean_seconds
                        variances.append(output.ttc_log_variance.float().cpu().numpy())
                        if expert == "A5":
                            elapsed = dt[:, 1]
                            support = output.sensor_support
                            pair_features = torch.cat(
                                (
                                    output.pair_tokens[:, -1],
                                    torch.stack(
                                        (elapsed, torch.log(elapsed + 1e-8), elapsed.reciprocal()),
                                        -1,
                                    ),
                                    support[:, -1:],
                                    torch.minimum(support[:, -2], support[:, -1]).unsqueeze(-1),
                                ),
                                -1,
                            ).float()
                    del output
                predicted.append(prediction.float().cpu().numpy())
                del model, prediction
                gc.collect()
            actual = np.stack(predicted, axis=1)
            expected = table["arrays"]["expert_ttc"][rows]
            if pair_features is None:
                raise ValueError("completed A5 feature extraction required for replay output")
            np.savez_compressed(
                args.output / f"{outer_name}_{role}.npz",
                actual_ttc=actual,
                expected_ttc=expected,
                variance=np.stack(variances, axis=1),
                pair_features=pair_features.cpu().numpy(),
                tokens=np.asarray(chosen),
            )
            record = {
                "family": family,
                "rows": len(chosen),
                "checkpoint_sha256": hashes,
                "max_abs_ttc_difference": np.abs(actual - expected).max(axis=0).tolist(),
                "finite": bool(np.isfinite(actual).all()),
                "resources": check_resources(),
            }
            write_new_json(args.output / f"{outer_name}_{role}.json", record)
            results.append(record)
            print(json.dumps(record), flush=True)
            del x, dt, pair_features
        write_new_json(
            args.output / "REPLAY_DIAGNOSTIC.json",
            {
                "status": "REPLAY_EXECUTED_PARITY_REVIEW_REQUIRED",
                "results": results,
                "seconds": time.perf_counter() - started,
                "initial_resources": initial,
                "optimizer_updates": 0,
                "scientific_fit": False,
                "cohort_sha256": sha256(args.cohort),
                "torch": torch.__version__,
                "precision": "FP32_TF32_DISABLED",
                "device": torch.cuda.get_device_name(),
            },
        )


if __name__ == "__main__":
    main()
