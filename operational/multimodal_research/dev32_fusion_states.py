"""Recover fusion states from sealed RGB histories and native event replay, without TTC."""

from __future__ import annotations

import hashlib
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np
import torch

from e_jepa_ttc.rgb_port.fusion import DualClockFusion
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.multimodal_validation.ancestry import historical_source_binding
from operational.multimodal_validation.common import ROOT, RUN, V12, admission, read, sha, write
from operational.rgb_port import transfer as engine
from operational.rgb_port.infer_experts import Fragments
from operational.rgb_port_external_fix.runtime import observation_contract
from operational.rgb_port_fusion_fix.fusion import deterministic_forward

from .common import OUT
from .model import frozen_state


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    manifest = V12 / "artifacts/evttc_transfer_20261008/QUERY_MANIFEST.json"
    rows = read(manifest)["rows"]
    previous = RUN / "transfer/dev32"
    native = read(previous / "PREDICTIONS_SEALED.json")
    assert native["device"] == "cpu" and native["query_manifest_sha256"] == sha(manifest)
    fragment_root = previous / "fragments"
    assert (fragment_root / "BINDING.json").is_file()
    fragments = Fragments(fragment_root, read(fragment_root / "BINDING.json"))
    assert fragments.digest == native["fragment_binding_sha256"]
    assert engine._fragment_inventory(fragments, len(rows)) == native["fragment_inventory_sha256"]
    base = OUT / "external_states/DEV32"
    base_seal = read(base / "SEALED.json")
    assert sha(base / "BINDING.json") == base_seal["binding_sha256"]
    destination = OUT / "external_fusion_states/DEV32"
    destination.mkdir(parents=True, exist_ok=True)
    binding: dict = dict(
        source_sha256=sha(__file__),
        historical_seal_sha256=sha(previous / "PREDICTIONS_SEALED.json"),
        base_state_seal_sha256=sha(base / "SEALED.json"),
        manifest_sha256=sha(manifest),
        parent_freeze_sha256=sha(RUN / "rounds/PRODUCERS_PAIR_ENDPOINT_FREEZE.json"),
        fusion_checkpoint_sha256=sha(RUN / "fits/F_TRUE/checkpoint_last.pt"),
        precision="CPU FP32 one thread; native events; sealed historical normalized RGB histories",
        schema="F_TRUE combine hidden160/raw_location/emitted_phase",
        targets_read=False,
        optimizer_updates=0,
    )
    binding_path = destination / "BINDING.json"
    if binding_path.exists():
        assert read(binding_path) == binding
    else:
        write(binding_path, binding)
    binding_sha = sha(binding_path)
    records = {}
    with ExitStack() as stack:
        stack.enter_context(historical_source_binding())
        stack.enter_context(observation_contract())
        stack.enter_context(patch.object(DualClockFusion, "forward", deterministic_forward))
        models = engine._models(
            RUN, ROOT / "configs/rgb_port/producers.json", RUN / "P_MANIFEST.json", "cpu"
        )
        for model in models.values():
            model.eval().requires_grad_(False)
        mean, scale = engine._normalizer(
            RUN / "features/H/E_NORMALIZER.json", "event", binding["parent_freeze_sha256"]
        )
        for index, row in enumerate(rows):
            admission()
            qid, query = row["query_id"], int(row["anchor_us"])
            key = hashlib.sha256(qid.encode()).hexdigest()
            path = destination / f"{key}.npz"
            receipt_path = path.with_suffix(".json")
            if receipt_path.exists():
                receipt = read(receipt_path)
                assert receipt["binding_sha256"] == binding_sha and sha(path) == receipt["sha256"]
                records[qid] = receipt
                continue
            saved = fragments.read(index)
            assert saved is not None
            historical, history = engine._restore_fragment(saved, row, torch.device("cpu"))
            assert historical["rgb_available"] and history
            assert len(engine._select_rgb_history(history, query)) == len(history)
            base_receipt = base_seal["queries"][qid]
            base_path = base / f"{key}.npz"
            assert sha(base_path) == base_receipt["sha256"]
            raw = engine.prepare_event(row)
            with torch.inference_mode():
                eraw, experts = engine._expert_block(
                    models,
                    "E",
                    torch.from_numpy(raw["own_events"]),
                    torch.full((len(engine.EVENT_LAGS_US), 2), 0.1),
                )
                ef = ((eraw - mean) / scale)[None]
                anchors = query - engine.EVENT_LAGS_US
                et = torch.from_numpy(engine._clock(anchors, np.full(len(anchors), query), query))[
                    None
                ]
                ev = torch.ones((1, len(anchors)), dtype=torch.bool)
                block, _, _ = frozen_state(models["E_CTX_MATCHED"], ef, et, ev, experts[-1:])
                with np.load(base_path, allow_pickle=False) as z:
                    assert np.array_equal(block.numpy(), z["event_block"]), qid
                rf = torch.stack([h[3] for h in history])[None]
                rt = torch.from_numpy(
                    engine._clock(
                        np.asarray([h[1] for h in history]),
                        np.asarray([h[2] for h in history]),
                        query,
                    )
                )[None]
                rv = torch.ones((1, len(history)), dtype=torch.bool)
                result = models["F_TRUE"](ef, et, ev, experts[-1:], rf, rt, rv)
                features = torch.cat(
                    [
                        result["hidden"],
                        result["raw_location"][:, None],
                        result["point_phase"][:, None],
                    ],
                    -1,
                ).numpy()
                prediction = float(phase_to_ttc(result["point_phase"])[0])
            temporary = path.with_suffix(".pending.npz")
            np.savez_compressed(temporary, features=features)
            temporary.replace(path)
            receipt = dict(
                query_id=qid,
                sequence_id=row["sequence_id"],
                anchor_us=query,
                binding_sha256=binding_sha,
                sha256=sha(path),
                event_state_exact=True,
                fusion_ttc=prediction,
                historical_fusion_ttc=historical["F_TRUE"],
                historical_delta_s=abs(prediction - historical["F_TRUE"]),
            )
            write(receipt_path, receipt)
            records[qid] = receipt
            print(index + 1, len(rows), qid, receipt["historical_delta_s"], flush=True)
    write(
        destination / "SEALED.json",
        dict(
            binding_sha256=binding_sha,
            population=len(rows),
            queries=records,
            targets_read=False,
            optimizer_updates=0,
            historical_max_delta_s=max(v["historical_delta_s"] for v in records.values()),
        ),
    )


if __name__ == "__main__":
    main()
