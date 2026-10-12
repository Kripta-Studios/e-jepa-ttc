"""Sensor-only external states in the verified historical GPU precision mode."""

from __future__ import annotations

import argparse
import hashlib
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from e_jepa_ttc.rgb_port.fusion import DualClockFusion
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.fcwd_rgb_registration.transfer import prepare_event, rgb_observations
from operational.multimodal_validation.ancestry import historical_source_binding
from operational.multimodal_validation.common import (
    MANIFEST,
    REG,
    ROOT,
    RUN,
    V12,
    read,
    registrations,
    sha,
    write,
)
from operational.rgb_port import transfer as engine
from operational.rgb_port_external_fix.runtime import observation_contract
from operational.rgb_port_fusion_fix.fusion import deterministic_forward

from .common import OUT, guard, setup
from .model import frozen_state


def extract(cohort: str) -> None:
    assert cohort == "FCWD", "Dev32 historical inference used CPU; keep its native CPU route"
    manifest = (
        MANIFEST
        if cohort == "FCWD"
        else V12 / "artifacts/evttc_transfer_20261008/QUERY_MANIFEST.json"
    )
    target = OUT / "external_states_gpu_native" / cohort
    target.mkdir(parents=True, exist_ok=True)
    rows = read(manifest)["rows"]
    assert len({r["query_id"] for r in rows}) == len(rows)
    binding: dict = dict(
        cohort=cohort,
        population=len(rows),
        manifest_sha256=sha(manifest),
        source_sha256=sha(__file__),
        model_source_sha256=sha(Path(__file__).with_name("model.py")),
        parent_freeze_sha256=sha(RUN / "rounds/PRODUCERS_PAIR_ENDPOINT_FREEZE.json"),
        head_sha256={
            name: sha(RUN / f"fits/{name}/checkpoint_last.pt")
            for name in ("E_CTX_MATCHED", "R_CTX", "F_TRUE")
        },
        normalizer_sha256={m: sha(RUN / f"features/H/{m}_NORMALIZER.json") for m in ("E", "R")},
        registration_seal_sha256=sha(REG / "REGISTRATION_SEALED.json")
        if cohort == "FCWD"
        else None,
        device="cuda_matmulTF32_false_cudnnTF32_true_one_CPU_thread",
        targets_read=False,
        optimizer_updates=0,
        use="frozen inference only; no external model or hyperparameter selection",
        numerical_scope="Historical GPU producer precision; full parent replay checked separately",
    )
    binding_path = target / "BINDING.json"
    if binding_path.exists():
        assert read(binding_path) == binding
    else:
        write(binding_path, binding)
    binding_sha = sha(binding_path)
    regs = registrations() if cohort == "FCWD" else {}
    with ExitStack() as stack:
        stack.enter_context(historical_source_binding())
        stack.enter_context(observation_contract())
        stack.enter_context(patch.object(DualClockFusion, "forward", deterministic_forward))
        models = engine._models(
            RUN, ROOT / "configs/rgb_port/producers.json", RUN / "P_MANIFEST.json", "cuda"
        )
        for model in models.values():
            model.eval().requires_grad_(False)
        normalizers = {
            m: tuple(
                x.cuda()
                for x in engine._normalizer(
                    RUN / f"features/H/{m}_NORMALIZER.json",
                    modality,
                    binding["parent_freeze_sha256"],
                )
            )
            for m, modality in (("E", "event"), ("R", "rgb"))
        }
        receipts = {}
        for index, row in enumerate(rows):
            guard()
            key = hashlib.sha256(row["query_id"].encode()).hexdigest()
            path, receipt_path = target / f"{key}.npz", target / f"{key}.json"
            if receipt_path.exists():
                receipt = read(receipt_path)
                assert receipt["binding_sha256"] == binding_sha
                assert receipt["query_id"] == row["query_id"] and receipt["sha256"] == sha(path)
                receipts[row["query_id"]] = receipt
                continue
            raw = (prepare_event if cohort == "FCWD" else engine.prepare_event)(row)
            sensor = torch.from_numpy(raw["own_events"]).cuda()
            query = int(row["anchor_us"])
            with torch.inference_mode():
                eraw, experts = engine._expert_block(
                    models,
                    "E",
                    sensor,
                    torch.full((len(engine.EVENT_LAGS_US), 2), 0.1, device="cuda"),
                )
                mean, scale = normalizers["E"]
                ef = ((eraw - mean) / scale)[None]
                anchors = query - np.asarray(engine.EVENT_LAGS_US)
                et = torch.from_numpy(engine._clock(anchors, np.full(len(anchors), query), query))[
                    None
                ].cuda()
                ev = torch.ones((1, len(anchors)), dtype=torch.bool, device="cuda")
                ee = experts[-1:]
                eb, _, eout = frozen_state(models["E_CTX_MATCHED"], ef, et, ev, ee)
                observations, reason = (
                    rgb_observations(row, regs[row["query_id"]])
                    if cohort == "FCWD"
                    else engine.raw_rgb_observations(row)
                )
                values: dict = dict(
                    event_phase=eout["point_phase"].cpu().numpy(), event_block=eb.cpu().numpy()
                )
                if observations:
                    timeline = []
                    mean, scale = normalizers["R"]
                    for pixels, meta in observations:
                        raw_r, expert_r = engine._expert_block(
                            models,
                            "R",
                            torch.from_numpy(pixels)[None].cuda(),
                            torch.tensor([meta["delta_t_s"]], dtype=torch.float32, device="cuda"),
                        )
                        timeline.append(
                            (
                                meta["first_sensor_ts_us"],
                                meta["anchor_us"],
                                meta["available_us"],
                                ((raw_r - mean) / scale)[0],
                                expert_r[0],
                            )
                        )
                    history = engine._select_rgb_history(timeline, query)
                    rf = torch.stack([v[3] for v in history])[None]
                    rt = torch.from_numpy(
                        engine._clock(
                            np.asarray([v[1] for v in history]),
                            np.asarray([v[2] for v in history]),
                            query,
                        )
                    )[None].cuda()
                    rv = torch.ones((1, len(history)), dtype=torch.bool, device="cuda")
                    rb, widths, rout = frozen_state(
                        models["R_CTX"], rf, rt, rv, history[-1][4][None]
                    )
                    fout = models["F_TRUE"](ef, et, ev, ee, rf, rt, rv)
                    values.update(
                        features=torch.cat([rb, eb], -1).cpu().numpy(),
                        rgb_width_logits=widths.cpu().numpy(),
                        rgb_phase=rout["point_phase"].cpu().numpy(),
                        fusion_phase=fout["point_phase"].cpu().numpy(),
                    )
                else:
                    values["fusion_phase"] = values["event_phase"]
            temp = path.with_suffix(".pending.npz")
            np.savez_compressed(temp, **values)
            temp.replace(path)
            receipt = dict(
                query_id=row["query_id"],
                sequence_id=row["sequence_id"],
                anchor_us=query,
                binding_sha256=binding_sha,
                sha256=sha(path),
                rgb_available=bool(observations),
                reason=reason,
                E_CTX=float(phase_to_ttc(eout["point_phase"])[0]),
            )
            write(receipt_path, receipt)
            receipts[row["query_id"]] = receipt
            print(cohort, index + 1, len(rows), row["query_id"], flush=True)
        write(
            target / "SEALED.json",
            dict(
                binding_sha256=binding_sha,
                population=len(rows),
                queries=receipts,
                targets_read=False,
                optimizer_updates=0,
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", choices=("FCWD",))
    args = parser.parse_args()
    setup()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(
        1024**3 / torch.cuda.get_device_properties(0).total_memory
    )
    extract(args.cohort)


if __name__ == "__main__":
    main()
