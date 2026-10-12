"""Recoverable, frozen test12 inference with native same-object event/RGB histories."""

from __future__ import annotations

import hashlib
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from e_jepa_ttc.rgb_port.data import TarFrameReader, decode_query
from e_jepa_ttc.rgb_port.fusion import DualClockFusion
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.multimodal_validation.ancestry import historical_source_binding
from operational.multimodal_validation.common import ROOT, RUN, read, sha, write
from operational.rgb_port import transfer as engine
from operational.rgb_port.infer_experts import clock_features, delta_matrix
from operational.rgb_port_external_fix.runtime import observation_contract
from operational.rgb_port_fusion_fix.fusion import deterministic_forward

from . import common
from .model import frozen_state
from .phase_only_anchor import PhaseOnlyAnchor

DEST = common.OUT / "test12_phase_only"


def store(directory: Path, identity: str, arrays: dict, binding: str) -> dict:
    key = hashlib.sha256(identity.encode()).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (key + ".npz")
    temporary = path.with_suffix(".pending.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    receipt = dict(identity=identity, binding_sha256=binding, sha256=sha(path))
    write(path.with_suffix(".json"), receipt)
    return receipt


def restore(directory: Path, identity: str, binding: str) -> dict | None:
    path = directory / (hashlib.sha256(identity.encode()).hexdigest() + ".npz")
    receipt_path = path.with_suffix(".json")
    if not receipt_path.exists():
        return None
    receipt = read(receipt_path)
    assert receipt["identity"] == identity and receipt["binding_sha256"] == binding
    assert receipt["sha256"] == sha(path)
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def context(ids: list[str], values: dict, meta: dict, query: int, normalizer: tuple) -> tuple:
    assert 0 < len(ids) <= 8 and len(set(ids)) == len(ids)
    raw = torch.from_numpy(np.stack([values[k]["raw"] for k in ids])).cuda()
    features = torch.zeros((1, 8, 17), device="cuda")
    features[0, -len(ids) :] = (raw - normalizer[0]) / normalizer[1]
    anchors = [int(meta[k]["anchor_us"]) for k in ids]
    arrivals = [int(meta[k]["available_us"]) for k in ids]
    timing = torch.zeros((1, 8, 4), device="cuda")
    timing[0, -len(ids) :] = torch.from_numpy(clock_features(anchors, arrivals, query)).cuda()
    valid = torch.zeros((1, 8), dtype=torch.bool, device="cuda")
    valid[0, -len(ids) :] = True
    experts = torch.from_numpy(values[ids[-1]]["experts"])[None].cuda()
    return features, timing, valid, experts


def main() -> None:
    common.setup()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_per_process_memory_fraction(
        1024**3 / torch.cuda.get_device_properties(0).total_memory
    )
    freeze_path = DEST / "MODEL_INPUT_FREEZE.json"
    freeze = read(freeze_path)
    binding = sha(freeze_path)
    for path, expected in freeze["source_sha256"].items():
        assert sha(ROOT / path) == expected, path
    for path, expected in freeze["parent_sha256"].items():
        assert sha(Path(path)) == expected, path
    inputs_path = DEST / "INPUTS.json"
    assert sha(inputs_path) == freeze["inputs_sha256"]
    inputs = read(inputs_path)
    for relative, receipt in inputs["rgb_sources"].items():
        stat = (Path(inputs["media_root"]) / relative).stat()
        assert stat.st_size == receipt["size_bytes"] and stat.st_mtime_ns == receipt["mtime_ns"]
    models_readout = {}
    for mode in ("EVENT", "ZERO"):
        path = common.OUT / f"fits/R_ANCHOR_PHASE_ONLY_{mode}_S7/checkpoint_last.pt"
        assert sha(path) == freeze["readout_sha256"][mode]
        model = PhaseOnlyAnchor(mode).eval().requires_grad_(False)
        model.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
        )
        models_readout[mode] = model
    with ExitStack() as stack:
        stack.enter_context(historical_source_binding())
        stack.enter_context(observation_contract())
        stack.enter_context(patch.object(DualClockFusion, "forward", deterministic_forward))
        models = engine._models(
            RUN, ROOT / "configs/rgb_port/producers.json", RUN / "P_MANIFEST.json", "cuda"
        )
        for model in models.values():
            model.eval().requires_grad_(False)
        normalizers = {}
        for prefix, modality in (("E", "event"), ("R", "rgb")):
            mean, scale = engine._normalizer(
                RUN / f"features/H/{prefix}_NORMALIZER.json",
                modality,
                freeze["producer_freeze_sha256"],
            )
            normalizers[prefix] = mean.cuda(), scale.cuda()
        evalues, rvalues = {}, {}
        for number, (token, job) in enumerate(inputs["event_observations"].items()):
            common.guard()
            value = restore(DEST / "event_observations", token, binding)
            if value is None:
                cache = Path(job["current_tensor_path"])
                assert sha(cache) == job["current_tensor_sha256"]
                with np.load(cache, allow_pickle=False) as z:
                    assert z["valid"][-1]
                    sensor = torch.from_numpy(z["events"][-1:]).cuda()
                delta = delta_matrix(torch.tensor([job["delta_t_s"]]), 3).cuda()
                with torch.inference_mode():
                    raw, experts = engine._expert_block(models, "E", sensor, delta)
                value = dict(raw=raw[0].cpu().numpy(), experts=experts[0].cpu().numpy())
                store(DEST / "event_observations", token, value, binding)
            evalues[token] = value
            if number % 100 == 0:
                print("TEST12_EVENT", number + 1, len(inputs["event_observations"]), flush=True)
        reader = TarFrameReader(
            Path(inputs["media_root"]), max_open=2, max_frame_cache_bytes=128 * 1024**2
        )
        stack.callback(reader.close)
        for number, (identity, row) in enumerate(inputs["rgb_observations"].items()):
            common.guard()
            value = restore(DEST / "rgb_observations", identity, binding)
            if value is None:
                decoded = decode_query(row, eap_root=Path(inputs["media_root"]), reader=reader)
                sensor = torch.from_numpy(decoded["rgb"])[None].cuda()
                delta = torch.from_numpy(decoded["delta_t_s"])[None].cuda()
                with torch.inference_mode():
                    raw, experts = engine._expert_block(models, "R", sensor, delta)
                value = dict(raw=raw[0].cpu().numpy(), experts=experts[0].cpu().numpy())
                store(DEST / "rgb_observations", identity, value, binding)
            rvalues[identity] = value
            if number % 100 == 0:
                print("TEST12_RGB", number + 1, len(inputs["rgb_observations"]), flush=True)
        receipts = {}
        for number, row in enumerate(inputs["queries"]):
            common.guard()
            token, query = row["sample_token"], row["query_time_us"]
            value = restore(DEST / "predictions", token, binding)
            if value is None:
                with torch.inference_mode():
                    ec = context(
                        row["event_history"],
                        evalues,
                        inputs["event_observations"],
                        query,
                        normalizers["E"],
                    )
                    eb, _, eo = frozen_state(models["E_CTX_MATCHED"], *ec)
                    value = dict(
                        E_CTX=phase_to_ttc(eo["point_phase"]).cpu().numpy(),
                        event_phase=eo["point_phase"].cpu().numpy(),
                    )
                    if row["rgb_history"]:
                        rc = context(
                            row["rgb_history"],
                            rvalues,
                            inputs["rgb_observations"],
                            query,
                            normalizers["R"],
                        )
                        rb, widths, ro = frozen_state(models["R_CTX"], *rc)
                        fo = models["F_TRUE"](*ec, rc[0], rc[1], rc[2])
                        features = torch.cat([rb, eb], -1).cpu()
                        value.update(
                            features=features.numpy(),
                            rgb_width_logits=widths.cpu().numpy(),
                            R_CTX=phase_to_ttc(ro["point_phase"]).cpu().numpy(),
                            F_TRUE=phase_to_ttc(fo["point_phase"]).cpu().numpy(),
                        )
                        for mode, model in models_readout.items():
                            prediction = model(features, widths.cpu())
                            value[mode] = phase_to_ttc(prediction["point_phase"]).numpy()
                            value[mode + "_phase"] = prediction["point_phase"].numpy()
                    else:
                        value.update(
                            R_CTX=np.asarray([np.nan]),
                            F_TRUE=value["E_CTX"],
                            EVENT=value["E_CTX"],
                            ZERO=value["E_CTX"],
                        )
                store(DEST / "predictions", token, value, binding)
            receipt_path = (
                DEST / "predictions" / (hashlib.sha256(token.encode()).hexdigest() + ".json")
            )
            receipts[token] = read(receipt_path)
            if number % 100 == 0:
                print("TEST12_HEADS", number + 1, len(inputs["queries"]), flush=True)
        write(
            DEST / "PREDICTIONS_SEALED.json",
            dict(
                binding_sha256=binding,
                queries=receipts,
                population=len(receipts),
                targets_read=False,
                optimizer_updates=0,
                official_score_available=False,
                external_submission=False,
            ),
        )


if __name__ == "__main__":
    main()
