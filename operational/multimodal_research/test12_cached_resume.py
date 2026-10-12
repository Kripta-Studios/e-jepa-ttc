"""Cache-only frozen recovery with bounded admission batches and exact saved-output parity."""

from __future__ import annotations

import hashlib
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import psutil
import torch

from e_jepa_ttc.rgb_port.fusion import DualClockFusion
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.multimodal_validation.ancestry import historical_source_binding
from operational.multimodal_validation.common import ROOT, RUN, read, sha, write
from operational.rgb_port import transfer as engine
from operational.rgb_port_external_fix.runtime import observation_contract
from operational.rgb_port_fusion_fix.fusion import deterministic_forward

from . import common
from .guard_scan import python_first
from .model import frozen_state
from .phase_only_anchor import PhaseOnlyAnchor
from .test12_infer import DEST, context, restore, store
from .waiting_owner import wrap as waiting_guard

AUTHORIZATION = common.authorize()


def check(index: int, batch: int) -> None:
    elapsed = (datetime.now(UTC) - datetime.fromisoformat(AUTHORIZATION["utc"])).total_seconds()
    if elapsed >= AUTHORIZATION["max_elapsed_seconds"]:
        raise RuntimeError("Authorized 18-hour execution window exhausted")
    if index % batch == 0:
        common.guard()


def run() -> None:
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
        for name, output in (("event", evalues), ("rgb", rvalues)):
            for number, identity in enumerate(inputs[name + "_observations"]):
                check(number, 128)
                value = restore(DEST / (name + "_observations"), identity, binding)
                assert value is not None, (
                    "Cache-only recovery requires complete sensor observations"
                )
                output[identity] = value
            print("RESTORED", name, len(output), flush=True)
        completed = []
        for row in inputs["queries"]:
            token = row["sample_token"]
            receipt_path = (
                DEST / "predictions" / (hashlib.sha256(token.encode()).hexdigest() + ".json")
            )
            if receipt_path.exists():
                completed.append(token)
        audit_tokens = set(
            sorted(
                completed,
                key=lambda v: hashlib.sha256(("RESUME_PARITY_20261012|" + v).encode()).hexdigest(),
            )[:16]
        )
        assert len(audit_tokens) == 16
        write(
            DEST / "CACHE_RESUME_PARITY_SELECTION.json",
            dict(
                tokens=sorted(audit_tokens), available_completed=len(completed), target_blind=True
            ),
        )
        parity = []
        receipts = {}
        for number, row in enumerate(inputs["queries"]):
            check(number, 32)
            token, query = row["sample_token"], row["query_time_us"]
            value = restore(DEST / "predictions", token, binding)
            previous = value
            if value is None or token in audit_tokens:
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
                if previous is not None:
                    assert value.keys() == previous.keys()
                    for key in value:
                        assert np.array_equal(value[key], previous[key], equal_nan=True), (
                            token,
                            key,
                        )
                    parity.append(token)
                else:
                    store(DEST / "predictions", token, value, binding)
            receipt_path = (
                DEST / "predictions" / (hashlib.sha256(token.encode()).hexdigest() + ".json")
            )
            receipts[token] = read(receipt_path)
            if number % 100 == 0:
                print("TEST12_HEADS", number + 1, len(inputs["queries"]), flush=True)
        common.guard()
        assert set(parity) == audit_tokens
        write(
            DEST / "CACHE_RESUME_PARITY.json",
            dict(
                status="PASS",
                tokens=sorted(parity),
                all_stored_arrays_bit_exact=True,
                source_sha256=sha(__file__),
                binding_sha256=binding,
            ),
        )
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


def main() -> None:
    receipt_path = DEST / "CACHE_RESUME_EXECUTION.json"
    assert not receipt_path.exists()
    ledger = common.accounting()
    assert ledger["physical_upper"] == 80000 and ledger["pending_upper"] == 0
    receipt = dict(
        utc_start=datetime.now(UTC).isoformat(),
        status="RUNNING",
        source_sha256=sha(__file__),
        admission_source_sha256=sha(Path(__file__).with_name("waiting_owner.py")),
        idle_owner_contract_sha256=sha(common.OUT / "WAITING_OWNER_CONTRACT.json"),
        model_input_freeze_sha256=sha(DEST / "MODEL_INPUT_FREEZE.json"),
        admission=(
            "Full unchanged live guard before each 128 cached observations and 32 heads, "
            "plus deadline every item and final full guard. Frozen inference only; "
            "fixed 1 GiB CUDA allocator cap. No optimizer steps. Python-first metadata filtering."
        ),
        accounting_before=ledger,
    )
    write(receipt_path, receipt)
    try:
        with (
            patch.object(psutil, "process_iter", python_first),
            patch.object(common, "guard", waiting_guard(common.guard)),
        ):
            run()
        assert common.accounting() == ledger
        receipt["status"] = "COMPLETE"
    except BaseException as error:
        receipt.update(status="STOPPED", error=str(error))
        raise
    finally:
        receipt["utc_end"] = datetime.now(UTC).isoformat()
        write(receipt_path, receipt)


if __name__ == "__main__":
    main()
