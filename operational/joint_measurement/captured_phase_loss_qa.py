"""Prospective native parity/speed and complete-state resume gates for a separate loss graph."""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from unittest.mock import patch

import numpy as np
import torch

from e_jepa_ttc.joint_measurement.loss import training_mass
from operational.range_transfer.common import freeze, read, sha

from . import adaptation_depth as a1
from . import persistent_adaptation as persistent
from .adaptation_runtime import admission as prior_admission
from .adaptation_runtime_qa import OUT as PRIOR
from .captured_phase_loss import backward as candidate_backward
from .common import ROOT, RUN, accounting, config, gpu_charge
from .compute_guard import guard_idle
from .geometry_capture import contiguous_subset
from .geometry_capture_qa import STATE_KEYS
from .parallel_training_sensor import ParallelTrainingSensors
from .phase_transport_loss import backward as reference_backward
from .qa import exact
from .train import committed_sources

OUT = RUN / "runtime/A1_loss_graph_v1"
QA = RUN / "qa/A1_loss_graph_v1"


def sources() -> dict:
    """Bind new code and the complete already-admitted runtime dependencies."""
    paths = (
        "operational/joint_measurement/captured_phase_loss.py",
        "operational/joint_measurement/captured_phase_loss_qa.py",
        "operational/joint_measurement/adaptation_loss_graph_runtime.py",
        "tests/unit/test_captured_phase_loss.py",
    )
    return {**read(PRIOR / "ADMISSION.json")["sources"], **{p: sha(ROOT / p) for p in paths}}


def register() -> dict:
    """Freeze gates before any additional CUDA workload or physical QA updates."""
    prior_admission()
    committed_sources(sources())
    record = {
        "sources": sources(),
        "prior_admission_sha256": sha(PRIOR / "ADMISSION.json"),
        "scientific_registration_sha256": sha(a1.REGISTRATION),
        "native": (
            "Next real P B32 from an immutable complete checkpoint captured at quiescence; "
            "four alternating paired prepared forward/backward trials, zero optimizer updates"
        ),
        "native_gate": (
            "Bitwise exact loss dict, all parameter gradients, CPU/CUDA RNG and unchanged "
            "weights; median warm candidate/reference time ratio <=0.85, first trial excluded"
        ),
        "complete_state_gate": (
            "Both A1 arms: current Session4 vs candidate2+2 vs candidate2 then fresh resume2; "
            "exact model/Adam/scheduler/sampler/all RNG/cursor curves, excluding timings only"
        ),
        "capture_gate": "One model capture and one loss capture retained across two scopes",
        "physical_technical_updates": 24,
        "scientific_updates": 0,
        "activation": (
            "Requires every proof; restore complete original scientific checkpoint, verify "
            "curve prefix, sample sequence, Adam/scheduler advance and no added recovery. "
            "This QA module does not install the candidate or change the live process."
        ),
        "preserved": (
            "Same phase terms, reduction order, microbatch ordering, FP32, model guards before "
            "backward, sign support, weights, sampler, optimizer and registered objective"
        ),
    }
    freeze(OUT / "REGISTRATION.json", record)
    return record


def checked_registration() -> dict:
    """Validate immutable registration and the previous admission before GPU use."""
    prior_admission()
    record = read(OUT / "REGISTRATION.json")
    if record["sources"] != sources() or record["scientific_registration_sha256"] != sha(
        a1.REGISTRATION
    ):
        raise ValueError("Loss-graph registration changed")
    return record


def native() -> dict:
    """Measure exact native derivatives and prepared timing at the quiescent checkpoint."""
    guard_idle()
    checked_registration()
    if (OUT / "NATIVE.json").exists():
        raise ValueError("Retain the completed native diagnostic; no silent rerun")
    witness = read(OUT / "QUIESCED.json")
    if witness["checkpoint_sha256"] != sha(OUT / "ORIGIN.pt"):
        raise ValueError("Quiescent scientific checkpoint changed")
    payload = torch.load(OUT / "ORIGIN.pt", map_location="cpu", weights_only=False)
    sensors = ParallelTrainingSensors(workers=8, limit_bytes=0)
    checks, timings = [], []
    try:
        sampler = torch.Generator().set_state(payload["sampler_generator_state"])
        ids = torch.randint(sensors.population, (32,), generator=sampler).tolist()
        batch = sensors.load(ids)
        masses = training_mass(
            sensors.rows.target_ttc.to_numpy(), sensors.rows.group_id.to_numpy(), role="P"
        )
        with gpu_charge("A1_native_loss_graph_zero_updates", reserve=900):
            a1.engine.setup(7)
            batches = [
                contiguous_subset(batch, list(range(i, i + 4))).to(torch.device("cuda"))
                for i in range(0, 32, 4)
            ]
            models = {
                name: a1.make_model("LAST-BLOCK")[0].cuda().train()
                for name in ("reference", "candidate")
            }
            for model in models.values():
                model.load_state_dict(payload["model_state_dict"])
            for iteration in range(4):
                order = (
                    ("reference", "candidate") if iteration % 2 == 0 else ("candidate", "reference")
                )
                observed, row = {}, {"iteration": iteration, "order": order, "seconds": {}}
                for name in order:
                    torch.set_rng_state(payload["torch_rng_state"])
                    torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
                    model = models[name]
                    model.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    losses = (reference_backward if name == "reference" else candidate_backward)(
                        model, batches, torch.from_numpy(masses[ids]).float(), sensors.population
                    )
                    torch.cuda.synchronize()
                    row["seconds"][name] = time.perf_counter() - start
                    observed[name] = {
                        "loss": losses,
                        "gradients": {
                            n: p.grad.detach().cpu().clone()
                            for n, p in model.named_parameters()
                            if p.grad is not None
                        },
                        "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(),
                    }
                check = {
                    key: exact(observed["reference"][key], observed["candidate"][key])
                    for key in observed["reference"]
                }
                check["weights_unchanged"] = all(
                    exact(
                        {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        payload["model_state_dict"],
                    )
                    for model in models.values()
                )
                checks.append(check)
                timings.append(row)
                if not all(check.values()):
                    break
        ratio = (
            statistics.median(
                row["seconds"]["candidate"] / row["seconds"]["reference"] for row in timings[1:]
            )
            if len(timings) > 1
            else None
        )
        passed = len(checks) == 4 and all(all(c.values()) for c in checks)
        passed = passed and ratio is not None and ratio <= 0.85
        result = {
            "status": "PASS" if passed else "REJECTED",
            "registration_sha256": sha(OUT / "REGISTRATION.json"),
            "origin_sha256": witness["checkpoint_sha256"],
            "origin_updates": payload["completed_updates"],
            "checks": checks,
            "timings": timings,
            "median_warm_ratio": ratio,
            "ids": ids,
            "optimizer_updates": 0,
            "scope": "Prepared real P B32 forward/backward; not end-to-end production throughput",
        }
        freeze(OUT / "NATIVE.json", result)
        return result
    finally:
        sensors.shutdown()


def gpu(mode: str) -> dict:
    """Check both persistence and process-resume against the admitted current runtime."""
    guard_idle()
    checked_registration()
    if read(OUT / "NATIVE.json")["status"] != "PASS":
        raise ValueError("Exact native parity and speed gate must pass before QA updates")
    if accounting()["technical"] + 12 > config()["caps"]["technical"]:
        raise RuntimeError("Technical reserve exhausted")
    target = QA / mode
    if target.exists():
        raise ValueError("Retain previous physical QA attempts; no implicit repeat")
    masses = np.full(8, 1 / 8)
    with gpu_charge(f"A1_loss_graph_QA_{mode}", reserve=900) as deadline:
        sensors = a1.geometry_train.SyntheticSensors()
        session = persistent.Session(target / "reference", mode, masses, synthetic=True)
        session.advance(sensors, limit=4, deadline=deadline)
        del session
        gc.collect()
        torch.cuda.empty_cache()
        with patch.object(persistent, "backward", candidate_backward):
            session = persistent.Session(target / "persistent", mode, masses, synthetic=True)
            session.advance(sensors, limit=2, deadline=deadline)
            ids = [id(p) for p in session.model.parameters()]
            loss_graph = session.model._v18_phase_loss
            session.advance(sensors, limit=4, deadline=deadline)
            retained = (
                session.capture_count == 1
                and loss_graph.capture_count == 1
                and session.model._v18_phase_loss is loss_graph
                and ids == [id(p) for p in session.model.parameters()]
            )
            del loss_graph, session
            gc.collect()
            torch.cuda.empty_cache()
            session = persistent.Session(target / "resumed", mode, masses, synthetic=True)
            session.advance(sensors, limit=2, deadline=deadline)
            del session
            gc.collect()
            torch.cuda.empty_cache()
            session = persistent.Session(target / "resumed", mode, masses, synthetic=True)
            session.advance(sensors, limit=4, deadline=deadline)
            del session
        sensors.close()
    payloads = [
        torch.load(target / name / "checkpoint_last.pt", map_location="cpu", weights_only=False)
        for name in ("reference", "persistent", "resumed")
    ]
    checks = {}
    for name, other in zip(("persistent", "resumed"), payloads[1:], strict=True):
        checks[name] = {key: exact(payloads[0][key], other[key]) for key in STATE_KEYS}
        checks[name]["curve"] = exact(payloads[0]["cursor"]["curve"], other["cursor"]["curve"])
        checks[name]["four_updates"] = (
            other["completed_updates"] == payloads[0]["completed_updates"] == 4
        )
    passed = retained and all(all(c.values()) for c in checks.values())
    result = {
        "status": "PASS" if passed else "REJECTED",
        "checks": checks,
        "same_parameters_one_model_and_loss_capture_two_scopes": retained,
        "physical_technical_updates": 12,
        "registration_sha256": sha(OUT / "REGISTRATION.json"),
    }
    freeze(target / "RESULT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "native", "gpu"))
    parser.add_argument("--arm", choices=a1.MODES, default=a1.MODES[0])
    args = parser.parse_args()
    result = (
        gpu(args.arm)
        if args.action == "gpu"
        else {"register": register, "native": native}[args.action]()
    )
    print({key: result.get(key) for key in ("status", "checks", "median_warm_ratio")}, flush=True)
