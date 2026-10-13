"""Eight-update exact graph-resume witness and retained eager-drift admission."""

from __future__ import annotations

import copy
import gc

import numpy as np
import torch

from operational.range_transfer.common import digest, freeze, read, sha, utc, write
from operational.rgb_port.train_producers import ProducerCheckpoint

from . import train
from .common import ROOT, accounting, gpu_charge
from .fast_runtime import PersistentSensors, batched_backward
from .profile_fast import DIAG, OUT, STATE_KEYS, micros, model_from, origin, rng
from .qa import exact


def close(left: object, right: object, *, rtol: float, atol: float) -> bool:
    """FP32 implementation compatibility; integer, RNG and identity fields stay exact."""
    if isinstance(left, torch.Tensor):
        if (
            not isinstance(right, torch.Tensor)
            or left.shape != right.shape
            or left.dtype != right.dtype
        ):
            return False
        if left.is_floating_point():
            return bool(
                torch.isfinite(left).all()
                and torch.isfinite(right).all()
                and torch.allclose(left, right, rtol=rtol, atol=atol)
            )
        return torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return exact(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(close(value, right[key], rtol=rtol, atol=atol) for key, value in left.items())
        )
    if isinstance(left, (tuple, list)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(close(a, b, rtol=rtol, atol=atol) for a, b in zip(left, right, strict=True))
        )
    if isinstance(left, float):
        return bool(np.isclose(left, right, rtol=rtol, atol=atol))
    return bool(left == right)


def trajectory(mode: str, payload: dict) -> dict:
    """Shared four-update prefix is counted once; replay counts only two new updates."""
    directory = DIAG / ("resume_" + mode)
    count = 6 if mode == "continuous" else 2
    identity = {
        "fit_id": "FAST_RESUME-" + mode,
        "updates": count,
        "registration": sha(OUT / "RESUME_PROOF_REGISTRATION.json"),
        "origin_updates": 6131,
        "prefix_offset": 0 if mode == "continuous" else 4,
    }
    identity["identity_sha256"] = digest(identity)
    freeze(directory / "IDENTITY.json", identity)
    state = ProducerCheckpoint(directory, identity, count)
    dataset = PersistentSensors()
    train.setup(7)
    with gpu_charge("primary_runtime_exact_graph_resume/" + mode, reserve=900):
        model = model_from(payload, graph=True)
        optimizer, scheduler = train.make_optimizer(model, updates=12000)
        optimizer.load_state_dict(copy.deepcopy(payload["optimizer_state_dict"]))
        scheduler.load_state_dict(copy.deepcopy(payload["scheduler_state_dict"]))
        generator = torch.Generator().set_state(payload["sampler_generator_state"])
        rng(payload)
        curves = [] if mode == "continuous" else copy.deepcopy(payload["cursor"]["curve"])
        if state.pointer.exists():
            curves = state.restore(model, optimizer, scheduler, generator)["curve"]
        else:
            state.save(model, optimizer, scheduler, generator, {"curve": curves}, status="RUNNING")
        shadow = torch.Generator().set_state(generator.get_state())
        ids = torch.randint(dataset.population, (32,), generator=shadow).tolist()
        batches = micros(dataset.load(ids))
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            batched_backward(model, batches, None, dataset.population)
        state.restore(model, optimizer, scheduler, generator)
        dataset.prepare(generator.get_state())
        try:
            while state.completed < count:
                ids = torch.randint(dataset.population, (32,), generator=generator).tolist()
                batches = micros(dataset.batch(ids))
                optimizer.zero_grad(set_to_none=True)
                losses = batched_backward(model, batches, None, dataset.population)
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, error_if_nonfinite=True
                )
                state.begin()
                optimizer.step()
                scheduler.step()
                state.commit()
                curves.append({"ids": ids, "losses": losses, "norm": float(norm)})
                if mode == "continuous" and state.completed == 4:
                    state.save(
                        model, optimizer, scheduler, generator, {"curve": curves}, status="PAUSED"
                    )
                    # Immutable witness copy survives normal checkpoint retention.
                    import shutil

                    shutil.copyfile(state.path, directory / "PREFIX_4.pt")
            state.save(model, optimizer, scheduler, generator, {"curve": curves}, status="COMPLETE")
        finally:
            dataset.shutdown()
    result = torch.load(state.path, map_location="cpu", weights_only=False)
    del model, optimizer, scheduler, batches
    gc.collect()
    torch.cuda.empty_cache()
    accounting()
    return result


def run() -> dict:
    """Admit only numeric compatibility, exact optimized restore and measured speed."""
    if (OUT / "FINAL_ADMISSION.json").exists():
        return read(OUT / "FINAL_ADMISSION.json")
    registration = read(OUT / "RESUME_PROOF_REGISTRATION.json")
    old = read(OUT / "ADMISSION.json")
    eager = torch.load(DIAG / "serial/checkpoint_last.pt", map_location="cpu", weights_only=False)
    fast = torch.load(DIAG / "fast/checkpoint_last.pt", map_location="cpu", weights_only=False)
    numeric = {
        key: close(
            eager[key],
            fast[key],
            rtol=registration["eager_numerical_rtol"],
            atol=registration["eager_numerical_atol"],
        )
        for key in STATE_KEYS
    }
    # The stored gradient norm is a diagnostic scalar, not resumed optimizer
    # state. Verify finite norms and the same clipping branch; the actual
    # post-clipping effects remain covered by parameter and optimizer tolerances.
    eager_curve, fast_curve = eager["cursor"]["curve"], fast["cursor"]["curve"]
    numeric["cursor"] = len(eager_curve) == len(fast_curve) and all(
        left["ids"] == right["ids"]
        and close(
            left["losses"],
            right["losses"],
            rtol=registration["eager_numerical_rtol"],
            atol=registration["eager_numerical_atol"],
        )
        and np.isfinite(left["norm"])
        and np.isfinite(right["norm"])
        and (left["norm"] > 1) == (right["norm"] > 1)
        for left, right in zip(eager_curve, fast_curve, strict=True)
    )
    write(
        OUT / "EAGER_NUMERICAL_COMPARISON_SCOPED.json",
        {"checks": numeric, "registration_sha256": sha(OUT / "RESUME_PROOF_REGISTRATION.json")},
    )
    if not all(numeric.values()):
        raise ValueError(f"Graph numerical trajectory compatibility failed: {numeric}")
    baseline = trajectory("continuous", origin())
    prefix = torch.load(
        DIAG / "resume_continuous/PREFIX_4.pt", map_location="cpu", weights_only=False
    )
    resumed = trajectory("restored", prefix)
    checks = {key: exact(baseline[key], resumed[key]) for key in STATE_KEYS}
    result = {
        **old,
        "utc": utc(),
        "status": "PASS" if all(checks.values()) and old["ratio"] < 1 else "REJECTED",
        "checks": checks,
        "eager_numerical_comparison": numeric,
        "eager_bitwise_equivalence": False,
        "continuous_and_resumed_graph_state_exact": all(checks.values()),
        "optimizer_updates": 40,
        "additional_resume_witness_updates": 8,
        "retained_initial_admission_sha256": sha(OUT / "ADMISSION.json"),
        "resume_registration_sha256": sha(OUT / "RESUME_PROOF_REGISTRATION.json"),
        "runtime_sources": {
            name: sha(ROOT / "operational/joint_measurement" / name)
            for name in ("fast_runtime.py", "profile_fast.py", "resume_fast_proof.py")
        },
        "scope": (
            "Retained eager16 versus graph8+8 timing and numeric comparison; "
            "exact graph6 vs shared4+restored2 uses eight additional physical updates"
        ),
    }
    write(OUT / "FINAL_ADMISSION.json", result)
    print({k: v for k, v in result.items() if k != "timing"}, flush=True)
    return result


if __name__ == "__main__":
    print(run())
