"""Independently inspect final RGB weights and optimizer/sampling continuity on CPU."""

from __future__ import annotations

import argparse
import math

import torch
from torch import nn

from operational.range_transfer.common import read, sha, utc, write

from .common import RUN
from .rgb_model import load_parent, pair_from_parent
from .rgb_train import REGISTRATION, registered


def inspect_payload(
    model: nn.Module,
    payload: dict,
    *,
    updates: int,
    population: int,
    batch_size: int,
    seed: int,
    frozen: bool,
) -> dict:
    """Check the actual tensors and regenerate every registered sample ID independently."""
    if payload["completed_updates"] != updates or payload["accumulation_index"] != 0:
        raise ValueError("RGB payload is not at its complete optimizer boundary")
    initial = model.state_dict()
    final = payload["model_state_dict"]
    if initial.keys() != final.keys() or any(
        initial[name].shape != final[name].shape or initial[name].dtype != final[name].dtype
        for name in initial
    ):
        raise ValueError("RGB endpoint topology changed")
    if not all(torch.isfinite(value).all() for value in final.values()):
        raise ValueError("Nonfinite RGB endpoint weights")
    changed = [
        name
        for name in initial
        if not torch.equal(
            initial[name].contiguous().reshape(-1).view(torch.uint8),
            final[name].contiguous().reshape(-1).view(torch.uint8),
        )
    ]
    encoder = [name for name in initial if name.startswith("encoder.")]
    encoder_changed = sorted(set(encoder) & set(changed))
    readout_changed = sorted(set(changed) - set(encoder))
    if not encoder or not readout_changed or bool(encoder_changed) == frozen:
        raise ValueError("Actual encoder/readout change contradicts the registered arm")
    curves = payload["cursor"]["curve"]
    timings = payload["cursor"]["timings"]
    if len(curves) != updates or len(timings) != updates:
        raise ValueError("RGB trajectory length changed")
    generator = torch.Generator().manual_seed(seed)
    for step, (row, elapsed) in enumerate(zip(curves, timings, strict=True), 1):
        expected_ids = torch.randint(population, (batch_size,), generator=generator).tolist()
        if row["update"] != step or row["ids"] != expected_ids:
            raise ValueError("RGB sample sequence or update cursor changed")
        numbers = [value for key, value in row.items() if key != "ids"]
        if not all(math.isfinite(value) for value in [*numbers, elapsed]) or elapsed < 0:
            raise ValueError("Nonfinite RGB curve or invalid timing")
    if not torch.equal(generator.get_state(), payload["sampler_generator_state"]):
        raise ValueError("RGB final sampler state differs from the registered sequence")
    optimizer = payload["optimizer_state_dict"]
    names = [
        name
        for prefix in (True, False)
        for name, parameter in model.named_parameters()
        if name.startswith("encoder.") == prefix and parameter.requires_grad
    ]
    ids = [index for group in optimizer["param_groups"] for index in group["params"]]
    if len(ids) != len(names) or len(set(ids)) != len(ids) or set(ids) != set(optimizer["state"]):
        raise ValueError("RGB optimizer parameter coverage changed")
    for index, name in zip(ids, names, strict=True):
        state = optimizer["state"][index]
        if float(state["step"]) != updates:
            raise ValueError("RGB Adam step differs from the endpoint")
        for key in ("exp_avg", "exp_avg_sq"):
            moment = state[key]
            if moment.shape != final[name].shape or not torch.isfinite(moment).all():
                raise ValueError("RGB Adam moment shape or finiteness changed")
    if payload["scheduler_state_dict"]["last_epoch"] != updates:
        raise ValueError("RGB scheduler step differs from the endpoint")
    return {
        "status": "PASS",
        "updates": updates,
        "sample_ids_regenerated": updates * batch_size,
        "sampler_final_state_exact": True,
        "encoder_tensors": len(encoder),
        "encoder_tensors_changed": encoder_changed,
        "readout_tensors_changed": readout_changed,
        "frozen_encoder_byte_identical": frozen and not encoder_changed,
        "adam_parameter_states": len(ids),
        "adam_step": updates,
        "scheduler_step": updates,
        "recovery_upper": payload["recovery_upper"],
        "optimizer_updates_in_verification": 0,
    }


def verify(fit_id: str) -> dict:
    """Admit endpoint hashes and parent identity, then inspect on CPU only."""
    torch.set_num_threads(1)
    record = registered()
    if fit_id not in record["fits"]:
        raise ValueError("Unknown RGB fit")
    directory = RUN / "rgb/fits" / fit_id
    receipt = read(directory / "CHECKPOINT_RECEIPT.json")
    pointer = read(directory / "CHECKPOINT_POINTER.json")
    journal = read(directory / "UPDATE_JOURNAL.json")
    identity = read(directory / "IDENTITY.json")
    endpoint = record["updates_per_fit"]
    checkpoint = directory / "checkpoint_versions" / pointer["version"]
    if (
        receipt["status"] != "COMPLETE"
        or receipt["completed_updates"] != endpoint
        or pointer["completed_updates"] != endpoint
        or journal["completed_updates"] != endpoint
        or journal["durable_updates"] != endpoint
        or journal["pending_update_upper"] != 0
        or sha(checkpoint) != receipt["checkpoint_sha256"]
        or pointer["checkpoint_sha256"] != receipt["checkpoint_sha256"]
        or identity["registration_sha256"] != sha(REGISTRATION)
    ):
        raise ValueError("RGB final endpoint is incomplete or its durable identity changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload["identity"] != identity
        or payload["identity_sha256"] != receipt["identity_sha256"]
        or journal["identity_sha256"] != receipt["identity_sha256"]
        or payload["recovery_upper"] != journal["recovery_upper"]
    ):
        raise ValueError("RGB checkpoint identity or recovery accounting changed")
    parent, parent_identity = load_parent()
    if parent_identity != record["parent"]:
        raise ValueError("RGB initial parent differs from registration")
    frozen = "FROZEN" in fit_id
    model = pair_from_parent(
        parent, correlation=record["variant"] == "CORR", frozen=frozen, seed=record["seed"]
    )
    result = inspect_payload(
        model,
        payload,
        updates=endpoint,
        population=record["input_identity"]["population"],
        batch_size=record["batch_size"],
        seed=record["seed"],
        frozen=frozen,
    )
    result.update(
        utc=utc(),
        fit_id=fit_id,
        checkpoint_sha256=receipt["checkpoint_sha256"],
        registration_sha256=sha(REGISTRATION),
        parent=parent_identity,
        cuda_initialized=torch.cuda.is_initialized(),
        scope="Stored final weights, optimizer and registered samples; no inference or retraining",
    )
    write(RUN / "verification/rgb_endpoints" / fit_id / "RESULT.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fit_id")
    print(verify(parser.parse_args().fit_id))
