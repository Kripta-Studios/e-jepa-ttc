"""Resume the unchanged C1 experiment with admitted complete graph capture."""

from __future__ import annotations

import argparse
from unittest.mock import patch

import torch

from operational.range_transfer.common import freeze, read, sha

from . import geometry_run, geometry_train, rgb_cached_run
from .common import ROOT, RUN
from .geometry_capture import contiguous_subset, owned_backward
from .geometry_capture_probe import OUT as PROBE
from .geometry_capture_qa import OUT as QA
from .geometry_capture_qa import STATE_KEYS, CapturedGeometry
from .geometry_graph_probe import no_live_training_driver
from .train import committed_sources

OUT = RUN / "runtime/geometry_captured_training_v1"
ORIGIN = RUN / "runtime/geometry_capture_v2"


class VerifiedCheckpoint(rgb_cached_run.VerifiedCheckpoint):
    """Reuse the eleven-state restore witness with C1-specific output ownership."""

    def restore(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        generator: torch.Generator,
    ) -> dict:
        with patch.object(rgb_cached_run, "OUT", OUT):
            return super().restore(model, optimizer, scheduler, generator)


def validate_proofs(probe: dict, qa: dict) -> None:
    """Require complete check sets; absent checks cannot silently admit a runtime."""
    probe_keys = {
        "losses",
        "gradients",
        "rng_exact",
        "weights_unchanged",
        "pinned_microbatch_views",
        "parameter_identity",
    }
    state_keys = {*STATE_KEYS, "curve"}
    if (
        probe.get("status") != "CANDIDATE_FOR_RESUME_QA"
        or set(probe.get("checks", {})) != probe_keys
        or not all(probe["checks"].values())
        or not 0 < probe.get("ratio", float("inf")) <= 0.9
        or qa.get("status") != "PASS"
        or set(qa.get("exact_graph_resume", {})) != state_keys
        or set(qa.get("eager_numerical_comparison", {})) != state_keys
        or not all(qa["exact_graph_resume"].values())
        or not all(qa["eager_numerical_comparison"].values())
    ):
        raise ValueError("C1 capture runtime requires complete numerical, speed and resume proof")


def admission() -> dict:
    """Bind preserved trajectory, passing proofs and executable overlay before updates."""
    geometry_train.registered()
    probe = read(PROBE / "RESULT.json")
    qa = read(QA / "RESULT.json")
    validate_proofs(probe, qa)
    if (
        probe["registration_sha256"] != sha(PROBE / "REGISTRATION.json")
        or qa["registration_sha256"] != sha(QA / "REGISTRATION.json")
        or qa["probe_result_sha256"] != sha(PROBE / "RESULT.json")
    ):
        raise ValueError("Capture proof registration changed")
    sources = {}
    for folder in (PROBE, QA):
        for path, expected in read(folder / "REGISTRATION.json")["sources"].items():
            if sha(ROOT / path) != expected:
                raise ValueError(f"Admitted capture source changed: {path}")
            sources[path] = expected
    for name in ("geometry_capture_run.py", "geometry_run.py", "rgb_cached_run.py"):
        path = "operational/joint_measurement/" + name
        sources[path] = sha(ROOT / path)
    committed_sources(sources)
    origin = read(ORIGIN / "DURABLE_ORIGIN.json")
    if sha(ORIGIN / "resume_origin.pt") != origin["checkpoint_sha256"]:
        raise ValueError("Preserved C1 trajectory changed")
    record = {
        "sources": sources,
        "C1_registration_sha256": sha(geometry_train.REGISTRATION),
        "origin": origin,
        "probe_result_sha256": sha(PROBE / "RESULT.json"),
        "resume_proof_sha256": sha(QA / "RESULT.json"),
        "scientific_recipe_unchanged": True,
        "runtime": "Complete forward/backward CUDA capture; owned accumulation; pinned slice views",
        "precision": "FP32_no_TF32",
        "effective_batch": 32,
        "microbatch": 4,
        "sensor_cache": "Existing bounded immutable-row cache and one-batch prefetch",
        "full_state_restore_required_before_first_update": True,
        "prepared_timing_ratio": probe["ratio"],
        "end_to_end_speedup_claim": False,
    }
    freeze(OUT / "REGISTRATION.json", record)
    return record


def run(*, once: bool = False) -> None:
    """Reuse the original sole-writer loop, budget, checkpoint and evaluation paths."""
    admission()
    no_live_training_driver()
    with (
        patch.object(geometry_train, "GeometricPairMeasurement", CapturedGeometry),
        patch.object(geometry_train, "backward", owned_backward),
        patch.object(geometry_train, "subset", contiguous_subset),
        patch.object(geometry_train, "ProducerCheckpoint", VerifiedCheckpoint),
    ):
        geometry_run.run(once=once)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--register-only", action="store_true")
    args = parser.parse_args()
    if args.register_only:
        print(admission())
    else:
        run(once=args.once)
