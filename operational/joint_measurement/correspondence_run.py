"""B1 runtime repair: allocate the fixed warp tensor before CUDA graph capture."""

from __future__ import annotations

import argparse
import copy
import shutil
from unittest.mock import patch

import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from operational.range_transfer.common import freeze, read, sha, utc

from . import correspondence_model as numerical
from . import correspondence_train as training
from .common import ROOT, RUN
from .qa import exact, fixture
from .train import committed_sources

OUT = RUN / "runtime/correspondence_buffered_capture_v1"


class BufferedCorrespondence(numerical.CorrespondenceCore):
    """Same arithmetic and constants; the immutable affine matrices live with the core."""

    warp_theta: Tensor

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__(encoder)
        self.register_buffer(
            "warp_theta",
            torch.tensor(
                [
                    [[1 / s, 0, -tx / 64 / s], [0, 1 / s, -ty / 64 / s]]
                    for s, tx, ty in numerical.TRANSFORMS
                ],
                dtype=torch.float32,
            ),
        )

    def warp(self, pixels: Tensor) -> Tensor:
        """Preserve original grid_sample operation order, shape and FP32 matrix values."""
        if len(pixels) != 4 or pixels.shape[-2:] != (128, 128) or pixels.dtype != torch.float32:
            raise ValueError("B1 runtime requires four native128 FP32 textures")
        grid = functional.affine_grid(self.warp_theta, list(pixels.shape), align_corners=False)
        return functional.grid_sample(pixels, grid, align_corners=False, padding_mode="zeros")

    def forward(self, pixels: Tensor) -> Tensor:
        """Reuse every loss operation; only the allocation of constant matrices changes."""
        with patch.object(numerical, "warped_texture", self.warp):
            return super().forward(pixels)


def sources() -> dict:
    """Bind this runtime overlay independently of the unchanged scientific recipe."""
    own = "operational/joint_measurement/correspondence_run.py"
    return {**training.source_binding(), own: sha(ROOT / own)}


def prototype() -> dict:
    """Prove exact CPU loss/gradient equivalence and preserve the original failed capture."""
    training.registered()
    committed_sources(sources())
    failure = OUT / "FAILED_FIRST_CAPTURE.json"
    if not failure.exists():
        folder = training.QA / "CORRESPONDENCE-STAGE1/continuous"
        journal = read(folder / "UPDATE_JOURNAL.json")
        if journal["completed_updates"] or journal["pending_update_upper"]:
            raise ValueError("Expected pre-update capture failure")
        OUT.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(folder / "checkpoint_last.pt", OUT / "failed_capture_origin.pt")
        log = training.DIRECTORY / "logs/cuda_qa_20261016.stderr.log"
        shutil.copyfile(log, OUT / "failed_capture.stderr.log")
        freeze(
            failure,
            {
                "utc": utc(),
                "journal": journal,
                "checkpoint_sha256": sha(OUT / "failed_capture_origin.pt"),
                "stderr_sha256": sha(OUT / "failed_capture.stderr.log"),
                "eager_updates_already_counted": 10,
                "scientific_updates": 0,
                "cause": "pixels.new_tensor copies CPU constants to CUDA inside capture",
                "repair": "Immutable affine matrices allocated as a core buffer before capture",
            },
        )
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model, _ = training.make_model(training.MODES[0], 1, synthetic=True)
    original = numerical.CorrespondenceCore(copy.deepcopy(model.encoder))
    buffered = BufferedCorrespondence(copy.deepcopy(model.encoder))
    pixels = fixture(128).events[:4, -1]
    initial = copy.deepcopy(model.encoder.state_dict())
    rng = torch.get_rng_state()
    a = original(pixels)
    after_original = torch.get_rng_state()
    torch.set_rng_state(rng)
    b = buffered(pixels)
    a.backward()
    b.backward()
    checks = {
        "warp_exact": torch.equal(numerical.warped_texture(pixels), buffered.warp(pixels)),
        "loss_exact": torch.equal(a, b),
        "gradients_exact": exact(
            [p.grad for p in original.encoder.parameters()],
            [p.grad for p in buffered.encoder.parameters()],
        ),
        "rng_exact": torch.equal(after_original, torch.get_rng_state()),
        "original_weights_unchanged": exact(initial, original.encoder.state_dict()),
        "buffered_weights_unchanged": exact(initial, buffered.encoder.state_dict()),
        "cuda_uninitialized": not torch.cuda.is_initialized(),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "optimizer_updates": 0,
        "sources": sources(),
        "loss": float(a.detach()),
        "prototype_revision": (
            "Compare forwards with identical dropout RNG. The first retained CPU test "
            "incorrectly used consecutive draws; zero optimizer updates in either test."
        ),
    }
    freeze(OUT / "CPU_PROOF.json", result)
    return result


def admission() -> dict:
    """Keep weights, target coordinates, sampler and optimizer unchanged across the repair."""
    training.registered()
    committed_sources(sources())
    proof = read(OUT / "CPU_PROOF.json")
    if proof["status"] != "PASS" or proof["sources"] != sources():
        raise ValueError("B1 runtime requires matching exact CPU arithmetic/gradient proof")
    record = {
        "sources": sources(),
        "scientific_registration_sha256": sha(training.REGISTRATION),
        "CPU_proof_sha256": sha(OUT / "CPU_PROOF.json"),
        "failure_sha256": sha(OUT / "FAILED_FIRST_CAPTURE.json"),
        "scientific_changes": [],
        "runtime_change": "Preallocate fixed affine matrices",
        "QA": "Original90-update plan; reuse durable eager10 and capture origin0",
        "checkpoint": "Restore original complete state; never restart completed updates",
    }
    freeze(OUT / "REGISTRATION.json", record)
    return record


def run(*, qa_only: bool) -> None:
    """Run the original trainer under the admitted allocation-only overlay."""
    admission()
    with patch.object(numerical, "CorrespondenceCore", BufferedCorrespondence):
        if qa_only:
            from .geometry_graph_probe import no_live_training_driver

            no_live_training_driver()
            print(training.qa(), flush=True)
        else:
            training.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype-cpu", action="store_true")
    parser.add_argument("--register-only", action="store_true")
    parser.add_argument("--qa-only", action="store_true")
    args = parser.parse_args()
    if args.prototype_cpu:
        print(prototype())
    elif args.register_only:
        print(admission())
    else:
        run(qa_only=args.qa_only)
