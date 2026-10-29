from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

from e_jepa_ttc.detection.temporal_detector import (
    MONODGP_SOURCE_SHA256,
    ProjectedFeatureBackend,
    ProjectedFeatureState,
    TemporalBatchContext,
    TemporalDetectorAdapter,
    TemporalFrameIdentity,
    admit_monodgp_temporal_source,
)
from e_jepa_ttc.detection.temporal_fusion import (
    CausalTemporalFeatureFusion,
    TemporalFusionSpec,
)


class _FakeBackend(ProjectedFeatureBackend):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(3, 4, 1)
        self.class_head = nn.Conv2d(4, 2, 1)
        self.velocity_head = nn.Conv2d(4, 2, 1)
        self.extract_count = 0
        self.decode_contexts: list[int] = []

    def extract_projected_features(self, images: torch.Tensor) -> ProjectedFeatureState:
        self.extract_count += 1
        features = self.projection(images)
        return ProjectedFeatureState((features,), self.extract_count)

    def decode_projected_features(
        self,
        state: ProjectedFeatureState,
        features: Sequence[torch.Tensor],
        calibs: torch.Tensor,
        targets: Sequence[Mapping[str, torch.Tensor]] | None,
        image_sizes: torch.Tensor,
        dn_args: object | None = None,
    ) -> dict[str, Any]:
        del calibs, targets, image_sizes, dn_args
        if not isinstance(state.decoder_context, int):
            raise TypeError("Fake decoder context must be an integer")
        context = state.decoder_context
        self.decode_contexts.append(context)
        return {
            "pred_logits": self.class_head(features[0]),
            "pred_velocity": self.velocity_head(features[0]),
            "context": context,
        }

    def baseline(self, images: torch.Tensor) -> dict[str, Any]:
        state = self.extract_projected_features(images)
        return self.decode_projected_features(
            state, state.features, torch.empty(0), None, torch.empty(0)
        )


def _fusion() -> CausalTemporalFeatureFusion:
    return CausalTemporalFeatureFusion(
        TemporalFusionSpec(channels=(4,), bottleneck_channels=3)
    )


def _frame(frame: str, *, generation: int = 0) -> TemporalFrameIdentity:
    return TemporalFrameIdentity("sequence", frame, generation)


def _context(
    control: str,
    history: tuple[TemporalFrameIdentity | None, TemporalFrameIdentity | None],
    *,
    current: TemporalFrameIdentity | None = None,
    version: str = "producer-v1",
) -> TemporalBatchContext:
    return TemporalBatchContext(
        producer_version=version,
        control=control,  # type: ignore[arg-type]
        current=(current or _frame("current"),),
        history=(history,),
    )


def _adapter(backend: _FakeBackend | None = None) -> TemporalDetectorAdapter:
    return TemporalDetectorAdapter(
        backend or _FakeBackend(),
        _fusion(),
        producer_version="producer-v1",
        maximum_history_age_us=500_000,
    )


def _fixed_args() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.randn(1, 3, 5, 7), torch.eye(3).unsqueeze(0), torch.tensor([[7, 5]])


def test_current_only_is_exact_split_backend_baseline() -> None:
    torch.manual_seed(7)
    baseline_backend = _FakeBackend()
    temporal_backend = _FakeBackend()
    temporal_backend.load_state_dict(baseline_backend.state_dict(), strict=True)
    images, calibs, sizes = _fixed_args()
    baseline = baseline_backend.baseline(images)
    result = _adapter(temporal_backend)(
        images,
        calibs,
        None,
        sizes,
        temporal_context=_context("current_only", (None, None)),
    )
    assert torch.equal(result["pred_logits"], baseline["pred_logits"])
    assert torch.equal(result["pred_velocity"], baseline["pred_velocity"])


@pytest.mark.parametrize("history_limit", [1, 2])
def test_history_recomputed_and_detached_while_current_keeps_gradient(history_limit: int) -> None:
    backend = _FakeBackend()
    adapter = _adapter(backend)
    images, calibs, sizes = _fixed_args()
    images.requires_grad_(True)
    history = torch.randn(1, 2, 3, 5, 7, requires_grad=True)
    mask = torch.tensor([[history_limit == 2, True]])
    ages = torch.tensor([[200_000.0 if history_limit == 2 else 0.0, 100_000.0]])
    context = _context(
        "correct_past",
        (_frame("old") if history_limit == 2 else None, _frame("recent")),
    )
    result = adapter(
        images,
        calibs,
        None,
        sizes,
        temporal_context=context,
        history_images=history,
        history_ages_us=ages,
        history_mask=mask,
        history_limit=history_limit,
    )
    result["pred_logits"].sum().backward()
    assert backend.extract_count == 2
    assert images.grad is not None and torch.count_nonzero(images.grad)
    assert history.grad is None


def test_repeated_current_uses_same_trainable_capacity() -> None:
    adapter = _adapter()
    images, calibs, sizes = _fixed_args()
    histories = images.detach().unsqueeze(1).repeat(1, 2, 1, 1, 1)
    current = _frame("current")
    result = adapter(
        images,
        calibs,
        None,
        sizes,
        temporal_context=_context("current_repeated", (current, current), current=current),
        history_images=histories,
        history_ages_us=torch.zeros(1, 2),
        history_mask=torch.ones(1, 2, dtype=torch.bool),
        history_limit=2,
    )
    result["pred_velocity"].sum().backward()
    assert adapter.fusion.output_projections()[0].weight.grad is not None


def test_context_is_call_local_and_generation_cannot_cross() -> None:
    backend = _FakeBackend()
    adapter = _adapter(backend)
    images, calibs, sizes = _fixed_args()
    first = adapter(
        images,
        calibs,
        None,
        sizes,
        temporal_context=_context("current_only", (None, None)),
    )
    second = adapter(
        images,
        calibs,
        None,
        sizes,
        temporal_context=_context(
            "current_only", (None, None), current=_frame("other", generation=1)
        ),
    )
    assert (first["context"], second["context"]) == (1, 2)
    assert backend.decode_contexts == [1, 2]

    with pytest.raises(ValueError, match="generation"):
        adapter(
            images,
            calibs,
            None,
            sizes,
            temporal_context=_context(
                "correct_past",
                (None, _frame("past", generation=0)),
                current=_frame("current", generation=1),
            ),
            history_images=torch.randn(1, 2, 3, 5, 7),
            history_ages_us=torch.tensor([[0.0, 100_000.0]]),
            history_mask=torch.tensor([[False, True]]),
            history_limit=1,
        )


def test_stale_producer_cross_sequence_age_and_mask_fail_closed() -> None:
    adapter = _adapter()
    images, calibs, sizes = _fixed_args()
    common = {
        "history_images": torch.randn(1, 2, 3, 5, 7),
        "history_ages_us": torch.tensor([[0.0, 100_000.0]]),
        "history_mask": torch.tensor([[False, True]]),
        "history_limit": 1,
    }
    with pytest.raises(ValueError, match="producer version"):
        adapter(
            images,
            calibs,
            None,
            sizes,
            temporal_context=_context("correct_past", (None, _frame("past")), version="stale"),
            **common,
        )
    with pytest.raises(ValueError, match="crosses sequence"):
        adapter(
            images,
            calibs,
            None,
            sizes,
            temporal_context=_context(
                "correct_past",
                (None, TemporalFrameIdentity("other", "past")),
            ),
            **common,
        )
    common["history_ages_us"] = torch.tensor([[0.0, 600_000.0]])
    with pytest.raises(ValueError, match="configured maximum"):
        adapter(
            images,
            calibs,
            None,
            sizes,
            temporal_context=_context("correct_past", (None, _frame("past"))),
            **common,
        )
    common["history_ages_us"] = torch.tensor([[0.0, 100_000.0]])
    common["history_mask"] = torch.tensor([[False, False]])
    with pytest.raises(ValueError, match="identities and history mask"):
        adapter(
            images,
            calibs,
            None,
            sizes,
            temporal_context=_context("correct_past", (None, _frame("past"))),
            **common,
        )


def test_strict_baseline_load_happens_before_adapter_attachment() -> None:
    backend = _FakeBackend()
    state = backend.state_dict()
    adapter = TemporalDetectorAdapter.from_strict_baseline(
        _FakeBackend(),
        state,
        _fusion(),
        producer_version="producer-v1",
        maximum_history_age_us=500_000,
    )
    assert set(adapter.backend.state_dict()) == set(state)
    missing = dict(state)
    missing.pop(next(iter(missing)))
    with pytest.raises(RuntimeError):
        TemporalDetectorAdapter.from_strict_baseline(
            _FakeBackend(),
            missing,
            _fusion(),
            producer_version="producer-v1",
            maximum_history_age_us=500_000,
        )


def test_source_admission_requires_exact_hash_and_marker(tmp_path: Path) -> None:
    source = Path(
        "output/detection_campaign/r0_d4096_seed7_v2/stage1/code/portable_source/"
        "lib/models/monodgp/monodgp.py"
    )
    receipt = admit_monodgp_temporal_source(source)
    assert receipt["source_sha256"] == MONODGP_SOURCE_SHA256
    changed = tmp_path / "monodgp.py"
    changed.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        admit_monodgp_temporal_source(changed)
