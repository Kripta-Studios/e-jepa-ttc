from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import torch
from torch import nn

from e_jepa_ttc.models.multimodal import DINOv3FeatureTeacher
from e_jepa_ttc.training.multimodal import _student_teacher_distillation_loss


def test_dinov3_distillation_stops_teacher_gradient() -> None:
    student = torch.randn(2, 1, 8, requires_grad=True)
    teacher = torch.randn(2, 1, 8, requires_grad=True)
    loss = _student_teacher_distillation_loss(
        student,
        teacher,
        torch.ones(2, 1, dtype=torch.bool),
    )
    loss.backward()

    assert student.grad is not None
    assert teacher.grad is None


def test_dinov3_teacher_accepts_a_local_download(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "dinov3-large"
    model_dir.mkdir()
    (model_dir / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "size": {"height": 224, "width": 224},
                "image_mean": [0.1, 0.2, 0.3],
                "image_std": [0.9, 0.8, 0.7],
            }
        ),
        encoding="utf-8",
    )

    class FakeAutoModel:
        calls: list[tuple[str, bool]] = []

        @classmethod
        def from_pretrained(cls, source: str, *, local_files_only: bool) -> nn.Module:
            cls.calls.append((source, local_files_only))
            return nn.Identity()

    transformers = types.ModuleType("transformers")
    transformers.AutoModel = FakeAutoModel  # type: ignore[attr-defined]
    hub = types.ModuleType("huggingface_hub")
    hub.hf_hub_download = lambda *_args, **_kwargs: "unused"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    teacher = DINOv3FeatureTeacher(str(model_dir))

    assert teacher.model_source == str(model_dir)
    assert FakeAutoModel.calls == [(str(model_dir), True)]
    assert teacher.input_size == (224, 224)
