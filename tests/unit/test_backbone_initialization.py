from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn

from e_jepa_ttc.detection import backbone_initialization


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(2)


class _Joiner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.body = _Backbone()


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.ModuleList([_Joiner()])


def _checkpoint(path: Path, *, extra: bool = False) -> tuple[dict[str, torch.Tensor], str]:
    source = _Backbone().state_dict()
    source["conv1.weight"] = torch.full_like(source["conv1.weight"], 3.0)
    source["bn1.running_mean"] = torch.tensor([4.0, 5.0])
    source["bn1.running_var"] = torch.tensor([6.0, 7.0])
    source["fc.weight"] = torch.zeros(4, 2)
    source["fc.bias"] = torch.zeros(4)
    if extra:
        source["unapproved.weight"] = torch.ones(1)
    torch.save(source, path)
    return source, hashlib.sha256(path.read_bytes()).hexdigest()


def test_load_backbone_is_strict_offline_and_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "weights.pth"
    source, digest = _checkpoint(path)
    monkeypatch.setattr(backbone_initialization, "OFFICIAL_SHA256", digest)
    model = _Model()

    receipt = backbone_initialization.load_imagenet_resnet50_backbone(model, path, digest)

    actual = model.backbone[0].body.state_dict()
    assert all(torch.equal(actual[name], source[name]) for name in actual)
    assert receipt["all_backbone_tensors_exact"] is True
    assert receipt["strict_load"] is True
    assert receipt["implicit_download"] is False
    assert receipt["ignored_keys"] == ["fc.bias", "fc.weight"]
    assert receipt["frozen_batch_norm_running_stat_tensor_count"] == 2


def test_load_backbone_rejects_file_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "weights.pth"
    _checkpoint(path)

    with pytest.raises(ValueError, match="pinned ImageNet"):
        backbone_initialization.load_imagenet_resnet50_backbone(
            model=_Model(), weights_path=path, expected_sha256="0" * 64
        )


def test_load_backbone_rejects_unapproved_checkpoint_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "weights.pth"
    _, digest = _checkpoint(path, extra=True)
    monkeypatch.setattr(backbone_initialization, "OFFICIAL_SHA256", digest)

    with pytest.raises(ValueError, match="Unexpected ImageNet checkpoint keys"):
        backbone_initialization.load_imagenet_resnet50_backbone(_Model(), path, digest)
