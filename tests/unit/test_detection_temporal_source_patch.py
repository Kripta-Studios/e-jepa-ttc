from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from e_jepa_ttc.detection.temporal_source_patch import (
    MONODGP_RELATIVE_PATH,
    R0_MONODGP_SHA256,
    TemporalSourceContract,
    materialize_temporal_source,
    portable_tree_hashes,
    transform_monodgp_source,
    unified_patch,
)


def _fake_source() -> str:
    return '''import torch
from torch import nn
from e_jepa_ttc.detection.velocity_extension import matched_velocity_l1

class MonoDGP(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_feature_levels = 1
        self.input_proj = nn.ModuleList([nn.Conv2d(3, 4, 1)])
        self.region_head = nn.Identity()
        self.class_head = nn.Conv2d(4, 2, 1)
        self.velocity_head = nn.Conv2d(4, 2, 1)

    def forward(self, images, calibs, targets, img_sizes, dn_args=None):
        features, pos = self.backbone(images)
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
        # region enhancement
        srcs = [self.region_head(value) for value in srcs]
        out = dict()
        out['pred_logits'] = self.class_head(srcs[0])
        out['pred_velocity'] = self.velocity_head(srcs[0])
        out['context_value'] = pos[0] + masks[0].float().sum()
        return out

    def backbone(self, images):
        mask = torch.zeros(images.shape[0], images.shape[2], images.shape[3], dtype=torch.bool)
        return [(images, mask)], [images.new_tensor(3.0)]
'''


def _contract(source: str) -> TemporalSourceContract:
    return TemporalSourceContract(source_sha256=hashlib.sha256(source.encode()).hexdigest())


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _fake_preregistration(
    source_root: Path, path: Path, source: str
) -> TemporalSourceContract:
    tree = portable_tree_hashes(source_root)
    tree_manifest = _canonical_sha256(tree)
    payload = {
        "schema": "eap_detection_campaign_runtime_preregistration_v1",
        "portable_source_sha256": tree,
        "portable_source_sha256_manifest": tree_manifest,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return TemporalSourceContract(
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        portable_tree_manifest_sha256=tree_manifest,
        runtime_preregistration_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def test_fake_forward_split_preserves_outputs_and_state_dict() -> None:
    source = _fake_source()
    transformed = transform_monodgp_source(source, _contract(source))
    original_namespace: dict[str, object] = {}
    transformed_namespace: dict[str, object] = {}
    exec(compile(source, "original.py", "exec"), original_namespace)  # noqa: S102
    exec(compile(transformed, "temporal.py", "exec"), transformed_namespace)  # noqa: S102
    original = cast(Any, original_namespace["MonoDGP"])()
    temporal = cast(Any, transformed_namespace["MonoDGP"])()
    temporal.load_state_dict(original.state_dict(), strict=True)
    assert set(original.state_dict()) == set(temporal.state_dict())
    images = torch.randn(2, 3, 5, 7)
    expected = original(images, torch.empty(0), None, torch.empty(0))
    actual = temporal(images, torch.empty(0), None, torch.empty(0))
    assert expected.keys() == actual.keys()
    for key in expected:
        assert torch.equal(expected[key], actual[key])
    state = temporal.extract_projected_features(images)
    split = temporal.decode_projected_features(
        state, state.features, torch.empty(0), None, torch.empty(0)
    )
    for key in expected:
        assert torch.equal(expected[key], split[key])


def test_real_r0_source_transforms_deterministically() -> None:
    source_path = Path(
        "output/detection_campaign/r0_d4096_seed7_v2/stage1/code/portable_source/"
    ) / MONODGP_RELATIVE_PATH
    source = source_path.read_text(encoding="utf-8")
    assert hashlib.sha256(source.encode()).hexdigest() == R0_MONODGP_SHA256
    first = transform_monodgp_source(source)
    second = transform_monodgp_source(source)
    assert first == second
    assert "class MonoDGP(ProjectedFeatureBackend):" in first
    assert first.count("def extract_projected_features") == 1
    assert first.count("def decode_projected_features") == 1
    assert unified_patch(source, first).startswith("--- a/lib/models/monodgp/monodgp.py")


def test_source_tamper_and_ambiguous_markers_fail_closed() -> None:
    source = _fake_source()
    contract = _contract(source)
    with pytest.raises(ValueError, match="SHA-256"):
        transform_monodgp_source(source + "\n", contract)
    ambiguous = source.replace(
        "        # region enhancement\n",
        "        # region enhancement\n        # region enhancement\n",
    )
    with pytest.raises(ValueError, match="ambiguous"):
        transform_monodgp_source(ambiguous, _contract(ambiguous))


def test_materialization_copies_only_source_change_and_binds_checkpoint(tmp_path: Path) -> None:
    source = _fake_source()
    source_root = tmp_path / "source"
    source_file = source_root / MONODGP_RELATIVE_PATH
    source_file.parent.mkdir(parents=True)
    source_file.write_text(source, encoding="utf-8", newline="")
    other = source_root / "unchanged.py"
    other.write_text("VALUE = 1\n", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    preregistration = tmp_path / "runtime_preregistration.json"
    contract = _fake_preregistration(source_root, preregistration, source)
    other.write_text("TAMPERED = True\n", encoding="utf-8")
    with pytest.raises(ValueError, match="tree differs"):
        materialize_temporal_source(
            source_root=source_root,
            destination=tmp_path / "tampered_destination",
            checkpoint=checkpoint,
            runtime_preregistration=preregistration,
            patch_output=tmp_path / "tampered.patch",
            manifest_output=tmp_path / "tampered.json",
            forbidden_destination=tmp_path / "forbidden",
            contract=contract,
        )
    other.write_text("VALUE = 1\n", encoding="utf-8")
    destination = tmp_path / "temporal"
    patch = tmp_path / "temporal.patch"
    manifest = tmp_path / "manifest.json"
    result = materialize_temporal_source(
        source_root=source_root,
        destination=destination,
        checkpoint=checkpoint,
        runtime_preregistration=preregistration,
        patch_output=patch,
        manifest_output=manifest,
        forbidden_destination=tmp_path / "forbidden",
        contract=contract,
    )
    assert result["changed_files"] == [MONODGP_RELATIVE_PATH.as_posix()]
    assert (destination / "unchanged.py").read_bytes() == other.read_bytes()
    assert patch.is_file() and manifest.is_file()
    with pytest.raises(FileExistsError):
        materialize_temporal_source(
            source_root=source_root,
            destination=destination,
            checkpoint=checkpoint,
            runtime_preregistration=preregistration,
            patch_output=tmp_path / "again.patch",
            manifest_output=tmp_path / "again.json",
            forbidden_destination=tmp_path / "forbidden",
            contract=contract,
        )


def test_active_portable_destination_is_rejected_before_copy(tmp_path: Path) -> None:
    source = _fake_source()
    source_root = tmp_path / "source"
    source_file = source_root / MONODGP_RELATIVE_PATH
    source_file.parent.mkdir(parents=True)
    source_file.write_text(source, encoding="utf-8", newline="")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    preregistration = tmp_path / "runtime_preregistration.json"
    contract = _fake_preregistration(source_root, preregistration, source)
    forbidden = tmp_path / "active"
    with pytest.raises(ValueError, match="aliases"):
        materialize_temporal_source(
            source_root=source_root,
            destination=forbidden,
            checkpoint=checkpoint,
            runtime_preregistration=preregistration,
            patch_output=tmp_path / "patch",
            manifest_output=tmp_path / "manifest",
            forbidden_destination=forbidden,
            contract=contract,
        )


@pytest.mark.parametrize(
    "case",
    [
        "destination_in_source",
        "source_in_destination",
        "patch_in_source",
        "manifest_in_destination",
        "aliased_receipts",
    ],
)
def test_path_containment_and_receipt_aliases_fail_before_copy(
    tmp_path: Path, case: str
) -> None:
    source = _fake_source()
    outer = tmp_path / "outer"
    source_root = outer / "source"
    source_file = source_root / MONODGP_RELATIVE_PATH
    source_file.parent.mkdir(parents=True)
    source_file.write_text(source, encoding="utf-8", newline="")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    preregistration = tmp_path / "runtime_preregistration.json"
    contract = _fake_preregistration(source_root, preregistration, source)
    destination = tmp_path / "destination"
    patch = tmp_path / "variant.patch"
    manifest = tmp_path / "variant.json"
    if case == "destination_in_source":
        destination = source_root / "child"
    elif case == "source_in_destination":
        destination = outer
    elif case == "patch_in_source":
        patch = source_root / "receipt.patch"
    elif case == "manifest_in_destination":
        manifest = destination / "receipt.json"
    else:
        manifest = patch
    with pytest.raises(ValueError, match="contain|outside|differ|distinct"):
        materialize_temporal_source(
            source_root=source_root,
            destination=destination,
            checkpoint=checkpoint,
            runtime_preregistration=preregistration,
            patch_output=patch,
            manifest_output=manifest,
            forbidden_destination=tmp_path / "forbidden",
            contract=contract,
        )
    assert portable_tree_hashes(source_root)[MONODGP_RELATIVE_PATH.as_posix()] == hashlib.sha256(
        source.encode()
    ).hexdigest()
