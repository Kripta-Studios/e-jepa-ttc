"""Real byte inventory tests; no scientific fits or expert forwards."""

import hashlib

import pytest

from e_jepa_ttc.simplex_t.bundle_creation import BundleMember, create_verified_bundle
from e_jepa_ttc.simplex_t.postprocessing_inventory import inventory_postprocessing


@pytest.fixture
def owned_output(tmp_path):
    output = tmp_path / "owned"
    output.mkdir()
    return output


def test_nested_payloads_and_empty_file(owned_output):
    tmp_path = owned_output
    (tmp_path / "weights").mkdir()
    data = b"fixture" * 200000
    (tmp_path / "weights" / "head.npz").write_bytes(data)
    (tmp_path / "empty.json").write_bytes(b"")
    result = inventory_postprocessing(tmp_path, resource_ok=lambda: True)
    assert result["files"] == 2 and result["bytes"] == len(data)
    assert result["members"]["weights/head.npz"]["sha256"] == hashlib.sha256(data).hexdigest()


@pytest.mark.parametrize("name", ["POSTPROCESSING.json", "head.npz.partial", "head.tmp"])
def test_refuses_unfinished_or_self_manifest(owned_output, name):
    tmp_path = owned_output
    (tmp_path / name).write_bytes(b"fixture")
    with pytest.raises(ValueError, match="unfinished"):
        inventory_postprocessing(tmp_path, resource_ok=lambda: True)


def test_resource_pause_during_payload(owned_output):
    tmp_path = owned_output
    (tmp_path / "head.npz").write_bytes(b"x" * 2000000)
    calls = 0

    def resource():
        nonlocal calls
        calls += 1
        return calls < 5

    with pytest.raises(InterruptedError, match="PAUSED_RESOURCE"):
        inventory_postprocessing(tmp_path, resource_ok=resource)
    assert (tmp_path / "head.npz").stat().st_size == 2000000


def test_mutation_during_hash_is_rejected(owned_output):
    tmp_path = owned_output
    path = tmp_path / "head.npz"
    path.write_bytes(b"old")
    calls = 0

    def resource():
        nonlocal calls
        calls += 1
        if calls == 4:
            path.write_bytes(b"changed")
        return True

    with pytest.raises(ValueError, match="grew|changed"):
        inventory_postprocessing(tmp_path, resource_ok=resource)


def test_inventory_pins_feed_verified_zip(tmp_path):
    output = tmp_path / "postprocessing"
    output.mkdir()
    (output / "weights.npz").write_bytes(b"synthetic weights fixture")
    inventory = inventory_postprocessing(output, resource_ok=lambda: True)

    def verify_inventory():
        assert inventory_postprocessing(output, resource_ok=lambda: True) == inventory

    result = create_verified_bundle(
        tmp_path / "fixture.zip",
        {
            name: BundleMember(output / name, pin["sha256"], pin["bytes"])
            for name, pin in inventory["members"].items()
        },
        validate_inventory_authority=verify_inventory,
        resource_ok=lambda: True,
    )
    assert result["members"] == 1
    assert result["optimizer_updates"] == 0
