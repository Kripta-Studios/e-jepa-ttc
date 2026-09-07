"""Real ZIP byte verification and failure retention, without scientific completion claims."""

import hashlib
import zipfile

import pytest

from e_jepa_ttc.simplex_t.bundle_creation import BundleMember, create_verified_bundle


@pytest.fixture
def inputs(tmp_path):
    path = tmp_path / "source.bin"
    payload = b"bounded inventory fixture\n" * 100
    path.write_bytes(payload)
    member = BundleMember(path, hashlib.sha256(payload).hexdigest(), len(payload))
    return tmp_path / "results.zip", {"evidence/source.bin": member}


def test_created_zip_matches_exact_inventory(inputs):
    archive, members = inputs
    calls = []
    result = create_verified_bundle(
        archive,
        members,
        validate_inventory_authority=lambda: calls.append(True),
        resource_ok=lambda: True,
    )
    assert len(calls) == 2
    assert result["sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert result["members"] == 1 and result["optimizer_updates"] == 0
    assert not archive.with_suffix(".zip.partial").exists()
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == list(members)
        assert (
            bundle.read("evidence/source.bin") == members["evidence/source.bin"].path.read_bytes()
        )


@pytest.mark.parametrize("failure", ["hash", "size", "authority", "pause", "final", "partial"])
def test_failure_never_publishes_unverified_bundle(inputs, failure):
    archive, members = inputs
    member = members["evidence/source.bin"]
    partial = archive.with_suffix(".zip.partial")
    if failure == "hash":
        members["evidence/source.bin"] = BundleMember(member.path, "f" * 64, member.bytes)
    elif failure == "size":
        members["evidence/source.bin"] = BundleMember(member.path, member.sha256, member.bytes + 1)
    elif failure == "final":
        archive.write_bytes(b"preserved")
    elif failure == "partial":
        partial.write_bytes(b"preserved")

    def authority():
        if failure == "authority":
            raise ValueError("unapproved inventory")

    with pytest.raises((ValueError, InterruptedError, FileExistsError)):
        create_verified_bundle(
            archive,
            members,
            validate_inventory_authority=authority,
            resource_ok=lambda: failure != "pause",
        )
    if failure == "final":
        assert archive.read_bytes() == b"preserved"
    else:
        assert not archive.exists()
    if failure == "hash":
        assert partial.exists()
    if failure == "partial":
        assert partial.read_bytes() == b"preserved"


def test_changed_authority_after_writing_retains_partial(inputs):
    archive, members = inputs
    calls = []

    def authority():
        calls.append(True)
        if len(calls) == 2:
            raise ValueError("inventory changed")

    with pytest.raises(ValueError, match="inventory changed"):
        create_verified_bundle(
            archive, members, validate_inventory_authority=authority, resource_ok=lambda: True
        )
    assert not archive.exists() and archive.with_suffix(".zip.partial").exists()


def test_bytes_changed_at_authority_boundary_are_verified_before_publication(inputs):
    archive, members = inputs
    calls = []

    def authority():
        calls.append(True)
        if len(calls) == 2:
            with zipfile.ZipFile(archive.with_suffix(".zip.partial"), "a") as bundle:
                bundle.writestr("unexpected.bin", b"changed")

    with pytest.raises(ValueError, match="missing, extra or duplicate"):
        create_verified_bundle(
            archive, members, validate_inventory_authority=authority, resource_ok=lambda: True
        )
    assert not archive.exists()


@pytest.mark.parametrize("name", ["../escape", "C:/file", "folder\\file", "/absolute"])
def test_unsafe_names_rejected_before_authority_or_writes(inputs, name):
    archive, members = inputs
    with pytest.raises(ValueError):
        create_verified_bundle(
            archive,
            {name: next(iter(members.values()))},
            validate_inventory_authority=lambda: pytest.fail("called"),
            resource_ok=lambda: True,
        )
    assert not archive.exists() and not archive.with_suffix(".zip.partial").exists()
