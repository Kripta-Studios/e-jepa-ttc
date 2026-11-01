"""Archived dependency proofs reject missing, duplicated, or changed source bytes."""

import hashlib
import io
import tarfile

import pytest

from scripts.audit_phase_native_source_closure import archive_source_hashes, formatting_equivalent


def make_archive(path, names):
    with tarfile.open(path, "w") as archive:
        for name, payload in names:
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_archive_source_bytes_are_bound_to_entire_archive_hash(tmp_path):
    path = tmp_path / "frozen.tar"
    digest = make_archive(path, [("frozen/src/model.py", b"original")])
    assert archive_source_hashes(path, digest, ["src/model.py"]) == {
        "src/model.py": hashlib.sha256(b"original").hexdigest()
    }
    with pytest.raises(ValueError, match="archive hash"):
        archive_source_hashes(path, "wrong", ["src/model.py"])


@pytest.mark.parametrize("entries", [[], [("one/src/model.py", b"a"), ("two/src/model.py", b"a")]])
def test_missing_or_ambiguous_archived_source_is_rejected(tmp_path, entries):
    path = tmp_path / "frozen.tar"
    digest = make_archive(path, entries)
    with pytest.raises(ValueError, match="unique archived source"):
        archive_source_hashes(path, digest, ["src/model.py"])


def test_formatting_proof_accepts_wrapping_but_rejects_numerical_and_comment_changes():
    original = "value = func(\n    first, second\n)\n"
    assert formatting_equivalent(original, "value = func(first, second)\n")
    assert not formatting_equivalent(original, "value = func(second, first)\n")
    assert not formatting_equivalent(original, original + "# unreviewed annotation\n")
