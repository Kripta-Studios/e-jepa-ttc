"""Solid packing preserves exact bytes and rejects unsafe or incomplete extraction."""

import json
import subprocess
import sys
import zipfile

import pytest

from operational.joint_measurement.solid_delivery import write_solid_archive
from operational.joint_measurement.unpack_solid import safe_name, verify


def test_solid_round_trip_and_standalone_extractor(tmp_path):
    source = tmp_path / "input.bin"
    source.write_bytes(bytes(range(256)) * 30)
    archive = tmp_path / "results.zip"
    files = {"source/á file.bin": source, "campaign/same.bin": source}
    receipt = write_solid_archive(archive, files, {"no_claim": True})
    assert receipt["status"] == "PASS"
    assert receipt["payload_members"] == 2
    with zipfile.ZipFile(archive) as outer:
        script = tmp_path / "unpack.py"
        script.write_bytes(outer.read("unpack.py"))
    extracted = tmp_path / "separate"
    result = subprocess.run(
        [sys.executable, "-B", str(script), str(archive), str(extracted)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["zip_sha256"] == receipt["zip_sha256"]
    for name in files:
        assert (extracted / name).read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        verify(archive, extracted)
    with zipfile.ZipFile(archive, "a") as outer:
        outer.writestr("unexpected", "not in manifest")
    with pytest.raises(ValueError, match="membership"):
        verify(archive)


@pytest.mark.parametrize("name", ["../x", "/absolute", "C:/x", "a\\b", "a/../b", "a. "])
def test_solid_rejects_escape_or_windows_alias(name):
    with pytest.raises(ValueError):
        safe_name(name)
