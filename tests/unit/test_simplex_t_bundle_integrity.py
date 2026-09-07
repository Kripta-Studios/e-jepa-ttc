"""Essential ZIP verification detects byte changes even with a valid ZIP CRC."""

import hashlib
import zipfile

import pytest

from e_jepa_ttc.simplex_t.bundle_integrity import verify_bundle


@pytest.mark.parametrize("mode", ["ok", "changed", "extra", "missing", "pause", "unsafe", "alias"])
def test_bundle_payload_inventory(tmp_path, mode):
    archive = tmp_path / "bundle.zip"
    payload = b"pinned scientific evidence"
    expected = {"evidence.json": hashlib.sha256(payload).hexdigest()}
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        if mode != "missing":
            bundle.writestr(
                "evidence.json", b"different bytes valid CRC" if mode == "changed" else payload
            )
        if mode == "extra":
            bundle.writestr("unexpected.json", payload)
    if mode == "unsafe":
        expected = {"../evidence.json": expected["evidence.json"]}
    if mode == "alias":
        expected["EVIDENCE.json"] = expected["evidence.json"]
    if mode == "ok":
        report = verify_bundle(archive, expected, resource_ok=lambda: True)
        assert report["members"] == 1 and report["uncompressed_bytes"] == len(payload)
    else:
        with pytest.raises((ValueError, InterruptedError)):
            verify_bundle(archive, expected, resource_ok=lambda: mode != "pause")
