"""Exercise the actual receiver protocol, including resumable binary payloads."""

import hashlib
import json
import subprocess
import sys

import pytest

from scripts import runpod_event_transfer_v3
from scripts.runpod_event_transfer_v2 import transfer_v1


@pytest.mark.parametrize("offset", [0, 3])
@pytest.mark.parametrize(
    "receiver", [transfer_v1.REMOTE_RECEIVER, runpod_event_transfer_v3.REMOTE_RECEIVER]
)
def test_receiver_emits_line_delimited_acknowledgements(tmp_path, offset, receiver):
    payload = b"sample\x00\nbytes"
    if offset:
        (tmp_path / ".sample.bin.partial").write_bytes(payload[:offset])
    header = {
        "relative": "sample.bin",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    process = subprocess.run(
        [sys.executable, "-u", "-c", receiver, str(tmp_path)],
        input=json.dumps(header).encode() + b"\n" + payload[offset:] + b'{"command":"END"}\n',
        capture_output=True,
        timeout=15,
        check=True,
    )
    replies = [json.loads(line) for line in process.stdout.splitlines()]
    assert [reply["status"] for reply in replies] == ["OFFSET", "OK", "COMPLETE"]
    assert replies[0]["offset"] == offset
    assert (tmp_path / "sample.bin").read_bytes() == payload
    assert (
        json.loads((tmp_path / "sample.bin.transfer.json").read_text())["sha256"]
        == header["sha256"]
    )
