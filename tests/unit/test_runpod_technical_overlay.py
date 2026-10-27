"""Technical overlay routing and filesystem rejection without Linux/GPU side effects."""

import pytest

from scripts.runpod_technical_overlay_v2 import select_mount, validate_command, validate_run


@pytest.mark.parametrize("command", ["producer", "finish-rotation", "", "--help"])
def test_scientific_commands_never_reach_runner(command):
    with pytest.raises(ValueError):
        validate_command([command])


@pytest.mark.parametrize("command", ["preflight", "resume-probe"])
def test_technical_commands_are_admitted(command):
    validate_command([command, "--run", "/opt/ejepa_technical/r4"])


def test_output_paths_are_restricted():
    validate_run("/opt/ejepa_technical/r4", "/opt/ejepa_bundle")
    for target in ["/workspace/r4", "/opt/ejepa_bundle/r4", "/opt/ejepa_technical/../other"]:
        with pytest.raises(ValueError):
            validate_run(target, "/opt/ejepa_bundle")


def test_global_fuse_is_rejected_even_under_opt():
    mounts = (
        "1 0 0:1 / / rw - overlay overlay rw\n"
        "2 1 0:2 / /opt/ejepa_technical rw - fuse.g global rw\n"
    )
    with pytest.raises(ValueError, match="not admitted"):
        select_mount("/opt/ejepa_technical/r4", mounts)
    assert select_mount("/opt/ejepa_bundle", mounts)["filesystem"] == "overlay"
