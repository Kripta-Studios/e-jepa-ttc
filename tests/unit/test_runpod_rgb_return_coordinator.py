"""The RGB return coordinator is resumable and fails closed scientifically."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from operational.runpod_migration.rgb_return_coordinator import (
    CONFIG_SCHEMA,
    Coordinator,
    RetryableRemoteError,
    SingleInstanceLock,
)


def test_remote_timeout_retries_but_scientific_timeout_does_not(tmp_path: Path) -> None:
    repository, config = _config(tmp_path)

    def timeout(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(command, 45)

    coordinator = Coordinator(config, repository, runner=timeout)
    with pytest.raises(RetryableRemoteError, match="transport timed out"):
        coordinator._run(["ssh", "example.test"], timeout=45)
    with pytest.raises(subprocess.TimeoutExpired):
        coordinator._run([sys.executable, "scientific_verify.py"], timeout=45)


def _config(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    repository = tmp_path / "repo"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    verifier = scripts / "verify_runpod_rgb_scientific_return_v1.py"
    verifier.write_text("# fixture\n", encoding="utf-8")
    identity = tmp_path / "id_ed25519"
    known_hosts = tmp_path / "known_hosts"
    identity.write_text("fixture", encoding="utf-8")
    known_hosts.write_text("fixture", encoding="utf-8")
    final = repository / "artifacts/runpod_returns/irtz8y3sigd621/final_rgb"
    incoming = final / "incoming"
    config: dict[str, Any] = {
        "schema": CONFIG_SCHEMA,
        "poll_seconds": 60,
        "ssh_host": "root@example.test",
        "ssh_port": 13608,
        "ssh_key": str(identity),
        "ssh_known_hosts": str(known_hosts),
        "remote_python": "/opt/ejepa_env/bin/python",
        "remote_export_entrypoint": (
            "/workspace/ejepa_operations/rgb_archive/scripts/runpod_rgb_complete_export_v1.py"
        ),
        "remote_run_roots": {
            str(rotation): {
                "a5": f"/workspace/ejepa_technical/r{rotation}_rgb_v3",
                "c2f": f"/workspace/ejepa_technical/r{rotation}_c2f_v3",
                "downstream": f"/workspace/ejepa_technical/r{rotation}_rgb_downstream_v1",
            }
            for rotation in (2, 3, 4)
        },
        "remote_sources": ["/workspace/ejepa_operations/rgb_queue.py"],
        "remote_authorization_root": "/workspace/ejepa_operations",
        "remote_coverage": "/workspace/ejepa_operations/RGB_DOWNSTREAM_COVERAGE.json",
        "remote_archive": "/opt/ejepa_rgb_return/RGB_COMPLETE.tar",
        "local_final_rgb_dir": str(final),
        "local_incoming_dir": str(incoming),
        "local_status_path": str(incoming / "COORDINATOR_STATUS.json"),
        "local_history_path": str(incoming / "COORDINATOR_HISTORY.jsonl"),
        "local_lock_path": str(incoming / "COORDINATOR.lock"),
        "local_python_executable": sys.executable,
        "scientific_verifier": str(verifier),
        "scientific_proof_output": str(final / "scientific_proofs/REUSE.json"),
    }
    return repository, config


class FakeRemote:
    def __init__(
        self,
        payload: bytes,
        *,
        waiting: bool = False,
        scientific_code: int = 0,
        existing_export: bool = False,
        archive: str = "/opt/ejepa_rgb_return/RGB_COMPLETE.tar",
    ) -> None:
        self.payload = payload
        self.sha = hashlib.sha256(payload).hexdigest()
        self.waiting = waiting
        self.scientific_code = scientific_code
        self.existing_export = existing_export
        self.archive = archive
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        if (
            command[0] == "ssh"
            and "runpod_rgb_complete_export_v1.py" not in command[-1]
            and ".tar.json" in command[-1]
        ):
            if not self.existing_export:
                return subprocess.CompletedProcess(
                    command, 0, json.dumps({"status": "MISSING"}), ""
                )
            value = {
                "schema": "ejepa_cloud_rgb_complete_export_v1",
                "status": "READY",
                "archive": self.archive,
                "receipt": self.archive + ".json",
                "archive_sha256": self.sha,
                "archive_bytes": len(self.payload),
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(value), "")
        if command[0] == "ssh" and "runpod_rgb_complete_export_v1.py" not in command[-1]:
            state = "WAITING" if self.waiting else "COMPLETE"
            return subprocess.CompletedProcess(command, 0, json.dumps({"status": state}), "")
        if command[0] == "ssh":
            value = {
                "schema": "ejepa_cloud_rgb_complete_export_v1",
                "status": "READY",
                "archive": self.archive,
                "receipt": self.archive + ".json",
                "archive_sha256": self.sha,
                "archive_bytes": len(self.payload),
            }
            return subprocess.CompletedProcess(command, 0, json.dumps(value), "")
        if command[0] == "scp":
            destination = Path(command[-1])
            if command[-2].endswith(".json"):
                value = {
                    "schema": "ejepa_cloud_rgb_complete_export_v1",
                    "status": "READY",
                    "archive": self.archive,
                    "receipt": self.archive + ".json",
                    "archive_sha256": self.sha,
                    "archive_bytes": len(self.payload),
                }
                destination.write_text(json.dumps(value), encoding="utf-8")
            else:
                destination.write_bytes(self.payload)
            return subprocess.CompletedProcess(command, 0, "", "")
        output = Path(command[command.index("--output") + 1])
        if self.scientific_code == 0:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(
                    {
                        "schema": "ejepa_local_rgb_scientific_reusability_v1",
                        "status": "PASS",
                        "archive_sha256": self.sha,
                        "scientific_reusability_verified": True,
                        "selected_root_path_maps_verified": True,
                        "three_root_path_maps_verified": True,
                        "active_fits_modified": False,
                    }
                ),
                encoding="utf-8",
            )
        return subprocess.CompletedProcess(command, self.scientific_code, "", "scientific failure")


class FakeAdmit:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[dict[str, Any]] = []

    def __call__(self, _source: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        self.root.mkdir(parents=True, exist_ok=True)
        return {
            "status": "PASS",
            "snapshot_root": str(self.root),
            "cleanup": {
                "status": "ELIGIBLE" if "reuse_verification" in kwargs else "NOT_EVALUATED"
            },
            "eviction": (
                {"status": "PASS"}
                if kwargs.get("expected_rotations") == (4,) and "reuse_verification" in kwargs
                else None
            ),
        }


def _individual_r4_config(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    repository, config = _config(tmp_path)
    base = repository / "artifacts/runpod_returns/irtz8y3sigd621/individual_r4"
    incoming = base / "incoming"
    config.update(
        {
            "rotations": [4],
            "remote_run_roots": {"4": config["remote_run_roots"]["4"]},
            "remote_archive": "/opt/ejepa_rgb_return/R4_COMPLETE.tar",
            "local_final_rgb_dir": str(base),
            "local_incoming_dir": str(incoming),
            "local_status_path": str(incoming / "COORDINATOR_STATUS.json"),
            "local_history_path": str(incoming / "COORDINATOR_HISTORY.jsonl"),
            "local_lock_path": str(incoming / "COORDINATOR.lock"),
            "scientific_proof_output": str(base / "scientific_proofs/REUSE.json"),
        }
    )
    return repository, config


def test_individual_r4_config_is_singleton_closed_and_completes(tmp_path: Path) -> None:
    repository, config = _individual_r4_config(tmp_path)
    remote = FakeRemote(b"individual-r4-tar", archive="/opt/ejepa_rgb_return/R4_COMPLETE.tar")
    admit = FakeAdmit(tmp_path / "r4-extracted")
    coordinator = Coordinator(config, repository, runner=remote, admit=admit)
    assert coordinator.config["rotations"] == (4,)
    assert len(coordinator.config["remote"]["result_paths"]) == 1
    argv = coordinator.config["remote"]["exporter_argv"]
    assert argv[argv.index("--rotation") + 1] == "4"
    assert argv[argv.index("--output") + 1] == "/opt/ejepa_rgb_return/R4_COMPLETE.tar"
    assert coordinator.run(once=True) == 0
    status = json.loads(Path(config["local_status_path"]).read_text(encoding="utf-8"))
    assert status["raw_r4_eviction_eligibility"] == "PASS_EXTERNAL_ACTION_ONLY"


def test_individual_r4_config_rejects_foreign_rotation_root(tmp_path: Path) -> None:
    repository, config = _individual_r4_config(tmp_path)
    config["remote_run_roots"]["3"] = config["remote_run_roots"]["4"]
    with pytest.raises(ValueError, match="equal the selected rotations"):
        Coordinator(config, repository)


def test_waits_for_all_three_complete_results_without_export(tmp_path: Path) -> None:
    repository, config = _config(tmp_path)
    remote = FakeRemote(b"tar", waiting=True)
    coordinator = Coordinator(config, repository, runner=remote)
    assert coordinator.run(once=True) == 3
    assert json.loads(Path(config["local_status_path"]).read_text())["stage"] == (
        "WAITING_REMOTE_RESULTS"
    )
    assert not any("runpod_rgb_complete_export_v1.py" in command[-1] for command in remote.commands)


def test_reuses_a_complete_partial_then_verifies_science_before_eligibility(
    tmp_path: Path,
) -> None:
    repository, config = _config(tmp_path)
    payload = b"complete-rgb-tar"
    remote = FakeRemote(payload)
    incoming = Path(config["local_incoming_dir"])
    incoming.mkdir(parents=True)
    (incoming / "RGB_COMPLETE.tar.partial").write_bytes(payload)
    admit = FakeAdmit(tmp_path / "extracted")
    coordinator = Coordinator(config, repository, runner=remote, admit=admit)
    assert coordinator.run(once=True) == 0
    assert len(admit.calls) == 2
    assert "reuse_verification" not in admit.calls[0]
    assert "reuse_verification" in admit.calls[1]
    archive_scps = [
        command
        for command in remote.commands
        if command[0] == "scp" and not command[-2].endswith(".json")
    ]
    assert archive_scps == []
    assert (incoming / "RGB_COMPLETE.tar").read_bytes() == payload


def test_scientific_failure_is_terminal_and_never_requests_cleanup_eligibility(
    tmp_path: Path,
) -> None:
    repository, config = _config(tmp_path)
    remote = FakeRemote(b"complete-rgb-tar", scientific_code=2)
    admit = FakeAdmit(tmp_path / "extracted")
    coordinator = Coordinator(config, repository, runner=remote, admit=admit)
    assert coordinator.run(once=True) == 5
    assert len(admit.calls) == 1
    assert "reuse_verification" not in admit.calls[0]
    status = json.loads(Path(config["local_status_path"]).read_text())
    assert status["stage"] == "SCIENTIFIC_VERIFICATION_FAILED"
    assert status["remote_deletion_performed"] is False


def test_existing_remote_export_is_reused_after_local_restart(tmp_path: Path) -> None:
    repository, config = _config(tmp_path)
    remote = FakeRemote(b"complete-rgb-tar", existing_export=True)
    admit = FakeAdmit(tmp_path / "extracted")
    coordinator = Coordinator(config, repository, runner=remote, admit=admit)
    assert coordinator.run(once=True) == 0
    exporter_calls = [
        command
        for command in remote.commands
        if command[0] == "ssh" and "runpod_rgb_complete_export_v1.py" in command[-1]
    ]
    assert exporter_calls == []


def test_connection_failure_is_retryable_and_durable(tmp_path: Path) -> None:
    repository, config = _config(tmp_path)

    def disconnected(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 255, "", "connection lost")

    coordinator = Coordinator(config, repository, runner=disconnected)
    assert coordinator.run(once=True) == 4
    status = json.loads(Path(config["local_status_path"]).read_text())
    assert status["stage"] == "RETRYABLE_CONNECTION_ERROR"


def test_poll_interval_is_fixed_to_sixty_seconds(tmp_path: Path) -> None:
    repository, config = _config(tmp_path)
    config["poll_seconds"] = 30
    try:
        Coordinator(config, repository)
    except ValueError as error:
        assert "poll interval" in str(error)
    else:
        raise AssertionError("Non-60-second RGB polling was accepted")


def test_single_process_lock_rejects_a_second_coordinator(tmp_path: Path) -> None:
    lock = tmp_path / "COORDINATOR.lock"
    with SingleInstanceLock(lock):
        with pytest.raises(RuntimeError, match="owns the lock"):
            with SingleInstanceLock(lock):
                raise AssertionError("Second coordinator acquired the same lock")


def test_ssh_and_scp_quote_windows_known_hosts_paths_with_spaces(tmp_path: Path) -> None:
    repository, config = _config(tmp_path)
    coordinator = Coordinator(config, repository)
    expected = f'UserKnownHostsFile="{Path(config["ssh_known_hosts"]).as_posix()}"'
    for command in (coordinator._ssh_base(), coordinator._scp_base()):
        index = command.index("StrictHostKeyChecking=yes")
        assert command[index - 1] == "-o"
        assert command[index + 1] == "-o"
        assert command[index + 2] == expected
        assert "\\" not in command[index + 2]
