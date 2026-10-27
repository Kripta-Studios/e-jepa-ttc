"""Bind a watchdog config to nine existing Linux workers; never start or stop them."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

try:
    from scripts import runpod_event_resource_watchdog_v1 as watchdog
except ModuleNotFoundError:
    import runpod_event_resource_watchdog_v1 as watchdog


def process_environment(pid: int, proc_root: Path) -> dict[str, str]:
    """Read selected inherited settings without publishing unrelated environment data."""
    pairs = (proc_root / str(pid) / "environ").read_bytes().split(b"\0")
    return {
        key.decode(): value.decode()
        for pair in pairs
        if b"=" in pair
        for key, value in [pair.split(b"=", 1)]
    }


def find_worker(
    task: dict[str, Any], run: str, proc_root: Path = Path("/proc")
) -> dict[str, Any]:
    """Reject duplicate owners and bind the exact current command and start time."""
    matches = []
    for directory in proc_root.iterdir():
        if not directory.name.isdecimal():
            continue
        identity = watchdog.proc_identity(int(directory.name), proc_root)
        if identity is None or identity["state"] == "Z":
            continue
        argv = identity["argv"]
        if str(task["entrypoint"]) not in argv:
            continue
        try:
            same = (
                watchdog.option(argv, "--run") == run
                and watchdog.option(argv, "--rotation") == str(task["rotation"])
            )
            if task["fit"] in watchdog.PRODUCERS:
                same = same and watchdog.option(argv, "--fit") == task["fit"]
            else:
                same = same and "--fit" not in argv and "producer" not in argv
        except ValueError:
            continue
        if same:
            matches.append(identity)
    if len(matches) != 1:
        raise ValueError(f"Expected one live owner for {task['id']}, found {len(matches)}")
    return matches[0]


def build(spec: dict[str, Any], proc_root: Path = Path("/proc")) -> dict[str, Any]:
    """Materialize authority hashes and live identities from an operator-owned spec."""
    if spec.get("schema") != "ejepa_event_watchdog_preparation_spec_v1":
        raise ValueError("Unknown preparation spec")
    environment = spec["required_environment"]
    if "PYTHONPATH" not in environment:
        raise ValueError("Explicit inherited PYTHONPATH is required")
    for name, value in environment.items():
        if os.environ.get(name) != value:
            raise ValueError(f"Preparation environment differs: {name}")
    authority = dict(spec["authority"])
    for key in (
        "pod_budget", "downstream_budget", "migration_binding", "legacy_controller_exclusion"
    ):
        path = Path(authority[f"{key}_path"]).resolve(strict=True)
        authority[f"{key}_path"] = str(path)
        authority[f"{key}_sha256"] = watchdog.digest(path)
    pod = watchdog.read(Path(authority["pod_budget_path"]))
    authority["global_deadline_utc"] = pod["execution_deadline_utc"]
    exclusion = watchdog.read(Path(authority["legacy_controller_exclusion_path"]))
    disabled_paths = exclusion.get("disabled_controller_paths", [])
    if not disabled_paths:
        raise ValueError("No explicit legacy-controller exclusion evidence")
    # Independently reject still-running shell controllers even if a stale receipt says PASS.
    for directory in proc_root.iterdir():
        if not directory.name.isdecimal():
            continue
        identity = watchdog.proc_identity(int(directory.name), proc_root)
        if identity and identity["state"] != "Z" and any(
            path in identity["argv"] for path in disabled_paths
        ):
            raise ValueError(f"Legacy controller is still running: PID {directory.name}")
    tasks = []
    for raw in spec["tasks"]:
        task = dict(raw)
        identity = find_worker(task, spec["run_root"], proc_root)
        inherited = process_environment(identity["pid"], proc_root)
        if any(inherited.get(key) != value for key, value in environment.items()):
            raise ValueError(f"Worker environment differs: {task['id']}")
        argv = identity["argv"]
        assignment = Path(watchdog.option(argv, "--assignment")).resolve(strict=True)
        runtime_files = [
            {"path": str(Path(path).resolve(strict=True)), "sha256": watchdog.digest(Path(path))}
            for path in task["runtime_files"]
        ]
        task.update(
            role_schema=(watchdog.PRODUCER_ROLE_SCHEMA if task["fit"] in watchdog.PRODUCERS
                         else watchdog.DOWNSTREAM_ROLE_SCHEMA),
            run=spec["run_root"], pid=identity["pid"],
            pid_start_ticks=identity["start_ticks"],
            pid_started_epoch=identity["started_epoch"],
            pid_cmdline_sha256=identity["cmdline_sha256"], argv=argv, adopted_argv=[argv],
            entrypoint_sha256=watchdog.digest(Path(task["entrypoint"])),
            assignment_path=str(assignment), assignment_sha256=watchdog.digest(assignment),
            runtime_files=runtime_files,
        )
        tasks.append(task)
    return {
        "schema": watchdog.SCHEMA, "bundle_root": spec["bundle_root"],
        "run_root": spec["run_root"], "required_environment": environment,
        "backoff_seconds": 60, "max_retries": 12, "poll_seconds": 10,
        "status_path": spec["status_path"], "lock_path": spec["lock_path"],
        "authority": authority, "tasks": tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to replace an existing watchdog config")
    config = build(watchdog.read(args.spec))
    pending = args.output.with_name(f".{args.output.name}.{os.getpid()}.validation")
    try:
        watchdog.atomic_json(pending, config)
        watchdog.validate_config(pending)
        # Recheck adopted workers after hashes; a cut here must not fabricate an owner.
        for task in config["tasks"]:
            current = watchdog.proc_identity(task["pid"])
            if (current is None or current["state"] == "Z"
                    or current["start_ticks"] != task["pid_start_ticks"]
                    or not watchdog.command_matches(task, current)):
                raise RuntimeError(f"Worker changed during preparation: {task['id']}")
        # Exclusive publication, never overwrite an authority created by another preparer.
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(config, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        pending.unlink(missing_ok=True)
    print(json.dumps({"status": "PASS", "config_sha256": watchdog.digest(args.output),
                      "tasks": len(config["tasks"]), "processes_started": 0,
                      "processes_stopped": 0, "optimizer_updates": 0}))


if __name__ == "__main__":
    main()
