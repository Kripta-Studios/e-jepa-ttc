"""Bind the explicit Pod2 authorization to three rotation-local downstream owners."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_once(path: Path, value: dict[str, object]) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"Downstream authority differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.pending")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def bind(run: Path, authorization: Path, budget_source: Path, output: Path) -> None:
    run, authorization, budget_source = map(
        Path.resolve, (run, authorization, budget_source)
    )
    authorized = json.loads(authorization.read_text(encoding="utf-8"))
    source = json.loads(budget_source.read_text(encoding="utf-8"))
    migration = run / "EVENT_MIGRATION_BINDING.json"
    if not (
        authorized.get("schema") == "ejepa_event_execution_authorization_v1"
        and authorized.get("status") == "AUTHORIZED"
        and authorized.get("downstream_queue_authorized") == ["PAIR_E", "E_CTX", "EVENT_OOF"]
        and authorized.get("maximum_total_usd") == 24
        and source.get("schema") == "ejepa_event_pod_budget_v1"
        and source.get("maximum_total_usd") == 24
        and source.get("execution_deadline_utc") == authorized.get("execution_deadline_utc")
    ):
        raise ValueError("Explicit EVENT downstream authorization differs")
    budget = {
        "schema": "ejepa_event_cloud_budget_authorization_v1",
        "execution_authorized": True,
        "maximum_total_usd": 24,
        "execution_deadline_utc": source["execution_deadline_utc"],
        "pod_id": source["pod_id"],
        "source_budget_sha256": digest(budget_source),
        "source_authorization_sha256": digest(authorization),
        "shared_with_six_event_producers": True,
    }
    write_once(output / "EVENT_BUDGET_24USD.json", budget)
    for rotation in (2, 3, 4):
        task = [{"rotation": rotation, "fit": "EVENT_DOWNSTREAM"}]
        folder = output / f"r{rotation}_downstream"
        exclusion = {
            "schema": "ejepa_event_downstream_local_exclusion_v1",
            "status": "EXCLUDED_FROM_LOCAL_EXECUTION",
            "tasks": task,
            "scientific_config_sha256": (
                "3268e2998391b40bb72b07b6fe2bfc873f3bf61f159ca581b28575ba068d767e"
            ),
        }
        exclusion_path = folder / "LOCAL_QUEUE_EXCLUSION.json"
        write_once(exclusion_path, exclusion)
        assignment = {
            "schema": "ejepa_cloud_assignment_v1",
            "execution_authorized": True,
            "tasks": task,
            "migration_binding_sha256": digest(migration),
            "local_queue_exclusion_receipt_sha256": digest(exclusion_path),
            "maximum_compute_usd": 8,
            "execution_deadline_utc": source["execution_deadline_utc"],
            "shared_budget_authorization_sha256": digest(output / "EVENT_BUDGET_24USD.json"),
        }
        write_once(folder / "EVENT_DOWNSTREAM_ASSIGNMENT.json", assignment)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--budget-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bind(args.run, args.authorization, args.budget_source, args.output)
