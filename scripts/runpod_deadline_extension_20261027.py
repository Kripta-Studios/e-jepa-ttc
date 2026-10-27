"""Apply the user's six-hour extension without rewriting historical authorities."""

from __future__ import annotations

import ast
import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def extension_deadline(original: datetime) -> datetime:
    """Admit only the bound Pod2 extension, preserving all prior receipt hashes."""
    path = Path(os.environ["EJEPA_DEADLINE_EXTENSION"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != os.environ["EJEPA_DEADLINE_EXTENSION_SHA256"]:
        raise ValueError("Deadline extension receipt changed")
    value = json.loads(path.read_text())
    old = datetime.fromisoformat(value["previous_deadline_utc"])
    new = datetime.fromisoformat(value["extended_deadline_utc"])
    if not (
        value["schema"] == "ejepa_explicit_deadline_extension_v1"
        and value["pod_id"] == "lreiivpf18vxu7"
        and value["authorized"] is True
        and value["additional_hours"] == 6
        and original == old
        and old.tzinfo is not None
        and new - old == timedelta(hours=6)
        and datetime.now(UTC) < new
    ):
        raise ValueError("Extension is invalid, expired, or for a different deadline")
    for name, expected in value["historical_authority_sha256"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Historical authority changed: {name}")
    return new


def rewritten_function(module: Any, name: str, replacements: dict[str, str]) -> Any:
    """Rewrite only exact named expressions; preserve every other validation."""
    tree = ast.parse(Path(module.__file__).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    targets = {ast.dump(ast.parse(k, mode="eval").body): v for k, v in replacements.items()}
    counts = dict.fromkeys(targets, 0)

    class Rewrite(ast.NodeTransformer):
        def visit(self, node: ast.AST) -> ast.AST:
            key = ast.dump(node)
            if key in targets:
                counts[key] += 1
                return ast.copy_location(ast.parse(targets[key], mode="eval").body, node)
            return super().visit(node)

    result = Rewrite().visit(function)
    if any(n != 1 for n in counts.values()):
        raise ValueError("Expected deadline expression differs from frozen implementation")
    code = ast.fix_missing_locations(ast.Module(body=[result], type_ignores=[]))
    namespace = dict(vars(module), extension_deadline=extension_deadline)
    exec(compile(code, "<authorized-six-hour-extension>", "exec"), namespace)
    return namespace[name]


def install_queue(queue: Any) -> None:
    validator = rewritten_function(queue, "_validate_authority", {
        'datetime.fromisoformat(str(assignment.get("execution_deadline_utc")))':
            'extension_deadline(datetime.fromisoformat(str(assignment.get("execution_deadline_utc"))))',
        'datetime.fromisoformat(str(budget.get("execution_deadline_utc")))':
            'extension_deadline(datetime.fromisoformat(str(budget.get("execution_deadline_utc"))))',
    })

    def validate(assignment_path: Path, budget_path: Path, exclusion_path: Path,
                 run: Path, rotation: int) -> Any:
        deadline, authority = validator(assignment_path, budget_path, exclusion_path, run, rotation)
        queue.atomic_write_json(run / f"DEADLINE_EXTENSION_ACK_R{rotation}.json", {
            "schema": "ejepa_deadline_extension_worker_ack_v1", "pid": os.getpid(),
            "rotation": rotation, "effective_deadline_utc": deadline.isoformat(),
            "extension_sha256": os.environ["EJEPA_DEADLINE_EXTENSION_SHA256"],
            "historical_authority": authority, "acknowledged_utc": datetime.now(UTC).isoformat(),
        })
        return deadline, authority

    queue._validate_authority = validate


def install_watchdog(module: Any) -> None:
    """Replace the base expiry check, retaining all layered integrity checks."""
    import sys

    base = sys.modules["ejepa_event_resource_watchdog_v1_v2"]
    # The original validator is captured by V2. Replace its code in-place so all
    # V2..V12 validation wrappers continue to execute, with the same globals.
    source = Path(base.__file__).with_name("runpod_event_resource_watchdog_v1.py")
    from types import SimpleNamespace
    shadow = SimpleNamespace(**vars(base))
    shadow.__file__ = str(source)
    replacement = rewritten_function(shadow, "validate_config", {
        "parse_deadline(deadline) > datetime.now(UTC)":
            "extension_deadline(parse_deadline(deadline)) > datetime.now(UTC)",
    })
    original = module.validate_config
    # Find the base validator captured by the nested validation wrappers.
    current = original
    while current.__closure__:
        candidates = [cell.cell_contents for cell in current.__closure__
                      if callable(cell.cell_contents) and getattr(cell.cell_contents, "__name__", "") in
                      {"validate_config", "validate"}]
        if len(candidates) != 1:
            raise ValueError("Unexpected watchdog validator chain")
        current = candidates[0]
    current.__globals__["extension_deadline"] = extension_deadline
    current.__code__ = replacement.__code__

    def validate(path: Path) -> dict[str, Any]:
        value = original(path)
        effective = extension_deadline(datetime.fromisoformat(value["authority"]["global_deadline_utc"]))
        if value.get("effective_global_deadline_utc") != effective.isoformat():
            raise ValueError("Effective watchdog deadline differs from authorized extension")
        value["authority"] = dict(value["authority"], global_deadline_utc=effective.isoformat())
        return value

    module.validate_config = validate
