"""Revalidate acknowledged historical producer bindings without expert inference."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256

from .coordination import verified_ack


def verify_acknowledged_producers(
    ack_path: Path, ack_sha256: str, *, resource_ok: Callable[[], bool]
) -> dict:
    """Rehash all ancestor bindings and validate the exact twelve producer families.

    The acknowledged audit is the authority for historical fit exclusions. This
    checks its current bytes and cross-family consistency, not a new fit audit
    reconstructed from labels. External teacher weights are not loaded or
    rehashed unless present in the acknowledged bindings. A resource pause is an
    exception before the next file, never a scientific failure or authorization.
    """
    ack = verified_ack(ack_path, ack_sha256)
    reference = ack["producers"]["authoritative_historical_manifest"]
    manifest = json.loads(Path(reference["path"]).read_text(encoding="utf-8"))
    roles = ack["interfaces"]["role_manifest"]["roles"]
    original, expansion = set(roles["original"]), set(roles["expansion"])
    if len(original) != 9 or original & expansion:
        raise ValueError("acknowledged original/expansion roles overlap or changed")
    rows = manifest["producers"]
    producers = {(p["outer_fold"], p["role"], p["expert"]): p for p in rows}
    expected = {
        (o, r, e)
        for o in range(3)
        for r in ("inner0", "inner1", "inner2", "outer_dev")
        for e in ("A5", "C2F", "PAIR")
    }
    if len(rows) != 36 or set(producers) != expected:
        raise ValueError("complete unique historical producer family grid required")
    bindings = manifest["input_bindings"].values()
    digests = {b["sha256"] for b in bindings}
    if any(p["checkpoint_sha256"] not in digests for p in rows):
        raise ValueError("producer checkpoint absent from authoritative bindings")
    outer_sets = []
    teacher_ids = set()
    for outer in range(3):
        held = set(producers[outer, "outer_dev", "A5"]["split_validation"]["dev_sequence_ids"])
        if not held or not held < original:
            raise ValueError("invalid outer evaluation group")
        outer_sets.append(held)
        for role in ("inner0", "inner1", "inner2", "outer_dev"):
            splits = []
            for expert in ("A5", "C2F"):
                producer = producers[outer, role, expert]
                split, initialization = (
                    producer["split_validation"],
                    producer["initialization_validation"],
                )
                train, dev = set(split["train_sequence_ids"]), set(split["dev_sequence_ids"])
                if (
                    split["split_relationships_validated"] is not True
                    or split["ancestry_validated"] is not True
                    or not train
                    or not dev
                    or train & dev
                    or train & held
                    or train | dev != (original if role == "outer_dev" else original - held)
                    or set(split["excluded_outer_dev_sequence_ids"]) != held
                    or producer["protocol_sha256"] not in digests
                    or initialization["initialization_ancestors"] != []
                    or initialization["checkpoint_initialization_validated"] is not True
                    or initialization["effective_training_sets_validated"] is not True
                ):
                    raise ValueError("historical fit exclusions or initialization changed")
                teacher_ids.add(initialization["representation_teacher_artifact_sha256"])
                splits.append((train, dev))
            if splits[0] != splits[1]:
                raise ValueError("A5/C2F producer training groups differ")
            pair = producers[outer, role, "PAIR"]
            if (
                pair["nested_a5_ancestor_sha256"]
                != producers[outer, role, "A5"]["checkpoint_sha256"]
                or pair["exact_dev_universe_validated"] is not True
            ):
                raise ValueError("PAIR changes its nested A5 ancestor or evaluation universe")
    if set.union(*outer_sets) != original or sum(map(len, outer_sets)) != len(original):
        raise ValueError("outer development groups must partition OLD sequences")
    if len(teacher_ids) != 1:
        raise ValueError("historical families change representation teacher")
    count, total_bytes = 0, 0
    for binding in bindings:
        if not resource_ok():
            raise RuntimeError("PAUSED_RESOURCE: historical binding revalidation")
        path = Path(binding["path"])
        if sha256(path) != binding["sha256"]:
            raise ValueError(f"historical ancestor bytes changed: {path.name}")
        count += 1
        total_bytes += path.stat().st_size
    if verified_ack(ack_path, ack_sha256) != ack:
        raise ValueError("acknowledged authority changed during revalidation")
    return {
        "status": "ACKNOWLEDGED_PRODUCER_BINDINGS_REVERIFIED",
        "ancestry_sha256": reference["sha256"],
        "producers": 36,
        "families": 12,
        "bindings_verified": count,
        "bytes_hashed": total_bytes,
        "representation_teacher_artifact_sha256": next(iter(teacher_ids)),
        "optimizer_updates": 0,
        "models_loaded": False,
        "scientific_execution_authorized": False,
        "claim_limit": (
            "Acknowledged historical audit and bound bytes; no new acquisition-independence "
            "or expanded time authority."
        ),
    }
