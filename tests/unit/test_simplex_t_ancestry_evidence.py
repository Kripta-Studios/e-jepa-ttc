"""Bounded producer-grid checks with synthetic metadata and no model loading."""

import json

import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t.ancestry_evidence import verify_acknowledged_producers


@pytest.mark.parametrize("change", ["", "missing", "pair", "split", "bytes", "resource"])
def test_acknowledged_grid_and_bytes(tmp_path, monkeypatch, change):
    source = tmp_path / "source.bin"
    source.write_bytes(b"synthetic producer binding")
    digest = sha256(source)
    original = [str(n) for n in range(9)]
    rows = []
    for outer in range(3):
        held = original[outer * 3 : outer * 3 + 3]
        rest = [s for s in original if s not in held]
        for inner, role in enumerate(("inner0", "inner1", "inner2", "outer_dev")):
            dev = held if inner == 3 else rest[inner * 2 : inner * 2 + 2]
            train = [s for s in rest if s not in dev]
            for expert in ("A5", "C2F", "PAIR"):
                rows.append(
                    dict(
                        outer_fold=outer,
                        role=role,
                        expert=expert,
                        checkpoint_sha256=digest,
                        protocol_sha256=digest,
                        nested_a5_ancestor_sha256=digest,
                        exact_dev_universe_validated=True,
                        split_validation=dict(
                            train_sequence_ids=train,
                            dev_sequence_ids=dev,
                            excluded_outer_dev_sequence_ids=held,
                            split_relationships_validated=True,
                            ancestry_validated=True,
                        ),
                        initialization_validation=dict(
                            initialization_ancestors=[],
                            checkpoint_initialization_validated=True,
                            effective_training_sets_validated=True,
                            representation_teacher_artifact_sha256="t" * 64,
                        ),
                    )
                )
    if change == "missing":
        rows.pop()
    elif change == "pair":
        rows[2]["nested_a5_ancestor_sha256"] = "x" * 64
    elif change == "split":
        rows[0]["split_validation"]["train_sequence_ids"] = original
    elif change == "bytes":
        source.write_bytes(b"changed")
    path = tmp_path / "ancestry.json"
    path.write_text(
        json.dumps(
            dict(producers=rows, input_bindings={"source": dict(path=str(source), sha256=digest)})
        ),
        encoding="utf-8",
    )
    ack = dict(
        producers=dict(authoritative_historical_manifest=dict(path=str(path), sha256=sha256(path))),
        interfaces=dict(role_manifest=dict(roles=dict(original=original, expansion=["extra"]))),
    )
    monkeypatch.setattr("e_jepa_ttc.simplex_t.ancestry_evidence.verified_ack", lambda *args: ack)
    if change:
        with pytest.raises(RuntimeError if change == "resource" else ValueError):
            verify_acknowledged_producers(path, "a" * 64, resource_ok=lambda: change != "resource")
    else:
        result = verify_acknowledged_producers(path, "a" * 64, resource_ok=lambda: True)
        assert result["producers"] == 36 and result["bindings_verified"] == 1
        assert result["models_loaded"] is False
        assert result["scientific_execution_authorized"] is False
