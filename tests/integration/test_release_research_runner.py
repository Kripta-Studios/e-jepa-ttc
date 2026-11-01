"""Exercise blind diagnostic -> seal -> target join, without external data or checkpoints."""

import hashlib
import json

import numpy as np
import pytest
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.release_research.protocol import sha256, write_json
from scripts.release_research import HEAD, diagnose, evaluate, mechanisms, report, verify_complete


def test_mechanism_bank_seals_all_artifacts(tmp_path):
    output = tmp_path / "mechanisms"
    mechanisms(output)
    assert verify_complete(output)["status"] == "COMPLETE"
    (output / "METRICS.json").write_text("{}")
    with pytest.raises(ValueError, match="identity"):
        verify_complete(output)


def test_frozen_trace_to_target_join_keeps_original_missing(tmp_path):
    screen = tmp_path / "screen"
    screen.mkdir()
    population = tmp_path / "population.npz"
    tokens = ["a:1", "b:1", "b:2"]
    groups = ["a", "b", "b"]
    np.savez(population, token=tokens, group=groups, target=[2.0, 3.0, 4.0])
    records = []
    for number, token in enumerate(tokens):
        raw = np.array([0.04], np.float32)
        prediction = float(phase_to_ttc(torch.from_numpy(raw))[0]) if number != 2 else None
        records.append(
            {
                "token": token,
                "group": groups[number],
                "query_us": 1000000,
                "results": {
                    "original": {
                        "native": [
                            prediction,
                            float(phase_to_ttc(torch.from_numpy(raw * 0.8))[0])
                            if prediction
                            else None,
                            prediction + 1.0 if prediction else None,
                        ]
                    }
                },
                "baseline_exact": True,
            }
        )
        if number == 2:
            continue
        folder = screen / "queries" / hashlib.sha256(token.encode()).hexdigest()[:24]
        (folder / "original").mkdir(parents=True)
        np.savez(
            folder / "original/CONTEXT.npz",
            **{
                "R_CTX.output_raw_location": raw * 0.8,
                HEAD + ".output_raw_location": raw,
                HEAD + ".output_ttc": phase_to_ttc(torch.from_numpy(raw)).numpy(),
            },
        )
        np.savez(folder / "PAIR_INPUTS.npz")
        write_json(folder / "CALLS.json", {"calls": []})
    write_json(screen / "PREDICTIONS.json", {"records": records})
    write_json(
        screen / "FREEZE.json",
        {
            "schema": "phase_canonical_screen149_v1",
            "tokens": tokens,
            "groups": groups,
            "inputs": {str(population): sha256(population)},
        },
    )
    artifacts = {str(p.relative_to(screen)): sha256(p) for p in screen.rglob("*") if p.is_file()}
    write_json(
        screen / "COMPLETE.json",
        {
            "status": "COMPLETE",
            "baseline_exact": True,
            "targets_read": False,
            "artifacts": artifacts,
            "freeze_sha256": sha256(screen / "FREEZE.json"),
        },
    )
    blind, scored = tmp_path / "blind", tmp_path / "scored"
    diagnose(screen, blind)
    assert verify_complete(blind)["status"] == "COMPLETE"
    captured = json.loads((blind / "PREDICTIONS.json").read_text())["records"][0]
    assert captured["RGB"] == records[0]["results"]["original"]["native"][1]
    assert captured["EVENT_CONTEXT"] == records[0]["results"]["original"]["native"][2]
    assert captured["RGB"] != captured["EVENT_CONTEXT"]
    assert '"target":' not in (blind / "PREDICTIONS.json").read_text()
    evaluate(blind, population, scored)
    assert verify_complete(scored)["status"] == "COMPLETE"
    assert '"newly_missing": 0' in (scored / "METRICS.json").read_text()
    report(scored, tmp_path / "report")
    assert verify_complete(tmp_path / "report")["status"] == "COMPLETE"
    with pytest.raises(ValueError, match="fresh"):
        diagnose(screen, blind)
