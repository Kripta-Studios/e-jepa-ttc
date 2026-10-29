"""Synthetic historical diagnostic contracts; no expert inference or updates."""

import json
from pathlib import Path

import numpy as np
import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t.replay_evidence import verify_coherent_replay, verify_historical_replay


@pytest.mark.parametrize("failure", ["", "prediction", "input", "tokens", "bytes"])
def test_historical_evidence(tmp_path: Path, failure: str) -> None:
    families, sources, raw, positions, precision, pairs = {}, {}, [], [], [], []
    for outer in range(3):
        for slot, inner in enumerate(("inner0", "inner1", "inner2", "outer_dev")):
            family = f"outer{outer}/{inner}"
            count = 6 if outer == 0 else 5
            tokens = [f"synthetic_{outer}_{slot}_{q}" for q in range(count)]
            families[family] = tokens
            name = family.replace("/", "_") + ".npz"
            np.savez(tmp_path / name, tokens=tokens, expected_ttc=np.ones((count, 3)))
            sources[name] = sha256(tmp_path / name)
            for token in tokens:
                raw.append(
                    {
                        "token": token,
                        "results": [
                            {
                                "window": w,
                                "bit_identical": failure != "input",
                                "max_abs_by_channel": [0.0],
                            }
                            for w in range(3)
                        ],
                    }
                )
                positions.append(
                    {
                        "family": family,
                        "token": token,
                        "max_abs_pair_token_by_cudnn_tf32": {"True": 0.0},
                        "abs_phase_error_by_cudnn_tf32": {"True": 0.0},
                    }
                )
            for expert in ("A5", "C2F"):
                precision.append(
                    {
                        "family": family,
                        "expert": expert,
                        "modes": {
                            "bf16_cudnn_tf32": {
                                "ttc": [2.0 if failure == "prediction" else 1.0] * count
                            }
                        },
                    }
                )
            pairs.append(
                {"family": family, "tokens": tokens, "pair_from_signed_features_ttc": [1.0] * count}
            )
    if failure == "tokens":
        positions[0]["token"] = "not_in_cohort"
    payloads = {
        "QUERY_CONTEXT_RAW_INPUT_64.json": {"queries": raw},
        "A5_HISTORICAL_BATCH_POSITION.json": {"results": positions},
        "HISTORICAL_DETERMINISTIC_VALIDATION.json": {"results": precision},
        "PAIR_HISTORICAL_GPU_LAYOUT.json": {"results": pairs},
    }
    for name, value in payloads.items():
        (tmp_path / name).write_text(json.dumps(value), encoding="utf-8")
        sources[name] = sha256(tmp_path / name)
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "status": "HISTORICAL64_COMPONENT_REPLAY_PARITY_PASSED",
                "families": families,
                "sources": sources,
            }
        ),
        encoding="utf-8",
    )
    pin = sha256(report)
    if failure == "bytes":
        (tmp_path / "PAIR_HISTORICAL_GPU_LAYOUT.json").write_text("{}", encoding="utf-8")
    if failure:
        with pytest.raises(ValueError):
            verify_historical_replay(tmp_path, report, pin)
    else:
        result = verify_historical_replay(tmp_path, report, pin)
        assert result["historical_predictions"] == 192
        assert result["scientific_stage_authorized"] is False


@pytest.mark.parametrize("failure", ["", "values", "dtype", "producer", "bytes", "runtime"])
def test_coherent_saved_array_validation(tmp_path: Path, monkeypatch, failure: str) -> None:
    # Historical validation is independently tested above; isolate coherent binding here.
    monkeypatch.setattr(
        "e_jepa_ttc.simplex_t.replay_evidence.verify_historical_replay", lambda *args: {}
    )
    old = tmp_path / "old"
    new = tmp_path / "new"
    old.mkdir()
    new.mkdir()
    families, sources, records, diagnostic = {}, {}, [], []
    for outer in range(3):
        for inner in ("inner0", "inner1", "inner2", "outer_dev"):
            family = f"outer{outer}/{inner}"
            count = 6 if outer == 0 else 5
            tokens = [f"{family}_{i}" for i in range(count)]
            families[family] = tokens
            name = family.replace("/", "_") + ".npz"
            ttc = np.ones((count, 3), dtype=np.float32)
            pair = np.ones((count, 133), dtype=np.float32)
            np.savez(old / name, actual_ttc=ttc, pair_features=pair)
            sources[f"old/{name}"] = sha256(old / name)
            np.savez(
                new / name,
                expert_ttc=ttc * (2 if failure == "values" else 1),
                pair_features=pair,
                tokens=tokens,
                known=np.ones((count, 2), dtype=bool),
                features145=np.ones(
                    (count, 145), dtype=np.float64 if failure == "dtype" else np.float32
                ),
            )
            producers = {expert: "a" * 64 for expert in ("A5", "C2F", "PAIR")}
            diagnostic.append({"family": family, "checkpoint_sha256": producers})
            records.append(
                {
                    "family": family,
                    "rows": count,
                    "checkpoint_sha256": {} if failure == "producer" else producers,
                    "output_sha256": "b" * 64 if failure == "bytes" else sha256(new / name),
                }
            )
    diagnostic_path = old / "REPLAY_DIAGNOSTIC.json"
    diagnostic_path.write_text(json.dumps({"results": diagnostic}), encoding="utf-8")
    sources["old/REPLAY_DIAGNOSTIC.json"] = sha256(diagnostic_path)
    historical = old / "report.json"
    historical.write_text(json.dumps({"families": families, "sources": sources}), encoding="utf-8")
    report = new / "QA.json"
    report.write_text(
        json.dumps(
            {
                "status": "COHERENT_FP32_CURRENT_EXTRACTOR_CHECKED",
                "rows": 64,
                "all_exact": True,
                "optimizer_updates": 0,
                "results": records,
                "runtime": {
                    "precision": "FP32",
                    "cudnn_tf32": failure == "runtime",
                    "matmul_tf32": False,
                },
            }
        ),
        encoding="utf-8",
    )
    args = (tmp_path, historical, sha256(historical), report, sha256(report))
    if failure:
        with pytest.raises(ValueError):
            verify_coherent_replay(*args)
    else:
        assert verify_coherent_replay(*args)["new_optimizer_updates"] == 0
