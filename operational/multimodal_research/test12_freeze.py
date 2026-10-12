"""Freeze one development-selected recipe and label-free official inputs before inference."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from e_jepa_ttc.rgb_port.data import history_observations
from operational.multimodal_validation.common import ROOT, RUN, V12, admission, read, sha, write
from operational.rgb_port import transfer as engine
from operational.rgb_port.infer_experts import _observation_signature

from . import test12_native_inputs as native
from .common import OUT
from .test12_infer import DEST

REVISION = "a814a946e60d37f7242358af5f344542177e50c5"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    args = parser.parse_args()
    DEST.mkdir(exist_ok=True)
    freeze_path = DEST / "MODEL_INPUT_FREEZE.json"
    assert not freeze_path.exists(), "Existing model/input freeze is immutable"
    parity_path = OUT / "test12_readiness/NATIVE_EVENT_PARITY.json"
    parity = read(parity_path)
    assert parity["status"] == "PASS" and parity["H_history_identities_exact"]
    assert parity["native_source_sha256"] == sha(Path(native.__file__))
    candidate_path = OUT / "studies/PHASE_ONLY/S7/V_RESULTS.json"
    candidate = read(candidate_path)
    assert candidate["candidate"] == "R_ANCHOR_EVENT" and candidate["complementary_event_gate"]
    rgb_index_path = OUT / "test12_readiness/LABEL_FREE_RGB_INDEX.parquet"
    readiness = read(OUT / "test12_readiness/READINESS.json")
    assert sha(rgb_index_path) == readiness["index_sha256"]
    rgb_index = pd.read_parquet(rgb_index_path)
    metadata_path = V12 / "artifacts/sota_evidence_20261010/sources/test_inputs.parquet"
    old_path = V12 / "artifacts/garlttc_submission_20261010/INPUT_FREEZE.json"
    old = read(old_path)
    assert sha(metadata_path) == old["input_sha256"] == readiness["source_sha256"]
    jobs = [
        native.native_job(row, args.media_root)
        for row in pd.read_parquet(metadata_path, columns=list(native.EVENT_COLUMNS)).to_dict(
            "records"
        )
    ]
    histories = native.native_histories(jobs)
    old_rows = {r["sample_token"]: (i, r) for i, r in enumerate(old["rows"])}
    eobs = {}
    for job in jobs:
        number, previous = old_rows[job["sample_token"]]
        assert np.array_equal(job["windows_us"], previous["windows"])
        assert np.array_equal(job["square_xyxy"], previous["square"])
        assert np.float32(job["delta_t_s"]) == np.float32(previous["delta"])
        assert job["raw_path"] == str(Path(previous["path"]).resolve())
        path = args.test_cache / f"query_{number:05d}.npz"
        receipt = read(path.with_suffix(".json"))
        assert receipt["input_freeze_sha256"] == sha(old_path)
        assert receipt["sample_token"] == job["sample_token"]
        job.update(current_tensor_path=str(path.resolve()), current_tensor_sha256=receipt["sha256"])
        eobs[job["sample_token"]] = job
    robs, queries = {}, []
    for row in rgb_index.to_dict("records"):
        selected = history_observations(row)
        ids = []
        for obs in selected:
            # Parquet nested arrays become plain JSON scalars before source signatures.
            obs = cast(dict, plain(obs))
            identity = obs["observation_id"]
            obs["available_us"] = max(obs["producer_available_us"])
            if identity in robs:
                assert _observation_signature(robs[identity]) == _observation_signature(obs)
            else:
                robs[identity] = obs
            ids.append(identity)
        queries.append(
            dict(
                sample_token=row["sample_token"],
                sequence_id=row["sequence_id"],
                track_id=row["track_id"],
                query_time_us=int(row["query_time_us"]),
                event_history=histories[row["sample_token"]],
                rgb_history=ids,
            )
        )
    assert len(queries) == len(eobs) == 6762
    shards = sorted({p for row in robs.values() for p in row["producer_shards"]})
    shard_receipts = {}
    for number, relative in enumerate(shards):
        admission()
        path = args.media_root / relative
        metadata = args.media_root / ".cache/huggingface/download" / (relative + ".metadata")
        lines = metadata.read_text(encoding="utf-8").splitlines()
        assert lines[0] == REVISION and len(lines[1]) == 64
        receipt_path = DEST / "rgb_sources" / (relative.replace("/", "_") + ".json")
        expected = dict(
            path=relative,
            size_bytes=path.stat().st_size,
            mtime_ns=path.stat().st_mtime_ns,
            sha256=lines[1],
            revision=REVISION,
        )
        if receipt_path.exists():
            assert read(receipt_path) == expected
        else:
            assert sha(path) == lines[1], relative
            write(receipt_path, expected)
        shard_receipts[relative] = expected
        print("RGB_SOURCE_VERIFIED", number + 1, len(shards), flush=True)
    inputs_path = DEST / "INPUTS.json"
    write(
        inputs_path,
        dict(
            media_root=str(args.media_root.resolve()),
            queries=queries,
            event_observations=eobs,
            rgb_observations=robs,
            rgb_sources=shard_receipts,
        ),
    )
    sources = dict(engine._source_bindings())
    for directory in (ROOT / "operational/multimodal_research", ROOT / "src/e_jepa_ttc/rgb_port"):
        for path in directory.glob("*.py"):
            if path.name.startswith("test12") or directory.name == "rgb_port":
                sources[path.relative_to(ROOT).as_posix()] = sha(path)
    for name in (
        "model.py",
        "phase_only_anchor.py",
        "phase_units.py",
        "guard_scan.py",
        "shared_run.py",
    ):
        path = ROOT / "operational/multimodal_research" / name
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
    parents = {
        str(path.resolve()): sha(path) for path in (RUN / "fits").glob("*/checkpoint_last.pt")
    }
    parents.update(
        {str(path.resolve()): sha(path) for path in (RUN / "features/H").glob("*_NORMALIZER.json")}
    )
    for path in (RUN / "P_MANIFEST.json", ROOT / "configs/rgb_port/producers.json"):
        parents[str(path.resolve())] = sha(path)
    write(
        freeze_path,
        dict(
            utc=datetime.now(UTC).isoformat(),
            candidate="PHASE_ONLY_EVENT_S7",
            selection="First registered phase-only seed; original2% V gates pass, "
            "confirmed by seeds13/23. "
            "GROUP_CROSSFIT fails V MAE. No choice by test scores or best external seed.",
            candidate_V_result_sha256=sha(candidate_path),
            inputs_sha256=sha(inputs_path),
            source_sha256=sources,
            parent_sha256=parents,
            producer_freeze_sha256=sha(RUN / "rounds/PRODUCERS_PAIR_ENDPOINT_FREEZE.json"),
            readout_sha256={
                m: sha(OUT / f"fits/R_ANCHOR_PHASE_ONLY_{m}_S7/checkpoint_last.pt")
                for m in ("EVENT", "ZERO")
            },
            input_parity_sha256=sha(parity_path),
            test_input_sha256=sha(metadata_path),
            parent_precision="CUDA FP32; matmul TF32 false; cuDNN TF32 true; batch1 observations",
            readout_precision="CPU FP32",
            fallback="Exact E_CTX where native RGB history unavailable",
            compared_systems=[
                "PHASE_ONLY_EVENT_S7",
                "PHASE_ONLY_ZERO_S7",
                "E_CTX",
                "R_CTX",
                "F_TRUE",
            ],
            training_scope="Original V13 P producers/PAIR and H refiners/readout; no TRAIN40 refit",
            test_targets_read=False,
            optimizer_updates=0,
            external_submission=False,
            official_score_available=False,
            official_SOTA=False,
        ),
    )
    print("FROZEN", len(queries), len(robs), "RGB observations", flush=True)


def plain(value: object) -> object:
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


if __name__ == "__main__":
    main()
