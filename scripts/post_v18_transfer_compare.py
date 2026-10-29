"""Frozen, resumable post-V18 transfer comparison; zero optimizer updates."""

from __future__ import annotations

import argparse
import importlib
import json
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import torch

from scripts.post_v18_transfer_inputs import CASES, ROOT, garl_inputs, population, read, sha, write

OUT = ROOT / "artifacts/post_v18_transfer_20261017"
DOC = ROOT / "docs/post_v18_transfer_20261017"
V12 = ROOT.parent / "e-jepa-ttc-v12-efficient-context"
V13 = ROOT.parent / "e-jepa-ttc-v13-rgb-port"
V16 = ROOT.parent / "e-jepa-ttc-v16-multimodal-validation"
RUN13 = V13 / "artifacts/rgb_port_20261008"
RUN16 = V16 / "artifacts/multimodal_research_20261012"
RUN18 = ROOT / "artifacts/joint_measurement_20261012"
PUBLIC = V12 / "artifacts/evttc_rgb_transfer_20261008/public_garl"


def sources() -> dict[str, str]:
    """Bind new adapter code before any scientific prediction."""
    return {
        name: sha(ROOT / name)
        for name in (
            "scripts/post_v18_transfer_inputs.py",
            "scripts/post_v18_transfer_compare.py",
            "operational/evttc_transfer/inputs.py",
            "operational/evttc_rgb_transfer/model.py",
            "operational/rgb_port/transfer.py",
            "operational/rgb_port_revision/transfer_inputs.py",
            "operational/joint_measurement/fusion_pair_extract.py",
            "operational/joint_measurement/fusion_history_extract.py",
            "operational/joint_measurement/geometry_model.py",
        )
    }


def register() -> None:
    """Record the user's new comparison scope without reopening V18 or training."""
    if (OUT / "REGISTRATION.json").exists():
        raise ValueError("Registration exists; verify it instead of overwriting")
    manifest = population()
    write(OUT / "QUERY_MANIFEST.json", manifest)
    bindings = {}
    freeze = read(RUN16 / "test12_phase_only/MODEL_INPUT_FREEZE.json")
    for name, expected in freeze["parent_sha256"].items():
        path = Path(name)
        if sha(path) != expected:
            raise ValueError(f"Frozen parent changed: {name}")
        bindings[name] = expected
    paths = [
        PUBLIC / "paper_ours_full.pth",
        PUBLIC / "configs/ablation/ours_full.yaml",
        RUN16 / "fits/R_ANCHOR_PHASE_ONLY_EVENT_S7/checkpoint_last.pt",
        RUN18 / "adaptive/C1_geometry/fits/PAIR-CORR-GEOMETRY-S7/checkpoint_last.pt",
        RUN18 / "fusion/exports/FUSION-TRUE-S7/model.torchscript.pt",
        RUN18 / "fusion/exports/FUSION-TRUE-S7/BINDING.json",
    ]
    for path in paths:
        bindings[str(path.resolve())] = sha(path)
    head = paths[2]
    if bindings[str(head.resolve())] != freeze["readout_sha256"]["EVENT"]:
        raise ValueError("V16 seed7 head differs")
    export = read(RUN18 / "fusion/exports/FUSION-TRUE-S7/COMPLETE.json")
    for name, expected in export["files"].items():
        if sha(RUN18 / "fusion/exports/FUSION-TRUE-S7" / name) != expected:
            raise ValueError("Portable fusion changed")
    write(
        OUT / "REGISTRATION.json",
        {
            "authorization": "User: haz esos analisis y comparaciones; Garl, EvTTC y eAP locales",
            "status": "AUTHORIZED_FROZEN_COMPARISON",
            "optimizer_updates": 0,
            "model_selection": False,
            "seed": 7,
            "test12_labels_or_submission": False,
            "population": "Complete author-interval 100ms triplets; all CCRm variants retained",
            "queries": len(manifest["rows"]),
            "cases": CASES,
            "manifest_sha256": sha(OUT / "QUERY_MANIFEST.json"),
            "sources": sources(),
            "bindings": bindings,
            "device": "CPU_FP32_two_threads",
            "GPU_seconds": 0,
            "max_wall_seconds_per_route": 7200,
            "models": [
                "Garl_full",
                "PHASE_ONLY_V16_S7",
                "FUSION_C1_V18_S7",
                "F_TRUE",
                "R_CTX",
                "C1_event",
            ],
            "primary_contrast": "FUSION_C1_V18_S7 minus PHASE_ONLY_V16_S7",
            "metrics": ["RTE_percent", "MAE_seconds", "sign_error", "coverage", "per_sequence"],
            "statistics": (
                "Paired sequence differences; sequence bootstrap seed7 2000 draws; "
                "exploratory with three covered sequences"
            ),
            "GT": (
                "Read after both prediction seals; ttc.csv column1 seconds, column4 TTC; "
                "nearest <=50ms; exclude nonfinite/zero targets explicitly"
            ),
            "native_own_route": (
                "Historical event/RGB shared ROI margin0.25 and context <=650ms; "
                "new C1 takes latest event pair"
            ),
            "Garl_route": (
                "Last two RGB frames and actual event segments of sliding triples; "
                "factor1 common max-edge squares; fixed published dT=.1"
            ),
            "mapping": (
                "Calibrated rotation-only four-corner mapping; translation omitted without "
                "depth. Author's exact mapping math is unspecified."
            ),
            "track_selection": (
                "Nearest previous car-box center; first box nearest image center; "
                "source-index tie break"
            ),
            "limitations": [
                "Not an exact Table VI reproduction: query list/checkpoint provenance unknown",
                "EvTTC already exposed development; no blind confirmation",
                "System comparison preserves different native ROIs and temporal support",
                "CCRm overlap variant not identified by author; no choice based on scores",
                "CPU timings describe this run, not Table VI hardware or an optimized service",
            ],
            "eAP_scope": (
                "Reuse complete sealed internal V comparisons; public Garl TRAIN exposure "
                "prevents an independent same-V generalization claim"
            ),
        },
    )
    print(
        json.dumps(
            {"registered_queries": len(manifest["rows"]), "inventory": manifest["inventory"]}
        )
    )


def verified() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Refuse altered sources, models or cohort on every invocation."""
    registration = read(OUT / "REGISTRATION.json")
    if registration["sources"] != sources() or registration["manifest_sha256"] != sha(
        OUT / "QUERY_MANIFEST.json"
    ):
        raise ValueError("Frozen adapter/population changed")
    for name, expected in registration["bindings"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"Bound dependency changed: {name}")
    return registration, read(OUT / "QUERY_MANIFEST.json")["rows"]


def frozen_modules() -> tuple[Any, Any]:
    """Import the already-frozen V16 head and state reader without invoking its runners."""
    import operational

    operational.__path__.append(str(V16 / "operational"))
    anchor = importlib.import_module(
        "operational.multimodal_research.phase_only_anchor"
    ).PhaseOnlyAnchor
    state = importlib.import_module("operational.multimodal_research.model").frozen_state
    return anchor, state


def run_garl(limit: int | None) -> None:
    """Predict with the unchanged released full model; no target reads."""
    from operational.evttc_rgb_transfer.model import FullGarl

    registration, rows = verified()
    torch.set_num_threads(2)
    model = FullGarl(PUBLIC, Path("E:/Garl-TTC"), device="cpu")
    started = time.monotonic()
    directory = OUT / "garl"
    write(directory / "MODEL.json", model.bindings)
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        path = directory / f"query_{index:04d}.json"
        if path.exists():
            saved = read(path)
            if saved["query_id"] != row["query_id"] or saved["registration_sha256"] != sha(
                OUT / "REGISTRATION.json"
            ):
                raise ValueError("Saved prediction identity changed")
            continue
        if time.monotonic() - started > registration["max_wall_seconds_per_route"]:
            raise TimeoutError("Route wall limit; fragments preserved")
        begin = time.perf_counter()
        prepared = garl_inputs(row)
        middle = time.perf_counter()
        prediction = (
            model.predict_sensor(prepared["sensor"]) if prepared["sensor"] is not None else None
        )
        end = time.perf_counter()
        value = None if prediction is None else float(cast(float, prediction["ttc"]))
        write(
            path,
            {
                "query_id": row["query_id"],
                "registration_sha256": sha(OUT / "REGISTRATION.json"),
                "status": prepared["status"] if prediction is None else "PREDICTED",
                "predictions": {
                    "Garl_full": value if value is not None and np.isfinite(value) else None
                },
                "input_sha256": prepared.get("sensor_sha256"),
                "heights": None if prediction is None else prediction["heights"],
                "seconds": {
                    "preprocess": middle - begin,
                    "model": end - middle,
                    "total": end - begin,
                },
                "targets_read": False,
                "optimizer_updates": 0,
            },
        )
        print(
            json.dumps(
                {"route": "garl", "done": index + 1, "total": len(rows), "seconds": end - begin}
            ),
            flush=True,
        )
    seal("garl", rows)


def run_own(limit: int | None) -> None:
    """Predict native frozen V16, V18 C1 fusion and historical comparators."""

    from e_jepa_ttc.rgb_port.fusion import DualClockFusion
    from e_jepa_ttc.simplex_t.phase import phase_to_ttc
    from operational.joint_measurement.fusion_history_extract import extract_history
    from operational.joint_measurement.fusion_pair_extract import extract_pair
    from operational.joint_measurement.geometry_model import GeometricPairMeasurement
    from operational.rgb_port import transfer as engine
    from operational.rgb_port_revision import migration

    registration, rows = verified()
    torch.set_num_threads(2)
    anchor, frozen_state = frozen_modules()
    observation_contract = importlib.import_module(
        "operational.rgb_port_external_fix.runtime"
    ).observation_contract
    deterministic_forward = importlib.import_module(
        "operational.rgb_port_fusion_fix.fusion"
    ).deterministic_forward
    with patch.object(migration, "ROOT", V13):
        models = engine._models(
            RUN13, V16 / "configs/rgb_port/producers.json", RUN13 / "P_MANIFEST.json", "cpu"
        )
    freeze_sha = sha(RUN13 / "rounds/PRODUCERS_PAIR_ENDPOINT_FREEZE.json")
    normalizers = {
        name: engine._normalizer(RUN13 / f"features/H/{name}_NORMALIZER.json", modality, freeze_sha)
        for name, modality in (("E", "event"), ("R", "rgb"))
    }
    head = anchor("EVENT").eval().requires_grad_(False)
    head.load_state_dict(
        torch.load(
            RUN16 / "fits/R_ANCHOR_PHASE_ONLY_EVENT_S7/checkpoint_last.pt",
            map_location="cpu",
            weights_only=False,
        )["model_state_dict"]
    )
    geometry = (
        GeometricPairMeasurement(cast(Any, models["E_A5_MATCHED"])).eval().requires_grad_(False)
    )
    geometry.load_state_dict(
        torch.load(
            RUN18 / "adaptive/C1_geometry/fits/PAIR-CORR-GEOMETRY-S7/checkpoint_last.pt",
            map_location="cpu",
            weights_only=False,
        )["model_state_dict"]
    )
    with (RUN18 / "fusion/exports/FUSION-TRUE-S7/model.torchscript.pt").open("rb") as stream:
        fusion = torch.jit.load(stream, map_location="cpu").eval()
    directory = OUT / "own"
    started = time.monotonic()
    with ExitStack() as stack:
        stack.enter_context(observation_contract())
        stack.enter_context(patch.object(DualClockFusion, "forward", deterministic_forward))
        stack.enter_context(torch.inference_mode())
        for index, row in enumerate(rows):
            if limit is not None and index >= limit:
                break
            path = directory / f"query_{index:04d}.json"
            if path.exists():
                saved = read(path)
                if saved["query_id"] != row["query_id"] or saved["registration_sha256"] != sha(
                    OUT / "REGISTRATION.json"
                ):
                    raise ValueError("Saved own prediction identity differs")
                continue
            if time.monotonic() - started > registration["max_wall_seconds_per_route"]:
                raise TimeoutError("Own route wall limit; fragments preserved")
            begin = time.perf_counter()
            raw = engine.prepare_event(row)
            sensor = torch.from_numpy(raw["own_events"])
            observations, reason = engine.raw_rgb_observations(row)
            prepared_at = time.perf_counter()
            query = row["anchor_us"]
            eraw, experts = engine._expert_block(
                models, "E", sensor, torch.full((len(engine.EVENT_LAGS_US), 2), 0.1)
            )
            mean, scale = normalizers["E"]
            ef = ((eraw - mean) / scale)[None]
            et = torch.from_numpy(
                engine._clock(
                    query - np.asarray(engine.EVENT_LAGS_US),
                    np.full(len(engine.EVENT_LAGS_US), query),
                    query,
                )
            )[None]
            ev = torch.ones((1, len(engine.EVENT_LAGS_US)), dtype=torch.bool)
            ee = experts[-1:]
            eb, _, eout = frozen_state(models["E_CTX_MATCHED"], ef, et, ev, ee)
            event_features, event_ttc = extract_pair(
                geometry, sensor[-1:, -2:], torch.tensor([0.1])
            )
            predictions = {"C1_event": float(event_ttc[0])}
            if not observations:
                raise ValueError("This paired cohort requires RGB: " + reason)
            timeline = []
            mean, scale = normalizers["R"]
            for pixels, meta in observations:
                raw_r, expert_r = engine._expert_block(
                    models,
                    "R",
                    torch.from_numpy(pixels)[None],
                    torch.tensor([meta["delta_t_s"]], dtype=torch.float32),
                )
                timeline.append(
                    (
                        meta["first_sensor_ts_us"],
                        meta["anchor_us"],
                        meta["available_us"],
                        ((raw_r - mean) / scale)[0],
                        expert_r[0],
                    )
                )
            history = engine._select_rgb_history(timeline, query)
            rf = torch.stack([v[3] for v in history])[None]
            rt = torch.from_numpy(
                engine._clock(
                    np.asarray([v[1] for v in history]), np.asarray([v[2] for v in history]), query
                )
            )[None]
            rv = torch.ones((1, len(history)), dtype=torch.bool)
            rb, widths, rout = frozen_state(models["R_CTX"], rf, rt, rv, history[-1][4][None])
            fout = models["F_TRUE"](ef, et, ev, ee, rf, rt, rv)
            rgb_features, rgb_ttc = extract_history(
                cast(Any, models["R_CTX"]), rf, rt, rv, history[-1][4][None]
            )
            fused = fusion(
                torch.from_numpy(rgb_features),
                torch.from_numpy(event_features),
                torch.from_numpy(rgb_ttc),
                torch.from_numpy(event_ttc),
                torch.ones(1, dtype=torch.bool),
            )
            predictions.update(
                {
                    "R_CTX": float(phase_to_ttc(rout["point_phase"])[0]),
                    "F_TRUE": float(phase_to_ttc(fout["point_phase"])[0]),
                    "PHASE_ONLY_V16_S7": float(head(torch.cat([rb, eb], -1), widths)["ttc"][0]),
                    "FUSION_C1_V18_S7": float(fused["ttc"][0]),
                }
            )
            end = time.perf_counter()
            write(
                path,
                {
                    "query_id": row["query_id"],
                    "registration_sha256": sha(OUT / "REGISTRATION.json"),
                    "predictions": predictions,
                    "rgb_history_length": len(history),
                    "seconds": {
                        "preprocess": prepared_at - begin,
                        "all_models_shared_compute": end - prepared_at,
                        "total": end - begin,
                    },
                    "targets_read": False,
                    "optimizer_updates": 0,
                },
            )
            print(
                json.dumps(
                    {"route": "own", "done": index + 1, "total": len(rows), "seconds": end - begin}
                ),
                flush=True,
            )
    seal("own", rows)


def seal(route: str, rows: list[dict[str, Any]]) -> None:
    """Seal only complete route coverage, without opening targets."""
    paths = [OUT / route / f"query_{index:04d}.json" for index in range(len(rows))]
    if all(path.exists() for path in paths):
        write(
            OUT / route / "SEALED.json",
            {
                "queries": len(rows),
                "registration_sha256": sha(OUT / "REGISTRATION.json"),
                "files": {path.name: sha(path) for path in paths},
                "targets_read": False,
                "optimizer_updates": 0,
            },
        )


def main() -> None:
    """Run separate registration and label-free prediction stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "garl", "own"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.action == "register":
        register()
    elif args.action == "garl":
        run_garl(args.limit)
    else:
        run_own(args.limit)


if __name__ == "__main__":
    main()
