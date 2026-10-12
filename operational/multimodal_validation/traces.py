"""Q2: read-only forward hooks, exact paired output checks and stage records."""

from __future__ import annotations

import argparse

# ruff: noqa: ANN401 -- frozen model APIs expose heterogeneous tensor mappings.
import dataclasses
import hashlib
import time
from contextlib import ExitStack
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.rgb_port.fusion import DualClockFusion
from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.fcwd_rgb_registration.transfer import prepare_event as fcwd_event
from operational.fcwd_rgb_registration.transfer import rgb_observations
from operational.rgb_port import transfer as engine
from operational.rgb_port.train_heads import load_head_endpoint
from operational.rgb_port_external_fix.runtime import observation_contract
from operational.rgb_port_fusion_fix.fusion import deterministic_forward

from .common import (
    MANIFEST,
    OUT,
    ROOT,
    RUN,
    V12,
    admission,
    csvwrite,
    read,
    registrations,
    sha,
    write,
)


def tensors(value: Any) -> dict[str, torch.Tensor]:
    if dataclasses.is_dataclass(value):
        return {
            k: getattr(value, k)
            for k in value.__dataclass_fields__
            if isinstance(getattr(value, k), torch.Tensor)
        }
    if isinstance(value, dict):
        return {k: v for k, v in value.items() if isinstance(v, torch.Tensor)}
    return {"output": value} if isinstance(value, torch.Tensor) else {}


def packed(tensor: torch.Tensor) -> dict[str, Any]:
    a = tensor.detach().cpu().numpy()
    result: dict[str, Any] = dict(
        shape=list(a.shape), dtype=str(a.dtype), sha256=hashlib.sha256(a.tobytes()).hexdigest()
    )
    if a.size:
        result.update(min=float(a.min()), max=float(a.max()), mean=float(a.astype(float).mean()))
    if a.size <= 8192:
        result["values"] = a.tolist()
    return result


class Recorder:
    def __init__(self, profile: bool = False) -> None:
        self.rows = []
        self.details = []
        self.query = ""
        self.scope = ""
        self.name = ""
        self.profile = profile
        self.repeat = 0

    def call(self, model: Any, *args: Any, **kwargs: Any) -> Any:
        if self.profile:
            start = time.perf_counter()
            with torch.inference_mode():
                value = model(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            self.rows.append(
                dict(
                    query_id=self.query,
                    scope=self.scope,
                    stage=self.name,
                    repeat=self.repeat,
                    model_ms=elapsed,
                )
            )
            return value
        with torch.inference_mode():
            baseline = model(*args, **kwargs)
        hooks = []
        captured = {}
        scores = []

        def observe(module: torch.nn.Module, inputs: tuple, output: Any) -> None:
            captured.update({k: packed(v) for k, v in tensors(output).items()})

        hooks.append(model.register_forward_hook(observe))
        if hasattr(model, "residual") and hasattr(model.residual, "scorer"):
            hooks.append(
                model.residual.scorer.register_forward_hook(
                    lambda m, a, o: scores.append(o.detach().clone())
                )
            )
        if hasattr(model, "location"):
            hooks.append(
                model.location.register_forward_pre_hook(
                    lambda m, a: captured.update(location_input=packed(a[0]))
                )
            )
        start = time.perf_counter()
        try:
            with torch.inference_mode():
                observed = model(*args, **kwargs)
        finally:
            for hook in hooks:
                hook.remove()
        bt, ot = tensors(baseline), tensors(observed)
        exact = all(torch.equal(bt[k], ot[k]) for k in bt)
        if not exact:
            raise ValueError("Instrumentation changed output: " + self.name)
        fields: dict[str, Any] = dict(
            query_id=self.query,
            scope=self.scope,
            stage=self.name,
            bitwise_parity=exact,
            instrumented_ms=(time.perf_counter() - start) * 1000,
        )
        if len(scores) == 2:
            captured["residual_pre_tanh"] = packed(0.5 * (scores[0] - scores[1]))
        if "raw_location" in ot:
            phase = ot["point_phase"]
            fields.update(
                raw_location=float(ot["raw_location"][0]),
                point_phase=float(phase[0]),
                prediction_ttc=float(phase_to_ttc(phase)[0]),
            )
            experts = args[3]
            fields["anchor_phase"] = float(experts.median(-1).values[0])
            fields["residual_phase"] = fields["raw_location"] - fields["anchor_phase"]
            fields["clamp_delta"] = fields["point_phase"] - fields["raw_location"]
        if "residual_log_height_ratio" in ot:
            residual = ot["residual_log_height_ratio"]
            fields["max_residual_fraction"] = (
                float(residual.abs().max()) / model.config.max_abs_log_ratio_residual
                if model.config.max_abs_log_ratio_residual
                else 0.0
            )
        self.rows.append(fields)
        self.details.append(
            dict(
                **fields,
                inputs={
                    str(i): packed(v) for i, v in enumerate(args) if isinstance(v, torch.Tensor)
                },
                outputs=captured,
            )
        )
        return observed


def h_controls(rec: Recorder) -> None:
    archives = {}
    for name in ("E_HISTORY", "R_HISTORY", "FUSION"):
        path = RUN / f"features/H/{name}.npz"
        assert sha(path) == read(path.with_suffix(".json"))["output_sha256"]
        with np.load(path, allow_pickle=False) as z:
            archives[name] = {k: z[k] for k in z.files}
    ref = archives["FUSION"]
    selected = []
    for group in sorted(set(ref["group_id"])):
        ids = np.flatnonzero(ref["group_id"] == group)
        selected.extend(
            sorted(
                ids, key=lambda i: hashlib.sha256(str(ref["sample_token"][i]).encode()).hexdigest()
            )[:4]
        )
    write(
        OUT / "H_CONTROLS_SEALED.json",
        dict(
            selection="4 tokens per H group ordered by SHA256 before inspecting prediction error",
            tokens=[str(ref["sample_token"][i]) for i in selected],
            optimizer_updates=0,
        ),
    )
    for name, cache in [
        ("E_CTX_MATCHED", "E_HISTORY"),
        ("R_CTX", "R_HISTORY"),
        ("F_TRUE", "FUSION"),
    ]:
        model = load_head_endpoint(RUN / "fits" / name, name, torch.device("cpu"))
        values = archives[cache]
        assert np.array_equal(values["sample_token"], ref["sample_token"])
        fields = (
            ["features", "timing", "valid", "expert_phase"]
            if name != "F_TRUE"
            else [
                "event_features",
                "event_timing",
                "event_valid",
                "event_expert_phase",
                "rgb_features",
                "rgb_timing",
                "rgb_valid",
            ]
        )
        for i in selected:
            rec.query = str(ref["sample_token"][i])
            rec.scope = "H_hash_control"
            rec.name = name
            rec.call(model, *[torch.from_numpy(values[f][i : i + 1]) for f in fields])


def external(rec: Recorder) -> list[dict]:
    # Exposed published diagnostic cases; never used as a training selection set.
    ids = [
        "CCRs-1-medium-100-overlap-100:00",
        "CCRs-1-medium-100-overlap-100:01",
        "CCRs-1-low-100-overlap-100:22",
        "CCRs-1-low-100-overlap-100:23",
        "CCRs-1-low-100-overlap-100:24",
        "CCRs-1-low-100-overlap-100:30",
        "CCRs-side-medium:07",
        "CCRm-medium-100-overlap-100:02",
    ]
    manifests = [
        ("DEV32", V12 / "artifacts/evttc_transfer_20261008/QUERY_MANIFEST.json"),
        ("FCWD", MANIFEST),
    ]
    loaded = time.perf_counter()
    models = engine._models(
        RUN, ROOT / "configs/rgb_port/producers.json", RUN / "P_MANIFEST.json", "cpu"
    )
    write(
        OUT / "MODEL_LOAD_COST.json",
        dict(
            ms=(time.perf_counter() - loaded) * 1000, device="cpu", threads=1, resources=admission()
        ),
    )
    normalizers = {
        m: engine._normalizer(
            RUN / f"features/H/{m}_NORMALIZER.json",
            mod,
            sha(RUN / "rounds/PRODUCERS_PAIR_ENDPOINT_FREEZE.json"),
        )
        for m, mod in [("E", "event"), ("R", "rgb")]
    }
    regs = registrations()
    historical = (
        pd.read_csv(OUT / "HISTORICAL_TRACE_PARITY.csv").to_dict("records")
        if rec.profile and (OUT / "HISTORICAL_TRACE_PARITY.csv").exists()
        else []
    )
    for cohort, manifest in manifests:
        rows = read(manifest)["rows"]
        if rec.profile:
            sequences = sorted({r["sequence_id"] for r in rows})
            pools = {
                s: sorted(
                    [r for r in rows if r["sequence_id"] == s],
                    key=lambda r: hashlib.sha256(r["query_id"].encode()).hexdigest(),
                )
                for s in sequences
            }
            chosen = [pools[sequences[i % len(sequences)]][i // len(sequences)] for i in range(32)]
            chosen.sort(key=lambda r: (r["sequence_id"], r["anchor_us"]))
            write(
                OUT / f"{cohort}_COST_SELECTION.json",
                dict(
                    query_ids=[r["query_id"] for r in chosen],
                    selection=(
                        "sequence-balanced hash; chronological per sequence; first pass "
                        "and repeated warm pass; no error selection"
                    ),
                    cohort=cohort,
                ),
            )
        elif cohort == "DEV32":
            chosen = [r for r in rows if r["query_id"] in ids]
            assert len(chosen) == len(ids)
        else:
            chosen = []
            for seq in sorted({r["sequence_id"] for r in rows}):
                for accepted in (False, True):
                    selected = sorted(
                        [
                            r
                            for r in rows
                            if r["sequence_id"] == seq
                            and regs[r["query_id"]]["accepted"] == accepted
                        ],
                        key=lambda r: hashlib.sha256(r["query_id"].encode()).hexdigest(),
                    )
                    chosen.extend(selected[:2])
        historical_csv = pd.read_csv(
            ROOT
            / (
                "docs/dev32_transfer_20261012/SCORED_PREDICTIONS.csv"
                if cohort == "DEV32"
                else "docs/fcwd_rgb_registration_20261012/SCORED_PREDICTIONS.csv"
            )
        ).set_index("query_id")
        for repeat, row in [
            (repeat, row) for repeat in range(2 if rec.profile else 1) for row in chosen
        ]:
            if rec.profile and any(
                r["query_id"] == row["query_id"]
                and r["scope"] == cohort
                and r["repeat"] == repeat
                and r["stage"] == "TOTAL_SHARED_E_R_F_ROUTES"
                for r in rec.rows
            ):
                continue
            admission()
            rec.repeat = repeat
            rec.query = row["query_id"]
            rec.scope = cohort

            class Wrapped:
                def __init__(self, name: str) -> None:
                    self.name = name

                def __call__(self, *a: Any, **kw: Any) -> Any:
                    rec.name = self.name
                    return rec.call(models[self.name], *a, **kw)

                def __getattr__(self, key: str) -> Any:
                    return getattr(models[self.name], key)

            wrapped: Any = {name: Wrapped(name) for name in models}
            start = time.perf_counter()
            registration_ms = 0.0
            if rec.profile and cohort == "FCWD":
                from operational.fcwd_rgb_registration.prepare import register_query

                from .common import REG

                fresh = register_query(row, read(REG / "BINDING.json")["calibration"])
                reference = regs[row["query_id"]]
                assert all(fresh.get(k) == reference.get(k) for k in fresh)
                registration_ms = (time.perf_counter() - start) * 1000
            event_start = time.perf_counter()
            prepared = (fcwd_event if cohort == "FCWD" else engine.prepare_event)(row)
            event_prepared = time.perf_counter()
            event_sensor = torch.from_numpy(prepared["own_events"])
            event_delta = torch.full((len(engine.EVENT_LAGS_US), 2), 0.1)
            with torch.inference_mode():
                eraw, eexpert = engine._expert_block(wrapped, "E", event_sensor, event_delta)
                mean, scale = normalizers["E"]
                ef = ((eraw - mean) / scale)[None]
                query = int(row["anchor_us"])
                anchors = query - np.asarray(engine.EVENT_LAGS_US)
                et = torch.from_numpy(engine._clock(anchors, np.full(len(anchors), query), query))[
                    None
                ]
                ev = torch.ones((1, len(anchors)), dtype=torch.bool)
                ee = eexpert[-1:]
                eout = wrapped["E_CTX_MATCHED"](ef, et, ev, ee)
                rgb_start = time.perf_counter()
                observations, reason = (
                    rgb_observations(row, regs[row["query_id"]])
                    if cohort == "FCWD"
                    else engine.raw_rgb_observations(row)
                )
                rgb_ready = time.perf_counter()
                values = {"E_CTX_MATCHED": float(phase_to_ttc(eout["point_phase"])[0])}
                if observations:
                    timeline = []
                    mean, scale = normalizers["R"]
                    for sensor, meta in observations:
                        raw, expert = engine._expert_block(
                            wrapped,
                            "R",
                            torch.from_numpy(sensor)[None],
                            torch.tensor([meta["delta_t_s"]], dtype=torch.float32),
                        )
                        timeline.append(
                            (
                                meta["first_sensor_ts_us"],
                                meta["anchor_us"],
                                meta["available_us"],
                                ((raw - mean) / scale)[0],
                                expert[0],
                            )
                        )
                    history = engine._select_rgb_history(timeline, query)
                    rf = torch.stack([v[3] for v in history])[None]
                    rt = torch.from_numpy(
                        engine._clock(
                            np.asarray([v[1] for v in history]),
                            np.asarray([v[2] for v in history]),
                            query,
                        )
                    )[None]
                    rv = torch.ones((1, len(history)), dtype=torch.bool)
                    rout = wrapped["R_CTX"](rf, rt, rv, history[-1][4][None])
                    fout = wrapped["F_TRUE"](ef, et, ev, ee, rf, rt, rv)
                    values.update(
                        R_CTX=float(phase_to_ttc(rout["point_phase"])[0]),
                        F_TRUE=float(phase_to_ttc(fout["point_phase"])[0]),
                    )
                else:
                    values["F_TRUE"] = values["E_CTX_MATCHED"]
            total_ms = (time.perf_counter() - start) * 1000
            if rec.profile:
                rec.rows.append(
                    dict(
                        query_id=rec.query,
                        scope=cohort,
                        stage="TOTAL_SHARED_E_R_F_ROUTES",
                        repeat=repeat,
                        model_ms=total_ms,
                    )
                )
                rec.rows.append(
                    dict(
                        query_id=rec.query,
                        scope=cohort,
                        stage="EVENT_HDF5_READ_ROI_VOXELS",
                        repeat=repeat,
                        model_ms=(event_prepared - event_start) * 1000,
                    )
                )
                rec.rows.append(
                    dict(
                        query_id=rec.query,
                        scope=cohort,
                        stage="RGB_HDF5_DECODE_CROP",
                        repeat=repeat,
                        model_ms=(rgb_ready - rgb_start) * 1000,
                    )
                )
                rec.rows.append(
                    dict(
                        query_id=rec.query,
                        scope=cohort,
                        stage="E_CTX_ROUTE_END_TO_END",
                        repeat=repeat,
                        model_ms=(rgb_start - event_start) * 1000,
                    )
                )
                rec.rows.append(
                    dict(
                        query_id=rec.query,
                        scope=cohort,
                        stage="REGISTRATION_SEARCH",
                        repeat=repeat,
                        model_ms=registration_ms,
                    )
                )
            else:
                write(
                    OUT / "trace_inputs" / f"{cohort}_{row['query_id'].replace(':', '_')}.json",
                    dict(
                        row=row,
                        rgb_history=[m for _, m in observations],
                        reason=reason,
                        registration=regs[row["query_id"]] if cohort == "FCWD" else None,
                        normalizers={
                            k: sha(RUN / f"features/H/{k}_NORMALIZER.json") for k in ("E", "R")
                        },
                        instrumented_total_ms=total_ms,
                    ),
                )
            for name, value in values.items():
                old = float(historical_csv.loc[row["query_id"], name])
                historical.append(
                    dict(
                        query_id=row["query_id"],
                        cohort=cohort,
                        repeat=repeat,
                        model=name,
                        cpu_prediction=value,
                        historical_prediction=old,
                        absolute_delta=abs(value - old),
                    )
                )
            print("traced", cohort, row["query_id"], flush=True)
            save(rec, historical)
    return historical


def save(rec: Recorder, historical: list[dict]) -> None:
    keys = sorted({k for row in rec.rows for k in row})
    csvwrite(OUT / "STAGE_ATTRIBUTION.csv", [{k: r.get(k, "") for k in keys} for r in rec.rows])
    if not rec.profile:
        write(OUT / "STAGE_TRACE_DETAILS.json", rec.details)
    if historical:
        csvwrite(OUT / "HISTORICAL_TRACE_PARITY.csv", historical)
    write(
        OUT / "TRACE_PARITY.json",
        dict(
            instrumented_calls=len(rec.rows),
            all_bitwise_equal=all(r["bitwise_parity"] for r in rec.rows)
            if not rec.profile
            else None,
            scope=(
                "same CPU inputs and precision, hooks return None; historical "
                "GPU-to-CPU separately reported"
            ),
            historical_comparisons=len(historical)
            if historical
            else (
                len(pd.read_csv(OUT / "HISTORICAL_TRACE_PARITY.csv"))
                if (OUT / "HISTORICAL_TRACE_PARITY.csv").exists()
                else 0
            ),
            optimizer_updates=0,
        ),
    )


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("h", "external", "profile"), required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.scope == "profile":
        OUT = OUT / "cpu_profile"
    rec = Recorder(profile=args.scope == "profile")
    if rec.profile and (OUT / "STAGE_ATTRIBUTION.csv").exists():
        rec.rows = pd.read_csv(OUT / "STAGE_ATTRIBUTION.csv").to_dict("records")
    if (OUT / "STAGE_TRACE_DETAILS.json").exists():
        rec.details = read(OUT / "STAGE_TRACE_DETAILS.json")
        rec.rows = [
            {k: v for k, v in r.items() if k not in ("inputs", "outputs")} for r in rec.details
        ]
    with ExitStack() as stack:
        from .ancestry import historical_source_binding

        stack.enter_context(historical_source_binding())
        stack.enter_context(observation_contract())
        stack.enter_context(patch.object(DualClockFusion, "forward", deterministic_forward))
        if args.scope == "h":
            h_controls(rec)
            historical = []
        else:
            historical = external(rec)
    save(rec, historical)
    print("trace complete", len(rec.rows), flush=True)


if __name__ == "__main__":
    main()
