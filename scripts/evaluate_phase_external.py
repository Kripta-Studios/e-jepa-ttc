"""Freeze sensor/weight contracts, predict Slider-500, then open its TTC labels."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from operational.evttc_competition.geometry import load_calibration, project_event_box_to_rgb
from operational.evttc_competition.prepare import prepare_query
from operational.evttc_competition.readers import SensorSequence
from operational.evttc_competition.runtime import CompetitionRuntime, PreparedObservation
from operational.evttc_transfer_audit.common import file_sha
from operational.phase_adapt_cache import atomic_receipt, save_npz
from operational.phase_adapt_training import configure_cuda, load_head
from operational.phase_campaign.data import read
from operational.phase_campaign.evaluation import bbox_ttc, metrics
from operational.phase_campaign.external import join_external_targets
from scripts.final_train40_transfer import HEAD, verify_candidate


def shifted(item: PreparedObservation, delta: int) -> PreparedObservation:
    return replace(item, timestamps_us=tuple(t + delta for t in item.timestamps_us),
                   first_sensor_ts_us=item.first_sensor_ts_us + delta,
                   anchor_us=item.anchor_us + delta, available_us=item.available_us + delta)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "candidate", "models", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    configure_cuda()
    torch.set_num_threads(2)
    _, parent = verify_candidate(args.candidate)
    heads = {str(s): args.models / f"seed_{s}" / "checkpoint.pt" for s in (7, 13, 21)}
    hashes = {s: file_sha(p) for s, p in heads.items()}
    source_hashes = {p.name: file_sha(p) for p in sorted(args.source.iterdir()) if p.is_file()}
    staged = args.output / "inputs" / "Slider-500"
    staged.mkdir(parents=True, exist_ok=True)
    for source_name, dest_name in (("slider_500.bag", "slider_500.bag"),
                                   ("calibParams.mat", "calibParams.mat"),
                                   ("bounding_box.csv", "bbox.csv")):
        dest = staged / dest_name
        if not dest.exists():
            if dest.suffix == ".bag":
                dest.hardlink_to(args.source / source_name)
            else:
                shutil.copyfile(args.source / source_name, dest)
        if file_sha(dest) != source_hashes[source_name]:
            raise ValueError("Changed external staged sensor input")
    calibration = load_calibration(staged, "slider_candidate")
    with SensorSequence(staged.parent, "Slider-500", slider_profile="event_aided_ttc") as reader:
        rows = reader.bbox_rows()
        config = {"parent": parent, "adapted_heads": hashes, "sources": source_hashes,
                  "selected_seed": 7, "topic_profile": "event_aided_ttc",
                  "queries": [{"frame_index": r["frame_index"], "anchor_us": r["anchor_us"]}
                              for r in rows],
                  "label_contract": "frame,time_seconds,depth,speed,TTC; identity plus 2us clock check",
                  "geometry": calibration.metadata,
                  "geometry_status": "provisional rotation only; event-plane boxes assumed",
                  "independence": "outside current head fit; acquisition relation unresolved",
                  "labels_opened_before_freeze": False,
                  "source_code": {str(p): file_sha(p) for p in (
                      Path(__file__).relative_to(Path.cwd()),
                      Path("operational/phase_campaign/external.py"),
                      Path("operational/evttc_competition/readers.py"),
                      Path("operational/evttc_competition/prepare.py"),
                      Path("operational/evttc_competition/runtime.py"))}}
        atomic_receipt(args.output / "FREEZE.json", config)
        atomic_receipt(args.output / "SENSORS.json", reader.source_metadata())
        # Fixed quartiles, selected without TTC values. Event boxes and RGB
        # rotation projections are drawn in their respective native image planes.
        for index in sorted({len(rows) // 4, len(rows) // 2, 3 * len(rows) // 4}):
            row = rows[index]
            image = Image.fromarray(reader.read_rgb(row["frame_index"]))
            draw = ImageDraw.Draw(image)
            draw.rectangle(project_event_box_to_rgb(row["box_xyxy"], calibration),
                           outline="red", width=5)
            image.save(args.output / f"overlay_rgb_{index}.png")
            raw = reader.read_events(max(reader.event_start_us, row["anchor_us"] - 100000),
                                     row["anchor_us"])
            count = np.zeros((reader.event_height, reader.event_width), dtype=np.float32)
            np.add.at(count, (raw[:, 1].astype(int), raw[:, 0].astype(int)), 1)
            count = np.log1p(count)
            image = Image.fromarray((255 * count / max(float(count.max()), 1)).astype(np.uint8)).convert("RGB")
            ImageDraw.Draw(image).rectangle(row["box_xyxy"], outline="red", width=2)
            image.save(args.output / f"overlay_events_{index}.png")
        if args.prepare_only:
            print(json.dumps({"status": "FROZEN_LABELS_UNOPENED", "queries": len(rows)}))
            return
        runtime = CompetitionRuntime.from_candidate(args.candidate, device="cuda")
        models = {s: load_head(p, hashes[s]).to("cuda") for s, p in heads.items()}
        captured: list[tuple[torch.Tensor, torch.Tensor]] = []

        def capture(module: Any, inputs: tuple[torch.Tensor, ...]) -> None:
            del module
            captured.append((inputs[0].detach().clone(), inputs[1].detach().clone()))

        hook = runtime.models[HEAD].register_forward_pre_hook(capture)
        predictions = {name: [] for name in ("F0", "F1_7", "F1_13", "F1_21", "RGB", "event",
                                            "bbox_height", "bbox_width", "bbox_area")}
        diagnostics: list[dict] = []
        physical: list[dict] = []
        durations = []
        for i, row in enumerate(rows):
            start = time.perf_counter()
            prepared = prepare_query(reader, rows, row, calibration)
            captured.clear()
            result = runtime.predict(prepared.events, prepared.rgb, row["anchor_us"])
            predictions["F0"].append(result.model_ttc)
            predictions["RGB"].append(result.rgb_ttc)
            predictions["event"].append(result.event_ttc)
            with torch.inference_mode():
                for seed, model in models.items():
                    value = float(model(*captured[-1])["ttc"].item()) if captured else result.model_ttc
                    predictions[f"F1_{seed}"].append(value)
            causal = rows[max(0, i - 1): i + 1]
            for axis in ("height", "width", "area"):
                predictions[f"bbox_{axis}"].append(bbox_ttc(
                    np.array([r["box_xyxy"] for r in causal]),
                    np.array([r["anchor_us"] for r in causal]), axis))
            durations.append(time.perf_counter() - start)
            diagnostics.append({"query": i, "runtime": asdict(result), "input": prepared.details,
                                "fused_head_called": bool(captured)})
            if i in {len(rows) // 4, len(rows) // 2, 3 * len(rows) // 4}:
                delta = 1234567
                translated = runtime.predict([shifted(x, delta) for x in prepared.events],
                                             [shifted(x, delta) for x in prepared.rgb],
                                             row["anchor_us"] + delta)
                future_events = list(prepared.events) + [shifted(prepared.events[-1], 1000000)] if prepared.events else []
                future_rgb = list(prepared.rgb) + [shifted(prepared.rgb[-1], 1000000)] if prepared.rgb else []
                future = runtime.predict(future_events, future_rgb, row["anchor_us"])
                physical.append({"query": i, "original": result.model_ttc,
                                 "clock_shift": translated.model_ttc, "future_append": future.model_ttc})
            print(f"external {i + 1}/{len(rows)}: {result.status}", flush=True)
        hook.remove()
        arrays = {name: np.asarray(values, dtype=np.float64) for name, values in predictions.items()}
        save_npz(args.output / "predictions_sealed.npz", arrays)
        atomic_receipt(args.output / "PREDICTIONS_SEALED.json", {
            "sha256": file_sha(args.output / "predictions_sealed.npz"),
            "freeze_sha256": file_sha(args.output / "FREEZE.json"), "labels_opened": False})
        atomic_receipt(args.output / "DIAGNOSTICS.json", {"rows": diagnostics, "physical": physical,
                        "latency_seconds": {"mean": float(np.mean(durations)),
                                            "p50": float(np.median(durations)),
                                            "p95": float(np.quantile(durations, .95))}})
        # The first label-value read happens only after predictions are sealed.
        labels = np.loadtxt(args.source / "gt_ttc.csv", delimiter=",", ndmin=2)
        target = join_external_targets(rows, labels)
        save_npz(args.output / "targets.npz", {"target": target})
        report = {"scope": config["independence"], "geometry_status": config["geometry_status"],
                  "population": len(rows), "metrics": {k: metrics(target, v) for k, v in arrays.items()},
                  "physical": physical, "official_equivalence": False}
        for name in ("RGB", "event", "bbox_height", "bbox_width", "bbox_area"):
            shared = np.isfinite(target) & np.isfinite(arrays[name]) & np.isfinite(arrays["F0"]) & np.isfinite(arrays["F1_7"])
            report.setdefault("shared_comparisons", {})[name] = {
                key: metrics(target[shared], arrays[key][shared]) for key in ("F0", "F1_7", name)}
        atomic_receipt(args.output / "REPORT.json", report)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
