"""Inventory actual photometric calibration availability before admitting sensor S1/S2."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np

from operational.evttc_transfer.inputs import _calibration, _calibration_sha256
from operational.release_research.protocol import ResearchRun, sha256, write_json
from scripts.release_observability import check_stat
from scripts.release_research import read, sources, verify_complete


def inventory_hdf(path: Path) -> dict:
    """Read schema, attributes and small calibration values, never sensor or target arrays."""
    schema, metadata = {}, {}
    with h5py.File(path, "r") as handle:

        def visit(name: str, obj: h5py.Group | h5py.Dataset | h5py.Datatype) -> None:
            if isinstance(obj, h5py.Dataset):
                schema[name] = {"shape": list(obj.shape), "dtype": str(obj.dtype)}
                if (
                    ("calib" in name.lower() or path.suffix == ".mat")
                    and obj.size is not None
                    and obj.size <= 64
                ):
                    value = np.asarray(obj)
                    if value.dtype.kind in "iufb" and np.isfinite(value).all():
                        metadata[name] = value.tolist()
            for key, value in obj.attrs.items():
                metadata[name + "@" + key] = str(value)

        visit("", handle)
        handle.visititems(visit)
    candidates = {
        kind: sorted(
            name
            for name in set(schema) | set(metadata)
            if any(word in name.lower() for word in words)
        )
        for kind, words in {
            "contrast_or_threshold": ("contrast", "threshold", "bias_diff", "c_plus", "c_minus"),
            "response": ("response", "gamma", "radiometric", "photometric"),
            "exposure": ("exposure",),
            "refractory": ("refract",),
            "gain": ("gain",),
            "light_reference": ("radiance", "irradiance", "reference_texture"),
        }.items()
    }
    return {
        "schema": schema,
        "calibration_values_and_attributes": metadata,
        "candidate_fields_by_name": candidates,
        "scope": "field discovery is not calibrated-value validation; no external absence claim",
    }


def audit(screen: Path, expanded: Path, output: Path) -> None:
    """Bind every development acquisition to its existing geometric calibration and RAW schema."""
    verify_complete(screen)
    verify_complete(expanded)
    rows = read(screen / "PREDICTIONS.json")["records"]
    queries = {}
    inputs = {
        **sources(),
        "source:scripts/release_sensor_inventory.py": Path(__file__),
        "source:scripts/release_observability.py": Path("scripts/release_observability.py"),
        "source:operational/evttc_transfer/inputs.py": Path("operational/evttc_transfer/inputs.py"),
        "source:src/e_jepa_ttc/models/causal_scale_ttc.py": Path(
            "src/e_jepa_ttc/models/causal_scale_ttc.py"
        ),
        "screen_complete": screen / "COMPLETE.json",
        "expanded_complete": expanded / "COMPLETE.json",
    }
    for row in rows:
        if row["group"] == "Slider-500" or row["group"] in queries:
            continue
        identity = hashlib.sha256(row["token"].encode()).hexdigest()[:24]
        path = expanded / "queries" / identity / "QUERY.json"
        queries[row["group"]] = read(path)
        inputs[row["group"] + ":query"] = path
    slider = Path("output/phase_campaign/Slider500/inputs/Slider-500/calibParams.mat")
    receipt = Path("output/phase_campaign/Slider500/SENSORS.json")
    if sha256(slider) != read(receipt)["files"]["calibParams.mat"]["sha256"]:
        raise ValueError("Slider calibration differs from original")
    inputs.update(slider_calibration=slider, slider_receipt=receipt)
    run = ResearchRun(
        output,
        inputs,
        {
            "experiment_id": "S1-photometric-readiness-v1",
            "targets_read": False,
            "optimizer_updates": 0,
            "scope": "local admitted development sources only",
        },
    )
    try:
        results = {}
        for group, query in queries.items():
            path = Path(query["raw_path"])
            check_stat(path, query["raw_stat"])
            with h5py.File(path, "r") as handle:
                if _calibration_sha256(_calibration(handle)) != query["calibration_sha256"]:
                    raise ValueError("Road calibration changed")
            results[group] = inventory_hdf(path)
            results[group].update(
                raw_stat=query["raw_stat"], geometric_calibration_sha256=query["calibration_sha256"]
            )
            check_stat(path, query["raw_stat"])
        results["Slider-500"] = inventory_hdf(slider)
        write_json(output / "INVENTORY.json", results)
        summary = {
            kind: sum(bool(item["candidate_fields_by_name"][kind]) for item in results.values())
            for kind in results["Slider-500"]["candidate_fields_by_name"]
        }
        write_json(
            output / "DECISION.json",
            {
                "family": "sensor",
                "decision": "diagnose",
                "conditions": [],
                "acquisitions": len(results),
                "field_candidate_acquisitions": summary,
                "known_photometric_scale_established": False,
                "S1_real_calibration_executed": False,
                "missing": [
                    "event contrast thresholds or identified common scale",
                    "anchored event-sensor texture",
                    "alignment and reference residual decomposition",
                ],
                "dense_features_interface": "CausalScaleTTC.forward(return_dense_features=True): "
                "endpoint_dense_features",
                "dense_features_note": "Interface inspected; no new forward or gradient test",
                "release_modified": False,
            },
        )
        lines = [
            "# Disponibilidad fotométrica para S1",
            "",
            f"Se inspeccionaron {len(queries)} RAW EvTTC y la calibración original Slider-500.",
            "Se conservaron esquema completo, atributos y valores pequeños de calibración.",
            "",
            "| Familia de campos | Adquisiciones con candidatos por nombre |",
            "|---|---:|",
        ]
        lines += [f"| {k} | {v} |" for k, v in summary.items()]
        lines += [
            "",
            "No se ha establecido una escala fotométrica identificada para S1.",
            "La exposición RGB no identifica umbrales de eventos ni linealiza el ISP.",
            "El resultado se limita a estas entradas locales; no afirma ausencia en otros datos.",
            "E_C2F expone endpoint_dense_features mediante return_dense_features=True;",
            "la caché PAIR133 no contiene esos mapas ni referencias fotométricas.",
            "Faltan una referencia anclada y calibración/identificación de escala antes de S2.",
            "No se ha aplicado una pérdida fotométrica sobre RGB provisional ni actualizado R0.",
        ]
        (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        run.finish()
    except Exception as error:
        run.fail(error)
        raise


def main() -> None:
    """Record the local prerequisites without creating a fictional calibrated sensor loss."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen", type=Path, default=Path("output/phase_timewarp/screen149_v1"))
    parser.add_argument(
        "--expanded", type=Path, default=Path("output/phase_observation_cache/expanded149")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.screen, args.expanded, args.output)


if __name__ == "__main__":
    main()
