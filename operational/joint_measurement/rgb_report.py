"""Regenerate the closed RGB comparison from admitted prediction and metric files."""

from __future__ import annotations

from operational.range_transfer.common import read, sha, write

from .common import ROOT, RUN


def build() -> dict:
    """Bind all displayed values to verified files; perform no new model inference."""
    directory = RUN / "rgb"
    evidence = read(directory / "evidence/COMPLETE.json")
    selection = read(directory / "SELECTION_FINAL.json")
    if evidence["status"] != "RGB_READY_FOR_INTERPRETATION" or evidence["selection"] != selection:
        raise ValueError("Complete RGB evidence and frozen selection are required")
    rows, bindings = [], {}
    for role in ("P", "H"):
        for name in ("R_A5-P2-FROZEN", "RGB-PAIR-CORR-FT-S7", "RGB-PAIR-CORR-FROZEN-S7"):
            key = f"{role}/{name}"
            folder = directory / "predictions" / role / name
            receipt = read(folder / "COMPLETE.json")
            metrics = read(folder / "METRICS.json")
            proof = evidence["verification"][key]
            prediction_hash = sha(folder / "predictions.parquet")
            metric_hash = sha(folder / "METRICS.json")
            if not (
                receipt["status"] == "COMPLETE"
                and proof["status"] == "PASS"
                and prediction_hash
                == receipt["sha256"]
                == evidence["cohorts"][key]
                == proof["prediction_sha256"]
                and metric_hash == proof["metrics_sha256"]
                and metrics["num_rows"] == receipt["rows"] == proof["rows"]
            ):
                raise ValueError(f"RGB report source failed admission: {key}")
            bindings[key] = {"predictions": prediction_hash, "metrics": metric_hash}
            rows.append(
                {
                    "role": role,
                    "model": name,
                    "rows": metrics["num_rows"],
                    "aggregate": metrics["overall_MiD"],
                    "MiDc": metrics["bins"]["crucial"]["group_macro_mid"],
                    "negative_MiD": metrics["bins"]["negative"]["group_macro_mid"],
                    "MAE_s": metrics["diagnostics"]["mae_s"],
                    "sign_error_rate": metrics["diagnostics"]["sign_error_rate"],
                }
            )
    result = {
        "status": selection["status"],
        "rows": rows,
        "source_bindings": bindings,
        "evidence_sha256": sha(directory / "evidence/COMPLETE.json"),
        "selection_sha256": sha(directory / "SELECTION_FINAL.json"),
        "official_comparison": False,
        "full_objective_complete": False,
    }
    write(directory / "evidence/REPORT_DATA.json", result)
    lines = [
        "# V18 RGB P2: fixed endpoints and development result",
        "",
        "Regenerate: `python -B -m operational.joint_measurement.rgb_report`.",
        "",
        f"Frozen decision: **{selection['status']}**. No system promotion or official claim.",
        "Two seed-7 fits completed 12,000 updates each before H evaluation. P is training;",
        "H contains eight held-out groups. Both routes use two RGB observations with their",
        "own sensor clocks. The native P2 reference has a different objective and architecture.",
        "",
        "| Role | Model | Rows | Aggregate | MiDc | Negative MiD | MAE (s) | Sign errors |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['role']} | {row['model']} | {row['rows']} | {row['aggregate']:.6f} | "
            f"{row['MiDc']:.6f} | {row['negative_MiD']:.6f} | {row['MAE_s']:.6f} | "
            f"{100 * row['sign_error_rate']:.4f}% |"
        )
    lines.extend(
        [
            "",
            "The aggregate and crucial-band gains coexist with worse negative TTC and sign",
            "errors, including on P. This is not solely a held-out transfer failure. Both",
            "variants fail the registered band and sign checks. Retain the positive signal",
            "without selecting either endpoint for V, replication or refit under these gates.",
            "",
            "Paired cost covers prepared tensors resident on the GPU through TTC emission;",
            "it excludes sensor loading and preprocessing. It is not full-pipeline latency.",
            "Six prediction tables, NumPy metric verification, group bootstraps and three",
            "portable exports are recorded in `rgb/evidence/COMPLETE.json`. Exact endpoint",
            "verification separately confirms encoder changes for FT and unchanged encoder",
            "weights for FROZEN. The historical R_CTX/F_TRUE systems remain comparators.",
            "",
            "C1 is an independently registered event geometry experiment enabled by the",
            "earlier real-temporal correspondence diagnostic. Its execution does not promote",
            "these rejected RGB endpoints. RGB-anchor fusion remains a separate prospective",
            "experiment; using historical H-trained heads must be labelled sequential supervision.",
            "",
        ]
    )
    output = ROOT / "docs/joint_measurement_20261012/RGB_RESULTS.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    return {"status": result["status"], "tables": len(rows), "report": str(output)}


if __name__ == "__main__":
    print(build())
