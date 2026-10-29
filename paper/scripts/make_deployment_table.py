"""Validate the R02 metadata projection and generate the paced-run setup table."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "revision/deployment_contract.json"
EXPECTED_SHA256 = "c97dbdc25d7d28089d31ff60de0d0fa24b8e8a82744e78fc832e784e39c4447c"


def tex(value: object) -> str:
    """Escape metadata text for a table cell."""
    text = str(value)
    for char, replacement in [("_", r"\_"), ("%", r"\%"), ("&", r"\&")]:
        text = text.replace(char, replacement)
    return text


def main() -> None:
    """Generate the table from the exact audited metadata projection."""
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("R02 deployment metadata hash mismatch")
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    run = data["L40_runtime"]
    execution = run["execution"]
    software = run["software"]
    init = run["excluded_initialization"]
    rows = [
        ("GPU", tex(run["hardware"]["gpu_product"])),
        (
            "Software",
            "Python "
            + tex(software["python"])
            + "; PyTorch "
            + tex(software["torch"])
            + "; CUDA build "
            + tex(software["cuda_version_recorded"]),
        ),
        ("Precision", "FP32; TF32 disabled for matrix multiplication and cuDNN"),
        (
            "CPU / preparation",
            f"{execution['torch_cpu_threads']} PyTorch CPU threads; "
            + f"{execution['rgb_workers']} RGB workers; "
            + f"queue capacity {execution['queue_capacity']}",
        ),
        (
            "Parity tolerance",
            "Absolute and relative tolerance both "
            + r"$10^{-5}$; all "
            + str(run["population"]["scheduled_updates"])
            + " outputs checked",
        ),
        ("RAW preload", f"{init['raw_preload_seconds']:.3f} s"),
        ("Model and service initialization", f"{init['model_and_service_seconds']:.3f} s"),
        (
            "Synthetic shape priming",
            f"{init['synthetic_shape_priming_seconds']:.3f} s; "
            + str(run["synthetic_priming"]["layout_count"])
            + " layouts; synthetic tensors only, no RAW/future sensor data",
        ),
        (
            "History initialization",
            f"{init['history_initialization_seconds']:.3f} s for "
            + f"{init['history_sensor_duration_us'] / 1000:g} ms of preceding sensor history",
        ),
        (
            "Source warmup",
            f"{init['source_warmup_wall_seconds']:.3f} s; "
            + str(init["source_warmup_frames"])
            + " already-available RGB frames",
        ),
        ("Unrecorded hardware fields", "CPU model and GPU driver version"),
    ]
    tolerance = run["verification"]
    if not (
        tolerance["parity_relative_tolerance"] == tolerance["parity_absolute_tolerance"] == 1e-5
    ):
        raise ValueError("Unexpected parity tolerance")
    if execution["cuda_matmul_allow_tf32"] or execution["cudnn_allow_tf32"]:
        raise ValueError("Unexpected TF32 configuration")
    lines = [
        r"\begin{table}[H]\centering\small",
        r"\caption{Paced-replay hardware, configuration, and initialization timings. "
        r"Startup stages were timed separately and excluded from the arrival clock; "
        r"unexpected shapes were processed within the measured update. "
        r"\evidence{L40,R02}}",
        r"\label{tab:runtime-setup}",
        r"\begin{tabularx}{\linewidth}{@{}p{.31\linewidth}X@{}}\toprule",
        r"Item & Recorded value\\\midrule",
    ]
    for index, (key, value) in enumerate(rows):
        if index == 5:
            lines.append(
                r"\midrule\multicolumn{2}{l}{\textbf{Outside the measured arrival clock}}\\"
            )
        if index == 10:
            lines.append(r"\midrule")
        lines.append(tex(key) + " & " + value + r"\\")
    lines.append(r"\bottomrule\end{tabularx}\end{table}")
    (ROOT / "tables/runtime_setup.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"evidence_id": "R02", "metadata_sha256_verified": EXPECTED_SHA256}))


if __name__ == "__main__":
    main()
