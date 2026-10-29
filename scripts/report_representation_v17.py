"""Describe all six endpoints, exact CSV replay, group bootstrap and H/V generalization."""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from e_jepa_ttc.simplex_t.phase import phase_to_ttc
from operational.range_transfer.analyze import metrics
from operational.range_transfer.common import read, sha, write
from operational.representation_audit.model import RepresentationRefiner
from operational.representation_audit.run import INPUTS, QUEUE, ROOT, RUN, cache


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    results = read(RUN / "RESULTS.json")
    endpoints = read(RUN / "QUEUE_FREEZE.json")
    h = cache("H")
    h_target = -0.1 / np.expm1(-h["target_phase"].astype(np.float64))
    h_groups = h["group_id"].astype(str)
    all_rows, clipping, profiles, frames = [], [], [], {}
    for arm, seed in QUEUE:
        name = f"{arm}-S{seed}"
        path = RUN / f"evaluation/{name}/PREDICTIONS.csv"
        frame = pd.read_csv(path, converters={"target_ttc": float, "prediction_ttc": np.float32})
        if sha(path) != read(path.parent / "RECEIPT.json")["predictions_sha256"]:
            raise ValueError("Predictions changed")
        reproduced = metrics(
            frame.target_ttc.to_numpy(),
            frame.prediction_ttc.to_numpy(np.float32),
            frame.group_id.to_numpy(),
        )
        if reproduced != results[name]:
            raise AssertionError("Saved metrics do not reproduce exactly")
        frames[name] = frame
        curve = pd.read_csv(RUN / f"fits/{name}/CURVE.csv")
        clipping.append(
            {
                "fit": name,
                "updates": len(curve),
                "clipped_fraction": float((curve.gradient_l2 > 1).mean()),
                "last250_loss": float(curve.loss.tail(250).mean()),
            }
        )
        endpoint = RUN / f"fits/{name}/checkpoint_last.pt"
        if sha(endpoint) != endpoints[name]["checkpoint_sha256"]:
            raise ValueError("Endpoint changed")
        payload = torch.load(endpoint, map_location="cpu", weights_only=False)
        model = RepresentationRefiner(arm == "G")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval().requires_grad_(False)
        h_pred = []
        with torch.inference_mode():
            for start in range(0, len(h_target), 128):
                args = [torch.from_numpy(h[k][start : start + 128].copy()) for k in INPUTS]
                h_pred.append(phase_to_ttc(model(*args)["raw_location"]).numpy())
            args = [torch.from_numpy(h[k][:1].copy()) for k in INPUTS]
            for _ in range(3):
                model(*args)
            samples = []
            for _ in range(40):
                start = time.perf_counter()
                model(*args)
                samples.append((time.perf_counter() - start) * 1000)
        h_score = metrics(h_target, np.concatenate(h_pred), h_groups)
        write(path.parent / "H_DIAGNOSTIC.json", h_score)
        profiles.append(
            {
                "fit": name,
                "median_ms": float(np.median(samples)),
                "p95_ms": float(np.quantile(samples, 0.95)),
                "scope": "CPU_FP32_head_only_batch1_cached_inputs_1thread_shared_host",
                "end_to_end": False,
            }
        )
        for role, score in (("H", h_score), ("V", results[name])):
            all_rows.append(
                {
                    "fit": name,
                    "role": role,
                    "MiDc": score["bins"]["crucial"]["group_macro_mid"],
                    "MiDs": score["bins"]["small"]["group_macro_mid"],
                    "MiDl": score["bins"]["large"]["group_macro_mid"],
                    "MiDn": score["bins"]["negative"]["group_macro_mid"],
                    "overall_MiD": score["overall_MiD"],
                    "MAE": score["diagnostics"]["mae_s"],
                }
            )
        write(
            RUN / f"fits/{name}/RUN_PROVENANCE.json",
            {
                "scientific_code_commit": read(RUN / "SCIENTIFIC_CODE_COMMIT.json"),
                "identity": payload["identity"],
                "checkpoint_sha256": sha(endpoint),
                "environment": read(RUN / "ENVIRONMENT.json"),
            },
        )
    draws = []
    for seed in (7, 13, 23):
        a, b = frames[f"Z-S{seed}"], frames[f"G-S{seed}"]
        if not a[["sample_token", "group_id", "target_ttc"]].equals(
            b[["sample_token", "group_id", "target_ttc"]]
        ):
            raise ValueError("Paired cohorts differ")
        groups = sorted(a.group_id.unique())
        diff = []
        for group in groups:
            mask = (a.group_id == group) & (a.target_ttc > 0.1) & (a.target_ttc <= 3)
            target = a.loc[mask, "target_ttc"].to_numpy(np.float64)
            if not len(target):
                continue
            errors = [
                10000
                * np.abs(
                    np.log1p(-0.1 / x.loc[mask, "prediction_ttc"].to_numpy(np.float64))
                    - np.log1p(-0.1 / target)
                ).mean()
                for x in (a, b)
            ]
            diff.append(errors[1] - errors[0])
        generator = np.random.default_rng(12345)
        values = np.asarray(diff)
        boot = values[generator.integers(len(values), size=(2000, len(values)))].mean(1)
        draws.append(
            {
                "seed": seed,
                "delta_MiDc": float(values.mean()),
                "CI95": np.quantile(boot, [0.025, 0.975]).tolist(),
                "groups": len(values),
            }
        )
    pd.DataFrame(all_rows).to_csv(RUN / "H_V_TABLE.csv", index=False)
    pd.DataFrame(clipping).to_csv(RUN / "CLIPPING.csv", index=False)
    write(RUN / "HEAD_ONLY_COST.json", profiles)
    write(
        RUN / "BOOTSTRAP.json",
        {
            "unit": "group",
            "replicates": 2000,
            "seed": 12345,
            "pairs": draws,
            "CI_is_selection_gate": False,
        },
    )
    write(
        RUN / "METRIC_REPLAY.json",
        {
            "status": "PASS",
            "fits": 6,
            "rows": sum(len(f) for f in frames.values()),
            "max_difference": 0,
        },
    )
    render()


def markdown(frame: pd.DataFrame) -> str:
    """Render a simple numeric table without additional optional dependencies."""

    def cell(value: object) -> str:
        return f"{value:.4f}" if isinstance(value, float) else str(value)

    lines = [
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
    ]
    lines += [
        "| " + " | ".join(cell(v) for v in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join(lines)


def render() -> None:
    table = pd.read_csv(RUN / "H_V_TABLE.csv")
    clipping = pd.read_csv(RUN / "CLIPPING.csv")
    draws = read(RUN / "BOOTSTRAP.json")["pairs"]
    mean = table.loc[table.role == "V"].copy()
    mean["arm"] = mean.fit.str.split("-").str[0]
    mean = (
        mean.groupby("arm")[["MiDc", "MiDs", "MiDl", "MiDn", "overall_MiD", "MAE"]]
        .mean()
        .reset_index()
    )
    mean.to_csv(RUN / "V_SEED_MEAN.csv", index=False)
    destination = ROOT / "docs/representation_audit_20261012/RESULTS.md"
    content = [
        "# V17 — cola A: bypass geométrico\n",
        f"Decisión: **{read(RUN / 'NEXT_DECISION.json')['status']}**. "
        "V es desarrollo interno expuesto.\n",
        "## Métricas regeneradas\n",
        markdown(table),
        "\n## Media aritmética de métricas entre tres semillas\n",
        markdown(mean),
        "\n## Clipping\n",
        markdown(clipping),
        "\n## Bootstrap pareado por grupos\n",
        markdown(pd.DataFrame(draws)),
        "\nLos resultados H son ajuste/diagnóstico; V es desarrollo expuesto. "
        "Los encoders se comparten entre semillas. Los intervalos no son un gate adicional. "
        "No se escoge semilla ni se promedian predicciones.\n",
        "Métricas reconstruidas exactamente de 98.592 filas CSV. El perfil mide sólo cabeza CPU, "
        "con inputs preparados y host compartido; no representa E2E. La contabilidad GPU incluye "
        "el tiempo de los procesos propios dentro de cada reserva; "
        "las reservas no cerradas se contabilizan completas.\n",
    ]
    destination.write_text("\n".join(content), encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.render_only:
        render()
    else:
        main()
