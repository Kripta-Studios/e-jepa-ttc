"""Render the saved foreground pilot using only the standard library."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    folder = root / "artifacts/representation_audit_20261012/foreground_diagnostic"
    with (folder / "P40.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    scope = json.loads((folder / "SCOPE.json").read_text(encoding="utf-8"))
    if len(rows) != 40 or scope["optimizer_updates"] != 0:
        raise ValueError("Wrong diagnostic scope")
    nonzero = {
        key: sum(math.isfinite(float(r[key])) and float(r[key]) > 0 for r in rows)
        for key in ("height_gradient_norm", "temporal_gradient_norm", "nll_gradient_norm")
    }
    lines = [
        "# Piloto P: sensibilidad del foreground nativo",
        "",
        "Se ejecutó el forward nativo A5 en CPU sobre 40 consultas P: ocho sin reemplazo "
        "en cada banda, con seed20261012. Se derivó sólo respecto a los parámetros de "
        "`encoder.foreground`; se verificó que todos los parámetros y buffers permanecieran "
        "idénticos. No hubo updates de optimizador.",
        "",
        "| Banda | n | Norma altura, media | Norma NLL, media | "
        "Coseno altura/NLL, media | Coseno <0 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for band in ("0-1", "1-3", "3-6", "6plus", "negative"):
        subset = [r for r in rows if r["band"] == band]
        height = statistics.mean(float(r["height_gradient_norm"]) for r in subset)
        nll = statistics.mean(float(r["nll_gradient_norm"]) for r in subset)
        cosines = [float(r["height_nll_cosine"]) for r in subset]
        lines.append(
            f"| {band} | {len(subset)} | {height:.6f} | {nll:.6f} | "
            f"{statistics.mean(cosines):.6f} | {sum(c < 0 for c in cosines)}/{len(cosines)} |"
        )
    lines += [
        "",
        "Gradientes finitos y no nulos: "
        + "; ".join(f"{key}: {value}/40" for key, value in nonzero.items())
        + ".",
        "",
        "Esto no respalda que la rama esté completamente muerta en estos controles. "
        "Tampoco demuestra que una pérdida domine el entrenamiento nativo: la pérdida "
        "de altura del piloto es MSE sobre log-altura, la temporal es MSE de su cambio "
        "y la NLL es sólo el componente de ratio de la pareja actual. Sus escalas y "
        "reducciones difieren; se excluyen top-k, riesgo y otros términos nativos.",
        "",
        "Las derivadas recorren el forward nativo completo, con el resto de pesos "
        "congelados, incluidas sus dependencias de geometría, tokens e incertidumbre. "
        "No son las derivadas de una futura rama que conservara esos tokens por separado.",
        "",
        "Una probabilidad de foreground próxima a0 o1 puede corresponder a una máscara "
        "bien delimitada; su fracción de saturación no demuestra por sí sola un fallo. "
        "El CSV conserva esa fracción y la dispersión espacial, sin reinterpretarlas como "
        "una métrica de calidad de segmentación.",
        "",
        "Es un piloto descriptivo P con pocos ejemplos por banda. No estima rendimiento "
        "poblacional, no añade evidencia H/V ni selecciona un checkpoint. Todos los "
        "tokens, targets, pérdidas, normas y cosenos se conservan en `P40.csv`.",
    ]
    (root / "docs/representation_audit_20261012/FOREGROUND_PILOT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    (folder / "REPORT_SOURCE.json").write_text(
        json.dumps(
            {
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "input_sha256": hashlib.sha256((folder / "P40.csv").read_bytes()).hexdigest(),
                "nonzero_finite_gradients": nonzero,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
