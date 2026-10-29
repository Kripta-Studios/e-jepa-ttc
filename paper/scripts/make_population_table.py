"""Regenerate the evaluation-population map from the distributed result ledgers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    official = json.loads((ROOT / "data/official_benchmark.json").read_text(encoding="utf-8"))
    competition = json.loads(
        (ROOT / "data/evttc_competition_summary.json").read_text(encoding="utf-8")
    )
    with (ROOT / "data/transfer_queries.csv").open(encoding="utf-8", newline="") as stream:
        transfer = list(csv.DictReader(stream))
    samples = int(official["phase_submission"]["metrics"]["num_samples"])
    population = competition["query_population"]
    rows = [
        (
            "Official GarlTTC Test",
            f"{samples:,} scored tokens",
            r"Private targets; band-weighted MiD and scorer-defined failures. \evidence{B01,B02}",
        ),
        (
            "Published Garl EvTTC",
            f"{competition['literature']['reported_sequences']} sequences",
            r"Published RTE; target/query correspondence to our run unverified. \evidence{T02}",
        ),
        (
            "EvTTC competition",
            f"{population['queries']} queries; {population['sequence_count']} sequences",
            r"Unscored predictions, coverage and disagreement. \evidence{T01}",
        ),
    ]
    for cohort, label in (
        ("extended_new", "EvTTC Extended-new"),
        ("historical46", "EvTTC Historical46"),
    ):
        valid = [row for row in transfer if row["cohort"] == cohort and not row["target_exclusion"]]
        groups = len({row["group_id"] for row in valid})
        rows.append(
            (
                label,
                f"{len(valid)} targets; {groups} sequences",
                r"Labeled, development-exposed transfer with annotated ROI. \evidence{L01}",
            )
        )
    lines = [
        r"\begin{table}[H]",
        r"\centering\small",
        r"\caption{Evaluation populations and the quantities each supports. Historical46 "
        r"and Extended-new share three sequence groups but no query identities.}",
        r"\label{tab:evaluation-populations}",
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}p{.24\linewidth}"
        r">{\raggedright\arraybackslash}p{.22\linewidth}X@{}}",
        r"\toprule Population & Size & Interpretation \\\midrule",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule\end{tabularx}", r"\end{table}"])
    (ROOT / "tables/evaluation_populations.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
