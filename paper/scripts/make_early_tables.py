"""Regenerate early-study LaTeX fragments from public table data."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    payload = json.loads((ROOT / "data/early_tables.json").read_text(encoding="utf-8"))
    for table in payload["tables"]:
        lines = [
            rf"\begin{{tabular}}{{{table['spec']}}}",
            r"\toprule",
            table["header"] + r" \\",
            r"\midrule",
            *(row + r" \\" for row in table["rows"]),
            r"\bottomrule",
            r"\end{tabular}",
        ]
        (ROOT / "data" / table["file"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"generated_early_tables": len(payload["tables"])}))


if __name__ == "__main__":
    main()
