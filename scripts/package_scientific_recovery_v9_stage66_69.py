"""Read-only scientific reporting and verified compact packaging; never fits."""

# ruff: noqa: E402, E501 -- script imports and prose report paragraphs
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from e_jepa_ttc.artifacts.essential_bundle_v10 import create_bundle
from e_jepa_ttc.artifacts.risk_geometry_v10 import PhaseLedger, atomic_json, binding, verify
from e_jepa_ttc.evaluation.risk_geometry_v10 import verify_coverage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve(strict=True)
    decision = json.loads((root / "NEXT_DECISION_V3.json").read_text())
    analysis_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    decision["analysis_commit"] = analysis_commit
    atomic_json(root / "NEXT_DECISION_V3.json", decision)
    for p in root.glob("stage*_seeds*/LEDGER.json"):
        PhaseLedger(p).require("DECIDED")
    for p in root.glob("stage*_seeds*/diagnostics*/DIAGNOSTIC_COVERAGE.json"):
        verify_coverage(p)
    endpoint_records = []
    updates = 0
    fits = []
    safe = root / "safe_model_arrays"
    safe.mkdir(exist_ok=True)
    for p in root.glob("stage*_seeds*/ALL_ENDPOINTS_FROZEN.json"):
        endpoints = json.loads(p.read_text())
        for name, record in endpoints.items():
            path = verify(record)
            endpoint_records.append(record)
            if path.suffix == ".pt":
                state = torch.load(path, map_location="cpu", weights_only=False)
                updates += state["update"]
                np.savez(
                    safe / (name + ".npz"),
                    **{k: v.numpy() for k, v in state["model"].items()},
                    mean=state["mean"],
                    std=state["std"],
                )
                atomic_json(
                    safe / (name + ".json"),
                    dict(identity=state["identity"], update=state["update"], source=record),
                )
            fits.append(name)
    resume = [binding(p) for p in root.glob("stage*_seeds*/*/update*.pt")]
    atomic_json(root / "RESUME_CHECKPOINT_INDEX.json", resume)
    atomic_json(
        root / "FIT_RECONCILIATION.json",
        dict(fits=fits, fit_count=len(fits), optimizer_updates=updates, endpoints=endpoint_records),
    )
    lines = [
        "# E-JEPA-TTC Stage66–69 final report",
        "",
        f"Terminal action: **{decision['next_action']}**.",
        "",
        f"Training commit: `{decision.get('training_commit', 'none')}`. Analysis commit: `{analysis_commit}`.",
        f"Executed scientific fits: {len(fits)}; optimizer updates: {updates}. Ridge fits have no optimizer updates.",
        "",
        "Historical Stage63–65 acceptance remains INTEGRITY_BLOCKED. Recovered fixed references: RISK17 146.0394511387526 MiD; RISK8 150.091686; CE17/R2 150.876372; RouterR 153.876800. Historical diagnostics are POSTHOC_NONSELECTABLE.",
        "",
        "The earlier garl_ttc key/path exposure is recorded as LEVEL_0_METADATA_ONLY per the user's explicit policy. Prior derived scientific outputs were not reused: the original verified package and producer tables were re-audited from safe commit 5a877f6e7560f3b537ffa1e61298a2a11d8d5dd8 in the clean-v2 worktree.",
        "",
        "No protected evaluation was authorized or intentionally opened by this campaign. The source journal covers this runner's accesses; it cannot certify unrelated processes. Claims are development-only on nine reused sequences. No expert refits, seed selection, TTC averaging, downloads, submission, push or PR.",
        "",
        "| Phase / arm | MiD |",
        "|---|---:|",
    ]
    for p in sorted(root.glob("stage*_seeds*/diagnostics*/DIAGNOSTICS.json")):
        data = json.loads(p.read_text())
        for arm, summary in data["summaries"].items():
            lines.append(
                f"| {p.parent.parent.name}/{p.parent.name}/{arm} | {summary['score']:.9f} |"
            )
        lines.extend(
            [
                "",
                "Paired primary-minus-reference evidence:",
                "",
                "```json",
                json.dumps(data["comparisons"], indent=2),
                "```",
                "",
            ]
        )
    lines.extend(["", "Phase decisions and conditional reasons:", ""])
    for stage in (67, 68, 69):
        paths = list(root.glob(f"stage{stage}_seeds*/DECISION.json"))
        if paths:
            for p in paths:
                lines.extend([f"{p.parent.name}:", "```json", p.read_text(), "```"])
        else:
            reason = (
                decision["next_action"]
                if stage == 68 and decision["next_action"] == "GEOMETRY_INSUFFICIENT_SUPPORT"
                else "NOT_AUTHORIZED_BY_PARENT_GATE"
            )
            lines.append(f"Stage{stage}: {reason}.")
    if (root / "GEOMETRY_SUPPORT.json").exists():
        lines += [
            "",
            "Train-only geometry support:",
            (root / "GEOMETRY_SUPPORT.json").read_text(),
            "Insufficient support is missing observability, not proof that event motion carries no information.",
        ]
    lines += [
        "",
        "QA details and exact baseline/new failure IDs are in QA_ACCEPTANCE.json and the linked logs. All attempted failures are retained separately from numerical conclusions. Replication, when authorized, covers only the new selector head conditional on seed-7 experts; seed-averaged diagnostics average losses only. Routing fields in those loss-average files describe the fixed deployment seed7.",
        "",
        "Matched Garl predictions/checkpoint ancestry are required for a direct benchmark superiority claim; the historical 144.353 scalar is not a matched comparator. See GARL_REFERENCE_STATUS.json and the future confirmation plan. No fresh benchmark was executed.",
        "",
        "Essential bundle: physical nested feature tables, original tiny ridge parameters, all new endpoint bytes plus safe inference arrays, per-row errors and selections, uncertainty draws, diagnostics, locks, source/attempt manifests and complete local resume index. Intermediate optimizer checkpoints are indexed locally; raw event streams and historical encoder checkpoints are not bundled.",
    ]
    (root / "CODEX_STAGE66_STAGE67_STAGE68_STAGE69_FINAL_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    files = {}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        relative = p.relative_to(root)
        if any(
            part.startswith("stage66_bundle_") or part == "__pycache__" for part in relative.parts
        ):
            continue
        if p.suffix in (".zip", ".sha256", ".tmp", ".h5", ".hdf5", ".pyc") or p.name.endswith(
            ".verification.json"
        ):
            continue
        if p.suffix == ".pt" and p.name != "update1500.pt":
            continue
        if p.name == "ACTIVE_OWNER.json":
            continue
        files["run/" + relative.as_posix()] = p
    paths = subprocess.check_output(
        [
            "git",
            "ls-files",
            "src",
            "scripts",
            "tests",
            "configs/protocol/scientific_recovery_v9_stage66_69.json",
            "docs/stage66_69_implementation.md",
        ],
        cwd=ROOT,
        text=True,
    ).splitlines()
    files.update({"code/" + p: ROOT / p for p in paths})
    archive = (
        root
        / f"E_JEPA_TTC_STAGE66_STAGE67_STAGE68_STAGE69_ESSENTIAL_RESULTS_{analysis_commit[:12]}.zip"
    )
    result = create_bundle(files, archive)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
