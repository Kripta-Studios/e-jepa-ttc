"""Finish frozen primary evidence without selecting an adaptive recipe or opening V."""

from __future__ import annotations

import argparse

from operational.range_transfer.common import freeze, read, sha, utc, write

from .bootstrap import compare
from .common import RUN, config
from .cost import finalize_selection, measure
from .evaluate import freeze_primary
from .export import export
from .report import build
from .verify_predictions import verify


def complete() -> dict:
    """Resume finite evidence work after all primary fits and P/H cohorts finish."""
    endpoints = freeze_primary()
    names = ["A5-P2-FROZEN", *config()["fits"]]
    # Admit every cohort before producing any comparative interpretation.
    cohorts = {}
    for role in ("P", "H"):
        for name in names:
            folder = RUN / "predictions" / role / name
            receipt = read(folder / "COMPLETE.json")
            if (
                receipt["status"] != "COMPLETE"
                or sha(folder / "predictions.parquet") != receipt["sha256"]
            ):
                raise ValueError(f"Primary evidence requires complete {role}/{name}")
            cohorts[f"{role}/{name}"] = receipt["sha256"]
    directory = RUN / "primary_evidence"
    binding = {"endpoints": endpoints, "cohorts": cohorts}
    freeze(directory / "BINDING.json", binding)

    timing = measure()
    if timing["status"] != "COMPLETE":
        return {"status": "PARTIAL_COST", "detail": timing}
    selection = finalize_selection()
    verification = {}
    for role in ("P", "H"):
        for name in names:
            folder = RUN / "predictions" / role / name
            result = verify(folder)
            verification[f"{role}/{name}"] = {
                "status": result["status"],
                "rows": result["rows"],
                "metrics_sha256": result["metrics_sha256"],
            }
        reference = RUN / "predictions" / role / names[0] / "predictions.parquet"
        for name in config()["fits"]:
            candidate = RUN / "predictions" / role / name / "predictions.parquet"
            compare(reference, candidate, RUN / "bootstrap" / role / name)
    exports = {name: export(name) for name in names}
    result = {
        "status": "PRIMARY_EVIDENCE_COMPLETE_RESEARCH_ONGOING",
        "completed_utc": utc(),
        "binding": binding,
        "verification": verification,
        "exports": exports,
        "selection": selection,
        "system_promotion": False,
        "official_confirmation": False,
        "next": (
            "Interpret P/H paired mechanisms; preregister enabled adaptive and RGB work. "
            "A negative primary selection does not close the authorized program."
        ),
    }
    # The timestamp may differ after a replay; the immutable input binding above may not.
    write(directory / "COMPLETE.json", result)
    build()
    return result


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(complete())
