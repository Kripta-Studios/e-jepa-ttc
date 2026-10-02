"""Small filesystem/callback adapters around unmodified frozen analysis functions."""

from __future__ import annotations

import copy
import inspect
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from runtime import (
    atomic_parquet_write,
    digest,
    publish_json,
    require_roots,
    resumable_hierarchical_losses,
)

UNCERTAINTY_EXISTS_GUARD = (
    "if output.exists():\n        raise FileExistsError("
    '"uncertainty output already exists; preserve prior evidence")'
)
UNCERTAINTY_DIRECTORY_GUARD = (
    'if output.is_file():\n        raise FileExistsError("uncertainty output is not a directory")'
)


def set_override(target: object, name: str, implementation: object) -> None:
    """Bind an explicit operational callback on a dynamically imported API."""
    setattr(target, name, implementation)


def resumable_function(
    function: Callable[..., Any], replacements: dict[str, str], overrides: dict[str, Any]
) -> Callable[..., Any]:
    """Change only explicit output guards; archive the exact transformation."""
    source = inspect.getsource(function)
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError(
                f"operational adapter source guard changed: {function.__name__}: {old}"
            )
        source = source.replace(old, new)
    namespace = {**function.__globals__, **overrides}
    exec(
        compile(source, inspect.getfile(function) + ":operational-output-guards", "exec"), namespace
    )
    return namespace[function.__name__]


class InputSnapshot:
    """Full admission at entry/final, stat change detection at durable boundaries."""

    def __init__(self, launch: dict) -> None:
        self.launch = launch
        self.roots = {k: Path(v) for k, v in launch["roots"].items()}
        self.freeze = json.loads(Path(launch["freeze"]).read_text(encoding="utf-8"))
        self.paths = {self.roots[p["root"]] / p["relative_path"] for p in self.freeze["files"]}
        self.paths.update(
            Path(launch[k])
            for k in ("freeze", "local_paths", "source_configuration", "evidence_profile")
        )
        self.paths.update(
            Path(launch["accounting"][k]) for k in ("journal", "ledger", "reconciliation")
        )
        for phase in launch["publications"].values():
            self.paths.update(Path(phase[k]) for k in ("publication", "endpoints"))
            manifest = json.loads(Path(phase["publication"]).read_text(encoding="utf-8"))
            self.paths.update(
                Path(phase["publication"]).parent / row["path"] for row in manifest["fits"].values()
            )
        self.stats = None
        self.last_check = 0.0
        self.original_admission: Callable[..., Any] | None = None
        self.admitted_result = None
        self.original_freeze_reader: Callable[..., dict] | None = None
        self.verified_freeze = None

    def capture(self) -> dict[str, tuple[int, int, int]]:
        require_roots(self.roots)
        return {
            str(p): (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino) for p in self.paths
        }

    def check(self) -> None:
        # Cheap resource checks are separate. Integrity is checked at most once
        # per second inside loader callbacks, and forcibly at chunk boundaries.
        if time.monotonic() - self.last_check < 1:
            return
        self.force_check()

    def force_check(self) -> None:
        current = self.capture()
        if self.stats is not None and current != self.stats:
            changed = [p for p, value in current.items() if self.stats.get(p) != value]
            raise ValueError(f"admitted input snapshot changed: {changed[:4]}")
        self.last_check = time.monotonic()

    def admission(self, *args: Any, **kwargs: Any) -> Any:
        self.check()
        if self.admitted_result is None:
            assert self.original_admission is not None
            before = self.capture()
            self.admitted_result = self.original_admission(*args, **kwargs)
            after = self.capture()
            if before != after:
                raise ValueError("inputs changed during complete admission")
            self.stats = after
        return self.admitted_result

    def read_freeze(
        self,
        path: Path,
        *,
        expected_sha256: str,
        roots: dict[str, Path],
        validate_prerequisites: Callable[[], None],
    ) -> dict:
        validate_prerequisites()
        self.check()
        if (
            Path(path) != Path(self.launch["freeze"])
            or expected_sha256 != self.launch["freeze_sha256"]
        ):
            raise ValueError("unexpected freeze requested by adapter")
        if self.verified_freeze is None:
            assert self.original_freeze_reader is not None
            self.verified_freeze = self.original_freeze_reader(
                path,
                expected_sha256=expected_sha256,
                roots=roots,
                validate_prerequisites=validate_prerequisites,
            )
        return self.verified_freeze


def install(launch: dict) -> tuple[InputSnapshot, Callable[..., Any], Callable[..., Any]]:
    """Install operation-only adapters after complete input admission is required."""
    import pandas as pd

    from e_jepa_ttc.simplex_t import (
        campaign_completion,
        candidate_interface,
        compact_phase,
        configured_postprocessing,
        followup_analysis,
        history_support,
        provenance_bundle,
        scientific_admission,
        scientific_freeze,
        sealed_analysis,
        stage_gate,
        uncertainty_analysis,
    )

    snapshot = InputSnapshot(launch)
    snapshot.original_admission = scientific_admission.validate_scientific_admission
    snapshot.original_freeze_reader = scientific_freeze.read_scientific_freeze
    original_support = history_support.frozen_train_history_support
    support_cache: dict[object, dict] = {}

    def frozen_support(
        sources: Any, freeze: dict, *, validate_frozen_sources: Callable[[], None]
    ) -> dict:
        snapshot.force_check()
        validate_frozen_sources()
        if freeze != snapshot.freeze:
            raise ValueError("TRAIN history support requested against another freeze")
        identity = sources
        if identity not in support_cache:
            support_cache[identity] = original_support(
                sources,
                freeze,
                validate_frozen_sources=validate_frozen_sources,
            )
        return copy.deepcopy(support_cache[identity])

    campaign_completion.frozen_train_history_support = frozen_support
    stage_gate.frozen_train_history_support = frozen_support
    configured_postprocessing.validate_scientific_admission = snapshot.admission
    for module in (
        campaign_completion,
        provenance_bundle,
        configured_postprocessing,
        sealed_analysis,
        compact_phase,
        candidate_interface,
        stage_gate,
    ):
        if hasattr(module, "read_scientific_freeze"):
            set_override(module, "read_scientific_freeze", snapshot.read_freeze)
    draws = (
        Path(launch["roots"]["work"])
        / "artifacts/simplex_t/scientific_campaign/T6/CHECKPOINTED_WORK/analysis"
        / "analyses/T2/paired_uncertainty/HIERARCHICAL_DRAWS.jsonl"
    )
    binding = {
        "scientific_freeze_sha256": launch["freeze_sha256"],
        "publications": {
            stage: {k: row[k] for k in ("publication_sha256", "endpoints_sha256")}
            for stage, row in launch["publications"].items()
        },
        "recipe_sha256": digest(Path(inspect.getfile(uncertainty_analysis))),
        "historical_recipe_sha256": digest(
            Path(launch["roots"]["work"]) / "src/e_jepa_ttc/evaluation/risk_geometry_v10.py"
        ),
    }

    def hierarchical(
        frame: pd.DataFrame, losses: np.ndarray, output: Path, check: Callable[[], None]
    ) -> tuple[np.ndarray, dict]:
        def boundary() -> None:
            snapshot.force_check()
            check()

        return resumable_hierarchical_losses(
            frame, losses, output, boundary, draws_path=draws, binding=binding
        )

    paired = resumable_function(
        uncertainty_analysis.paired_uncertainty,
        {
            UNCERTAINTY_EXISTS_GUARD: UNCERTAINTY_DIRECTORY_GUARD,
            "output.mkdir(parents=True)": "output.mkdir(parents=True, exist_ok=True)",
        },
        {"hierarchical_losses": hierarchical, "write_new_json": publish_json},
    )
    replacements = {
        "output.exists() or output.resolve()": "output.resolve()",
        "output.mkdir(parents=True)": "output.mkdir(parents=True, exist_ok=True)",
    }
    follow = resumable_function(
        followup_analysis.analyze_followup_phase,
        replacements,
        {"paired_uncertainty": paired, "write_new_json": publish_json},
    )
    three = resumable_function(
        followup_analysis.analyze_three_seed_family,
        replacements,
        {"paired_uncertainty": paired, "write_new_json": publish_json},
    )
    # Diagnostic tables are small independent commits; make their publication
    # atomic. Inputs are immutable and the final stage manifest binds all bytes.
    original_parquet = pd.DataFrame.to_parquet

    def atomic_parquet(frame: pd.DataFrame, path: Path, *args: Any, **kwargs: Any) -> object:
        return atomic_parquet_write(frame, Path(path), original_parquet, args, kwargs)

    set_override(pd.DataFrame, "to_parquet", atomic_parquet)
    return snapshot, follow, three
