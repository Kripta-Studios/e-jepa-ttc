"""Build a sanitized, hash-verified publication of the 2026-10-08 campaign.

The publisher is deliberately read-only with respect to scientific artifacts.  It
copies a small allowlist into a documentation tree, records both source and
published hashes, and creates a separate essential-evidence ZIP.  A complete
publication is refused while any required campaign branch is unfinished.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

SCORING_FILES = (
    "METADATA.json",
    "METHOD_METRICS.csv",
    "PAIRED.csv",
    "PER_SEQUENCE.csv",
    "REPORT.json",
    "SCORED_ROWS.csv",
)
TEXT_SUFFIXES = {".json", ".md", ".txt", ".csv", ".py", ".yaml", ".yml"}
EXCLUDED_NAMES = {"PLAN.json", "QUERY_MANIFEST.json", "WRITER.lock"}
MAX_PUBLIC_FILE_BYTES = 5 * 1024 * 1024
OWNED_OUTPUT_NAMES = {
    ".gitattributes",
    "README.md",
    "NEXT_DECISION.json",
    "regenerate.py",
    "evidence",
    "tables",
    "SOURCE_INVENTORY.json",
    "SHA256SUMS.txt",
}


@dataclass(frozen=True)
class Branch:
    """Publication readiness of one independently auditable campaign branch."""

    name: str
    status: str
    complete: bool
    path: str | None
    detail: str


def sha256(path: Path) -> str:
    """Return the physical SHA-256 of *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _verified_scoring(directory: Path) -> bool:
    manifest = directory / "SHA256.json"
    if not manifest.is_file():
        return False
    hashes = _json(manifest)
    if set(hashes) != set(SCORING_FILES):
        return False
    for name in SCORING_FILES:
        path = directory / name
        expected = hashes.get(name)
        if not path.is_file() or not isinstance(expected, str) or sha256(path) != expected:
            return False
    return True


def _expanded_complete(artifacts: Path) -> bool:
    """Require the canonical final campaign result and every hash it binds."""
    try:
        result_path = artifacts / "CAMPAIGN_RESULT.json"
        result = _json(result_path)
        metrics = artifacts / "expanded_metrics"
        baseline = artifacts / "dev32_expanded"
        full = artifacts / "dev32_expanded_rgb"
        freeze = artifacts / "CAMPAIGN_FREEZE.json"
        freeze_value = _json(freeze)
        state = _json(artifacts / "PIPELINE_STATE.json")
        from operational.sota_eval import campaign

        device = str(freeze_value.get("device", ""))
        event_backend = str(freeze_value.get("event_backend", "direct"))
        full_backend = str(freeze_value.get("full_backend", "direct"))
        source_hashes = freeze_value.get("sources", {})
        source_current = isinstance(source_hashes, dict) and all(
            (campaign.ROOT / relative).is_file() and sha256(campaign.ROOT / relative) == expected
            for relative, expected in source_hashes.items()
        )
        return bool(
            result.get("status") == "COMPLETE"
            and result.get("optimizer_updates") == 0
            and _verified_scoring(metrics)
            and Path(str(freeze_value.get("baseline", ""))).resolve() == baseline.resolve()
            and Path(str(freeze_value.get("full", ""))).resolve() == full.resolve()
            and Path(str(freeze_value.get("metrics", ""))).resolve() == metrics.resolve()
            and source_current
            and campaign._sealed(baseline, "event", device, event_backend)
            and campaign._scored(baseline)
            and campaign._sealed(full, "full", device, full_backend)
            and campaign._scored(full)
            and result.get("campaign_freeze_sha256") == sha256(freeze)
            and result.get("baseline_preservation_sha256")
            == sha256(full / "BASELINE_PRESERVATION.json")
            and result.get("full_prediction_seal_sha256")
            == sha256(full / "PREDICTIONS_SEALED.json")
            and result.get("scored_predictions_sha256") == sha256(full / "SCORED_PREDICTIONS.csv")
            and result.get("metrics_sha256") == sha256(metrics / "SHA256.json")
            and state.get("status") == "COMPLETE"
            and state.get("result_sha256") == sha256(result_path)
        )
    except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def _fcwd_complete(artifacts: Path) -> bool:
    """Validate the canonical FCWD prediction, target-join and scoring lineage."""
    try:
        output = artifacts / "fcwd_inference"
        scoring = output / "scoring"
        manifest = _canonical_fcwd_manifest(artifacts)
        receipt = _json(output / "SCORING_COMPLETE.json")
        contract = _json(scoring / "TARGET_JOIN_CONTRACT.json")
        coverage = _json(scoring / "MODEL_COVERAGE.json")
        seal = output / "PREDICTIONS_SEALED.json"
        source_value = _json(output / "SOURCE_FREEZE.json")
        inference_value = _json(output / "INFERENCE_FREEZE.json")
        from operational.sota_eval import fcwd_run, followups

        fcwd_run.verify_seal(output, manifest)
        followups.verify_fcwd(
            output,
            manifest,
            Path(str(source_value["campaign"])),
            Path(str(source_value["public_full_dir"])),
            Path(str(source_value["code_root"])),
        )
        source_current = (
            fcwd_run.source_binding(
                manifest,
                Path(str(source_value["campaign"])),
                Path(str(source_value["public_full_dir"])),
                Path(str(source_value["code_root"])),
                str(source_value["device"]),
            )
            == source_value
        )
        return bool(
            receipt.get("status") == "COMPLETE"
            and receipt.get("queries") == 630
            and receipt.get("optimizer_updates") == 0
            and receipt.get("methods") == list(fcwd_run.METHODS[:4])
            and receipt.get("planned_methods") == list(fcwd_run.METHODS)
            and source_current
            and inference_value.get("source_freeze_sha256") == sha256(output / "SOURCE_FREEZE.json")
            and inference_value.get("manifest_sha256") == sha256(manifest)
            and inference_value.get("optimizer_updates") == 0
            and inference_value.get("targets_read") is False
            and _verified_scoring(scoring)
            and receipt.get("prediction_seal_sha256") == sha256(seal)
            and receipt.get("scoring_manifest_sha256") == sha256(scoring / "SHA256.json")
            and receipt.get("target_join_contract_sha256")
            == sha256(scoring / "TARGET_JOIN_CONTRACT.json")
            and contract.get("status") == "COMPLETE"
            and contract.get("prediction_seal_sha256") == sha256(seal)
            and contract.get("source_freeze_sha256") == sha256(output / "SOURCE_FREEZE.json")
            and contract.get("query_manifest_sha256") == sha256(manifest)
            and contract.get("model_coverage_sha256") == sha256(scoring / "MODEL_COVERAGE.json")
            and contract.get("joined_predictions_sha256")
            == sha256(scoring / "SCORED_PREDICTIONS.csv")
            and contract.get("scoring_manifest_sha256") == sha256(scoring / "SHA256.json")
            and contract.get("labels_opened_only_after_verified_prediction_seal") is True
            and coverage.get("head_selection_performed") is False
            and coverage.get("full_model", {}).get("status") == "DEPENDENCY_UNAVAILABLE"
            and coverage.get("full_model", {}).get("not_a_negative_model_result") is True
        )
    except (FileNotFoundError, KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _canonical_fcwd_manifest(artifacts: Path) -> Path:
    """Resolve the population manifest and reject any divergent inference copy."""
    canonical = artifacts / "fcwd_population" / "QUERY_MANIFEST.json"
    if not canonical.is_file():
        raise FileNotFoundError(f"Missing canonical FCWD query manifest: {canonical}")
    inference_copy = artifacts / "fcwd_inference" / "QUERY_MANIFEST.json"
    if inference_copy.exists() and (
        not inference_copy.is_file() or sha256(inference_copy) != sha256(canonical)
    ):
        raise ValueError("FCWD inference manifest copy differs from canonical population manifest")
    return canonical


def _cost_complete(artifacts: Path) -> bool:
    """Validate the canonical fixed-eight cost summary and its two freezes."""
    try:
        output = artifacts / "cost"
        from operational.sota_eval.followups import verify_cost

        verify_cost(output)
        summary = _json(output / "SYSTEM_COST_SUMMARY.json")
        execution = output / "EXECUTION_FREEZE.json"
        models = output / "MODEL_FREEZE.json"
        execution_value = _json(execution)
        model_value = _json(models)
        selected = execution_value.get("selected_queries", [])
        fragments = sorted(output.glob("FRAGMENT_*.json"), key=str)
        failures = list(output.glob("FAILURE_*.json"))
        selected_ids = {row.get("query_id") for row in selected}
        fragment_ids: set[object] = set()
        fragments_valid = len(fragments) == 8 and not failures
        for path in fragments:
            fragment = _json(path)
            raw = output / str(fragment.get("raw_csv", ""))
            fragment_ids.add(fragment.get("query_id"))
            fragments_valid = fragments_valid and bool(
                fragment.get("status") == "COMPLETE"
                and fragment.get("optimizer_updates") == 0
                and fragment.get("execution_freeze_sha256") == sha256(execution)
                and fragment.get("model_freeze_sha256") == sha256(models)
                and raw.is_file()
                and fragment.get("raw_csv_sha256") == sha256(raw)
            )
        source = Path(str(execution_value.get("source", "")))
        return bool(
            summary.get("schema") == "sota_system_cost_fixed8_v1"
            and summary.get("status") == "COMPLETE"
            and summary.get("query_count") == 8
            and summary.get("optimizer_updates") == 0
            and execution_value.get("status") == "EXECUTION_FROZEN"
            and model_value.get("status") == "MODELS_FROZEN_BEFORE_MEASUREMENT"
            and model_value.get("execution_freeze_sha256") == sha256(execution)
            and summary.get("execution_freeze_sha256") == sha256(execution)
            and summary.get("model_freeze_sha256") == sha256(models)
            and set(summary.get("metrics", {}))
            == {
                "h8_system_three_heads",
                "garl_event_only_shared_preparation",
                "garl_full_rgb_event",
            }
            and len(selected) == 8
            and len(selected_ids) == 8
            and fragment_ids == selected_ids
            and fragments_valid
            and source.is_file()
            and execution_value.get("source_sha256") == sha256(source)
        )
    except (FileNotFoundError, KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _baseline_complete(artifacts: Path) -> bool:
    """Require the failure, corrected-CMax and conditional-pilot evidence as one unit."""
    try:
        root = artifacts / "baselines"
        failure = _json(root / "FAILURE_COUNTS.json")
        diagnosis = _json(root / "CMAX_REFERENCE_DIAGNOSIS.json")
        pilot = root / "pilot_scored"
        hashes = _json(pilot / "SHA256.json")
        return bool(
            failure.get("status") == "COMPLETE"
            and diagnosis.get("status") == "COMPLETE"
            and diagnosis.get("old_variant", {}).get("successful") == 0
            and diagnosis.get("corrected_variant", {}).get("successful") == 15
            and set(hashes) == {"METHOD_METRICS.csv", "REPORT.json", "SCORED_ROWS.csv"}
            and all(sha256(pilot / name) == expected for name, expected in hashes.items())
        )
    except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def inspect_branches(artifacts: Path) -> list[Branch]:
    """Inspect authoritative completion evidence without inferring missing results."""
    scoring = artifacts / "scoring_existing"
    historical_ok = _verified_scoring(scoring)
    historical_detail = "946 GT-eligible rows; exploratory frozen historical cohort"
    branches = [
        Branch(
            "scoring_existing_946",
            "COMPLETE_EXPLORATORY" if historical_ok else "INCOMPLETE",
            historical_ok,
            str(scoring) if scoring.exists() else None,
            historical_detail,
        )
    ]

    expanded = artifacts / "expanded_metrics"
    expanded_ok = _expanded_complete(artifacts)
    branches.append(
        Branch(
            "expanded_metrics",
            "COMPLETE" if expanded_ok else "RUNNING_OR_ABSENT",
            expanded_ok,
            str(expanded) if expanded_ok else None,
            "Full expanded cohort; no result is inferred while inference/scoring runs",
        )
    )

    asset_manifest = artifacts / "fcwd" / "ASSET_MANIFEST.json"
    assets_ok = False
    if asset_manifest.is_file():
        value = _json(asset_manifest)
        assets_ok = (
            value.get("status") == "COMPLETE"
            and not value.get("missing_assets")
            and not value.get("missing_calibration_folders")
            and value.get("targets_read") is False
        )
    branches.append(
        Branch(
            "fcwd_assets",
            "COMPLETE" if assets_ok else "INCOMPLETE",
            assets_ok,
            str(asset_manifest) if asset_manifest.exists() else None,
            "Three public FCWD sequences; small transfer set with uncertified ancestry",
        )
    )

    fcwd_seal = artifacts / "fcwd_inference" / "SCORING_COMPLETE.json"
    fcwd_ok = _fcwd_complete(artifacts)
    branches.append(
        Branch(
            "fcwd_metrics",
            "COMPLETE" if fcwd_ok else "NOT_YET_SCORED",
            fcwd_ok,
            str(fcwd_seal) if fcwd_ok else None,
            "H8/event-only comparison; unavailable comparators remain null",
        )
    )

    parity = artifacts / "garl_parity" / "RESULT.json"
    parity_dir = parity.parent
    parity_hashes = _json(parity_dir / "SHA256.json") if parity.is_file() else {}
    parity_ok = bool(
        parity.is_file()
        and _json(parity).get("status") == "PASSED"
        and set(parity_hashes) == {"AUDIT_BINDING.json", "REPORT.md", "RESULT.json"}
        and all(sha256(parity_dir / name) == expected for name, expected in parity_hashes.items())
        and _json(parity_dir / "AUDIT_BINDING.json").get("result_sha256") == sha256(parity)
    )
    branches.append(
        Branch(
            "garl_parity",
            "PASSED" if parity_ok else "INCOMPLETE",
            parity_ok,
            str(parity) if parity.exists() else None,
            "Native preprocessing parity and declared EvTTC transfer adaptations",
        )
    )

    cost = artifacts / "cost" / "SYSTEM_COST_SUMMARY.json"
    cost_ok = _cost_complete(artifacts)
    branches.append(
        Branch(
            "system_cost",
            "COMPLETE" if cost_ok else "NOT_YET_COMPLETE",
            cost_ok,
            str(cost) if cost_ok else None,
            "Fixed-eight host observation; warm-cache and architecture contexts differ",
        )
    )

    baseline = artifacts / "baselines" / "FAILURE_COUNTS.json"
    baseline_ok = _baseline_complete(artifacts)
    branches.append(
        Branch(
            "baseline_failure_inventory",
            "COMPLETE_DIAGNOSTIC" if baseline_ok else "INCOMPLETE",
            baseline_ok,
            str(baseline) if baseline.exists() else None,
            "Development-32 coverage diagnostic; scorable-subset metrics remain conditional",
        )
    )

    contract = artifacts / "official_contract" / "STATUS.json"
    contract_value = _json(contract) if contract.is_file() else {}
    contract_status = contract_value.get("status", "ABSENT")
    contract_complete = contract_value.get("official_reproduction_ready") is True
    branches.append(
        Branch(
            "official_contract",
            str(contract_status),
            contract_complete,
            str(contract) if contract.exists() else None,
            "Metadata evidence may be complete while official reproduction remains blocked",
        )
    )

    r1 = artifacts / "R1_STATUS.json"
    r1_status = _json(r1).get("status") if r1.is_file() else "ABSENT"
    branches.append(
        Branch(
            "r1_measurement",
            str(r1_status),
            r1_status == "COMPLETE",
            str(r1) if r1.exists() else None,
            "Resume state is reported literally and never promoted by inference",
        )
    )
    return branches


def _replace_roots(text: str, repo: Path, data_roots: list[str]) -> str:
    variants = {str(repo.resolve()), str(repo.resolve()).replace("\\", "/")}
    for value in sorted(variants, key=len, reverse=True):
        escaped = json.dumps(value, ensure_ascii=False)[1:-1]
        text = text.replace(escaped, "${REPO}").replace(value, "${REPO}")
    for raw in sorted(set(data_roots), key=len, reverse=True):
        for value in {raw, raw.replace("\\", "/")}:
            escaped = json.dumps(value, ensure_ascii=False)[1:-1]
            text = text.replace(escaped, "${DATA}").replace(value, "${DATA}")
    return text


def _data_roots(artifacts: Path) -> list[str]:
    roots: list[str] = [str(Path.home())]
    manifest = artifacts / "fcwd" / "ASSET_MANIFEST.json"
    if manifest.is_file():
        value = _json(manifest)
        if isinstance(value.get("dataset_root"), str):
            roots.append(value["dataset_root"])
    population = artifacts / "dev32_expanded" / "POPULATION_FREEZE.json"
    if population.is_file():
        value = _json(population)
        if isinstance(value.get("data_root"), str):
            roots.append(value["data_root"])
    return roots


def _copy_sanitized(source: Path, destination: Path, repo: Path, roots: list[str]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() == ".json":
        raw = json.loads(source.read_text(encoding="utf-8-sig"))

        def clean(value: object) -> object:
            if isinstance(value, dict):
                cleaned: dict[str, object] = {}
                for key, item in value.items():
                    sanitized_key = str(clean(key))
                    if sanitized_key in cleaned:
                        raise ValueError(f"Sanitization key collision in {source}: {sanitized_key}")
                    cleaned[sanitized_key] = clean(item)
                return cleaned
            if isinstance(value, list):
                return [clean(item) for item in value]
            if isinstance(value, str):
                changed = _replace_roots(value, repo, roots)
                windows = PureWindowsPath(changed)
                if windows.is_absolute() and not changed.startswith(("${REPO}", "${DATA}")):
                    tail = [part for part in windows.parts[1:] if part not in {"\\", "/"}]
                    return "${DATA}/" + "/".join(tail[-4:])
                return changed
            return value

        destination.write_text(
            json.dumps(clean(raw), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    elif source.suffix.lower() in TEXT_SUFFIXES:
        text = source.read_bytes().decode("utf-8-sig")
        destination.write_bytes(_replace_roots(text, repo, roots).encode("utf-8"))
    else:
        shutil.copyfile(source, destination)


def _refresh_published_checksum_manifests(
    evidence: Path, inventory: list[dict[str, Any]]
) -> None:
    """Make nested checksum manifests describe published bytes after sanitization.

    A source manifest is retained as provenance through ``source_sha256`` in the
    inventory.  Only safe, present sibling paths are rehashed.  A manifest that
    cannot be made locally verifiable is explicitly labelled source-only rather
    than being published with checksums that describe different bytes.
    """
    inventory_by_path = {str(item["published_path"]): item for item in inventory}
    for manifest in sorted(evidence.rglob("SHA256SUMS.txt"), key=lambda path: path.as_posix()):
        original = manifest.read_text(encoding="utf-8").splitlines()
        parsed: list[tuple[str, Path]] = []
        reason: str | None = None
        for line in original:
            fields = line.split("  ", 1)
            if len(fields) != 2 or len(fields[0]) != 64:
                reason = "malformed checksum entry"
                break
            name = fields[1]
            relative = PurePosixPath(name)
            if (
                relative.is_absolute()
                or not relative.parts
                or ".." in relative.parts
                or "\\" in name
            ):
                reason = "unsafe checksum path"
                break
            target = (manifest.parent / Path(*relative.parts)).resolve()
            parent = manifest.parent.resolve()
            if target == manifest.resolve() or parent not in target.parents or not target.is_file():
                reason = "checksum target is not present in the published directory"
                break
            parsed.append((name, target))
        if reason is None:
            manifest.write_text(
                "".join(f"{sha256(target)}  {name}\n" for name, target in parsed),
                encoding="utf-8",
            )
            mode = "REHASHED_PUBLISHED_BYTES"
        else:
            manifest.write_text(
                "# SOURCE_ONLY_MANIFEST: not locally verifiable after publication filtering; "
                f"{reason}. Original source bytes are bound by SOURCE_INVENTORY.json.\n"
                + "\n".join(f"# {line}" for line in original)
                + "\n",
                encoding="utf-8",
            )
            mode = "SOURCE_ONLY_NOT_LOCALLY_VERIFIABLE"
        relative_manifest = manifest.relative_to(evidence.parent).as_posix()
        item = inventory_by_path[relative_manifest]
        item["published_sha256"] = sha256(manifest)
        item["sanitized"] = item["source_sha256"] != item["published_sha256"]
        item["bytes"] = manifest.stat().st_size
        item["nested_checksum_manifest_mode"] = mode


def _allowlisted_sources(artifacts: Path, branches: list[Branch]) -> list[Path]:
    complete = {branch.name: branch.complete for branch in branches}
    roots = [
        artifacts / name
        for name in (
            "official_contract",
            "r1_resume",
            "root_qa",
            "fcwd_runner_qa",
            "prefetch",
            "full_prefetch",
        )
    ]
    if complete["scoring_existing_946"]:
        roots.append(artifacts / "scoring_existing")
    if complete["garl_parity"]:
        roots.append(artifacts / "garl_parity")
    if complete["fcwd_assets"]:
        roots.append(artifacts / "fcwd")
    if complete["baseline_failure_inventory"]:
        roots.append(artifacts / "baselines")
    if complete["expanded_metrics"]:
        roots.extend(
            [
                artifacts / "expanded_metrics",
                artifacts / "dev32_expanded_rgb",
            ]
        )
    if complete["fcwd_metrics"]:
        roots.append(artifacts / "fcwd_inference")
    if complete["system_cost"]:
        roots.append(artifacts / "cost")
    singletons = [
        artifacts / "ACCOUNTING.json",
        artifacts / "R1_STATUS.json",
        artifacts / "R1_STATUS_BEFORE_PAUSE.json",
        artifacts / "R1_PAUSE_OWNERSHIP.json",
        artifacts / "CAMPAIGN_FREEZE.json",
        artifacts / "CAMPAIGN_RESULT.json",
        artifacts / "PIPELINE_STATE.json",
        artifacts / "dev32_expanded" / "POPULATION_FREEZE.json",
        artifacts / "dev32_expanded" / "INFERENCE_FREEZE.json",
        artifacts / "dev32_expanded" / "CALIBRATION_AUDIT.json",
        artifacts / "dev32_expanded" / "STATE.json",
        artifacts / "dev32_expanded" / "PREFETCH_PRESERVATION_VERIFIED.json",
        artifacts / "dev32_expanded" / "REUSE.json",
        artifacts / "fcwd_population" / "FCWD_INPUT_FREEZE.json",
    ]
    found: set[Path] = {path for path in singletons if path.is_file()}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if (
                path.is_file()
                and path.name not in EXCLUDED_NAMES
                and path.stat().st_size <= MAX_PUBLIC_FILE_BYTES
                and not (
                    "predictions" in path.relative_to(root).parts and path.name.startswith("query_")
                )
                and "checkpoint" not in path.name.lower()
                and path.suffix.lower() not in {".pt", ".pth", ".ckpt", ".h5", ".hdf5"}
            ):
                found.add(path)
    return sorted(found, key=lambda path: path.as_posix())


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _table(branches: list[Branch], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("branch", "status", "complete", "detail"))
        for branch in branches:
            writer.writerow(
                (branch.name, branch.status, str(branch.complete).lower(), branch.detail)
            )


METRIC_COLUMNS = (
    "branch",
    "method",
    "total_rows",
    "gt_eligible",
    "coverage_on_eligible_gt",
    "failure_rate_on_eligible_gt",
    "status",
    "complete_cohort_metrics__rte_percent",
    "complete_cohort_metrics__mae_seconds",
    "complete_cohort_metrics__median_absolute_error_seconds",
    "complete_cohort_metrics__rmse_seconds",
    "conditional_on_scorable_predictions_metrics__rte_percent",
    "conditional_on_scorable_predictions_metrics__mae_seconds",
    "conditional_on_scorable_predictions_metrics__median_absolute_error_seconds",
    "conditional_on_scorable_predictions_metrics__rmse_seconds",
)
COST_COLUMNS = (
    "system",
    "stage",
    "count",
    "mean_ms",
    "p50_ms",
    "p95_ms",
    "minimum_ms",
    "maximum_ms",
    "samples_per_second_from_mean",
)
BASELINE_COVERAGE_COLUMNS = (
    "variant",
    "equation",
    "requested",
    "successful",
    "failed_retained",
    "success_fraction",
    "interpretation",
)
PILOT_CONDITIONAL_COLUMNS = (
    "method",
    "cohort_queries",
    "truth_exposed_queries",
    "prediction_coverage_all_queries",
    "prediction_coverage_on_exposed_truth",
    "complete_exposed_cohort_status",
    "conditional_support",
    "conditional_mae_seconds",
    "conditional_rmse_seconds",
    "conditional_median_absolute_error_seconds",
    "conditional_signed_bias_seconds",
    "conditional_metrics_not_rankable",
)


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _result_rows(
    artifacts: Path, branches: list[Branch]
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    complete = {branch.name: branch.complete for branch in branches}
    metric_sources: list[tuple[str, Path]] = []
    if complete["scoring_existing_946"]:
        metric_sources.append(("scoring_existing_946", artifacts / "scoring_existing"))
    if complete["expanded_metrics"]:
        metric_sources.append(("expanded_metrics", artifacts / "expanded_metrics"))
    if complete["fcwd_metrics"]:
        metric_sources.append(("fcwd_metrics", artifacts / "fcwd_inference" / "scoring"))
    metrics: list[dict[str, object]] = []
    for branch, directory in metric_sources:
        for source in _csv_rows(directory / "METHOD_METRICS.csv"):
            metrics.append(
                {
                    column: branch if column == "branch" else source.get(column, "")
                    for column in METRIC_COLUMNS
                }
            )

    costs: list[dict[str, object]] = []
    if complete["system_cost"]:
        summary = _json(artifacts / "cost" / "SYSTEM_COST_SUMMARY.json")
        for system, stages in sorted(summary["metrics"].items()):
            for stage in ("cpu_prepare", "gpu_inference", "sequential_end_to_end"):
                values = stages.get(stage, {})
                costs.append(
                    {
                        column: system
                        if column == "system"
                        else stage
                        if column == "stage"
                        else values.get(column, "")
                        for column in COST_COLUMNS
                    }
                )

    failures: list[dict[str, object]] = []
    if complete["baseline_failure_inventory"]:
        baseline = _json(artifacts / "baselines" / "FAILURE_COUNTS.json")
        for method, reasons in sorted(baseline.get("failure_counts", {}).items()):
            for reason, count in sorted(reasons.items()):
                failures.append({"method": method, "reason": reason, "count": count})
    baseline_coverage: list[dict[str, object]] = []
    baseline_conditional: list[dict[str, object]] = []
    if complete["baseline_failure_inventory"]:
        diagnosis = _json(artifacts / "baselines" / "CMAX_REFERENCE_DIAGNOSIS.json")
        old, corrected = diagnosis["old_variant"], diagnosis["corrected_variant"]
        baseline_coverage = [
            {
                "variant": "initial_local_cmax",
                "equation": old["equation"],
                "requested": old["requested"],
                "successful": old["successful"],
                "failed_retained": old["requested"] - old["successful"],
                "success_fraction": old["successful"] / old["requested"],
                "interpretation": (
                    "Incorrect exponential/latest-event warp in the initial local adapter; "
                    "not a failure of the published CMax method"
                ),
            },
            {
                "variant": "affine_corrected_cmax_reference",
                "equation": corrected["equation"],
                "requested": corrected["requested"],
                "successful": corrected["successful"],
                "failed_retained": corrected["failed_retained"],
                "success_fraction": corrected["success_fraction"],
                "interpretation": (
                    "Corrected affine reference-time local adaptation; diagnostic, not a "
                    "paper-result reproduction"
                ),
            },
        ]
        pilot = _csv_rows(artifacts / "baselines" / "pilot_scored" / "METHOD_METRICS.csv")
        baseline_conditional = [
            {column: row.get(column, "") for column in PILOT_CONDITIONAL_COLUMNS}
            for row in pilot
            if row.get("method") in {"cmax_reference", "strttc_adapted"}
        ]
    return metrics, costs, failures, baseline_coverage, baseline_conditional


def _dict_csv(path: Path, rows: list[dict[str, object]], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _readme(
    branches: list[Branch],
    metrics: list[dict[str, object]],
    costs: list[dict[str, object]],
    failures: list[dict[str, object]],
    baseline_coverage: list[dict[str, object]],
    baseline_conditional: list[dict[str, object]],
) -> str:
    lines = [
        "# Campaña TTC 2026-10-08 — evidencia reproducible",
        "",
        "Este directorio es una vista sanitizada y regenerable de artefactos medidos. No "
        "afirma estado del arte ni convierte ramas en ejecución en resultados.",
        "",
        "## Estado de las ramas",
        "",
        "| Rama | Estado | Lectura |",
        "|---|---|---|",
    ]
    for branch in branches:
        lines.append(f"| `{branch.name}` | `{branch.status}` | {branch.detail} |")
    lines += [
        "",
        "## Alcance científico",
        "",
        "`scoring_existing_946` es exploratorio y procede de un conjunto histórico de 946 "
        "targets elegibles. Los resultados expandidos sólo se publican cuando su manifest "
        "de hashes verifica los seis productos del scorer. Los intervalos bootstrap son "
        "descriptivos, por secuencia o grupo completo, sin corrección por multiplicidad. "
        "Las métricas sobre predicciones puntuables sí se incluyen, etiquetadas como "
        "condicionales y acompañadas por cobertura y tasa de fallo; no sustituyen la "
        "cohorte completa.",
        "",
        "FCWD contiene tres secuencias públicas: sirve como transferencia pequeña y no "
        "certifica ascendencia independiente. Los assets están completos, pero eso no hace "
        "usable la calibración full: el MAT es MCOS, las cajas son 1280×720 de eventos y "
        "RGB es 1920×1200 sin transformación verificada. Event-only tiene 630 consultas "
        "viables; full está bloqueado por el mapping de entrada, no por un resultado negativo.",
        "La población FCWD de 630 consultas sólo cubre esas tres secuencias: sus intervalos "
        "por clúster tienen poca independencia. La expansión dev32 añade timestamps a las "
        "32 secuencias ya observadas; no constituye un holdout nuevo.",
        "",
        "H8 usa sólo eventos durante esta inferencia, pero su entrenamiento no fue SSL JEPA "
        "puro: empleó teacher RGB DINO y supervisión geométrica. Las semillas 7/13/23 son "
        "tres cabezas H8 sobre productores compartidos, no tres réplicas independientes de "
        "los productores. Ningún error Garl se filtra por rendimiento; fallos y cobertura "
        "permanecen en la cohorte y las vistas puntuables se etiquetan condicionales.",
        "H8 limita su salida nativa a ±60 s; Garl conserva la división nativa sin clipping. "
        "Cuando sus dos alturas predichas son casi iguales, Garl puede emitir TTC finitos "
        "extremos. Por eso se muestran conjuntamente MAE y mediana, sin eliminar filas. "
        "La mediana no tiene intervalo bootstrap en este análisis; una mediana menor no "
        "se describe como una diferencia estadísticamente demostrada. El diagnóstico "
        "de extremos y cualquier sensibilidad post hoc se conservan separadamente en "
        "`evidence/root_qa/full_outlier_audit/`; no sustituyen las métricas nativas.",
        "",
        "El inventario de baselines de desarrollo retiene fallos. Las métricas del subconjunto "
        "puntuable se publican sólo como condicionales, con cobertura explícita. Los costes "
        "son observaciones del host sobre ocho "
        "consultas, con caché caliente y contextos distintos; no demuestran causalidad de "
        "arquitectura ni son comparables directamente con latencias publicadas. El protocolo "
        "mide batch 1 en float32 sin TF32 y separa preparación, GPU y E2E. H8 y EO "
        "se miden con un único worker GPU del proyecto en el escritorio Windows WDDM; "
        "los clientes gráficos ambientales y la telemetría quedan registrados. No se "
        "afirma exclusividad física ni ausencia de contención del escritorio. H8 y EO "
        "comparten la preparación cruda; la preparación/E2E de EO es conservadora porque "
        "también construye history8. El piloto de tres consultas no prueba un 2,59× sostenido.",
        "",
        "La paridad Garl separa reproducción nativa de adaptaciones de transferencia EvTTC. "
        "La Tabla VI publicada no comparte un protocolo demostrado idéntico con nuestro RTE, "
        "por lo que no se presenta como comparación directa ni como reproducción oficial. "
        "El contrato oficial conserva dependencias bloqueadas y `AUTHORS_DRAFT.txt` es un "
        "borrador no enviado. El presupuesto restante no admite el plan inicial de nuevo "
        "entrenamiento; esta campaña registra cero updates. R1 registra 486 pares "
        "preexistentes preservados en un "
        "límite de grupo y registra cero avances de prefijo. Los grupos guardados no "
        "releen payloads, aunque CachedEventReader sí abre handles; cachés OS/HDF5 no están "
        "controladas y el timing final puede mezclar tramos antiguos y nuevos. No se afirma "
        "timing bit-exact ni una medición cold/warm ininterrumpida. Se informa el estado "
        "literal del recibo vigente; R1 y el coste del sistema TRAIN40 son mediciones distintas.",
        "",
        "Fuentes primarias: [EvTTC](https://nail-hnu.github.io/EventAidedTTC/), "
        "[Garl-TTC](https://github.com/NAIL-HNU/Garl-TTC) y "
        "[FCWD](https://nail-hnu.github.io/EventAidedTTC/).",
        "",
        "Ejecute `python regenerate.py --verify` desde este directorio para verificar hashes "
        "y reproducir exactamente las tablas desde la evidencia publicada. Esta verificación "
        "comprueba `SHA256SUMS.txt`; el `.sha256` del ZIP es el ancla externa sólo si se "
        "conserva u obtiene por separado. Ningún manifiesto autocontenido protege frente a la "
        "sustitución coordinada del snapshot y de todos sus hashes.",
    ]
    if metrics:
        lines += [
            "",
            "## Resultados medidos",
            "",
            "| Rama | Método | Población | GT elegibles | Cobertura | "
            "RTE cohorte completa (%) | MAE (s) | Mediana AE (s) | "
            "RTE condicional (%) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in metrics:
            lines.append(
                (
                    "| {branch} | {method} | {total} | {eligible} | {coverage} | "
                    "{rte} | {mae} | {median} | {conditional} |"
                ).format(
                    branch=row["branch"],
                    method=row["method"],
                    total=row["total_rows"],
                    eligible=row["gt_eligible"],
                    coverage=row["coverage_on_eligible_gt"] or "n/a",
                    rte=row["complete_cohort_metrics__rte_percent"] or "n/a",
                    mae=row["complete_cohort_metrics__mae_seconds"] or "n/a",
                    median=row["complete_cohort_metrics__median_absolute_error_seconds"] or "n/a",
                    conditional=row["conditional_on_scorable_predictions_metrics__rte_percent"]
                    or "n/a",
                )
            )
        lines += [
            "",
            "Cobertura se refiere exclusivamente a GT elegibles. Las filas sin GT válido "
            "permanecen en la población y su tratamiento sigue la regla predeclarada. "
            "Para la rama `expanded_metrics`, los contrastes pareados y sus intervalos por "
            "secuencia/grupo completos están en `evidence/expanded_metrics/PAIRED.csv`; no "
            "cubren las filas históricas ni FCWD de esta tabla. Describen esos cinco sistemas "
            "con sus distintos contratos de salida y no una superioridad intrínseca de "
            "arquitectura.",
        ]
    if costs:
        lines += [
            "",
            "## Coste observado",
            "",
            "| Sistema | Etapa | Media (ms) | p50 (ms) | p95 (ms) |",
            "|---|---|---:|---:|---:|",
        ]
        for row in costs:
            lines.append(
                f"| {row['system']} | {row['stage']} | {row['mean_ms']} | "
                f"{row['p50_ms']} | {row['p95_ms']} |"
            )
    if failures:
        lines += [
            "",
            "## Fallos de baselines en desarrollo-32",
            "",
            "Se conservan todos los fallos. Las métricas condicionales, cuando existen, no "
            "se presentan como rendimiento de cohorte completa.",
            "",
            "| Método | Resultado o causa | Conteo |",
            "|---|---|---:|",
        ]
        lines.extend(f"| {row['method']} | {row['reason']} | {row['count']} |" for row in failures)
    if baseline_coverage:
        lines += [
            "",
            "## Diagnóstico CMax",
            "",
            "El adaptador CMax local inicial obtuvo 0/32 porque usaba un warp exponencial "
            "referido al último evento. Esa formulación era incorrecta para la expansión afín "
            "de inverse-TTC. La adaptación corregida usa tiempo de referencia fijo y obtuvo "
            "15/32. Ambos son diagnósticos locales; no son fallos ni reproducciones del método "
            "publicado.",
            "",
            "| Variante | Éxitos | Solicitadas | Interpretación |",
            "|---|---:|---:|---|",
        ]
        lines.extend(
            f"| {row['variant']} | {row['successful']} | {row['requested']} | "
            f"{row['interpretation']} |"
            for row in baseline_coverage
        )
    if baseline_conditional:
        lines += [
            "",
            "## Piloto condicional de baselines geométricos",
            "",
            "Los soportes se muestran en la tabla. Son cohortes de éxito distintas: "
            "las cifras condicionales describen los "
            "casos puntuables y no forman un ranking entre métodos.",
            "",
            "| Método | Soporte | MAE condicional (s) | RMSE condicional (s) | No rankeable |",
            "|---|---:|---:|---:|---|",
        ]
        lines.extend(
            f"| {row['method']} | {row['conditional_support']} | "
            f"{row['conditional_mae_seconds']} | {row['conditional_rmse_seconds']} | "
            f"{row['conditional_metrics_not_rankable']} |"
            for row in baseline_conditional
        )
    return "\n".join(lines) + "\n"


REGENERATE = r'''"""Verify evidence, rerun every published scorer, and rebuild all tables."""
from __future__ import annotations
import argparse, csv, hashlib, json, subprocess, sys, tempfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
SCORE_OUTPUTS = ("METADATA.json", "METHOD_METRICS.csv", "PAIRED.csv",
                 "PER_SEQUENCE.csv", "REPORT.json", "SCORED_ROWS.csv", "SHA256.json")
METRIC_COLUMNS = ("branch", "method", "total_rows", "gt_eligible",
    "coverage_on_eligible_gt", "failure_rate_on_eligible_gt", "status",
    "complete_cohort_metrics__rte_percent", "complete_cohort_metrics__mae_seconds",
    "complete_cohort_metrics__median_absolute_error_seconds",
    "complete_cohort_metrics__rmse_seconds",
    "conditional_on_scorable_predictions_metrics__rte_percent",
    "conditional_on_scorable_predictions_metrics__mae_seconds",
    "conditional_on_scorable_predictions_metrics__median_absolute_error_seconds",
    "conditional_on_scorable_predictions_metrics__rmse_seconds")
COST_COLUMNS = ("system", "stage", "count", "mean_ms", "p50_ms", "p95_ms",
                "minimum_ms", "maximum_ms", "samples_per_second_from_mean")
BASELINE_COVERAGE_COLUMNS = ("variant", "equation", "requested", "successful",
    "failed_retained", "success_fraction", "interpretation")
PILOT_CONDITIONAL_COLUMNS = ("method", "cohort_queries", "truth_exposed_queries",
    "prediction_coverage_all_queries", "prediction_coverage_on_exposed_truth",
    "complete_exposed_cohort_status", "conditional_support", "conditional_mae_seconds",
    "conditional_rmse_seconds", "conditional_median_absolute_error_seconds",
    "conditional_signed_bias_seconds", "conditional_metrics_not_rankable")

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def rows(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]

def write_csv(path: Path, values, columns) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader(); writer.writerows(values)

def verify_snapshot() -> None:
    manifest = ROOT / "SHA256SUMS.txt"
    if not manifest.is_file():
        raise SystemExit("missing top-level SHA256SUMS.txt")
    seen = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise SystemExit("malformed top-level SHA256SUMS.txt")
        expected, name = fields
        relative = PurePosixPath(name)
        if (relative.is_absolute() or not relative.parts or ".." in relative.parts
                or "\\" in name or name in seen):
            raise SystemExit(f"unsafe or duplicate snapshot path: {name}")
        seen.add(name)
        path = (ROOT / Path(*relative.parts)).resolve()
        if ROOT.resolve() not in path.parents or not path.is_file() or digest(path) != expected:
            raise SystemExit(f"snapshot hash mismatch: {name}")

def rerun(job, scorer: Path) -> None:
    reference, input_csv = ROOT / job["reference"], ROOT / job["input"]
    config = read_json(reference / "METADATA.json")["configuration"]
    with tempfile.TemporaryDirectory(prefix="sota_scoring_verify_") as temporary:
        command = [sys.executable, str(scorer), "--input", str(input_csv),
                   "--output-dir", temporary, "--models", *config["methods"],
                   "--bootstrap-draws", str(config["bootstrap_draws"]),
                   "--seed", str(config["seed"])]
        if config.get("group_column"):
            command += ["--group-column", config["group_column"]]
        run = subprocess.run(command, check=False, capture_output=True, text=True)
        if run.returncode:
            raise SystemExit(f"scoring regeneration failed for {job['branch']}: {run.stderr}")
        for name in SCORE_OUTPUTS:
            if digest(Path(temporary) / name) != digest(reference / name):
                raise SystemExit(f"scoring regeneration differs for {job['branch']}: {name}")

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    verify_snapshot()
    inventory = read_json(ROOT / "SOURCE_INVENTORY.json")
    for item in inventory["files"]:
        path = ROOT / item["published_path"]
        if not path.is_file() or digest(path) != item["published_sha256"]:
            raise SystemExit(f"hash mismatch: {path}")
    regeneration = read_json(ROOT / "tables" / "REGENERATION.json")
    scorer = ROOT / regeneration["scorer"]
    for job in regeneration["scoring_jobs"]:
        rerun(job, scorer)
    temporary_tables = tempfile.TemporaryDirectory(prefix="sota_tables_verify_")
    generated = Path(temporary_tables.name)
    state = read_json(ROOT / "NEXT_DECISION.json")
    branch_rows = state["branches"]
    target = generated / "branch_status.csv"
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(("branch", "status", "complete", "detail"))
        for row in branch_rows:
            writer.writerow((row["name"], row["status"],
                             str(row["complete"]).lower(), row["detail"]))
    metric_rows = []
    for job in regeneration["scoring_jobs"]:
        for source in rows(ROOT / job["reference"] / "METHOD_METRICS.csv"):
            metric_rows.append({
                column: job["branch"] if column == "branch" else source.get(column, "")
                for column in METRIC_COLUMNS})
    write_csv(generated / "metrics.csv", metric_rows, METRIC_COLUMNS)
    cost_rows = []
    cost_path = ROOT / "evidence" / "cost" / "SYSTEM_COST_SUMMARY.json"
    if cost_path.is_file():
        for system, stages in read_json(cost_path)["metrics"].items():
            for stage in ("cpu_prepare", "gpu_inference", "sequential_end_to_end"):
                values = stages.get(stage, {})
                cost_rows.append({
                    column: system if column == "system" else
                    stage if column == "stage" else values.get(column, "")
                    for column in COST_COLUMNS})
    write_csv(generated / "system_cost.csv", cost_rows, COST_COLUMNS)
    failure_rows = []
    failure_path = ROOT / "evidence" / "baselines" / "FAILURE_COUNTS.json"
    if failure_path.is_file():
        failure_source = read_json(failure_path)
        for method, reasons in failure_source.get("failure_counts", {}).items():
            for reason, count in reasons.items():
                failure_rows.append({"method": method, "reason": reason, "count": count})
    write_csv(generated / "baseline_failures.csv", failure_rows,
              ("method", "reason", "count"))
    coverage_rows, conditional_rows = [], []
    diagnosis_path = ROOT / "evidence" / "baselines" / "CMAX_REFERENCE_DIAGNOSIS.json"
    pilot_path = ROOT / "evidence" / "baselines" / "pilot_scored" / "METHOD_METRICS.csv"
    if diagnosis_path.is_file():
        diagnosis = read_json(diagnosis_path)
        old, corrected = diagnosis["old_variant"], diagnosis["corrected_variant"]
        coverage_rows = [{
            "variant": "initial_local_cmax", "equation": old["equation"],
            "requested": old["requested"], "successful": old["successful"],
            "failed_retained": old["requested"] - old["successful"],
            "success_fraction": old["successful"] / old["requested"],
            "interpretation": "Incorrect exponential/latest-event warp in the initial local "
                              "adapter; not a failure of the published CMax method"}, {
            "variant": "affine_corrected_cmax_reference",
            "equation": corrected["equation"], "requested": corrected["requested"],
            "successful": corrected["successful"],
            "failed_retained": corrected["failed_retained"],
            "success_fraction": corrected["success_fraction"],
            "interpretation": "Corrected affine reference-time local adaptation; diagnostic, "
                              "not a paper-result reproduction"}]
    if pilot_path.is_file():
        conditional_rows = [
            {column: row.get(column, "") for column in PILOT_CONDITIONAL_COLUMNS}
            for row in rows(pilot_path)
            if row.get("method") in {"cmax_reference", "strttc_adapted"}]
    write_csv(generated / "baseline_coverage.csv", coverage_rows,
              BASELINE_COVERAGE_COLUMNS)
    write_csv(generated / "baseline_conditional_metrics.csv", conditional_rows,
              PILOT_CONDITIONAL_COLUMNS)
    results = {"metrics": metric_rows, "system_cost": cost_rows,
               "baseline_failures": failure_rows, "baseline_coverage": coverage_rows,
               "baseline_conditional_metrics": conditional_rows}
    (generated / "RESULTS.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name, expected in state["generated_tables"].items():
        if name == "REGENERATION.json":
            candidate = ROOT / "tables" / name
        else:
            candidate = generated / name
        if not candidate.is_file() or digest(candidate) != expected:
            raise SystemExit(f"regenerated table differs: {name}")
        published = ROOT / "tables" / name
        if not published.is_file() or digest(published) != expected:
            raise SystemExit(f"published table hash mismatch: {name}")
    temporary_tables.cleanup()
    if args.verify:
        print(f"verified {len(inventory['files'])} evidence files and "
              f"{len(regeneration['scoring_jobs'])} exact scoring runs")

if __name__ == "__main__":
    main()
'''


def publish(
    artifacts: Path,
    output: Path,
    bundle: Path,
    *,
    require_complete: bool,
    repo: Path | None = None,
) -> dict[str, Any]:
    """Create one atomic documentation snapshot and an essential evidence ZIP."""
    artifacts = artifacts.resolve()
    output = output.resolve()
    bundle = bundle.resolve()
    repo = (repo or Path.cwd()).resolve()
    docs_root = (repo / "docs").resolve()
    artifacts_root = (repo / "artifacts").resolve()
    sidecar = bundle.with_suffix(bundle.suffix + ".sha256").resolve()
    if docs_root not in output.parents or output in {repo, artifacts} or not output.name:
        raise ValueError(f"Unsafe publication output: {output}")
    if artifacts_root not in bundle.parents or output == bundle or output in bundle.parents:
        raise ValueError(f"Unsafe publication bundle: {bundle}")
    if output == sidecar or output in sidecar.parents or sidecar == bundle:
        raise ValueError(f"Unsafe publication sidecar: {sidecar}")
    branches = inspect_branches(artifacts)
    required = {
        "scoring_existing_946",
        "expanded_metrics",
        "fcwd_assets",
        "fcwd_metrics",
        "garl_parity",
        "system_cost",
        "baseline_failure_inventory",
    }
    blockers = [b.name for b in branches if b.name in required and not b.complete]
    branch_complete = {branch.name: branch.complete for branch in branches}
    if require_complete and blockers:
        raise RuntimeError(f"Publication incomplete; blocked branches: {', '.join(blockers)}")

    roots = _data_roots(artifacts)
    sources = _allowlisted_sources(artifacts, branches)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sota_publication_", dir=output.parent) as temporary:
        stage = Path(temporary) / "publication"
        evidence = stage / "evidence"
        inventory: list[dict[str, Any]] = []
        publications = [(source, source.relative_to(artifacts)) for source in sources]
        historical_metadata = artifacts / "scoring_existing" / "METADATA.json"
        if branch_complete["scoring_existing_946"] and historical_metadata.is_file():
            input_hash = _json(historical_metadata).get("input_sha256")
            for candidate in artifacts.parent.rglob("SCORED_PREDICTIONS.csv"):
                if isinstance(input_hash, str) and sha256(candidate) == input_hash:
                    publications.append(
                        (candidate, Path("scoring_existing/input/SCORED_PREDICTIONS.csv"))
                    )
                    break
        for script_name in (
            "publication.py",
            "scoring.py",
            "campaign.py",
            "followups.py",
            "fcwd_run.py",
            "fcwd_score.py",
            "cost.py",
        ):
            script = repo / "operational" / "sota_eval" / script_name
            if script.is_file():
                publications.append((script, Path("scripts") / script_name))
        destinations: set[Path] = set()
        for source, relative in publications:
            if relative in destinations:
                continue
            destinations.add(relative)
            destination = evidence / relative
            _copy_sanitized(source, destination, repo, roots)
            try:
                source_label = "${REPO}/" + source.relative_to(repo).as_posix()
            except ValueError:
                source_label = source.name
            inventory.append(
                {
                    "source_path": source_label,
                    "source_sha256": sha256(source),
                    "published_path": destination.relative_to(stage).as_posix(),
                    "published_sha256": sha256(destination),
                    "sanitized": sha256(source) != sha256(destination),
                    "bytes": destination.stat().st_size,
                }
            )
        _refresh_published_checksum_manifests(evidence, inventory)
        (
            metric_rows,
            cost_rows,
            failure_rows,
            baseline_coverage_rows,
            baseline_conditional_rows,
        ) = _result_rows(artifacts, branches)
        scoring_jobs = []
        if branch_complete["scoring_existing_946"]:
            scoring_jobs.append(
                {
                    "branch": "scoring_existing_946",
                    "reference": "evidence/scoring_existing",
                    "input": "evidence/scoring_existing/input/SCORED_PREDICTIONS.csv",
                }
            )
        if branch_complete["expanded_metrics"]:
            scoring_jobs.append(
                {
                    "branch": "expanded_metrics",
                    "reference": "evidence/expanded_metrics",
                    "input": "evidence/dev32_expanded_rgb/SCORED_PREDICTIONS.csv",
                }
            )
        if branch_complete["fcwd_metrics"]:
            scoring_jobs.append(
                {
                    "branch": "fcwd_metrics",
                    "reference": "evidence/fcwd_inference/scoring",
                    "input": "evidence/fcwd_inference/scoring/SCORED_PREDICTIONS.csv",
                }
            )
        for job in scoring_jobs:
            if not (stage / job["input"]).is_file():
                raise RuntimeError(f"Missing published scorer input for {job['branch']}")
        _write_json(
            stage / "tables" / "REGENERATION.json",
            {"scorer": "evidence/scripts/scoring.py", "scoring_jobs": scoring_jobs},
        )
        _table(branches, stage / "tables" / "branch_status.csv")
        _dict_csv(stage / "tables" / "metrics.csv", metric_rows, METRIC_COLUMNS)
        _dict_csv(stage / "tables" / "system_cost.csv", cost_rows, COST_COLUMNS)
        _dict_csv(
            stage / "tables" / "baseline_failures.csv",
            failure_rows,
            ("method", "reason", "count"),
        )
        _dict_csv(
            stage / "tables" / "baseline_coverage.csv",
            baseline_coverage_rows,
            BASELINE_COVERAGE_COLUMNS,
        )
        _dict_csv(
            stage / "tables" / "baseline_conditional_metrics.csv",
            baseline_conditional_rows,
            PILOT_CONDITIONAL_COLUMNS,
        )
        _write_json(
            stage / "tables" / "RESULTS.json",
            {
                "metrics": metric_rows,
                "system_cost": cost_rows,
                "baseline_failures": failure_rows,
                "baseline_coverage": baseline_coverage_rows,
                "baseline_conditional_metrics": baseline_conditional_rows,
            },
        )
        table_hash = sha256(stage / "tables" / "branch_status.csv")
        branch_rows = []
        for branch in branches:
            row = dict(branch.__dict__)
            if branch.path is not None:
                row["path"] = _replace_roots(branch.path, repo, roots)
            branch_rows.append(row)
        next_actions = []
        if not branch_complete["expanded_metrics"]:
            next_actions.append(
                "Completar y sellar dev32_expanded, dev32_expanded_rgb, scoring y "
                "expanded_metrics bajo CAMPAIGN_RESULT.json canónico."
            )
        if not branch_complete["fcwd_metrics"]:
            next_actions.append(
                "Ejecutar y sellar las 630 consultas FCWD event-only y su scoring; mantener "
                "full como DEPENDENCY_UNAVAILABLE hasta resolver event→RGB."
            )
        if not branch_complete["system_cost"]:
            next_actions.append(
                "Ejecutar el protocolo de coste fixed-8 con un único worker GPU del proyecto "
                "y verificar sus ocho "
                "fragments RAW contra ambos freezes."
            )
        if not branch_complete["official_contract"]:
            next_actions.append(
                "Resolver con los autores el protocolo Table VI, ascendencia, mapping FCWD, "
                "calibración portable e innercar_bbox.csv; los borradores siguen sin enviar."
            )
        if not branch_complete["r1_measurement"]:
            next_actions.append(
                "Continuar R1 desde los pares preservados del recibo vigente, sin otro "
                "trabajo GPU del proyecto, después de accuracy/cost; "
                "reportar el timing interrumpido con su caveat de caché."
            )
        state = {
            "schema": "sota_campaign_publication_v1",
            "status": "COMPLETE" if not blockers else "PARTIAL_BLOCKED",
            "claim": "No SOTA claim; measured evidence only",
            "branches": branch_rows,
            "blocked_branches": blockers,
            "blocked_or_pending_branches": [
                branch.name for branch in branches if not branch.complete
            ],
            "next_actions": next_actions
            or ["Verificar el checkout limpio y publicar el snapshot completo."],
            "generated_table_sha256": table_hash,
            "generated_tables": {
                name: sha256(stage / "tables" / name)
                for name in (
                    "branch_status.csv",
                    "metrics.csv",
                    "system_cost.csv",
                    "baseline_failures.csv",
                    "baseline_coverage.csv",
                    "baseline_conditional_metrics.csv",
                    "RESULTS.json",
                    "REGENERATION.json",
                )
            },
        }
        _write_json(stage / "NEXT_DECISION.json", state)
        (stage / "README.md").write_text(
            _readme(
                branches,
                metric_rows,
                cost_rows,
                failure_rows,
                baseline_coverage_rows,
                baseline_conditional_rows,
            ),
            encoding="utf-8",
        )
        (stage / "regenerate.py").write_text(REGENERATE, encoding="utf-8")
        _write_json(
            stage / "SOURCE_INVENTORY.json",
            {"schema": "sanitized_source_inventory_v1", "files": inventory},
        )
        # Git must retain the physical bytes covered by SHA-256, including CSV CRLF.
        (stage / ".gitattributes").write_bytes(
            b"* -text whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol\n"
        )
        sums = []
        for path in sorted(stage.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_file() and path.name != "SHA256SUMS.txt":
                sums.append(f"{sha256(path)}  {path.relative_to(stage).as_posix()}")
        (stage / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")

        output.mkdir(parents=True, exist_ok=True)
        backup = Path(temporary) / "backup"
        backup.mkdir()
        installed: list[Path] = []
        try:
            for name in sorted(OWNED_OUTPUT_NAMES):
                child = (output / name).resolve()
                if child.parent != output:
                    raise ValueError(f"Unsafe publication child: {child}")
                if child.exists():
                    shutil.move(str(child), backup / name)
            for child in stage.iterdir():
                destination = output / child.name
                shutil.move(str(child), destination)
                installed.append(destination)
        except Exception:
            for child in reversed(installed):
                if child.is_dir():
                    shutil.rmtree(child)
                elif child.exists():
                    child.unlink()
            for child in backup.iterdir():
                shutil.move(str(child), output / child.name)
            raise

    bundle.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="sota_bundle_", dir=bundle.parent) as bundle_temp:
        transaction = Path(bundle_temp)
        temporary_bundle = transaction / bundle.name
        temporary_sidecar = transaction / sidecar.name
        with zipfile.ZipFile(temporary_bundle, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            bundle_paths: list[Path] = []
            for name in sorted(OWNED_OUTPUT_NAMES):
                owned = output / name
                if owned.is_file():
                    bundle_paths.append(owned)
                elif owned.is_dir():
                    bundle_paths.extend(path for path in owned.rglob("*") if path.is_file())
            for path in sorted(bundle_paths, key=lambda item: item.as_posix()):
                archive.write(path, path.relative_to(output).as_posix())
            for source in sorted(
                [
                    repo / "operational" / "sota_eval" / "publication.py",
                    repo / "operational" / "sota_eval" / "scoring.py",
                ],
                key=str,
            ):
                if source.is_file():
                    archive.write(source, f"scripts/{source.name}")
        digest = sha256(temporary_bundle)
        temporary_sidecar.write_text(f"{digest}  {bundle.name}\n", encoding="ascii")
        old_bundle, old_sidecar = transaction / "old.zip", transaction / "old.sha256"
        bundle_installed = sidecar_installed = False
        try:
            if bundle.exists():
                os.replace(bundle, old_bundle)
            if sidecar.exists():
                os.replace(sidecar, old_sidecar)
            os.replace(temporary_bundle, bundle)
            bundle_installed = True
            os.replace(temporary_sidecar, sidecar)
            sidecar_installed = True
        except Exception:
            if sidecar_installed and sidecar.exists():
                sidecar.unlink()
            if bundle_installed and bundle.exists():
                bundle.unlink()
            if old_bundle.exists():
                os.replace(old_bundle, bundle)
            if old_sidecar.exists():
                os.replace(old_sidecar, sidecar)
            raise
    return {"status": state["status"], "blocked_branches": blockers, "bundle_sha256": digest}


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/sota_campaign_20261008"))
    parser.add_argument("--output", type=Path, default=Path("docs/sota_campaign_20261008"))
    parser.add_argument(
        "--bundle", type=Path, default=Path("artifacts/sota_campaign_20261008/SOTA_ESSENTIAL.zip")
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--allow-partial", action="store_true")
    mode.add_argument("--require-complete", action="store_true")
    arguments = parser.parse_args()
    result = publish(
        arguments.artifacts,
        arguments.output,
        arguments.bundle,
        require_complete=arguments.require_complete,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
