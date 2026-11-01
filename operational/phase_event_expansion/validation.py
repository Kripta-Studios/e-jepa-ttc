"""Strict identity and provenance gates; no inference or target scoring."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from operational.phase_event_expansion.model import ExpansionResult, SolverConfig
from operational.phase_event_expansion.synthetic import make_fixture
from scripts.benchmark_phase_event_expansion_synthetic import KINDS, score_fixture


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf8"))


def _relative_source(name: str) -> Path:
    value = name.replace("\\", "/")
    roots = ("operational/", "scripts/", "tests/", "docs/")
    for root in roots:
        if value.startswith(root):
            break
        marker = "/" + root
        if marker in value:
            value = root + value.split(marker, 1)[1]
            break
    else:
        raise ValueError("Unknown archived source root")
    path = Path(value)
    if ".." in path.parts:
        raise ValueError("Unsafe archived source path")
    return path


def validate_confirmation(directory: Path) -> dict[str, Any]:
    """Verify the known 31/47 confirmation, regenerating fixtures but no predictions."""
    config, summary = _read(directory / "CONFIG.json"), _read(directory / "SUMMARY.json")
    recipe = asdict(SolverConfig(objective="temporal_ncc", reference="endpoint"))
    if (
        config.get("cohort") != "confirmation"
        or config.get("seeds") != [31, 47]
        or config.get("kinds") != list(KINDS)
        or config.get("events_per_fixture") != 4096
        or config.get("solver") != recipe
    ):
        raise ValueError("Confirmation cohort or exact solver recipe differs")
    archived = {}
    for name, expected in config.get("source_sha256", {}).items():
        relative = _relative_source(name)
        path = directory / "executed_source" / relative
        if not path.is_file() or _sha(path) != expected:
            raise ValueError("Archived confirmation source changed")
        key = relative.as_posix()
        if key in archived:
            raise ValueError("Duplicate archived source")
        archived[key] = expected
    for name in (
        "operational/phase_event_expansion/model.py",
        "operational/phase_event_expansion/synthetic.py",
        "scripts/benchmark_phase_event_expansion_synthetic.py",
    ):
        if archived.get(name) != _sha(Path(name)):
            raise ValueError(
                "Current solver, fixture or assessment source differs from confirmation"
            )
    expected_ids = {(seed, kind) for seed in (31, 47) for kind in KINDS}
    rows = summary.get("rows", [])
    if len(rows) != 12 or {(r.get("seed"), r.get("kind")) for r in rows} != expected_ids:
        raise ValueError("Missing, duplicate or wrong confirmation fixture")
    hashes = {}
    for row in rows:
        seed, kind = row["seed"], row["kind"]
        fixture = make_fixture(seed, kind, count=4096)
        if (
            row.get("split") != "evaluation"
            or row.get("truth") != {"q": fixture.q, "translation": fixture.translation.tolist()}
            or row.get("events_sha256") != hashlib.sha256(fixture.events.tobytes()).hexdigest()
        ):
            raise ValueError("Confirmation fixture truth, split or event hash differs")
        result = ExpansionResult(**row["result"])
        if result.valid and (
            result.q is None
            or result.q == 0
            or result.ttc_seconds is None
            or not np.isfinite(result.ttc_seconds)
            or not np.isclose(result.ttc_seconds * result.q, 1, rtol=1e-12, atol=0)
        ):
            raise ValueError("Inconsistent endpoint prediction")
        assessment = score_fixture(fixture, result)
        if row.get("assessment") != assessment or not assessment["passed"]:
            raise ValueError("Confirmation assessment failed independent recomputation")
        path = directory / f"{seed}_{kind}.json"
        if not path.is_file() or _read(path) != row:
            raise ValueError("Confirmation summary differs from fixture receipt")
        hashes[path.name] = _sha(path)
    if (
        summary.get("all_synthetic_gates_pass") is not True
        or summary.get("evaluation_total") != 12
        or summary.get("evaluation_passed") != 12
    ):
        raise ValueError("Confirmation summary counters differ")
    return {
        "config_sha256": _sha(directory / "CONFIG.json"),
        "summary_sha256": _sha(directory / "SUMMARY.json"),
        "archived_source_sha256": archived,
        "fixture_receipt_sha256": hashes,
    }


def bind_confirmation(directory: Path) -> Path:
    """Explicitly add an immutable binding after independent legacy-evidence validation."""
    destination = directory / "CONFIRMATION_BINDING.json"
    if destination.exists():
        raise FileExistsError("Confirmation binding already exists")
    evidence = validate_confirmation(directory)
    receipt = {
        "schema": "phase_event_expansion_confirmation_binding_v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "validation_source_sha256": _sha(Path(__file__)),
        "validation": "31/47 x six kinds; regenerated events/truth; recomputed assessments; "
        "exact endpoint temporal_ncc recipe; all archived sources verified; no solver rerun",
        **evidence,
    }
    with destination.open("x", encoding="utf8") as stream:
        stream.write(json.dumps(receipt, indent=2) + "\n")
    return destination


def require_confirmation_binding(directory: Path) -> dict[str, Any]:
    """Require an explicit binding and revalidate its complete evidence chain."""
    path = directory / "CONFIRMATION_BINDING.json"
    if not path.is_file():
        raise ValueError("Missing explicit confirmation binding; validate and bind separately")
    receipt = _read(path)
    if receipt.get("schema") != "phase_event_expansion_confirmation_binding_v1":
        raise ValueError("Unknown confirmation binding")
    for name, filename in (("config_sha256", "CONFIG.json"), ("summary_sha256", "SUMMARY.json")):
        if receipt.get(name) != _sha(directory / filename):
            raise ValueError("Confirmation binding differs from configuration or summary")
    evidence = validate_confirmation(directory)
    if any(receipt.get(key) != value for key, value in evidence.items()):
        raise ValueError("Confirmation binding evidence changed")
    return receipt


def validate_cache_manifest(manifest: dict, freeze: dict) -> None:
    """Enforce all 77 unique ordered identities and their frozen calibration."""
    rows = manifest.get("rows", [])
    if (
        manifest.get("sequence_id") != "Slider-500"
        or freeze.get("sequence_id") != "Slider-500"
        or manifest.get("queries") != 77
        or len(rows) != 77
        or manifest.get("status") != "complete"
        or manifest.get("labels_in_cache") is not False
        or manifest.get("calibration") != freeze.get("calibration")
    ):
        raise ValueError("Invalid Slider500 manifest population or calibration")
    for index, row in enumerate(rows):
        if (
            row.get("query_id") != f"Slider-500:{index}"
            or row.get("row_index") != index
            or row.get("file") != f"query_{index:05d}.npz"
        ):
            raise ValueError("Missing, duplicate, reordered or unsafe cache identity")
    identities = [{"frame_index": r["frame_index"], "anchor_us": r["query_us"]} for r in rows]
    if identities != freeze.get("queries"):
        raise ValueError("Manifest clocks/frame identities differ from freeze")


def validate_cached_window(arrays: dict[str, np.ndarray], row: dict) -> None:
    """Bind numeric sensor payload to its manifest clock, ROI and causal bounds."""
    if set(arrays) != {"x", "y", "t_us", "p", "query_us", "roi_xyxy"}:
        raise ValueError("Unexpected sensor cache fields")
    query = arrays["query_us"]
    if query.shape != () or query.dtype.kind not in "iu" or int(query) != row["query_us"]:
        raise ValueError("Cached query clock differs from manifest")
    clocks = arrays["t_us"]
    if any(
        arrays[k].shape != clocks.shape or arrays[k].dtype.kind not in "iu"
        for k in ("x", "y", "t_us", "p")
    ):
        raise ValueError("Noninteger or misaligned cached events")
    if (
        clocks.ndim != 1
        or len(clocks) != row["selected_count"]
        or not np.array_equal(arrays["roi_xyxy"], row["roi_xyxy"])
        or np.any(np.diff(clocks) < 0)
        or np.any(clocks < row["window_start_us"])
        or np.any(clocks >= row["window_end_exclusive_us"])
        or np.any(clocks > int(query))
        or not np.isin(arrays["p"], [0, 1]).all()
    ):
        raise ValueError("Cached events violate frozen ROI/count/causal window")


def validate_population_tokens(tokens: np.ndarray) -> None:
    """Prevent dictionary lookup from silently collapsing population identities."""
    if tokens.ndim != 1 or len(set(map(str, tokens))) != len(tokens):
        raise ValueError("Population tokens must be a unique vector")
