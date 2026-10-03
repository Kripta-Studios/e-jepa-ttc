"""Fragmented publication and prospective comparisons for the six C0 recipes.

Importing this module does not import Torch or evaluate development predictions.
Each family must first seal every endpoint, or provide a deadline-fixed partial
inventory containing only complete three-fold arms. Historical sources stay read-only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import pandas as pd

    from e_jepa_ttc.simplex_t.campaign_sources import CampaignSources


class ResourceGuard(Protocol):
    """The registered engine resource checker used at safe analysis boundaries."""

    def check(self) -> None: ...


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/simplex_t/nocturnal_20261003"
EXEC = OUT / "execution"
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "operational/simplex_t_closure")]

FAMILIES = {
    "N2": ("FULL_C0", "A5_ONLY_C0", "C2F_ONLY_C0", "A5_PAIR_C0"),
    "N3": ("SET_AGE_C0", "SET_NOTIME_C0"),
}
CONTRASTS = (
    ("FULL_C0", "H8@7"),
    ("A5_ONLY_C0", "FULL_C0"),
    ("C2F_ONLY_C0", "FULL_C0"),
    ("A5_PAIR_C0", "FULL_C0"),
    ("SET_AGE_C0", "FULL_C0"),
    ("SET_NOTIME_C0", "SET_AGE_C0"),
)


def record(path: Path) -> dict[str, Any]:
    """Read an explicit immutable receipt, never discover a latest run."""
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    """Return the complete file SHA-256."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _runtime() -> ModuleType:
    import runtime

    return runtime


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    _runtime().atomic_json(path, value)


def immutable_json(path: Path, value: dict[str, Any]) -> None:
    _runtime().publish_json(path, value)


def immutable_npz(path: Path, arrays: dict[str, Any]) -> str:
    """Commit numeric arrays atomically; preserve any conflicting old export."""
    import numpy as np

    if path.exists():
        with np.load(path, allow_pickle=False) as prior:
            if set(prior.files) != set(arrays) or any(
                not np.array_equal(prior[key], value, equal_nan=True)
                for key, value in arrays.items()
            ):
                raise ValueError(f"conflicting numeric export: {path}")
    else:
        stream = io.BytesIO()
        np.savez_compressed(stream, **arrays)
        _runtime().atomic_bytes(path, stream.getvalue())
    return digest(path)


def cached_inputs(arrays: dict[str, Any]) -> tuple[Path, str]:
    """Deduplicate prepared inputs by names, shapes, dtypes and exact bytes."""
    import numpy as np

    identity = hashlib.sha256()
    for key in sorted(arrays):
        array = np.ascontiguousarray(arrays[key])
        identity.update(json.dumps([key, str(array.dtype), list(array.shape)]).encode())
        identity.update(array.tobytes())
    path = EXEC / "cached_inputs" / (identity.hexdigest() + ".npz")
    return path, immutable_npz(path, arrays)


def write_table(frame: pd.DataFrame, parquet: Path, csv_path: Path) -> None:
    """Replace only complete prediction/table bytes."""
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    _runtime().atomic_bytes(parquet, buffer.getvalue())
    _runtime().atomic_bytes(csv_path, frame.to_csv(index=False, float_format="%.17g").encode())


def _check(resources: ResourceGuard) -> None:
    resources.check()


def _seal(family: str, pin: str, partial_inventory: Path | None) -> tuple[dict, list[dict]]:
    """Validate all authorized family IDs before loading any development source."""
    if family not in FAMILIES:
        raise ValueError("family must be N2 or N3")
    path = EXEC / (family + "_ENDPOINTS.json")
    if partial_inventory is None:
        seal = record(path)
        if seal["protocol_sha256"] != pin or not seal.get("all_family_frozen_before_evaluation"):
            raise ValueError("family endpoint protocol differs")
        rows = seal["fits"]
        wanted = set(FAMILIES[family])
        if {row["arm"] for row in rows} != wanted or len(rows) != len(wanted) * 3:
            raise ValueError("all family endpoints must precede first evaluation")
    else:
        # This explicit inventory is frozen by the supervisor after the deadline.
        from datetime import UTC, datetime

        window = record(OUT / "WINDOW_AUTHORIZATION.json")
        if datetime.now(UTC) < datetime.fromisoformat(window["deadline_utc"]):
            raise ValueError("partial-family evaluation is reserved for the deadline inventory")
        seal = record(partial_inventory)
        if seal["protocol_sha256"] != pin or seal["family"] != family:
            raise ValueError("partial inventory protocol/family differs")
        if not seal.get("frozen_before_first_evaluation"):
            raise ValueError("partial evaluation inventory must precede scores")
        rows = seal["fits"]
        wanted = {row["arm"] for row in rows}
        if not wanted <= set(FAMILIES[family]):
            raise ValueError("partial inventory contains an unauthorized recipe")
    ids = {(row["arm"], row["fold"], row["seed"]) for row in rows}
    expected = {(arm, fold, 7) for arm in wanted for fold in range(3)}
    if not rows or len(ids) != len(rows) or ids != expected:
        raise ValueError("only complete three-fold seed7 arms may be evaluated")
    return seal, rows


def _historical_h8(p: dict) -> pd.DataFrame:
    """Load only preregistered H8 seed7 predictions, without rerunning its head."""
    import pandas as pd

    controls = p.get("historical_controls", p.get("historical_h8_controls", {}))
    selected = []
    for row in controls.values():
        fit = row["endpoint"]["fit"]
        if fit["name"] != "TPR-D1-H8-C160" or fit["seed"] != 7:
            continue
        path = Path(row["prediction"])
        if digest(path) != row["prediction_sha256"]:
            raise ValueError("historical H8 comparator changed")
        selected.append(pd.read_parquet(path))
    if len(selected) != 3:
        raise ValueError("three preregistered H8 seed7 comparator folds required")
    frame = (
        pd.concat(selected, ignore_index=True).sort_values("sample_token").reset_index(drop=True)
    )
    _full_cohort(frame)
    return frame


def _full_cohort(frame: pd.DataFrame) -> None:
    if (
        len(frame) != 8192
        or frame.sample_token.duplicated().any()
        or frame.sequence_id.nunique() != 9
        or set(frame.outer_fold) != {0, 1, 2}
    ):
        raise ValueError("complete OLD_DEV8192 nine-sequence cohort required")


def _sources(p: dict) -> CampaignSources:
    from e_jepa_ttc.simplex_t.configuration_preflight import open_acknowledged_source_configuration

    launch = p["launch"]
    return open_acknowledged_source_configuration(
        Path(launch["local_paths"]),
        Path(launch["source_configuration"]),
        launch["source_configuration_sha256"],
    )[0]


def _wrapped_dev_pin(p: dict, fold: int, arm: str) -> str:
    binding = p["sources"][str(fold)]["wrapped"][arm]
    return binding.get("dev_sha256", binding.get("dev", ""))


def publish_family(
    p: dict, pin: str, family: str, resources: ResourceGuard, partial_inventory: Path | None = None
) -> tuple[dict[str, Any], dict]:
    """Infer sealed endpoints in independently committed batches of128 queries."""
    import numpy as np
    import pandas as pd
    import torch

    from e_jepa_ttc.simplex_t.current_inputs import load_current_inputs
    from e_jepa_ttc.simplex_t.evaluation import prediction_frame
    from e_jepa_ttc.simplex_t.registry import FitSpec
    from operational.simplex_t_cost_context import engine
    from operational.simplex_t_cost_context.model import MaskedSource, anchor_from_allowed

    seal, rows = _seal(family, pin, partial_inventory)
    seal_path = partial_inventory or EXEC / (family + "_ENDPOINTS.json")
    torch.set_num_threads(4)
    if torch.get_num_interop_threads() > 2:
        torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    # Validate every endpoint before any OLD_DEV source or head forward.
    for row in rows:
        _check(resources)
        model, state = engine.load_new_endpoint(row, p, pin)
        del model, state
    historical = _historical_h8(p)
    frames: dict[str, Any] = {"H8@7": historical}
    exports = {}
    sources = _sources(p)
    try:
        for row in rows:
            _check(resources)
            arm, fold = row["arm"], row["fold"]
            parent = sources.source(FitSpec("T2", "TPR-D1-H8-C160", fold, 7), "outer_dev")
            if parent.identity_sha256 != p["sources"][str(fold)]["parent_dev_sha256"]:
                raise ValueError("historical D1/H8 OLD_DEV parent identity changed")
            source = MaskedSource(parent, arm)
            if source.identity_sha256 != _wrapped_dev_pin(p, fold, arm):
                raise ValueError("prospective masked OLD_DEV identity changed")
            model, state = engine.load_new_endpoint(row, p, pin)
            weight_path = EXEC / "weights" / (row["key"].replace("/", "__") + ".npz")
            weight_sha = immutable_npz(
                weight_path,
                {name: value.detach().cpu().numpy() for name, value in state["model"].items()},
            )
            del state
            root = EXEC / "publication" / row["key"]
            root.mkdir(parents=True, exist_ok=True)
            payloads, anchors, fragments = [], [], []
            for start in range(0, source.population, 128):
                _check(resources)
                stop = min(start + 128, source.population)
                x, times, valid, experts, truth, mass = source.gather(torch.arange(start, stop))
                input_path, input_sha = cached_inputs(
                    dict(
                        features=x.numpy(),
                        times=times.numpy(),
                        valid=valid.numpy(),
                        experts=experts.numpy(),
                        target_phase=truth.numpy(),
                        mass=mass.numpy(),
                    )
                )
                output_path = root / f"batch_{start:05d}.npz"
                receipt_path = root / f"batch_{start:05d}.json"
                expected = dict(
                    start=start,
                    stop=stop,
                    input_path=str(input_path.relative_to(EXEC)),
                    input_sha256=input_sha,
                    endpoint_sha256=row["checkpoint_sha256"],
                    protocol_sha256=pin,
                    source_sha256=source.identity_sha256,
                )
                if receipt_path.exists():
                    receipt = record(receipt_path)
                    if {key: receipt[key] for key in expected} != expected:
                        raise ValueError("inference fragment binding changed")
                    if digest(output_path) != receipt["payload_sha256"]:
                        raise ValueError("inference fragment payload changed")
                else:
                    with torch.inference_mode():
                        batch = {
                            key: value.detach().cpu().numpy()
                            for key, value in model(x, times, valid, experts).items()
                        }
                    if not all(np.isfinite(value).all() for value in batch.values()):
                        raise ValueError("nonfinite prospective endpoint output")
                    payload_sha = immutable_npz(output_path, batch)
                    immutable_json(receipt_path, dict(**expected, payload_sha256=payload_sha))
                    print(
                        json.dumps(
                            dict(
                                status="INFERENCE_FRAGMENT_COMMITTED",
                                family=family,
                                key=row["key"],
                                confirmed_queries=stop,
                            )
                        ),
                        flush=True,
                    )
                with np.load(output_path, allow_pickle=False) as archive:
                    payloads.append({key: archive[key].copy() for key in archive.files})
                anchors.append(anchor_from_allowed(experts, arm).detach().numpy())
                fragments.append(
                    dict(
                        receipt_path=str(receipt_path.relative_to(EXEC)),
                        receipt_sha256=digest(receipt_path),
                        input_path=str(input_path.relative_to(EXEC)),
                        input_sha256=input_sha,
                        output_path=str(output_path.relative_to(EXEC)),
                        output_sha256=digest(output_path),
                        start=start,
                        stop=stop,
                    )
                )
            outputs = {
                key: np.concatenate([batch[key] for batch in payloads]) for key in payloads[0]
            }
            current = load_current_inputs(
                sources.historical_root,
                fold,
                "outer_dev",
                ancestry_sha256=sources.ancestry_sha256,
                allowed_sequences=sources.allowed_sequences,
            )
            template = (
                historical.loc[historical.outer_fold == fold]
                .set_index("sample_token")
                .loc[current["metadata"].sample_token]
                .reset_index()
            )
            if not np.array_equal(
                template.target_ttc.to_numpy(), current["metadata"].target_ttc.to_numpy()
            ):
                raise ValueError("query targets differ from independent canonical table")
            # Historical table experts differ slightly from compiled current observations.
            # Use the exact compiled expert output used by the historical H8 publication.
            current_ids = parent.history[:, -1]
            full_experts = np.load(
                sources.folds[fold].path / "expert_ttc.npy", mmap_mode="r", allow_pickle=False
            )[current_ids]
            expected_experts = template[[f"expert{i}_ttc" for i in range(3)]].to_numpy()
            if not np.array_equal(full_experts, expected_experts):
                raise ValueError("compiled expert values differ from published H8 control")
            metadata_names = [
                "sample_token",
                "sequence_id",
                "track_id",
                "target_ttc",
                "anchor_us",
                "history_span_us",
                "roi_age_us",
            ]
            metadata = template[metadata_names].copy()
            frame = prediction_frame(
                metadata, full_experts, outputs, parent.history[:, -8:], arm=arm, seed=7, fold=fold
            )
            frame["source_sha256"] = source.identity_sha256
            frame["context_semantics"] = (
                "RETROSPECTIVE_CURRENT_QUERY_ROI_NOT_VERIFIED_OBJECT_HISTORY"
            )
            frame["hull_reference"] = "ALL_THREE_FROZEN_EXPERTS_DIAGNOSTIC_ONLY"
            frame["allowed_anchor_phase"] = np.concatenate(anchors).astype(np.float64)
            frame["cost_auxiliary_active"] = False
            frame["cost_supervision_active"] = False
            frame["expert_selection_used"] = False
            frame["cost_fields_semantics"] = "INACTIVE_COMPATIBILITY_PLACEHOLDERS_NOT_SELECTION"
            write_table(frame, root / "PREDICTIONS.parquet", root / "PREDICTIONS.csv")
            publication = dict(
                protocol_sha256=pin,
                family=family,
                endpoints_seal_sha256=digest(seal_path),
                endpoint_sha256=row["checkpoint_sha256"],
                queries=len(frame),
                source_sha256=source.identity_sha256,
                predictions_sha256=digest(root / "PREDICTIONS.parquet"),
                csv_sha256=digest(root / "PREDICTIONS.csv"),
                compiled_expert_reference=True,
                full_hull_diagnostic_only=True,
                inference_fragments=len(fragments),
            )
            immutable_json(root / "PUBLICATION.json", publication)
            exports[row["key"]] = dict(
                arm=arm,
                fold=fold,
                seed=7,
                checkpoint=row["checkpoint"],
                checkpoint_sha256=row["checkpoint_sha256"],
                train_source_sha256=row["train_source_sha256"],
                dev_source_sha256=source.identity_sha256,
                weights_path=str(weight_path.relative_to(EXEC)),
                weights_sha256=weight_sha,
                predictions_path=str((root / "PREDICTIONS.parquet").relative_to(EXEC)),
                predictions_sha256=publication["predictions_sha256"],
                publication_path=str((root / "PUBLICATION.json").relative_to(EXEC)),
                publication_sha256=digest(root / "PUBLICATION.json"),
                fragments=fragments,
            )
            frames.setdefault(arm, []).append(frame)
            del source, parent, model, payloads, outputs, full_experts
            sources.release()
            gc.collect()
    finally:
        sources.release()
    for arm in {row["arm"] for row in rows}:
        frames[arm] = (
            pd.concat(frames[arm], ignore_index=True)
            .sort_values("sample_token")
            .reset_index(drop=True)
        )
        _full_cohort(frames[arm])
    export_path = EXEC / "ANALYSIS_EXPORT_INDEX.json"
    index: dict[str, Any] = (
        record(export_path)
        if export_path.exists()
        else dict(
            schema="simplex_t_cost_context_cached_export_v1",
            protocol_sha256=pin,
            fits={},
            families={},
        )
    )
    if index["protocol_sha256"] != pin:
        raise ValueError("export protocol changed")
    for key, value in exports.items():
        if key in index["fits"] and index["fits"][key] != value:
            raise ValueError("completed export changed")
        index["fits"][key] = value
    index["families"][family] = dict(
        seal_path=str(seal_path),
        seal_sha256=digest(seal_path),
        arms=sorted({row["arm"] for row in rows}),
        incomplete_family=partial_inventory is not None,
    )
    atomic_json(export_path, index)
    # A later family extends the union index. Its addition must not change an
    # earlier family's bootstrap binding or immutable RESULTS receipt.
    family_index = dict(
        schema=index["schema"],
        protocol_sha256=pin,
        fits={
            key: value for key, value in index["fits"].items() if value["arm"] in FAMILIES[family]
        },
        families={family: index["families"][family]},
    )
    immutable_json(EXEC / "analysis" / family / "EXPORT_INDEX.json", family_index)
    return frames, family_index


def _prior_frame(arm: str, pin: str) -> pd.DataFrame:
    import pandas as pd

    index = record(EXEC / "ANALYSIS_EXPORT_INDEX.json")
    if index["protocol_sha256"] != pin:
        raise ValueError("prior-family export protocol differs")
    selected = [row for row in index["fits"].values() if row["arm"] == arm]
    if len(selected) != 3 or {row["fold"] for row in selected} != {0, 1, 2}:
        raise ValueError("comparator needs all three previously frozen/published folds")
    parts = []
    for row in selected:
        path = EXEC / row["predictions_path"]
        if digest(path) != row["predictions_sha256"]:
            raise ValueError("prior-family comparator prediction changed")
        parts.append(pd.read_parquet(path))
    frame = pd.concat(parts, ignore_index=True).sort_values("sample_token").reset_index(drop=True)
    _full_cohort(frame)
    return frame


def analyze_family(
    p: dict, pin: str, family: str, frames: dict, resources: ResourceGuard, index: dict
) -> dict:
    """Apply fixed paired loss contrasts; retain all nine independent sequences."""
    import numpy as np
    import pandas as pd

    from e_jepa_ttc.evaluation.exact_sequence_v10 import exact_sequence_diagnostic
    from e_jepa_ttc.evaluation.stage63_65 import strict_macro_mass
    from e_jepa_ttc.simplex_t.diagnostic_summary import summarize_diagnostics
    from e_jepa_ttc.simplex_t.practical_comparison import paired_practical_comparison

    pending = []
    active = []
    for candidate, reference in CONTRASTS:
        if candidate not in FAMILIES[family] or candidate not in frames:
            continue
        if reference not in frames:
            try:
                frames[reference] = _prior_frame(reference, pin)
            except (ValueError, FileNotFoundError) as error:
                pending.append(
                    dict(candidate=candidate, reference=reference, dependency=str(error))
                )
                continue
        active.append((candidate, reference))
    base = frames["H8@7"]
    identity = ["sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc"]
    for frame in frames.values():
        if not frame[identity].equals(base[identity]):
            raise ValueError("paired population identity differs")
    names = sorted(frames)
    losses = np.column_stack([frames[name].loss.to_numpy(np.float64) for name in names])
    mass = strict_macro_mass(base.target_ttc.to_numpy(), base.sequence_id.to_numpy())
    root = EXEC / "analysis" / family
    root.mkdir(parents=True, exist_ok=True)
    scores = mass @ losses
    _check(resources)
    sampled, bootstrap = _runtime().resumable_hierarchical_losses(
        base.rename(columns={"target_ttc": "target_ttc_s"}),
        losses,
        root / "bootstrap",
        lambda: _check(resources),
        draws_path=Path(p["bootstrap_draws"]),
        binding=dict(
            protocol_sha256=pin,
            family=family,
            export_index_fits_sha256=hashlib.sha256(
                json.dumps(index["fits"], sort_keys=True).encode()
            ).hexdigest(),
        ),
        chunk_size=256,
    )
    comparisons = {}
    strata = []
    metric_rows = []
    for name, score in zip(names, scores, strict=True):
        metric_rows.append(
            dict(
                kind="arm",
                family=family,
                id=name,
                candidate=name,
                reference="",
                MiD=float(score),
                reference_MiD=None,
                delta=None,
                hierarchical_ci95_lower=None,
                hierarchical_ci95_upper=None,
                sequence_ci95_lower=None,
                sequence_ci95_upper=None,
                sign_delta=None,
                crucial_delta=None,
                precision_screen_pass=None,
                efficiency_status="NOT_MEASURED_HERE",
                system_substitution_supported=False,
                queries=8192,
                independent_sequences=9,
                seed=7,
                exploratory=True,
            )
        )
    for candidate, reference in active:
        a, b = names.index(candidate), names.index(reference)
        delta = losses[:, a] - losses[:, b]
        perseq = []
        for axis, labels in (("sequence", sorted(base.sequence_id.unique())), ("fold", [0, 1, 2])):
            values = (
                base.sequence_id.to_numpy() if axis == "sequence" else base.outer_fold.to_numpy()
            )
            for label in labels:
                selected = values == label
                value = float((mass[selected] / mass[selected].sum()) @ delta[selected])
                strata.append(
                    dict(
                        candidate=candidate,
                        reference=reference,
                        axis=axis,
                        group=str(label),
                        queries=int(selected.sum()),
                        global_mass=float(mass[selected].sum()),
                        delta=value,
                    )
                )
                if axis == "sequence":
                    perseq.append(value)
        practical = paired_practical_comparison(frames[candidate], frames[reference])
        ci = np.percentile(sampled[:, a] - sampled[:, b], [2.5, 97.5]).tolist()
        seq = exact_sequence_diagnostic(np.array(perseq))
        precision = bool(
            ci[1] < 2.0
            and practical["weighted_sign_error_delta"] <= 0.005
            and practical["crucial_bucket_mid_delta"] <= 5
            and practical["candidate_finite_fraction"] == 1
        )
        key = candidate + "_minus_" + reference
        comparisons[key] = dict(
            candidate=candidate,
            reference=reference,
            candidate_MiD=float(scores[a]),
            reference_MiD=float(scores[b]),
            paired_loss_delta=float(mass @ delta),
            hierarchical_ci95=ci,
            sequence_only=seq,
            practical=practical,
            engineering_screen=dict(
                ci_upper_strictly_below_2=bool(ci[1] < 2),
                sign_increase_max_0p005=bool(practical["weighted_sign_error_delta"] <= 0.005),
                crucial_increase_max_5=bool(practical["crucial_bucket_mid_delta"] <= 5),
                precision_screen_pass=precision,
                latency_reduction_required=0.20,
                latency_scope="UNMEASURED",
                qualifies_head_efficiency=False,
                qualifies_system_substitution=False,
            ),
            exploratory=True,
            confirmatory=False,
        )
        metric_rows.append(
            dict(
                kind="contrast",
                family=family,
                id=key,
                candidate=candidate,
                reference=reference,
                MiD=float(scores[a]),
                reference_MiD=float(scores[b]),
                delta=float(mass @ delta),
                hierarchical_ci95_lower=ci[0],
                hierarchical_ci95_upper=ci[1],
                sequence_ci95_lower=seq["ci95"][0],
                sequence_ci95_upper=seq["ci95"][1],
                sign_delta=practical["weighted_sign_error_delta"],
                crucial_delta=practical["crucial_bucket_mid_delta"],
                precision_screen_pass=precision,
                efficiency_status="COST_MEASUREMENT_REQUIRED",
                system_substitution_supported=False,
                queries=8192,
                independent_sequences=9,
                seed=7,
                exploratory=True,
            )
        )
    metric = pd.DataFrame(metric_rows)
    _runtime().atomic_bytes(
        root / "METRICS_COSTE_CONTEXTO.csv",
        metric.to_csv(index=False, float_format="%.17g").encode(),
    )
    _runtime().atomic_bytes(
        root / "PAIRED_SEQUENCE_FOLD.csv",
        pd.DataFrame(strata).to_csv(index=False, float_format="%.17g").encode(),
    )
    diagnostics = [
        summarize_diagnostics(frames[arm], history_length=8)
        for arm in FAMILIES[family]
        if arm in frames
    ]
    _runtime().atomic_bytes(
        root / "DIAGNOSTICS.csv",
        pd.concat(diagnostics, ignore_index=True)
        .to_csv(index=False, float_format="%.17g")
        .encode(),
    )
    immutable_npz(
        root / "PAIRED_LOSSES.npz",
        dict(mass=mass, **{name: losses[:, i] for i, name in enumerate(names)}),
    )
    report = dict(
        schema="simplex_t_prospective_cost_context_results_v1",
        status="COMPLETE_EXPLORATORY_FAMILY_ANALYSIS"
        if not pending and not index["families"][family]["incomplete_family"]
        else "AVAILABLE_COMPLETE_ARMS_ANALYZED_EXPLORATORY_WITH_GAPS",
        protocol_sha256=pin,
        family=family,
        arms=[arm for arm in FAMILIES[family] if arm in frames],
        scores={name: float(scores[i]) for i, name in enumerate(names)},
        comparisons=comparisons,
        pending_comparators=pending,
        bootstrap=bootstrap,
        independent_sequences=9,
        seed=7,
        ensemble=False,
        historical_candidate="TPR-D1-H8-C160",
        candidate_replaced=False,
        confirmatory=False,
        full_hull_diagnostic_reference=True,
        latency_measured_here=False,
        input_scope="NORMALIZED_CACHED_H8_CONTEXTS",
        raw_reconstruction=False,
        export_index_sha256=digest(root / "EXPORT_INDEX.json"),
        metrics_sha256=digest(root / "METRICS_COSTE_CONTEXTO.csv"),
    )
    immutable_json(root / "RESULTS.json", report)
    # Aggregate family CSVs only; retain family reports as independently bound receipts.
    family_tables = [
        # pandas supports round_trip; the installed overload stub omits it.
        pd.read_csv(  # type: ignore[reportCallIssue]
            EXEC / "analysis" / f / "METRICS_COSTE_CONTEXTO.csv",
            float_precision="round_trip",  # type: ignore[reportArgumentType]
        )
        for f in FAMILIES
        if (EXEC / "analysis" / f / "METRICS_COSTE_CONTEXTO.csv").exists()
    ]
    _runtime().atomic_bytes(
        OUT / "METRICS_COSTE_CONTEXTO.csv",
        pd.concat(family_tables, ignore_index=True)
        .to_csv(index=False, float_format="%.17g")
        .encode(),
    )
    atomic_json(
        root / "ANALYSIS_RECEIPT.json",
        dict(
            status=report["status"],
            results_sha256=digest(root / "RESULTS.json"),
            metrics_sha256=report["metrics_sha256"],
            export_index_sha256=report["export_index_sha256"],
            comparisons=len(comparisons),
            accepted_draws=bootstrap["valid_draws"],
            draw_attempts=bootstrap["attempts"],
            completed_arms=len(report["arms"]),
        ),
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=tuple(FAMILIES), required=True)
    parser.add_argument("--partial-inventory", type=Path)
    args = parser.parse_args()
    from operational.simplex_t_cost_context import engine

    p, pin = engine.protocol()
    resources = engine.Resources()
    _check(resources)
    engine.validate_pins(p, full=True)
    with engine.Lease():
        frames, index = publish_family(p, pin, args.family, resources, args.partial_inventory)
        report = analyze_family(p, pin, args.family, frames, resources, index)
    print(
        json.dumps(
            dict(
                status=report["status"],
                family=args.family,
                arms=report["arms"],
                export_index_sha256=report["export_index_sha256"],
                metrics_sha256=report["metrics_sha256"],
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
