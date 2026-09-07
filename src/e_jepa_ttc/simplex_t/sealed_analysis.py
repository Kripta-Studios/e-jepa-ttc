"""Bind all analysis arms to actual sealed endpoints, frozen sources and OLD targets."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase

from .arms import resolve_arm
from .expert_phase import expert_benchmark_phase
from .factorial_analysis import paired_factor_effects
from .phase_inference import validated_phase
from .phase_manifest import fit_key
from .published_predictions import iter_published_predictions
from .scientific_freeze import read_scientific_freeze
from .stage_gate import CanonicalPublication
from .uncertainty_analysis import paired_uncertainty


def load_sealed_analysis_arms(
    binding: CanonicalPublication,
    *,
    stage: str,
    freeze: Path,
    freeze_sha256: str,
    roots: dict[str, Path],
    expected_queries: pd.DataFrame,
    validate_authority_and_qa: Callable[[], None],
    resource_ok: Callable[[], bool],
) -> dict[tuple[str, int], pd.DataFrame]:
    """Return complete arms by (canonical name, seed), never averaged TTC.

    The caller must obtain expected_queries from the independent historical OLD
    loader. This routine verifies the full actual freeze and all endpoint states,
    retains every query and control, and recomputes the stored benchmark loss.
    It does not select candidates, fit, open confirmation or write a final report.
    A failure returns no partially validated arm collection.
    """
    record = read_scientific_freeze(
        freeze,
        expected_sha256=freeze_sha256,
        roots=roots,
        validate_prerequisites=validate_authority_and_qa,
    )
    frozen_flags = record["source_contract"]["availability"]
    flags = binding.availability
    if (
        set(flags) != set(frozen_flags)
        or any(
            type(value) is not bool or (value and not frozen_flags[key])
            for key, value in flags.items()
        )
        or any(flags[key] != frozen_flags[key] for key in ("d1", "density"))
    ):
        raise ValueError("analysis availability changes frozen pools or enables absent arms")
    graph, endpoints = validated_phase(
        binding.endpoints,
        binding.checkpoint_root,
        manifest_sha256=binding.endpoints_sha256,
        freeze_sha256=freeze_sha256,
        stage=stage,
        availability=flags,
        resource_ok=resource_ok,
    )
    expected_specs = {fit_key(spec): spec for spec in graph if spec.stage == stage}
    for key, endpoint in endpoints.items():
        if endpoint["train_source_sha256"] != record["source_identities"][key]["inner_oof"]:
            raise ValueError("sealed training source differs from the scientific freeze")
    identity = ["sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc"]
    if (
        not set(identity) <= set(expected_queries)
        or len(expected_queries) != 8192
        or expected_queries.sample_token.duplicated().any()
        or expected_queries.sequence_id.nunique() != 9
        or expected_queries[identity].isna().to_numpy().any()
    ):
        raise ValueError("independently verified full OLD targets required")
    cohort = expected_queries.loc[:, identity].sort_values("sample_token").reset_index(drop=True)

    def validate() -> None:
        validate_authority_and_qa()
        if sha256(freeze) != freeze_sha256 or sha256(binding.endpoints) != binding.endpoints_sha256:
            raise ValueError("analysis freeze or endpoint seal changed")
        for endpoint in endpoints.values():
            if not resource_ok():
                raise InterruptedError("PAUSED_RESOURCE: analysis endpoint verification")
            if sha256(endpoint["resolved_checkpoint"]) != endpoint["checkpoint_sha256"]:
                raise ValueError("analysis endpoint checkpoint changed")

    collected: dict[tuple[str, int], list[pd.DataFrame]] = {}
    seen = set()
    for spec, frame in iter_published_predictions(
        binding.publication,
        manifest_sha256=binding.publication_sha256,
        endpoint_manifest_sha256=binding.endpoints_sha256,
        freeze_sha256=freeze_sha256,
        stage=stage,
        availability=flags,
        expected_queries=expected_queries,
        validate_prerequisites=validate,
        resource_ok=resource_ok,
    ):
        key = fit_key(spec)
        if key not in expected_specs or key in seen:
            raise ValueError("duplicate or unregistered analysis fit")
        seen.add(key)
        if not {*identity, "prediction_ttc_s", "prediction_phase", "loss"} <= set(frame):
            raise ValueError("analysis prediction fields missing")
        actual = frame.loc[:, identity].sort_values("sample_token").reset_index(drop=True)
        if not actual.equals(cohort.loc[cohort.outer_fold == spec.fold].reset_index(drop=True)):
            raise ValueError("published targets differ from independent OLD targets")
        if not frame.source_sha256.eq(record["source_identities"][key]["outer_dev"]).all():
            raise ValueError("analysis development source differs from scientific freeze")
        mode = resolve_arm(spec, graph).model.output_mode
        phase_function = expert_benchmark_phase if mode == "selector" else benchmark_phase
        phase = phase_function(frame.prediction_ttc_s.to_numpy(np.float64))
        loss = 10000 * np.abs(phase - benchmark_phase(frame.target_ttc.to_numpy(np.float64)))
        if not np.array_equal(phase, frame.prediction_phase.to_numpy()) or not np.array_equal(
            loss, frame.loss.to_numpy()
        ):
            raise ValueError("published phase or loss differs from emitted TTC arithmetic")
        collected.setdefault((spec.name, spec.seed), []).append(frame)
    if seen != set(expected_specs):
        raise ValueError("incomplete phase cannot supply analysis arms")
    result = {}
    for key, parts in collected.items():
        frame = (
            pd.concat(parts, ignore_index=True).sort_values("sample_token").reset_index(drop=True)
        )
        if len(parts) != 3 or not frame[identity].equals(cohort):
            raise ValueError("analysis arm does not retain all three OLD folds")
        result[key] = frame
    validate()
    return result


def analyze_sealed_t2(
    binding: CanonicalPublication,
    *,
    output: Path,
    freeze: Path,
    freeze_sha256: str,
    roots: dict[str, Path],
    expected_queries: pd.DataFrame,
    validate_authority_and_qa: Callable[[], None],
    resource_ok: Callable[[], bool],
) -> dict:
    """Publish all T2 contrasts and paired uncertainty, never selecting a winner.

    Use the registered primary H1-C160 reference, with identical shared bootstrap
    draws across every seed7 arm/control. Signed per-query factorial effects are
    saved separately, not disguised as nonnegative benchmark losses. Interrupted
    output is retained; a new analysis directory is required for retry.
    """
    target = output.resolve()
    work = roots["work"].resolve(strict=True)
    if target == work or not target.is_relative_to(work) or output.exists():
        raise ValueError("new analysis directory inside companion required")
    arms = load_sealed_analysis_arms(
        binding,
        stage="T2",
        freeze=freeze,
        freeze_sha256=freeze_sha256,
        roots=roots,
        expected_queries=expected_queries,
        validate_authority_and_qa=validate_authority_and_qa,
        resource_ok=resource_ok,
    )
    if any(seed != 7 for _, seed in arms):
        raise ValueError("T2 analysis cannot mix replicate seeds")
    frames = {name: frame for (name, _), frame in arms.items()}
    primary = "D1" if binding.availability["d1"] else "D0"
    reference = f"TPR-{primary}-H1-C160"
    effects = paired_factor_effects(
        pd.concat(list(frames.values()), ignore_index=True),
        expected_queries,
        d1_available=binding.availability["d1"],
    )

    def boundary() -> None:
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: sealed T2 analysis")
        validate_authority_and_qa()
        if (
            sha256(freeze) != freeze_sha256
            or sha256(binding.publication) != binding.publication_sha256
            or sha256(binding.endpoints) != binding.endpoints_sha256
        ):
            raise ValueError("T2 analysis input seals changed")

    boundary()
    output.mkdir(parents=True)
    effects_path = output / "FACTOR_EFFECTS.parquet"
    effects.to_parquet(effects_path, index=False)
    uncertainty = paired_uncertainty(
        frames,
        reference=reference,
        output=output / "paired_uncertainty",
        resource_check=boundary,
    )
    boundary()
    result = {
        "schema": "simplex_t_sealed_t2_analysis_v1",
        "status": "T2_FACTORIAL_AND_PAIRED_UNCERTAINTY_COMPLETE_NOT_T6",
        "freeze_sha256": freeze_sha256,
        "publication_sha256": binding.publication_sha256,
        "endpoints_sha256": binding.endpoints_sha256,
        "reference": reference,
        "arm_names": sorted(frames),
        "query_count_per_arm": 8192,
        "factor_effects": {"path": effects_path.name, "sha256": sha256(effects_path)},
        "uncertainty": {
            "path": "paired_uncertainty/PAIRED_UNCERTAINTY.json",
            "sha256": sha256(output / "paired_uncertainty/PAIRED_UNCERTAINTY.json"),
            "status": uncertainty["status"],
        },
        "scope": "REUSED_OLD_DEVELOPMENT_NOT_CONFIRMATORY",
        "sequence_id_establishes_independent_acquisition": False,
        "controls_can_replace_primary": False,
        "optimizer_updates": 0,
        "holdout_opened": False,
    }
    write_new_json(output / "T2_ANALYSIS.json", result)
    return result
