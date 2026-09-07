"""Sealed T3/T4/T5 analysis with fixed family and same-seed H1 references."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json

from .diagnostic_summary import summarize_diagnostics
from .replication_summary import three_seed_losses
from .sealed_analysis import load_sealed_analysis_arms
from .stage_gate import CanonicalPublication
from .uncertainty_analysis import paired_uncertainty


def analyze_followup_phase(
    binding: CanonicalPublication,
    *,
    stage: str,
    output: Path,
    freeze: Path,
    freeze_sha256: str,
    roots: dict[str, Path],
    expected_queries: pd.DataFrame,
    scalar_seed7: CanonicalPublication | None,
    validate_authority_and_qa: Callable[[], None],
    resource_ok: Callable[[], bool],
) -> dict:
    """Report exploration or replication without promoting a control to primary.

    T3 uses the pinned T2 primary H1/H8 seed7 pair as references, retaining all
    context/Transformer arms. T4 compares its three latent arms with latent H1.
    T5 compares H8 with H1 separately for each family and seed13/23. Cross-seed
    summaries and joint three-seed inference remain separate analyses; this
    routine never averages signed TTC. Caller derives OLD from historical inputs.
    """
    if stage not in {"T3", "T4", "T5"}:
        raise ValueError("registered follow-up stage required")
    work = roots["work"].resolve(strict=True)
    if output.exists() or output.resolve() == work or not output.resolve().is_relative_to(work):
        raise ValueError("new companion analysis output required")
    if (stage == "T3") != (scalar_seed7 is not None):
        raise ValueError("T3 alone requires a pinned scalar seed7 publication")

    def load(item: CanonicalPublication, phase: str) -> dict[tuple[str, int], pd.DataFrame]:
        return load_sealed_analysis_arms(
            item,
            stage=phase,
            freeze=freeze,
            freeze_sha256=freeze_sha256,
            roots=roots,
            expected_queries=expected_queries,
            validate_authority_and_qa=validate_authority_and_qa,
            resource_ok=resource_ok,
        )

    arms = load(binding, stage)
    primary = "D1" if binding.availability["d1"] else "D0"
    groups: dict[str, tuple[str, dict[str, pd.DataFrame]]] = {}
    references = [binding]
    if stage == "T3":
        assert scalar_seed7 is not None
        if any(scalar_seed7.availability[k] != binding.availability[k] for k in ("d1", "density")):
            raise ValueError("T3 reference changes frozen data pools")
        base = load(scalar_seed7, "T2")
        names = [f"TPR-{primary}-H{h}-C160" for h in (1, 8)]
        frames = {name: base[name, 7] for name in names}
        frames.update({name: frame for (name, seed), frame in arms.items() if seed == 7})
        if len(frames) != len(arms) + 2:
            raise ValueError("T3 contains unexpected seeds or aliases a baseline")
        groups["context_seed7"] = (names[0], frames)
        references.append(scalar_seed7)
    elif stage == "T4":
        if any(seed != 7 for _, seed in arms):
            raise ValueError("T4 requires seed7")
        groups["latent_seed7"] = (
            f"LATENT-{primary}-H1-C160",
            {name: frame for (name, _), frame in arms.items()},
        )
    else:
        for family in ("TPR", "LATENT"):
            flag = "replicate_scalar" if family == "TPR" else "replicate_latent"
            if not binding.availability[flag]:
                continue
            for seed in (13, 23):
                names = [f"{family}-{primary}-H{h}-C160" for h in (1, 8)]
                groups[f"{family}_seed{seed}"] = (
                    names[0],
                    {name: arms[name, seed] for name in names},
                )
        if sum(len(frames) for _, frames in groups.values()) != len(arms):
            raise ValueError("T5 cannot omit or add a family/seed arm")
    if not groups:
        raise ValueError("no registered analysis groups")

    def boundary() -> None:
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: follow-up analysis")
        validate_authority_and_qa()
        if sha256(freeze) != freeze_sha256 or any(
            sha256(item.publication) != item.publication_sha256
            or sha256(item.endpoints) != item.endpoints_sha256
            for item in references
        ):
            raise ValueError("follow-up analysis input seals changed")

    boundary()
    output.mkdir(parents=True)
    records = {}
    for group, (reference, frames) in groups.items():
        boundary()
        destination = output / group
        paired_uncertainty(frames, reference=reference, output=destination, resource_check=boundary)
        diagnostics = pd.concat(
            [
                summarize_diagnostics(frame, history_length=int(name.split("-")[2][1:]))
                for name, frame in frames.items()
            ],
            ignore_index=True,
        )
        diagnostics.to_parquet(destination / "DIAGNOSTICS.parquet", index=False)
        records[group] = {
            "reference": reference,
            "arms": sorted(frames),
            "files": {
                name: sha256(destination / name)
                for name in ("PAIRED_UNCERTAINTY.json", "DIAGNOSTICS.parquet")
            },
        }
    boundary()
    report = {
        "schema": "simplex_t_followup_analysis_v1",
        "stage": stage,
        "status": "FOLLOWUP_PAIRED_DIAGNOSTICS_COMPLETE_NOT_T6",
        "freeze_sha256": freeze_sha256,
        "groups": records,
        "publications": [
            {"sha256": item.publication_sha256, "endpoints_sha256": item.endpoints_sha256}
            for item in references
        ],
        "cross_seed_analysis_complete": False,
        "rapid_change_and_sign_transition_diagnostics_complete": False,
        "controls_can_replace_primary": False,
        "ttc_ensembling_performed": False,
        "scope": "REUSED_OLD_DEVELOPMENT_NOT_CONFIRMATORY",
        "optimizer_updates": 0,
        "holdout_opened": False,
    }
    write_new_json(output / "FOLLOWUP_ANALYSIS.json", report)
    return report


def analyze_three_seed_family(
    seed7: CanonicalPublication,
    replicates: CanonicalPublication,
    *,
    family: str,
    output: Path,
    freeze: Path,
    freeze_sha256: str,
    roots: dict[str, Path],
    expected_queries: pd.DataFrame,
    validate_authority_and_qa: Callable[[], None],
    resource_ok: Callable[[], bool],
) -> dict:
    """Bind seed7 and T5 to the same freeze, then publish paired mean-loss analysis."""
    if family not in {"TPR", "LATENT"}:
        raise ValueError("canonical scalar or latent family required")
    flag = "replicate_scalar" if family == "TPR" else "replicate_latent"
    if not replicates.availability[flag] or any(
        seed7.availability[key] != replicates.availability[key] for key in ("d1", "density")
    ):
        raise ValueError("replication family unavailable or frozen pools differ")
    work = roots["work"].resolve(strict=True)
    if output.exists() or output.resolve() == work or not output.resolve().is_relative_to(work):
        raise ValueError("new companion three-seed output required")
    inputs = []
    for binding, stage in ((seed7, "T2" if family == "TPR" else "T4"), (replicates, "T5")):
        inputs.append(
            load_sealed_analysis_arms(
                binding,
                stage=stage,
                freeze=freeze,
                freeze_sha256=freeze_sha256,
                roots=roots,
                expected_queries=expected_queries,
                validate_authority_and_qa=validate_authority_and_qa,
                resource_ok=resource_ok,
            )
        )
    pool = "D1" if replicates.availability["d1"] else "D0"
    names = [f"{family}-{pool}-H{h}-C160" for h in (1, 8)]
    selected = {(name, 7): inputs[0][name, 7] for name in names}
    selected.update({(name, seed): inputs[1][name, seed] for name in names for seed in (13, 23)})
    averaged, summary = three_seed_losses(selected, family=family, pool=pool)

    def boundary() -> None:
        if not resource_ok():
            raise InterruptedError("PAUSED_RESOURCE: three-seed analysis")
        validate_authority_and_qa()
        if sha256(freeze) != freeze_sha256 or any(
            sha256(binding.publication) != binding.publication_sha256
            or sha256(binding.endpoints) != binding.endpoints_sha256
            for binding in (seed7, replicates)
        ):
            raise ValueError("three-seed input seals changed")

    boundary()
    output.mkdir(parents=True)
    losses = pd.concat(
        [frame.assign(loss_series=name) for name, frame in averaged.items()], ignore_index=True
    )
    losses.to_parquet(output / "MEAN_LOSSES_NOT_TTC_PREDICTIONS.parquet", index=False)
    paired_uncertainty(
        averaged, reference=names[0], output=output / "uncertainty", resource_check=boundary
    )
    boundary()
    report = {
        "schema": "simplex_t_sealed_three_seed_analysis_v1",
        "status": "THREE_SEED_LOSS_ANALYSIS_COMPLETE_NOT_T6",
        "family": family,
        "freeze_sha256": freeze_sha256,
        "summary": summary,
        "seed7_publication_sha256": seed7.publication_sha256,
        "replicates_publication_sha256": replicates.publication_sha256,
        "uncertainty_conditions_on_seeds": [7, 13, 23],
        "optimization_seeds_resampled": False,
        "files": {
            name: sha256(output / name)
            for name in (
                "MEAN_LOSSES_NOT_TTC_PREDICTIONS.parquet",
                "uncertainty/PAIRED_UNCERTAINTY.json",
            )
        },
        "scope": "REUSED_OLD_DEVELOPMENT_NOT_CONFIRMATORY",
        "optimizer_updates": 0,
        "holdout_opened": False,
    }
    write_new_json(output / "THREE_SEED_ANALYSIS.json", report)
    return report
