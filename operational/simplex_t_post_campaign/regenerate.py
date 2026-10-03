"""Regenerate included loss contrasts and timing quantiles; optionally replay nine frozen heads."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def sha(path: Path) -> str:
    """Hash full payload bytes."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path: Path) -> dict:
    """Read an explicitly bound receipt."""
    return json.loads(path.read_bytes())


def verify(root: Path, output: Path, *, heads: bool, create_reference: bool) -> None:
    """Verify only included cached sources, with zero optimizer updates and no raw calls."""
    manifest_path = root / "CONTENT_MANIFEST.json"
    if manifest_path.exists():
        for name, row in record(manifest_path)["members"].items():
            path = (root / name).resolve(strict=True)
            if not path.is_relative_to(root) or sha(path) != row["sha256"]:
                raise ValueError(f"included file differs: {name}")
    sys.path[:0] = [str(root / "provenance"), str(root / "provenance/src")]
    guards = __import__("runpy").run_path(str(root / "resource_guard.py"))
    guard = guards["resource_guard"]
    guard(root)
    import numpy as np
    import pandas as pd

    from e_jepa_ttc.evaluation.stage63_65 import strict_macro_mass

    index = record(root / "INPUT_INDEX.json")
    grouped = {}
    for row in index["predictions"]:
        if sha(root / row["included_path"]) != row["sha256"]:
            raise ValueError("included prediction payload SHA differs")
        grouped.setdefault(row["model"], []).append(pd.read_parquet(root / row["included_path"]))
    frames = {
        k: pd.concat(v, ignore_index=True).sort_values("sample_token").reset_index(drop=True)
        for k, v in grouped.items()
    }
    base = frames["H8@7"]
    if (
        len(frames) != 12
        or len(base) != 8192
        or base.sample_token.duplicated().any()
        or base.sequence_id.nunique() != 9
    ):
        raise ValueError("twelve complete nine-sequence cohorts required")
    mass = strict_macro_mass(base.target_ttc.to_numpy(), base.sequence_id.to_numpy())
    scores = {}
    diagnostics_rows = pd.read_csv(root / "results/MODEL_DIAGNOSTICS.csv")
    for label, f in frames.items():
        if not f[["sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc"]].equals(
            base[["sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc"]]
        ):
            raise ValueError("included query pairing differs")
        loss = 10000 * np.abs(f.prediction_phase.to_numpy() - f.target_phase.to_numpy())
        if not np.array_equal(loss, f.loss.to_numpy()):
            raise ValueError("canonical phase losses differ")
        scores[label] = float(mass @ loss)
    published = record(root / "results/PREDICTION_INVENTORY.json")["scores"]
    score_error = max(abs(scores[label] - published[label]) for label in scores)
    pairs = pd.read_csv(root / "results/PAIRED_STRATA.csv")
    errors = []
    target = base.target_ttc.to_numpy()
    buckets = np.select(
        [target > 6, target > 3, target > 0],
        ["positive_6_10", "positive_3_6", "positive_0_3"],
        default="negative_10_0",
    )
    for values in pairs.to_dict(orient="records"):
        row = SimpleNamespace(**values)
        if row.candidate.startswith("H16_"):
            seeds = (13, 23) if row.candidate.startswith("H16_NEW") else (7, 13, 23)
            delta = np.mean(
                [
                    frames[f"H16@{s}"].loss.to_numpy() - frames[f"H8@{s}"].loss.to_numpy()
                    for s in seeds
                ],
                axis=0,
            )
        else:
            delta = frames[row.candidate].loss.to_numpy() - frames[row.reference].loss.to_numpy()
        if row.group == "global":
            mask = np.ones(len(base), bool)
        elif row.group == "sequence":
            mask = (base.sequence_id == row.key).to_numpy()
        elif row.group == "fold":
            mask = (base.outer_fold == int(row.key)).to_numpy()
        elif row.group == "bucket":
            mask = buckets == row.key
        else:
            seq, bucket = row.key.split("/")
            mask = (base.sequence_id == seq).to_numpy() & (buckets == bucket)
        if int(mask.sum()) != row.queries:
            raise ValueError("stratum support differs")
        errors.append(abs(float(mass[mask] @ delta[mask] / mass[mask].sum()) - row.delta_MiD))
    # Rebuild every all-query episode; annotation IDs do not establish online tracking.
    work = base.assign(_position=np.arange(len(base))).sort_values(
        ["sequence_id", "track_id", "anchor_us", "sample_token"]
    )
    episode_lookup = {}
    for (sequence, track), group in work.groupby(["sequence_id", "track_id"], sort=True):
        numbers = (group.anchor_us.diff().fillna(0) > 100000).cumsum()
        for episode, piece in group.groupby(numbers, sort=True):
            episode_lookup[(str(sequence), str(track), int(episode))] = piece._position.to_numpy(
                int
            )
    archived_episodes = pd.read_csv(
        root / "results/ALL_ERROR_EPISODES.csv", dtype={"sequence_id": str, "track_id": str}
    )
    episode_errors = []
    deltas = {}
    for candidate, reference in (
        archived_episodes[["candidate", "reference"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        if candidate.startswith("H16_"):
            seeds = (13, 23) if candidate.startswith("H16_NEW") else (7, 13, 23)
            deltas[(candidate, reference)] = np.mean(
                [
                    frames[f"H16@{s}"].loss.to_numpy() - frames[f"H8@{s}"].loss.to_numpy()
                    for s in seeds
                ],
                axis=0,
            )
        else:
            deltas[(candidate, reference)] = (
                frames[candidate].loss.to_numpy() - frames[reference].loss.to_numpy()
            )
    for values in archived_episodes.to_dict(orient="records"):
        row = SimpleNamespace(**values)
        positions = episode_lookup[(row.sequence_id, row.track_id, row.episode)]
        if len(positions) != row.queries:
            raise ValueError("episode support differs")
        delta = deltas[(row.candidate, row.reference)]
        total = mass[positions].sum()
        episode_errors.append(
            abs(float(mass[positions] @ delta[positions] / total) - row.delta_MiD)
        )
    # Full descriptive diagnostics: rebuild each registered stratum from its physical frame.
    diagnostics_errors = []
    for values in diagnostics_rows.to_dict(orient="records"):
        row = SimpleNamespace(**values)
        f = frames[row.model]
        if row.group == "global":
            mask = np.ones(len(base), bool)
        elif row.group == "sequence":
            mask = (base.sequence_id == row.key).to_numpy()
        elif row.group == "fold":
            mask = (base.outer_fold == int(row.key)).to_numpy()
        elif row.group == "bucket":
            mask = buckets == row.key
        else:
            seq, bucket = row.key.split("/")
            mask = (base.sequence_id == seq).to_numpy() & (buckets == bucket)
        wt = mass[mask] / mass[mask].sum()
        if row.queries != int(mask.sum()):
            raise ValueError("diagnostic stratum support differs")
        diagnostics_errors.append(abs(float(wt @ f.loss.to_numpy()[mask]) - row.MiD))
        for field in (
            "wrong_sign",
            "phase_support_saturation",
            "finite_ttc_cap",
            "phase_interval_covers",
            "phase_interval_width",
            "cold_start",
            "escape_gain",
            "escape_harm",
        ):
            diagnostics_errors.append(
                abs(float(wt @ f[field].to_numpy()[mask]) - getattr(row, field))
            )
    timing = pd.read_csv(root / "results/MATCHED_HEAD_COST.csv")
    timing_error = []
    for values in timing.to_dict(orient="records"):
        row = SimpleNamespace(**values)
        raw = [
            v
            for frag in range(20)
            for v in record(
                root / f"results/timings/b{row.block - 1}_f{frag:02d}_{row.model}.json"
            )["raw_ms"]
        ]
        timing_error += [
            abs(float(np.median(raw)) - row.p50_ms),
            abs(float(np.quantile(raw, 0.95)) - row.p95_ms),
        ]
    numerical_error = None
    if heads:
        guard(root)
        import torch

        from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
        from e_jepa_ttc.simplex_t.phase import phase_to_ttc
        from operational.simplex_t_cost_context.model import build_model

        torch.set_num_threads(4)
        torch.set_num_interop_threads(2)
        torch.use_deterministic_algorithms(True)
        arrays = {}
        with torch.inference_mode():
            for label, row in index["profile"]["models"].items():
                guard(root)
                for kind in ("weights", "inputs"):
                    if sha(root / row[kind + "_path"]) != row[kind + "_sha256"]:
                        raise ValueError("prepared head payload SHA differs")
                c = row["constructor"]
                model = (
                    TemporalRefiner(TemporalConfig(**c["config"]))
                    if c["kind"] == "historical_temporal"
                    else build_model(c["arm"])
                )
                model.cpu().float().eval()
                with np.load(root / row["weights_path"], allow_pickle=False) as z:
                    model.load_state_dict(
                        {k: torch.from_numpy(z[k].copy()) for k in z.files}, strict=True
                    )
                with np.load(root / row["inputs_path"], allow_pickle=False) as z:
                    if set(z.files) != {"features", "times", "valid", "experts"}:
                        raise ValueError("only target-free prepared inputs admitted")
                    xs = tuple(
                        torch.from_numpy(z[k].copy())
                        for k in ("features", "times", "valid", "experts")
                    )
                y = model(*xs)
                for key, value in y.items():
                    arrays[label + "__" + key] = value.numpy()
                arrays[label + "__ttc"] = phase_to_ttc(y["point_phase"].to(torch.float64)).numpy()
                if not all(np.isfinite(value).all() for value in arrays.values()):
                    raise ValueError("nonfinite frozen-head replay")
        reference = root / "HEAD_REFERENCE.npz"
        if create_reference:
            if reference.exists():
                raise ValueError("reference already exists; cannot overwrite")
            np.savez_compressed(reference, **arrays)
        else:
            with np.load(reference, allow_pickle=False) as z:
                if set(z.files) != set(arrays):
                    raise ValueError("reference fields differ")
                numerical_error = max(float(np.max(np.abs(arrays[k] - z[k]))) for k in z.files)
            if numerical_error > 1e-6:
                raise ValueError("frozen-head inference differs")
    result = dict(
        status="VERIFIED",
        models=len(scores),
        queries_per_model=len(base),
        query_prediction_rows=sum(len(f) for f in frames.values()),
        scores=scores,
        max_score_difference=score_error,
        max_paired_stratum_difference=max(errors),
        max_episode_difference=max(episode_errors),
        episode_rows=len(archived_episodes),
        singleton_episodes=sum(len(v) == 1 for v in episode_lookup.values()),
        episode_count=len(episode_lookup),
        max_diagnostic_difference=max(diagnostics_errors),
        max_timing_quantile_difference=max(timing_error),
        matched_measurements=13500,
        head_replay_models=9 if heads else 0,
        head_replay_batch=64 if heads else None,
        maximum_numeric_head_difference=numerical_error,
        optimizer_updates=0,
        raw_reconstruction=False,
        experts_executed=False,
        scope="published_losses_and_prepared_heads",
        timing_samples_recollected=False,
    )
    if (
        max(
            score_error,
            max(errors),
            max(timing_error),
            max(episode_errors),
            max(diagnostics_errors),
        )
        > 1e-9
    ):
        raise ValueError("included result reconciliation differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "scores"}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--heads", action="store_true")
    parser.add_argument(
        "--create-reference",
        action="store_true",
        help="Builder only; never overwrite a numerical reference",
    )
    args = parser.parse_args()
    if args.create_reference and not args.heads:
        parser.error("--create-reference requires --heads")
    verify(
        args.root.resolve(strict=True),
        args.output,
        heads=args.heads,
        create_reference=args.create_reference,
    )


if __name__ == "__main__":
    main()
