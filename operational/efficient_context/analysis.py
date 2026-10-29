"""Fragmented inference and paired, sequence-respecting development analysis."""

from __future__ import annotations

import io
from pathlib import Path

from .common import Campaign, atomic_bytes, atomic_json, digest, npz, read, release, spec


def controls(c: Campaign, seed: int) -> dict:
    """Read hash-bound historical predictions without rerunning any historic fit."""
    import pandas as pd

    rows = list(c.h16["historical_controls"].values())
    if seed != 7:
        seal = read(c.h16_path.parent / "ENDPOINTS.json")
        for row in seal["fits"]:
            if row["seed"] == seed:
                path = c.h16_path.parent / "publication" / row["key"] / "PREDICTIONS.parquet"
                pub = read(path.parent / "PUBLICATION.json")
                rows.append(
                    {
                        "prediction": str(path),
                        "prediction_sha256": pub["predictions_sha256"],
                        "endpoint": {"fit": {"name": "TPR-D1-H16-C160", "seed": seed}},
                    }
                )
    result = {}
    if seed == 7:
        publication = (
            c.historical / "artifacts/simplex_t/scientific_campaign/publication/T2_PREDICTIONS.json"
        )
        binding = read(
            c.historical
            / (
                "artifacts/simplex_t/scientific_campaign/T6/CHECKPOINTED_WORK/control/CLOSURE_INPUT_BINDING.json"
            )
        )["publications"]["T2"]
        if digest(publication) != binding["publication_sha256"]:
            raise ValueError("historical H1 publication changed")
        inventory = read(publication)
        for key, record in inventory["fits"].items():
            if "TPR-D1-H1-C160" in key and key.endswith("seed7"):
                path = publication.parent / record["path"]
                if digest(path) != record["sha256"]:
                    raise ValueError("historical H1 predictions changed")
                result.setdefault("H1", []).append(pd.read_parquet(path))
    for row in rows:
        fit = row["endpoint"]["fit"]
        if fit["seed"] != seed:
            continue
        path = Path(row["prediction"])
        if digest(path) != row["prediction_sha256"]:
            raise ValueError("historical prediction changed")
        label = "H16" if "H16" in fit["name"] else "H8"
        result.setdefault(label, []).append(pd.read_parquet(path))
    return {
        k: pd.concat(v, ignore_index=True).sort_values("sample_token").reset_index(drop=True)
        for k, v in result.items()
    }


def evaluate(c: Campaign, seed: int) -> None:
    """Require all three frozen endpoints before the first new OLD_DEV forward."""
    import numpy as np
    import pandas as pd
    import torch

    from e_jepa_ttc.efficient_context.analytical import transport_ewma
    from e_jepa_ttc.efficient_context.sparse_history import WIDE_SLOTS, WideSource
    from e_jepa_ttc.simplex_t.current_inputs import load_current_inputs
    from e_jepa_ttc.simplex_t.endpoint import load_endpoint
    from e_jepa_ttc.simplex_t.evaluation import prediction_frame
    from e_jepa_ttc.simplex_t.model import TemporalConfig

    p = c.freeze()
    pin = digest(c.out / "PROTOCOL.json")
    seal = read(c.out / f"ENDPOINTS_seed{seed}.json")
    if len(seal["fits"]) != 3 or seal["protocol_sha256"] != pin:
        raise ValueError("all endpoints must be frozen before OLD_DEV evaluation")
    for row in seal["fits"]:
        if digest(Path(row["checkpoint"])) != row["checkpoint_sha256"]:
            raise ValueError("sealed endpoint changed before evaluation")
    frames = controls(c, seed)
    s = c.sources()
    result = []
    ewma = []
    counts = []
    for row in seal["fits"]:
        c.require_resources()
        fold = row["fold"]
        parent = s.source(spec(s, fold), "outer_dev")
        if parent.identity_sha256 != c.h16["sources"][str(fold)]["dev_sha256"]:
            raise ValueError("historical OLD_DEV identity changed")
        source = WideSource(parent)
        model = load_endpoint(
            Path(row["checkpoint"]),
            TemporalConfig(**row["model"]),
            seed=seed,
            freeze_sha256=pin,
            train_source_sha256=p["sources"][str(fold)]["wide_sha256"],
            endpoint_sha256=row["checkpoint_sha256"],
        )
        chunks = []
        ewma_chunks = []
        for start in range(0, source.population, 128):
            c.require_resources()
            stop = min(start + 128, source.population)
            path = c.out / f"publication/seed{seed}/fold{fold}/batch_{start:05d}.npz"
            receipt = path.with_suffix(".json")
            x, t, v, e, y, m = source.gather(torch.arange(start, stop))
            if receipt.exists():
                r = read(receipt)
                if r["checkpoint_sha256"] != row["checkpoint_sha256"] or r[
                    "payload_sha256"
                ] != digest(path):
                    raise ValueError("inference fragment changed")
            else:
                with torch.inference_mode():
                    batch = {k: v.numpy() for k, v in model(x, t, v, e).items()}
                npz(path, **batch)
                atomic_json(
                    receipt,
                    {
                        "checkpoint_sha256": row["checkpoint_sha256"],
                        "payload_sha256": digest(path),
                        "start": start,
                        "stop": stop,
                    },
                )
            with np.load(path, allow_pickle=False) as z:
                chunks.append({k: z[k].copy() for k in z.files})
            if seed == 7:
                idx = parent.history[start:stop, -8:]
                valid = idx >= 0
                raw = parent.features[np.maximum(idx, 0), 8:11]
                phases = np.median(raw, axis=-1)
                anchors = parent.anchor_us[np.maximum(idx, 0)]
                ages = (anchors[:, -1, None] - anchors) / 1e6
                point, diag = transport_ewma(
                    torch.from_numpy(phases), torch.from_numpy(ages), torch.from_numpy(valid)
                )
                ewma_chunks.append(point.numpy())
                counts.append(diag)
        current = load_current_inputs(
            s.historical_root,
            fold,
            "outer_dev",
            ancestry_sha256=s.ancestry_sha256,
            allowed_sequences=s.allowed_sequences,
        )
        template = frames["H16"].loc[frames["H16"].outer_fold == fold]
        template = (
            template.set_index("sample_token").loc[current["metadata"].sample_token].reset_index()
        )
        if not np.array_equal(template.target_ttc, current["metadata"].target_ttc):
            raise ValueError("OLD_DEV target mapping differs")
        experts = np.load(s.folds[fold].path / "expert_ttc.npy", mmap_mode="r")[
            parent.history[:, -1]
        ]
        if not np.array_equal(experts, template[[f"expert{i}_ttc" for i in range(3)]].to_numpy()):
            raise ValueError("expert query mapping differs")
        metadata = template[
            [
                "sample_token",
                "sequence_id",
                "track_id",
                "target_ttc",
                "anchor_us",
                "history_span_us",
                "roi_age_us",
            ]
        ].copy()
        outputs = {k: np.concatenate([v[k] for v in chunks]) for k in chunks[0]}
        frame = prediction_frame(
            metadata,
            experts,
            outputs,
            parent.history[:, WIDE_SLOTS],
            arm="TPR-D1-H8WIDE-C160",
            seed=seed,
            fold=fold,
        )
        result.append(frame)
        if seed == 7:
            f = template.copy()
            from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase
            from e_jepa_ttc.simplex_t.phase import phase_to_ttc

            f["prediction_ttc_s"] = phase_to_ttc(
                torch.from_numpy(np.concatenate(ewma_chunks))
            ).numpy()
            f["loss"] = 10000 * np.abs(
                benchmark_phase(f.prediction_ttc_s.to_numpy())
                - benchmark_phase(f.target_ttc.to_numpy())
            )
            ewma.append(f)
        del parent, source, model, current, chunks
        release(s)
    frames["WIDE"] = (
        pd.concat(result, ignore_index=True).sort_values("sample_token").reset_index(drop=True)
    )
    for name, f in frames.items():
        if len(f) != 8192 or f.sample_token.duplicated().any():
            raise ValueError("complete OLD8192 population required")
        b = io.BytesIO()
        f.to_parquet(b, index=False)
        atomic_bytes(c.out / f"analysis/seed{seed}/{name}.parquet", b.getvalue())
    summary = compare(c, seed, frames)
    if seed == 7 and not (c.out / "EWMA_TRANSPORT_CV_RESULTS.json").exists():
        ewma_frame = (
            pd.concat(ewma, ignore_index=True).sort_values("sample_token").reset_index(drop=True)
        )
        from e_jepa_ttc.simplex_t.practical_comparison import paired_practical_comparison

        diag = paired_practical_comparison(ewma_frame, frames["H8"])
        diag.update(
            status="COMPLETE",
            optimizer_updates=0,
            anchor_scope="hash-bound original producer observation anchor and current ROI",
            physical_label_acquisition_independently_replayed=False,
            rejected_terms=sum(v["rejected_terms"] for v in counts),
            valid_terms=sum(v["valid_terms"] for v in counts),
            zero_phase_terms=sum(v["zero_phase_terms"] for v in counts),
            historical_ewma_modified=False,
        )
        b = io.BytesIO()
        ewma_frame.to_parquet(b, index=False)
        atomic_bytes(c.out / "analysis/EWMA_TRANSPORT_CV.parquet", b.getvalue())
        atomic_json(c.out / "EWMA_TRANSPORT_CV_RESULTS.json", diag)
    if seed == 7:
        atomic_json(c.out / "H8_WIDE_RESULTS.json", summary)
    else:
        atomic_json(c.out / f"H8_WIDE_RESULTS_seed{seed}.json", summary)
    print(f"WIDE_SEED{seed}_ANALYSIS_COMPLETE", flush=True)


def compare(c: Campaign, seed: int, frames: dict) -> dict:
    """Compute both historical interval methods, masses and prospective guardrails."""
    import numpy as np

    from e_jepa_ttc.evaluation.exact_sequence_v10 import exact_sequence_diagnostic
    from e_jepa_ttc.evaluation.stage63_65 import strict_macro_mass
    from e_jepa_ttc.simplex_t.practical_comparison import paired_practical_comparison
    from operational.simplex_t_closure.runtime import resumable_hierarchical_losses

    names = sorted(frames)
    base = frames["H8"]
    identity = ["sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc"]
    if any(not f[identity].equals(base[identity]) for f in frames.values()):
        raise ValueError("paired scientific populations differ")
    losses = np.column_stack([frames[k].loss.to_numpy(np.float64) for k in names])
    mass = strict_macro_mass(base.target_ttc.to_numpy(), base.sequence_id.to_numpy())
    sampled, receipt = resumable_hierarchical_losses(
        base.rename(columns={"target_ttc": "target_ttc_s"}),
        losses,
        c.out / f"analysis/seed{seed}/bootstrap",
        c.require_resources,
        draws_path=Path(c.h16["bootstrap_draws"]),
        binding={
            "protocol_sha256": digest(c.out / "PROTOCOL.json"),
            "endpoints_sha256": digest(c.out / f"ENDPOINTS_seed{seed}.json"),
        },
    )
    comparisons = {}
    for name in ("H8", "H16"):
        diag = paired_practical_comparison(frames["WIDE"], frames[name])
        delta = sampled[:, names.index("WIDE")] - sampled[:, names.index(name)]
        diag["hierarchical_ci95"] = np.percentile(delta, [2.5, 97.5]).tolist()
        diag["sequence_only"] = exact_sequence_diagnostic(
            np.asarray(list(diag["sequence_deltas"].values()))
        )
        diag["noninferiority_margin_mid"] = 2
        diag["noninferiority_supported"] = diag["hierarchical_ci95"][1] <= 2
        comparisons[name] = diag
    r = comparisons["H8"]
    gate = (
        r["point_delta"] <= -1
        and r["sequence_wins"] >= 6
        and r["candidate_finite_fraction"] == 1
        and r["weighted_sign_error_delta"] <= 0.005
        and r["crucial_bucket_mid_delta"] <= 5
    )
    npz(
        c.out / f"analysis/seed{seed}/PAIRED_LOSSES.npz",
        mass=mass,
        losses=losses,
        names=np.asarray(names),
    )
    return {
        "status": "COMPLETE",
        "seed": seed,
        "comparisons": comparisons,
        "replication_authorized": bool(gate) if seed == 7 else False,
        "replication_rule": "prospective_all_five_conditions",
        "bootstrap": receipt,
        "scientific_updates": 7500,
        "encoder_replicas": False,
        "head_replicas": True,
        "confirmation": False,
        "candidate_promoted": False,
    }
