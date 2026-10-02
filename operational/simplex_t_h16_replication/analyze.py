"""Publish H16 only after all six endpoints, then apply fixed paired analyses."""

from __future__ import annotations

import gc
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from common import (
    ARM,
    OUT,
    Lease,
    Resources,
    atomic_bytes,
    atomic_json,
    csv,
    digest,
    historical_spec,
    npz,
    protocol,
    publish_json,
    record,
    sources,
    validate_pins,
)


def parquet(frame: pd.DataFrame, path: Path) -> None:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    atomic_bytes(path, buffer.getvalue())


def historical_frames(p: dict) -> dict:
    import pandas as pd

    frames = {}
    for row in p["historical_controls"].values():
        r = row["endpoint"]["fit"]
        path = Path(row["prediction"])
        if digest(path) != row["prediction_sha256"]:
            raise ValueError("historical control prediction changed")
        frames.setdefault(f"H{16 if r['name'] == ARM else 8}@{r['seed']}", []).append(
            pd.read_parquet(path)
        )
    result = {}
    for key, parts in frames.items():
        f = pd.concat(parts, ignore_index=True).sort_values("sample_token").reset_index(drop=True)
        if len(f) != 8192 or f.sample_token.duplicated().any() or f.sequence_id.nunique() != 9:
            raise ValueError("historical full-cohort control incomplete")
        result[key] = f
    return result


def publish(p: dict, pin: str, resources: Resources) -> dict:
    import numpy as np
    import pandas as pd
    import torch
    from e_jepa_ttc.simplex_t.arms import resolve_arm
    from e_jepa_ttc.simplex_t.current_inputs import load_current_inputs
    from e_jepa_ttc.simplex_t.endpoint import load_endpoint
    from e_jepa_ttc.simplex_t.evaluation import prediction_frame

    seal = record(OUT / "ENDPOINTS.json")
    if (
        seal["protocol_sha256"] != pin
        or len(seal["fits"]) != 6
        or not seal["all_six_frozen_before_evaluation"]
    ):
        raise ValueError("all six independent endpoints must be sealed before evaluation")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    # Validate EVERY model identity before the first dev source or model forward.
    from e_jepa_ttc.simplex_t.model import TemporalConfig

    for row in seal["fits"]:
        resources.check()
        model = load_endpoint(
            OUT / row["checkpoint"],
            TemporalConfig(**row["model"]),
            seed=row["seed"],
            freeze_sha256=pin,
            train_source_sha256=row["train_source_sha256"],
            endpoint_sha256=row["checkpoint_sha256"],
        )
        del model
    frames = historical_frames(p)
    s = sources(p)
    for row in seal["fits"]:
        resources.check()
        validate_pins(p, full=False)
        spec = historical_spec(s, row["fold"])
        source = s.source(spec, "outer_dev")
        if source.identity_sha256 != p["sources"][str(row["fold"])]["dev_sha256"]:
            raise ValueError("OLD_DEV H16 binding changed")
        binding = resolve_arm(spec, s.graph)
        model = load_endpoint(
            OUT / row["checkpoint"],
            binding.model,
            seed=row["seed"],
            freeze_sha256=pin,
            train_source_sha256=row["train_source_sha256"],
            endpoint_sha256=row["checkpoint_sha256"],
        )
        chunks = []
        root = OUT / "publication" / row["key"]
        root.mkdir(parents=True, exist_ok=True)
        for start in range(0, source.population, 128):
            resources.check()
            stop = min(start + 128, source.population)
            receipt = root / f"batch_{start:05d}.json"
            payload = root / f"batch_{start:05d}.npz"
            input_path = OUT / "cached_inputs" / f"H16_fold{row['fold']}_batch{start:05d}.npz"
            x, timing, valid, experts, truth, mass = source.gather(torch.arange(start, stop))
            npz(
                input_path,
                dict(
                    features=x.numpy(),
                    times=timing.numpy(),
                    valid=valid.numpy(),
                    experts=experts.numpy(),
                    target_phase=truth.numpy(),
                    mass=mass.numpy(),
                ),
            )
            if receipt.exists():
                r = record(receipt)
                if r != dict(
                    start=start,
                    stop=stop,
                    input_sha256=digest(input_path),
                    endpoint_sha256=row["checkpoint_sha256"],
                    payload_sha256=digest(payload),
                ):
                    raise ValueError("inference fragment changed")
            else:
                with torch.inference_mode():
                    batch = {k: v.numpy() for k, v in model(x, timing, valid, experts).items()}
                if not all(np.isfinite(v).all() for v in batch.values()):
                    raise ValueError("nonfinite H16 output")
                npz(payload, batch)
                publish_json(
                    receipt,
                    dict(
                        start=start,
                        stop=stop,
                        input_sha256=digest(input_path),
                        endpoint_sha256=row["checkpoint_sha256"],
                        payload_sha256=digest(payload),
                    ),
                )
                print(
                    json.dumps(
                        dict(
                            status="INFERENCE_FRAGMENT_COMMITTED",
                            seed=row["seed"],
                            fold=row["fold"],
                            queries=stop,
                        )
                    ),
                    flush=True,
                )
            with np.load(payload, allow_pickle=False) as a:
                chunks.append({k: a[k].copy() for k in a.files})
        outputs = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        template = frames["H16@7"].loc[frames["H16@7"].outer_fold == row["fold"]]
        # Load canonical fold order and raw experts independently from acknowledged tables.
        current = load_current_inputs(
            s.historical_root,
            row["fold"],
            "outer_dev",
            ancestry_sha256=s.ancestry_sha256,
            allowed_sequences=s.allowed_sequences,
        )
        template = (
            template.set_index("sample_token").loc[current["metadata"].sample_token].reset_index()
        )
        if not np.array_equal(
            template.target_ttc.to_numpy(), current["metadata"].target_ttc.to_numpy()
        ):
            raise ValueError("control targets differ from independently verified cohort")
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
        experts = current["arrays"]["expert_ttc"]
        frame = prediction_frame(
            metadata,
            experts,
            outputs,
            source.history[:, -16:],
            arm=ARM,
            seed=row["seed"],
            fold=row["fold"],
        )
        frame["source_sha256"] = source.identity_sha256
        parquet(frame, root / "PREDICTIONS.parquet")
        csv(root / "PREDICTIONS.csv", frame)
        publish_json(
            root / "PUBLICATION.json",
            dict(
                protocol_sha256=pin,
                endpoints_seal_sha256=digest(OUT / "ENDPOINTS.json"),
                queries=len(frame),
                predictions_sha256=digest(root / "PREDICTIONS.parquet"),
                source_sha256=source.identity_sha256,
            ),
        )
        frames.setdefault(f"H16@{row['seed']}", []).append(frame)
        del source, model, chunks, outputs
        s.release()
        gc.collect()
    for seed in (13, 23):
        frames[f"H16@{seed}"] = (
            pd.concat(frames[f"H16@{seed}"], ignore_index=True)
            .sort_values("sample_token")
            .reset_index(drop=True)
        )
    s.release()
    # Published compact weights for all new and reused heads, no optimizer execution.
    from e_jepa_ttc.simplex_t.training import load_checkpoint

    controls = [(k, Path(v["checkpoint"])) for k, v in p["historical_controls"].items()]
    for key, path in controls + [(v["key"], OUT / v["checkpoint"]) for v in seal["fits"]]:
        state = load_checkpoint(path)
        npz(
            OUT / "weights" / (key.replace("/", "__") + ".npz"),
            {k: v.numpy() for k, v in state["model"].items()},
        )
    return frames


def analysis(p: dict, pin: str, frames: dict, resources: Resources) -> None:
    import numpy as np
    import pandas as pd
    from e_jepa_ttc.evaluation.exact_sequence_v10 import exact_sequence_diagnostic
    from e_jepa_ttc.evaluation.stage63_65 import strict_macro_mass
    from e_jepa_ttc.simplex_t.practical_comparison import paired_practical_comparison
    from runtime import resumable_hierarchical_losses

    base = frames["H8@7"]
    identity = ["sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc"]
    for f in frames.values():
        if not f[identity].equals(base[identity]):
            raise ValueError("paired query identities differ")
    for seeds, label in (((13, 23), "new13_23"), ((7, 13, 23), "all7_13_23")):
        for h in (8, 16):
            f = base.copy()
            f["loss"] = np.mean(
                [frames[f"H{h}@{seed}"].loss.to_numpy(np.float64) for seed in seeds], axis=0
            )
            frames[f"H{h}@{label}"] = f
    names = sorted(frames)
    losses = np.column_stack([frames[k].loss.to_numpy(np.float64) for k in names])
    mass = strict_macro_mass(base.target_ttc.to_numpy(), base.sequence_id.to_numpy())
    sampled, bootstrap = resumable_hierarchical_losses(
        base.rename(columns={"target_ttc": "target_ttc_s"}),
        losses,
        OUT / "analysis/bootstrap",
        resources.check,
        draws_path=Path(p["bootstrap_draws"]),
        binding=dict(protocol_sha256=pin, endpoints_sha256=digest(OUT / "ENDPOINTS.json")),
        chunk_size=256,
    )
    scores = mass @ losses
    comparisons, rows = {}, []
    for label in ("7", "13", "23", "new13_23", "all7_13_23"):
        a, b = names.index("H16@" + label), names.index("H8@" + label)
        delta = losses[:, a] - losses[:, b]
        perseq = []
        for seq in sorted(base.sequence_id.unique()):
            mask = base.sequence_id.to_numpy() == seq
            perseq.append(float((mass[mask] / mass[mask].sum()) @ delta[mask]))
            rows.append(
                dict(
                    comparison=label,
                    axis="sequence",
                    group=seq,
                    delta=perseq[-1],
                    queries=int(mask.sum()),
                    global_mass=float(mass[mask].sum()),
                )
            )
        for fold in range(3):
            mask = base.outer_fold.to_numpy() == fold
            rows.append(
                dict(
                    comparison=label,
                    axis="fold",
                    group=str(fold),
                    delta=float((mass[mask] / mass[mask].sum()) @ delta[mask]),
                    queries=int(mask.sum()),
                    global_mass=float(mass[mask].sum()),
                )
            )
        if label in {"7", "13", "23"}:
            practical = paired_practical_comparison(frames["H16@" + label], frames["H8@" + label])
        else:
            selected = (13, 23) if label == "new13_23" else (7, 13, 23)
            practical = {
                k: float(np.mean([comparisons[str(seed)]["practical"][k] for seed in selected]))
                for k in (
                    "weighted_sign_error_delta",
                    "crucial_bucket_mid_delta",
                    "candidate_finite_fraction",
                )
            }
        hi = np.percentile(sampled[:, a] - sampled[:, b], [2.5, 97.5]).tolist()
        seq = exact_sequence_diagnostic(np.asarray(perseq))
        comparisons[label] = dict(
            h16_mid=float(scores[a]),
            h8_mid=float(scores[b]),
            paired_loss_delta=float(mass @ delta),
            hierarchical_ci95=hi,
            sequence_only=seq,
            practical=practical,
            guardrails_pass=(
                practical["candidate_finite_fraction"] == 1.0
                and practical["weighted_sign_error_delta"] <= 0.005
                and practical["crucial_bucket_mid_delta"] <= 3
            ),
            seed7_previously_observed=label == "7",
            averaged_losses_not_ttc=label not in {"7", "13", "23"},
        )
    from e_jepa_ttc.simplex_t.diagnostic_summary import summarize_diagnostics
    from e_jepa_ttc.simplex_t.temporal_diagnostics import sampled_temporal_diagnostics

    diagnostic = []
    temporal = []
    for label in ("7", "13", "23"):
        for h in (8, 16):
            f = frames[f"H{h}@{label}"]
            diagnostic.append(summarize_diagnostics(f, history_length=h))
            _, t = sampled_temporal_diagnostics(f)
            temporal.append(t)
    csv(OUT / "analysis/DIAGNOSTICS.csv", pd.concat(diagnostic, ignore_index=True))
    csv(OUT / "analysis/TEMPORAL_DIAGNOSTICS.csv", pd.concat(temporal, ignore_index=True))
    csv(OUT / "analysis/PAIRED_SEQUENCE_FOLD.csv", pd.DataFrame(rows))
    csv(
        OUT / "analysis/METRICS.csv",
        pd.DataFrame(
            [
                dict(comparison=k, **{f: r[f] for f in ("h16_mid", "h8_mid", "paired_loss_delta")})
                for k, r in comparisons.items()
            ]
        ),
    )
    arrays = dict(
        mass=mass, **{k.replace("@", "_"): frames[k].loss.to_numpy(np.float64) for k in names}
    )
    npz(OUT / "analysis/PAIRED_LOSSES.npz", arrays)
    parquet(base[identity], OUT / "analysis/COHORT.parquet")
    allc = comparisons["all7_13_23"]
    if allc["paired_loss_delta"] >= 0:
        conclusion = "NEGATIVE_DESCRIPTIVE"
    elif (
        allc["hierarchical_ci95"][1] < 0
        and allc["sequence_only"]["ci95"][1] < 0
        and all(comparisons[k]["paired_loss_delta"] < 0 for k in ("13", "23"))
        and all(c["guardrails_pass"] for c in comparisons.values())
    ):
        conclusion = "POSITIVE_LOCAL_OLD_DEV"
    else:
        conclusion = "UNCERTAIN"
    publish_json(
        OUT / "analysis/RESULTS.json",
        dict(
            protocol_sha256=pin,
            comparisons=comparisons,
            bootstrap=bootstrap,
            conclusion=conclusion,
            independent_sequences=9,
            ensemble=False,
            confirmation_opened=False,
            scientific_updates=15000,
        ),
    )
    publish_json(
        OUT / "NEXT_DECISION.json",
        dict(
            status="CLOSED_INDEPENDENT_H16_REPLICATION",
            conclusion=conclusion,
            historical_candidate="TPR-D1-H8-C160",
            candidate_replaced=False,
            future_optimizer_updates_authorized=0,
            stage76_opened=False,
            push=False,
            decision="Deliver all results; no rescue fit, sweep or automatic promotion.",
        ),
    )
    lines = [
        "# Réplica independiente H16 — resultado local",
        "",
        f"Conclusión preregistrada: **{conclusion}**. "
        "Seis endpoints2500, 15000 updates científicos.",
        "",
        "| Comparación H16−H8 | H16 MiD | H8 MiD | Delta | CI95 jerárquico | CI95 secuencias |",
        "|---|---:|---:|---:|---|---|",
    ]
    for label, r in comparisons.items():
        lines.append(
            f"| {label} | {r['h16_mid']:.9f} | {r['h8_mid']:.9f} | "
            f"{r['paired_loss_delta']:.9f} | {r['hierarchical_ci95']} | "
            f"{r['sequence_only']['ci95']} |"
        )
    lines += [
        "",
        "Seed7 es exploratoria ya observada; seeds13/23 son nuevas. "
        "Los resúmenes promedian pérdidas emparejadas, no TTC. Nueve escenas "
        "independientes; replicación de cabezas, no de expertos.",
        "",
        "Los guardrails, diferencias por secuencia/fold, errores de signo, bucket crucial, "
        "cobertura, capping y escapes están en analysis/. No se demuestra mecanismo "
        "cronológico, reacción AEB, tracking persistente ni incertidumbre calibrada.",
        "",
        "La cabeza usa inputs normalizados cacheados. No se mide aquí el coste end-to-end "
        "ni se promete regeneración desde eventos crudos. El historial retrospectivo "
        "conserva el ROI actual. H8 continúa siendo el candidato histórico. "
        "No se abrieron holdouts ni Stage76; no se hizo push.",
        "",
        "Protocolo: [PROTOCOL.md](PROTOCOL.md). Resultados completos: "
        "[RESULTS.json](analysis/RESULTS.json). "
        "Contabilidad: [ACCOUNTING.json](ACCOUNTING.json).",
    ]
    atomic_bytes(OUT / "FINAL_REPORT_H16.md", ("\n".join(lines) + "\n").encode())
    atomic_json(
        OUT / "STATUS.json",
        dict(
            status="ANALYSIS_COMPLETE_DELIVERY_PENDING",
            endpoint_count=6,
            scientific_saved_updates=15000,
            bootstrap_fragments=33,
            accepted_draws=8192,
            conclusion=conclusion,
        ),
    )


def main() -> None:
    p, pin = protocol()
    resources = Resources()
    resources.check()
    validate_pins(p, full=True)
    with Lease():
        frames = publish(p, pin, resources)
        analysis(p, pin, frames, resources)


if __name__ == "__main__":
    main()
