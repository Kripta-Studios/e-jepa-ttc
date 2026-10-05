"""Six fixed Garl-H1/H8 fits and paired development evaluation, never partial scoring."""

from __future__ import annotations

import argparse
import gc
import io
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, cast

from .common import ROOT, Campaign, Lease, atomic_bytes, atomic_json, digest, npz, read
from .garl_recovery import completed_source, recover_head_transaction, verified_fragment

if TYPE_CHECKING:
    from e_jepa_ttc.efficient_context.garl_head import NativeHeadSource


@contextmanager
def numeric_binding() -> Iterator[None]:
    """Scope lambda_cost0 in this single worker; preserve canonical engine bytes."""
    from e_jepa_ttc.efficient_context.garl_head import objective
    from e_jepa_ttc.simplex_t import training

    original = training.training_loss
    training.training_loss = objective
    try:
        yield
    finally:
        training.training_loss = original


def source(c: Campaign, fold: int, role: str, length: int) -> NativeHeadSource:
    """Bind native fragments to the one shared H8 TRAIN-only normalizer."""
    import numpy as np
    import torch

    from e_jepa_ttc.efficient_context.garl_head import NativeHeadSource
    from e_jepa_ttc.simplex_t.cache import Normalizer
    from e_jepa_ttc.simplex_t.training import state_digest

    folder = c.out / f"garl/features/fold{fold}/{role}"
    receipt = completed_source(folder, digest(folder / "BINDING.json"))
    if receipt is None:
        raise ValueError("native head source is not completely published")
    with np.load(c.out / f"garl_heads/normalizers/fold{fold}.npz", allow_pickle=False) as z:
        normalizer = Normalizer(z["mean"], z["scale"], str(z["ids_hash"]))
    with np.load(folder / "SOURCE.npz", allow_pickle=False) as z:
        payload = {k: z[k].copy() for k in ("features", "times", "history", "truth", "mass")}
    identity = state_digest(
        {
            "source": receipt["source_sha256"],
            "length": length,
            "normalizer_mean": torch.from_numpy(normalizer.mean),
            "normalizer_scale": torch.from_numpy(normalizer.scale),
            "normalizer_ids": normalizer.consumed_ids_sha256,
        }
    )
    return NativeHeadSource(
        **payload, normalizer=normalizer, identity_sha256=identity, length=length
    )


def prepare(c: Campaign) -> dict:
    """Generate INNER-OOF inputs and seal all six fit recipes before training."""
    import numpy as np

    from e_jepa_ttc.efficient_context.garl_head import normalize

    from .garl_features import build

    path = c.out / "garl_heads/PROTOCOL.json"
    c.freeze()
    if path.exists():
        protocol = read(path)
        for file in protocol["files"]:
            if digest(Path(file["path"])) != file["sha256"]:
                raise ValueError("native temporal scientific implementation changed")
        if protocol["producer_endpoints_sha256"] != digest(c.out / "garl/ENDPOINTS.json"):
            raise ValueError("native producer endpoint seal changed")
        return protocol
    fits = []
    for fold in range(3):
        build(c, fold, "inner_oof")
        with np.load(
            c.out / f"garl/features/fold{fold}/inner_oof/SOURCE.npz", allow_pickle=False
        ) as z:
            stats = normalize(z["features"], z["history"])
        npz(
            c.out / f"garl_heads/normalizers/fold{fold}.npz",
            mean=stats.mean,
            scale=stats.scale,
            ids_hash=np.asarray(stats.consumed_ids_sha256),
        )
        for length in (1, 8):
            native = source(c, fold, "inner_oof", length)
            fits.append(
                {
                    "key": f"GARL_H{length}/fold{fold}/seed7",
                    "fold": fold,
                    "length": length,
                    "source_sha256": native.identity_sha256,
                    "updates": 2500,
                }
            )
    protocol = {
        "fits": fits,
        "scientific_updates": 15000,
        "seed": 7,
        "lambda_cost": 0,
        "feature_count": 3,
        "hidden": 160,
        "device": "CPU_FP32",
        "batch": 128,
        "producer_endpoints_sha256": digest(c.out / "garl/ENDPOINTS.json"),
        "files": [
            {"path": str(p), "sha256": digest(p)}
            for p in (
                Path(__file__),
                Path(__file__).with_name("garl_features.py"),
                Path(__file__).with_name("garl_recovery.py"),
                ROOT / "src/e_jepa_ttc/efficient_context/garl_head.py",
                ROOT / "src/e_jepa_ttc/simplex_t/training.py",
            )
        ],
    }
    atomic_json(path, protocol)
    return protocol


def train(c: Campaign) -> bool:
    """Use the canonical full-state2500-update CPU engine on native3-column inputs."""
    from e_jepa_ttc.efficient_context.garl_head import NativeHeadConfig
    from e_jepa_ttc.simplex_t.model import TemporalConfig
    from e_jepa_ttc.simplex_t.training import fit, learning_rate, load_checkpoint
    from e_jepa_ttc.simplex_t.work_budget import EngineWorkJournal, WorkBudget

    from .budget import require

    protocol = prepare(c)
    recover_head_transaction(c)
    pin = digest(c.out / "garl_heads/PROTOCOL.json")
    budget = WorkBudget(
        c.out / "garl_heads/PHYSICAL_WORK.json",
        {v["key"]: 2500 for v in protocol["fits"]},
        technical_reserved=0,
    )
    endpoints = []

    def allowed() -> bool:
        try:
            require(c)
            return True
        except InterruptedError:
            return False

    for fit_record in protocol["fits"]:
        require(c)
        native = source(c, fit_record["fold"], "inner_oof", fit_record["length"])
        if native.identity_sha256 != fit_record["source_sha256"]:
            raise ValueError("native head TRAIN identity differs from freeze")
        folder = c.out / "garl_heads/fits" / fit_record["key"]
        checkpoint = folder / "checkpoint_last.pt"
        with numeric_binding():
            result = fit(
                native,
                cast(TemporalConfig, NativeHeadConfig()),
                folder,
                seed=7,
                freeze_sha256=pin,
                resource_ok=allowed,
                resume=checkpoint.exists(),
                journal=EngineWorkJournal(budget, fit_record["key"]),
                device="cpu",
            )
        if result["status"] != "COMPLETED":
            atomic_json(c.out / "garl_heads/STATUS.json", result)
            return False
        state = load_checkpoint(checkpoint)
        npz(folder / "WEIGHTS.npz", **{k: v.numpy() for k, v in state["model"].items()})
        atomic_bytes(
            folder / "TRAINING_CURVE.csv",
            (
                "update,loss,learning_rate\n"
                + "".join(
                    f"{i + 1},{value:.17g},{learning_rate(i + 1):.17g}\n"
                    for i, value in enumerate(state["losses"])
                )
            ).encode(),
        )
        endpoints.append(
            {**fit_record, "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint)}
        )
        del native, state
        gc.collect()
    atomic_json(
        c.out / "garl_heads/ENDPOINTS.json",
        {"fits": endpoints, "protocol_sha256": pin, "all_six_frozen_before_evaluation": True},
    )
    return True


def evaluate(c: Campaign) -> None:
    """Keep every OLD8192 query; compare native, recalibration and eight-observation context."""
    import numpy as np
    import pandas as pd
    import torch

    from e_jepa_ttc.efficient_context.garl_head import model
    from e_jepa_ttc.evaluation.exact_sequence_v10 import exact_sequence_diagnostic
    from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase, strict_macro_mass
    from e_jepa_ttc.simplex_t.evaluation import prediction_frame
    from e_jepa_ttc.simplex_t.expert_phase import expert_benchmark_phase
    from e_jepa_ttc.simplex_t.practical_comparison import paired_practical_comparison
    from e_jepa_ttc.simplex_t.training import load_checkpoint
    from operational.simplex_t_closure.runtime import resumable_hierarchical_losses

    from .analysis import controls
    from .garl_features import build

    prepare(c)
    seal = read(c.out / "garl_heads/ENDPOINTS.json")
    if len(seal["fits"]) != 6 or not seal["all_six_frozen_before_evaluation"]:
        raise ValueError("all six head endpoints required before OLD_DEV forward")
    if digest(c.out / "garl_heads/PROTOCOL.json") != seal["protocol_sha256"]:
        raise ValueError("native temporal head freeze changed")
    frames = controls(c, 7)
    collected = {"GARL_H1": [], "GARL_H8": [], "GARL_NATIVE": []}
    for fold in range(3):
        build(c, fold, "outer_dev")
        folder = c.out / f"garl/features/fold{fold}/outer_dev"
        with np.load(folder / "SOURCE.npz", allow_pickle=False) as z:
            tokens, ttc = z["tokens"].copy(), z["native_ttc"].copy()
        template = frames["H8"].set_index("sample_token").loc[tokens].reset_index()
        metadata = template[
            [
                "sample_token",
                "sequence_id",
                "track_id",
                "target_ttc",
                "outer_fold",
                "anchor_us",
                "history_span_us",
                "roi_age_us",
            ]
        ].copy()
        native_frame = metadata.copy()
        native_frame["prediction_ttc_s"] = ttc
        native_frame["loss"] = 10000 * abs(
            expert_benchmark_phase(ttc) - benchmark_phase(metadata.target_ttc.to_numpy())
        )
        collected["GARL_NATIVE"].append(native_frame)
        for length in (1, 8):
            source_input = source(c, fold, "outer_dev", length)
            row = next(v for v in seal["fits"] if v["fold"] == fold and v["length"] == length)
            if digest(Path(row["checkpoint"])) != row["checkpoint_sha256"]:
                raise ValueError("sealed native head changed")
            state = load_checkpoint(Path(row["checkpoint"]))
            if state["identity"]["freeze"] != seal["protocol_sha256"]:
                raise ValueError("native head checkpoint parent differs")
            head = model().float().eval()
            head.load_state_dict(state["model"], strict=True)
            chunks = []
            for start in range(0, source_input.population, 128):
                c.require_resources()
                stop = min(start + 128, source_input.population)
                fragment = c.out / f"garl_heads/publication/fold{fold}/H{length}_{start:05d}.npz"
                receipt = fragment.with_suffix(".json")
                fragment_binding = {
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "source_identity_sha256": source_input.identity_sha256,
                    "head_protocol_sha256": seal["protocol_sha256"],
                    "start": start,
                    "stop": stop,
                }
                if not verified_fragment(fragment, fragment_binding):
                    x, t, v, e, _, _ = source_input.gather(torch.arange(start, stop))
                    with torch.inference_mode():
                        output = {k: value.numpy() for k, value in head(x, t, v, e).items()}
                    npz(fragment, **output)
                    atomic_json(
                        receipt,
                        {**fragment_binding, "sha256": digest(fragment)},
                    )
                with np.load(fragment, allow_pickle=False) as z:
                    chunks.append({k: z[k].copy() for k in z.files})
            output = {k: np.concatenate([v[k] for v in chunks]) for k in chunks[0]}
            frame = prediction_frame(
                metadata,
                np.repeat(ttc[:, None], 3, axis=1),
                output,
                source_input.history[:, -length:],
                arm=f"GARL_H{length}",
                seed=7,
                fold=fold,
            )
            collected[f"GARL_H{length}"].append(frame)
    for name, parts in collected.items():
        frames[name] = pd.concat(parts).sort_values("sample_token").reset_index(drop=True)
        if len(frames[name]) != 8192 or frames[name].sample_token.duplicated().any():
            raise ValueError("complete native OLD8192 population required")
        stream = io.BytesIO()
        frames[name].to_parquet(stream, index=False)
        atomic_bytes(c.out / f"garl_heads/analysis/{name}.parquet", stream.getvalue())
    names = ["GARL_NATIVE", "GARL_H1", "GARL_H8", "H8", "H16"]
    identity = ["sample_token", "sequence_id", "track_id", "outer_fold", "target_ttc"]
    if any(not frames[k][identity].equals(frames["H8"][identity]) for k in names):
        raise ValueError("native paired scientific query identities differ")
    for name in ("H8", "H16"):
        stream = io.BytesIO()
        frames[name].to_parquet(stream, index=False)
        atomic_bytes(c.out / f"garl_heads/analysis/{name}.parquet", stream.getvalue())
    losses = np.column_stack([frames[k].loss.to_numpy(np.float64) for k in names])
    mass = strict_macro_mass(
        frames["H8"].target_ttc.to_numpy(), frames["H8"].sequence_id.to_numpy()
    )
    npz(
        c.out / "garl_heads/analysis/PAIRED_LOSSES.npz",
        losses=losses,
        mass=mass,
        names=np.asarray(names),
    )
    draws, receipt = resumable_hierarchical_losses(
        frames["H8"].rename(columns={"target_ttc": "target_ttc_s"}),
        losses,
        c.out / "garl_heads/analysis/bootstrap",
        c.require_resources,
        draws_path=Path(c.h16["bootstrap_draws"]),
        binding={"endpoints_sha256": digest(c.out / "garl_heads/ENDPOINTS.json")},
    )
    comparisons = {}
    for candidate, reference in (
        ("GARL_H8", "GARL_H1"),
        ("GARL_H1", "GARL_NATIVE"),
        ("GARL_H8", "H8"),
    ):
        result = paired_practical_comparison(frames[candidate], frames[reference])
        delta = draws[:, names.index(candidate)] - draws[:, names.index(reference)]
        result["hierarchical_ci95"] = np.percentile(delta, [2.5, 97.5]).tolist()
        result["sequence_only"] = exact_sequence_diagnostic(
            np.asarray(list(result["sequence_deltas"].values()))
        )
        comparisons[candidate + "-" + reference] = result
    atomic_json(
        c.out / "GARL_CONTEXT_RESULTS.json",
        {
            "status": "COMPLETE",
            "comparisons": comparisons,
            "bootstrap": receipt,
            "scientific_head_updates": 15000,
            "confirmation": False,
            "same_window_encoding_or_privileges": False,
        },
    )


def main() -> int:
    """Resume source-native heads only after the native producer family is sealed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "train", "evaluate", "all"))
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/campaign/efficient_context_v1.json"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    c = Campaign(args.protocol)
    with Lease(c.out):
        if args.action == "prepare":
            prepare(c)
        elif args.action == "train":
            return 0 if train(c) else 3
        elif args.action == "evaluate":
            evaluate(c)
        elif train(c):
            evaluate(c)
        else:
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
