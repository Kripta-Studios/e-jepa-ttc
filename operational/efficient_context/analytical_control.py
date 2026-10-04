"""Independent zero-update EWMA control from pinned C: observations and anchors."""

from __future__ import annotations

import argparse
import gc
import io
from pathlib import Path

from .common import ROOT, Campaign, Lease, atomic_bytes, atomic_json, digest, npz, read


def run(c: Campaign) -> None:
    """Do not require Garl labels/raw to analyze already bound original observations."""
    import numpy as np
    import pandas as pd
    import torch

    from e_jepa_ttc.efficient_context.analytical import transport_ewma
    from e_jepa_ttc.evaluation.exact_sequence_v10 import exact_sequence_diagnostic
    from e_jepa_ttc.evaluation.stage63_65 import benchmark_phase, strict_macro_mass
    from e_jepa_ttc.simplex_t.context_sources import load_context_sources
    from e_jepa_ttc.simplex_t.phase import phase_to_ttc
    from e_jepa_ttc.simplex_t.practical_comparison import paired_practical_comparison
    from operational.simplex_t_closure.runtime import resumable_hierarchical_losses

    from .analysis import controls

    c.freeze()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    config_path = Path(c.launch["source_configuration"])
    if digest(config_path) != c.launch["source_configuration_sha256"]:
        raise ValueError("historical source configuration changed")
    config = read(config_path)
    original = config["original"]
    historical = Path(c.launch["roots"]["historical"])
    protocol = {
        "parent_sha256": digest(c.out / "PROTOCOL.json"),
        "kind": "EWMA_TRANSPORT_CV_H8",
        "optimizer_updates": 0,
        "tau_seconds": 0.3,
        "slots": "last8 of originalH16",
        "source_configuration_sha256": digest(config_path),
        "anchor_verification": "compiled int64 anchors, current lineage, original role index",
        "physical_GT_acquisition_independently_replayed": False,
        "cap_sensitivity": "past median phases at absolute60s replaced with phase0; present retained",
        "files": [
            {"path": str(p), "sha256": digest(p)}
            for p in (
                Path(__file__),
                ROOT / "src/e_jepa_ttc/efficient_context/analytical.py",
                ROOT / "src/e_jepa_ttc/simplex_t/context_sources.py",
            )
        ],
    }
    seal = c.out / "analytical/PROTOCOL.json"
    if seal.exists() and read(seal) != protocol:
        raise ValueError("analytical-control freeze changed")
    if not seal.exists():
        atomic_json(seal, protocol)
    frames = controls(c, 7)
    computed = []
    diagnostics = []
    for fold in range(3):
        c.require_resources()
        binding = original["folds"][str(fold)]
        compiled = c.historical / binding["path"]["relative_path"]
        sources = load_context_sources(
            compiled,
            c.historical / original["index_root"]["relative_path"],
            c.historical / original["dedup_root"]["relative_path"],
            historical,
            compiled_manifest_sha256=binding["sha256"],
            ancestry_sha256=original["ancestry_sha256"],
            allowed_sequences=set(config["original_sequences"]),
            feature_count=17,
        )
        parent = sources["outer_dev"]
        template = frames["H16"].loc[frames["H16"].outer_fold == fold].copy()
        from e_jepa_ttc.simplex_t.current_inputs import load_current_inputs

        metadata = load_current_inputs(
            historical,
            fold,
            "outer_dev",
            ancestry_sha256=original["ancestry_sha256"],
            allowed_sequences=set(config["original_sequences"]),
        )["metadata"]
        template = template.set_index("sample_token").loc[metadata.sample_token].reset_index()
        if not np.array_equal(template.target_ttc, metadata.target_ttc):
            raise ValueError("EWMA target query mapping differs from historical publication")
        experts = np.load(compiled / "expert_ttc.npy", mmap_mode="r", allow_pickle=False)
        output = []
        sensitivity = []
        cap_count = []
        infinite_count = []
        for start in range(0, parent.population, 128):
            c.require_resources()
            stop = min(parent.population, start + 128)
            history = parent.history[start:stop, -8:]
            valid = history >= 0
            idx = np.maximum(history, 0)
            phases = np.median(parent.features[idx, 8:11], axis=-1)
            anchors = parent.anchor_us[idx]
            ages = (anchors[:, -1, None] - anchors) / 1e6
            original_ttc = experts[idx]
            capped = (
                valid[..., None] & np.isfinite(original_ttc) & np.isclose(abs(original_ttc), 60)
            )
            infinite = valid[..., None] & np.isinf(original_ttc)
            point, diag = transport_ewma(
                torch.from_numpy(phases), torch.from_numpy(ages), torch.from_numpy(valid)
            )
            altered = phases.copy()
            median_capped = valid & np.isclose(
                abs(phase_to_ttc(torch.from_numpy(phases)).numpy()), 60
            )
            median_capped[:, -1] = False
            altered[median_capped] = 0
            sensitive, _ = transport_ewma(
                torch.from_numpy(altered), torch.from_numpy(ages), torch.from_numpy(valid)
            )
            path = c.out / f"analytical/fragments/fold{fold}_{start:05d}.npz"
            npz(
                path,
                phases=phases,
                ages=ages,
                anchors_us=anchors,
                valid=valid,
                original_ttc=original_ttc,
                capped_flags=capped,
                infinite_flags=infinite,
                point_phase=point.numpy(),
                sensitivity_phase=sensitive.numpy(),
            )
            atomic_json(
                path.with_suffix(".json"),
                {
                    "sha256": digest(path),
                    "source_sha256": parent.identity_sha256,
                    "start": start,
                    "stop": stop,
                },
            )
            output.append(point.numpy())
            sensitivity.append(sensitive.numpy())
            cap_count.append(capped.sum((1, 2)))
            infinite_count.append(infinite.sum((1, 2)))
            diagnostics.append(diag)
        columns = [
            "sample_token",
            "sequence_id",
            "track_id",
            "outer_fold",
            "target_ttc",
            "anchor_us",
            "history_span_us",
            "roi_age_us",
        ]
        frame = template[columns].copy()
        frame["prediction_ttc_s"] = phase_to_ttc(torch.from_numpy(np.concatenate(output))).numpy()
        frame["sensitivity_ttc_s"] = phase_to_ttc(
            torch.from_numpy(np.concatenate(sensitivity))
        ).numpy()
        frame["loss"] = 10000 * abs(
            benchmark_phase(frame.prediction_ttc_s.to_numpy())
            - benchmark_phase(frame.target_ttc.to_numpy())
        )
        frame["cap60_terms"] = np.concatenate(cap_count)
        frame["infinite_terms"] = np.concatenate(infinite_count)
        computed.append(frame)
        del parent, sources, experts
        gc.collect()
    frame = pd.concat(computed).sort_values("sample_token").reset_index(drop=True)
    sensitivity_frame = frame.copy()
    sensitivity_frame["prediction_ttc_s"] = frame.sensitivity_ttc_s
    sensitivity_frame["loss"] = 10000 * abs(
        benchmark_phase(frame.sensitivity_ttc_s.to_numpy())
        - benchmark_phase(frame.target_ttc.to_numpy())
    )
    reference = frames["H8"]
    losses = np.column_stack((frame.loss, reference.loss, sensitivity_frame.loss))
    bootstrap, receipt = resumable_hierarchical_losses(
        frame.rename(columns={"target_ttc": "target_ttc_s"}),
        losses,
        c.out / "analytical/bootstrap",
        c.require_resources,
        draws_path=Path(c.h16["bootstrap_draws"]),
        binding={"protocol_sha256": digest(seal)},
    )
    result = paired_practical_comparison(frame, reference)
    result.update(
        status="COMPLETE",
        optimizer_updates=0,
        historical_ewma_modified=False,
        bootstrap=receipt,
        hierarchical_ci95=np.percentile(bootstrap[:, 0] - bootstrap[:, 1], [2.5, 97.5]).tolist(),
        sequence_only=exact_sequence_diagnostic(
            np.asarray(list(result["sequence_deltas"].values()))
        ),
        rejected_terms=sum(v["rejected_terms"] for v in diagnostics),
        valid_terms=sum(v["valid_terms"] for v in diagnostics),
        zero_phase_terms=sum(v["zero_phase_terms"] for v in diagnostics),
        cap60_terms=int(frame.cap60_terms.sum()),
        infinite_terms=int(frame.infinite_terms.sum()),
        cap_sensitivity=paired_practical_comparison(sensitivity_frame, reference),
        physical_label_acquisition_independently_replayed=False,
        confirmation=False,
    )
    stream = io.BytesIO()
    frame.to_parquet(stream, index=False)
    atomic_bytes(c.out / "analysis/EWMA_TRANSPORT_CV.parquet", stream.getvalue())
    npz(
        c.out / "analytical/PAIRED_LOSSES.npz",
        losses=losses,
        mass=strict_macro_mass(frame.target_ttc.to_numpy(), frame.sequence_id.to_numpy()),
    )
    atomic_json(c.out / "EWMA_TRANSPORT_CV_RESULTS.json", result)
    print("ANALYTICAL_CONTROL_COMPLETE", result["candidate_score"], flush=True)


def main() -> int:
    """Execute the independent analytical branch even while raw/Garl are absent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "configs/campaign/efficient_context_v1.json"
    )
    args = parser.parse_args()
    c = Campaign(args.protocol)
    with Lease(c.out):
        run(c)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
