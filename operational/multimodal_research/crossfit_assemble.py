"""Assemble one group-out-of-fold physical-state row per H query before attaching H targets."""

from __future__ import annotations

import numpy as np

from operational.multimodal_validation.common import read, sha, write

from .common import OUT
from .crossfit_data import DEST, save


def main() -> None:
    base = OUT / "cache/H/states.npz"
    assert sha(base) == read(base.parent / "SEALED.json")["states_sha256"]
    with np.load(base, allow_pickle=False) as z:
        tokens, groups = z["sample_token"], z["group_id"]
    features = np.zeros((len(tokens), 340), np.float32)
    widths = np.zeros((len(tokens), 2), np.float32)
    counts = np.zeros((len(tokens), 2), np.int64)
    sources = {}
    for fold in range(4):
        for m, modality in enumerate(("R", "E")):
            folder = DEST / f"fold{fold}" / modality
            seal = read(folder / "OOF_PREDICTIONS_SEALED.json")
            fold_seal = read(folder / "SEALED.json")
            assert seal["fold_seal_sha256"] == sha(folder / "SEALED.json")
            assert not set(fold_seal["train_groups"]) & set(fold_seal["heldout_groups"])
            fit = OUT / f"fits/GROUP_CROSSFIT_F{fold}_{modality}_S7"
            endpoint = read(fit / "CHECKPOINT_RECEIPT.json")
            assert endpoint["status"] == "COMPLETE" and endpoint["completed_updates"] == 2500
            assert (
                seal["checkpoint_sha256"]
                == endpoint["checkpoint_sha256"]
                == sha(fit / "checkpoint_last.pt")
            )
            path = folder / "OOF_PREDICTIONS.npz"
            assert sha(path) == seal["sha256"]
            with np.load(path, allow_pickle=False) as z:
                ids = z["global_index"]
                assert len(ids) == len(set(ids.tolist())) == seal["population"]
                assert np.array_equal(tokens[ids], z["sample_token"])
                assert np.array_equal(groups[ids], z["group_id"])
                assert set(z["group_id"]) == set(fold_seal["heldout_groups"])
                assert np.isfinite(z["block"]).all() and not z["block"][:, :160].any()
                features[ids, m * 170 : (m + 1) * 170] = z["block"]
                counts[ids, m] += 1
                if modality == "R":
                    widths[ids] = z["width_logits"]
            sources[str(folder.relative_to(OUT))] = sha(folder / "OOF_PREDICTIONS_SEALED.json")
    assert (counts == 1).all()
    # Fixed producer experts and sensor clocks must not change with context folds.
    with np.load(base, allow_pickle=False) as z:
        for start in (162, 332):
            assert np.array_equal(
                features[:, start : start + 8], z["features"][:, start : start + 8]
            )
    destination = DEST / "cache/H"
    unlabelled = dict(
        features=features, rgb_width_logits=widths, sample_token=tokens, group_id=groups
    )
    digest = save(destination / "OOF_INPUTS.npz", unlabelled)
    write(
        destination / "OOF_INPUTS_SEALED.json",
        dict(
            sha256=digest,
            source_sha256=sha(__file__),
            fold_prediction_seals=sources,
            population=len(tokens),
            each_query_held_out_once_per_modality=True,
            targets_read_for_assembly=False,
            hidden_coordinates_zero=True,
            optimizer_updates=0,
        ),
    )
    # Meta-stage H labels are attached only after every held-out input is sealed.
    with np.load(base, allow_pickle=False) as z:
        values = dict(
            **unlabelled,
            role=z["role"],
            target_phase=z["target_phase"],
            mass=z["mass"],
            rgb_phase=features[:, 161],
            event_phase=features[:, 331],
        )
    binding = dict(
        source_sha256=sha(__file__),
        unlabelled_seal_sha256=sha(destination / "OOF_INPUTS_SEALED.json"),
        original_H_cache_sha256=sha(base),
        states_are_group_OOF=True,
        normalization="Per-fold unique training observations only",
        upstream="Frozen P-only producers/PAIR; H contexts exclude the target group",
        deployment="Original full-H contexts and normalizers; only the small corrector is new",
        schema="Physical170 blocks with all320 hidden coordinates zero; PhaseOnlyAnchor only",
    )
    write(destination / "BINDING.json", binding)
    digest = save(destination / "states.npz", values)
    write(
        destination / "SEALED.json",
        dict(
            states_sha256=digest,
            binding_sha256=sha(destination / "BINDING.json"),
            population=len(tokens),
            optimizer_updates=0,
        ),
    )
    print("Sealed", len(tokens), "group-OOF physical states", flush=True)


if __name__ == "__main__":
    main()
