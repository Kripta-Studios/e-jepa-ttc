"""Group-separated H context caches with training-group-only unique-observation normalizers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from e_jepa_ttc.rgb_port.normalization import fit_normalizer
from operational.multimodal_validation.common import RUN, read, sha, write

from .common import OUT

DEST = OUT / "group_crossfit"
SALT = "EJEPA_GROUP_CROSSFIT_20261012"


def group_folds(groups: np.ndarray) -> np.ndarray:
    unique = sorted(
        set(map(str, groups)), key=lambda g: hashlib.sha256((SALT + g).encode()).hexdigest()
    )
    if len(unique) != 8:
        raise ValueError("The registered four-fold experiment requires the eight H groups")
    mapping = {group: index % 4 for index, group in enumerate(unique)}
    return np.array([mapping[str(g)] for g in groups], dtype=np.int64)


def observation_split(ids: np.ndarray, valid: np.ndarray, train: np.ndarray) -> tuple[list, list]:
    training = sorted(set(ids[train][valid[train]].tolist()))
    heldout = sorted(set(ids[~train][valid[~train]].tolist()))
    if set(training) & set(heldout):
        raise ValueError("An observation crosses the held-out group boundary")
    return training, heldout


def save(path: Path, values: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            assert set(z.files) == set(values)
            assert all(np.array_equal(z[k], v) for k, v in values.items())
    else:
        pending = path.with_suffix(".pending.npz")
        np.savez_compressed(pending, **values)
        pending.replace(path)
    return sha(path)


def main() -> None:
    spec = OUT / "GROUP_CROSSFIT_PREREGISTRATION.json"
    assert read(spec)["planned_updates"] == 25000
    reference_tokens = reference_groups = None
    receipts = []
    for modality, name in (("E", "event"), ("R", "rgb")):
        directory = RUN / "features/H"
        history_path = directory / f"{modality}_RAW/history.npz"
        observation_path = directory / f"{modality}_OBSERVATIONS.npz"
        original_path = directory / f"{modality}_HISTORY.npz"
        assert (
            sha(history_path)
            == read(history_path.parent / "REAL_EXPERT_RECEIPT.json")["files"]["history.npz"]
        )
        assert sha(observation_path) == read(observation_path.with_suffix(".json"))["output_sha256"]
        assert sha(original_path) == read(original_path.with_suffix(".json"))["output_sha256"]
        with np.load(history_path, allow_pickle=False) as z:
            history = {k: z[k] for k in z.files}
        with np.load(observation_path, allow_pickle=False) as z:
            observations = {k: z[k] for k in z.files}
        tokens, groups = history["sample_token"], history["group_id"]
        assert set(history["role"]) == set(observations["role"]) == {"H"}
        if reference_tokens is None:
            reference_tokens, reference_groups = tokens, groups
        assert reference_groups is not None
        assert np.array_equal(tokens, reference_tokens) and np.array_equal(groups, reference_groups)
        folds = group_folds(groups)
        ids, valid = history["observation_id"], history["valid"]
        lookup = {str(identity): i for i, identity in enumerate(observations["observation_id"])}
        assert len(lookup) == len(observations["observation_id"])
        raw = np.zeros((*valid.shape, 17), dtype=np.float32)
        raw[valid] = observations["features"][[lookup[str(key)] for key in ids[valid]]]
        original_normalizer = read(directory / f"{modality}_NORMALIZER.json")
        replay = np.zeros_like(raw)
        replay[valid] = (raw[valid] - np.array(original_normalizer["mean"], np.float32)) / np.array(
            original_normalizer["scale"], np.float32
        )
        with np.load(original_path, allow_pickle=False) as z:
            assert np.array_equal(replay, z["features"]), "Raw history reconstruction must be exact"
            for key in ("timing", "valid", "sample_token", "group_id", "target_phase", "mass"):
                assert np.array_equal(history[key], z[key]), key
        for fold in range(4):
            train = folds != fold
            training_ids, heldout_ids = observation_split(ids, valid, train)
            normalizer = fit_normalizer(
                observations["features"][[lookup[str(key)] for key in training_ids]],
                training_ids,
                fit_role="H",
                expected_role="H",
                producer_sha256=str(observations["endpoint_freeze_sha256"]),
                schema_sha256=str(observations["schema_sha256"]),
                modality=name,
            )
            folder = DEST / f"fold{fold}" / modality
            folder.mkdir(parents=True, exist_ok=True)
            normalizer_path = folder / "NORMALIZER.json"
            if normalizer_path.exists():
                old = read(normalizer_path)
                assert (
                    old["mean"] == normalizer.mean.tolist()
                    and old["scale"] == normalizer.scale.tolist()
                )
            else:
                normalizer.save(normalizer_path)
            transformed = np.zeros_like(raw)
            transformed[valid] = normalizer.transform(raw[valid])
            normalized_mass = history["mass"][train].astype(np.float64)
            normalized_mass /= normalized_mass.sum()
            train_values = dict(
                features=transformed[train],
                timing=history["timing"][train],
                valid=valid[train],
                expert_phase=history["current_expert_phase"][train],
                target_phase=history["target_phase"][train],
                mass=normalized_mass.astype(np.float32),
                sample_token=tokens[train],
                group_id=groups[train],
                role=history["role"][train],
            )
            heldout_values = dict(
                features=transformed[~train],
                timing=history["timing"][~train],
                valid=valid[~train],
                expert_phase=history["current_expert_phase"][~train],
                sample_token=tokens[~train],
                group_id=groups[~train],
                global_index=np.flatnonzero(~train),
            )
            receipt = dict(
                fold=fold,
                modality=modality,
                train_groups=sorted(set(groups[train])),
                heldout_groups=sorted(set(groups[~train])),
                train_queries=int(train.sum()),
                heldout_queries=int((~train).sum()),
                train_observations=len(training_ids),
                heldout_observations=len(heldout_ids),
                observation_sets_disjoint=True,
                train_sha256=save(folder / "train.npz", train_values),
                heldout_sha256=save(folder / "heldout.npz", heldout_values),
                normalizer_sha256=sha(normalizer_path),
                raw_reconstruction_exact=True,
                normalizer_fit_ids_sha256=normalizer.unique_ids_sha256,
                source_sha256=sha(__file__),
                protocol_sha256=sha(spec),
                raw_history_sha256=sha(history_path),
                raw_observations_sha256=sha(observation_path),
                optimizer_updates=0,
            )
            write(folder / "SEALED.json", receipt)
            receipts.append(receipt)
    assert reference_tokens is not None and reference_groups is not None
    write(
        DEST / "FOLDS_SEALED.json",
        dict(
            receipts=receipts,
            group_folds=dict(
                zip(reference_groups.tolist(), group_folds(reference_groups).tolist(), strict=True)
            ),
            population=len(reference_tokens),
            source_sha256=sha(__file__),
            protocol_sha256=sha(spec),
            targets_used_for_assignment=False,
            heldout_targets_excluded_from_context_caches=True,
        ),
    )
    print(
        [(r["fold"], r["modality"], r["train_queries"], r["heldout_queries"]) for r in receipts],
        flush=True,
    )


if __name__ == "__main__":
    main()
