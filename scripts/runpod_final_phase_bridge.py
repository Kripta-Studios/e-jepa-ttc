"""Build PHASE20 from sealed EVENT/RGB OOF returns, without producer inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import time
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_verified(archive: Path, destination: Path, expected: str) -> None:
    """Extract regular manifest-bound files only, retaining immutable original bytes."""
    if digest(archive) != expected:
        raise ValueError("Archive checksum mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as source:
        manifest = json.load(source.extractfile("MANIFEST.json"))
        files = manifest["files"]
        members = source.getmembers()
        if {m.name for m in members} != {"MANIFEST.json", *files}:
            raise ValueError("Archive coverage differs from manifest")
        for member in members:
            target = destination / member.name
            if not member.isfile() or not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive member")
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                temporary = target.with_suffix(target.suffix + ".pending")
                with source.extractfile(member) as incoming, temporary.open("wb") as outgoing:
                    import shutil

                    shutil.copyfileobj(incoming, outgoing)
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
                os.replace(temporary, target)
            if member.name in files:
                bound = files[member.name]
                if target.stat().st_size != bound["bytes"] or digest(target) != bound["sha256"]:
                    raise ValueError(f"Archive member changed: {member.name}")


def context(root: Path, name: str, *, deadline: float) -> tuple:
    """Use the original frozen state extraction and the exact admitted head."""
    import numpy as np
    import torch

    from operational.final_refit.features import frozen_context_state
    from operational.final_refit.recipe import StageRecipe, canonical_sha256
    from operational.final_refit.scientific_oof_features import normalize_histories
    from operational.final_refit.training import _model

    admission = read(root / "fits" / name / "ENDPOINT_ADMISSION.json")
    if (
        admission["status"] != "PASS"
        or admission["technical"] is not False
        or admission["scientific_endpoint"] is not True
        or canonical_sha256({k: v for k, v in admission.items() if k != "admission_payload_sha256"})
        != admission["admission_payload_sha256"]
    ):
        raise ValueError("Invalid endpoint admission")
    receipt = read(root / "fits" / name / "CHECKPOINT_RECEIPT.json")
    checkpoint = root / "fits" / name / "checkpoint_last.pt"
    if (
        receipt["status"] != "COMPLETE"
        or receipt["completed_updates"] != admission["updates"]
        or digest(checkpoint) != admission["checkpoint_sha256"]
        or digest(root / "fits" / name / "CHECKPOINT_RECEIPT.json")
        != admission["checkpoint_receipt_sha256"]
    ):
        raise ValueError("Head checkpoint differs from scientific endpoint")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload["identity_sha256"] != admission["checkpoint_identity_sha256"]:
        raise ValueError("Head identity mismatch")
    norm_path = root / "normalizers" / (name + ".json")
    norm = read(norm_path)
    cache = read(root / "caches" / (name + ".json"))
    fragment = root / "generation" / name / "fragment.npz"
    fragment_receipt = read(fragment.with_suffix(".json"))
    if (
        digest(fragment) != fragment_receipt["output_sha256"]
        or fragment_receipt["targets_used"] is not False
        or norm["parents"] != admission["parent_checkpoint_sha256"]
        or fragment_receipt["parents"] != norm["parents"]
        or cache["normalizers"] != {name: digest(norm_path)}
    ):
        raise ValueError("Normalizer, fragment or parent mismatch")
    if set(norm["groups"]) != set(admission["train_group_ids"]):
        raise ValueError("Normalizer training groups differ")
    with np.load(fragment, allow_pickle=False) as data:
        raw = {k: data[k] for k in data.files}
    if set(raw["group_id"].astype(str)) != set(admission["generation_group_ids"]):
        raise ValueError("Generation groups differ")
    if set(admission["generation_group_ids"]) & set(admission["train_group_ids"]):
        raise ValueError("OOF leakage")
    stage = StageRecipe(
        name,
        "temporal_head",
        admission["updates"],
        32,
        0.0,
        (),
        True,
        "final_refit_context_cache_v1",
    )
    model = _model(stage).to("cuda").float().eval().requires_grad_(False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    features = normalize_histories(
        raw, np.asarray(norm["mean"], np.float32), np.asarray(norm["scale"], np.float32)
    )
    states, widths = [], []
    for start in range(0, len(features), 128):
        if time.time() >= deadline:
            raise TimeoutError("Authorized deadline reached")
        sl = slice(start, start + 128)
        block, width, _ = frozen_context_state(
            model,
            torch.from_numpy(features[sl]).cuda(),
            *[torch.from_numpy(raw[k][sl]).cuda() for k in ("timing", "valid", "expert_phase")],
        )
        states.append(block[:, 160:170].cpu().numpy())
        widths.append(width.cpu().numpy())
    return raw, np.concatenate(states), np.concatenate(widths), admission["checkpoint_sha256"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotation", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--rgb-root", type=Path, required=True)
    parser.add_argument("--event-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch

    from operational.final_refit.scientific_oof_pipeline import write_fragment

    torch.set_num_threads(2)
    rgb_root = args.rgb_root / f"roots/r{args.rotation}_downstream/rotations/{args.rotation}"
    event_root = args.event_root / f"rotations/{args.rotation}"
    event, e, _, eh = context(event_root, "E_CTX", deadline=args.deadline)
    rgb, r, widths, rh = context(rgb_root, "R_CTX", deadline=args.deadline)
    for key in ("sample_token", "group_id", "source_ordinal"):
        if not np.array_equal(event[key], rgb[key]):
            raise ValueError(f"Cross-modal row alignment mismatch: {key}")
    write_fragment(
        args.output,
        {
            "features": np.concatenate((r, e), 1),
            "rgb_width_logits": widths,
            **{k: rgb[k] for k in ("sample_token", "group_id", "source_ordinal")},
        },
        parents={"R_CTX": rh, "E_CTX": eh},
    )
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "rotation": args.rotation,
                "rows": len(r),
                "output_sha256": digest(args.output),
                "optimizer_updates": 0,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
