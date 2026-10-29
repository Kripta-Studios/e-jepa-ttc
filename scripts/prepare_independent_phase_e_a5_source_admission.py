"""Freeze E_A5 metadata or admit one real CPU batch without fitting a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _sha256(path: Path) -> str:
    with path.resolve(strict=True).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _metadata(args: argparse.Namespace) -> dict[str, Any]:
    from operational.independent_refit.producer_contract import (
        build_producer_plan,
        canonical_sha256,
    )
    from operational.independent_refit.producer_train import build_run_contract

    output = args.evidence_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Evidence directory already exists: {output}")
    plan = build_producer_plan(
        component="E_A5",
        rotation=0,
        train40_protocol_path=args.train40_protocol,
        rgb_recipe_path=args.rgb_recipe,
        fold_contract_path=args.fold_contract,
        raw_history_receipt_path=args.raw_history,
        h5_preflight_path=args.h5_preflight,
        teacher_manifest_path=args.teacher_manifest,
        teacher_role_receipt_path=args.teacher_role_receipt,
    )
    contract = build_run_contract(
        plan=plan,
        h5_admission_path=args.h5_admission,
        input_manifest_path=args.input_manifest,
        prepare_freeze_path=args.prepare_freeze,
        crosswalk_receipt_path=args.crosswalk_receipt,
        crosswalk_path=args.crosswalk,
        teacher_manifest_path=args.teacher_manifest,
        teacher_role_receipt_path=args.teacher_role_receipt,
    )
    launch = {
        "python": str(Path(sys.executable).resolve()),
        "module": "scripts/train_independent_phase_e_a5.py",
        "output": str(args.fit_output.resolve()),
        "input_root": str(args.input_root.resolve(strict=True)),
        "gpu_launch_authorized": False,
        "first_batch_admitted": False,
        "argv": [
            "--output", str(args.fit_output.resolve()),
            "--train40-protocol", str(args.train40_protocol.resolve(strict=True)),
            "--rgb-recipe", str(args.rgb_recipe.resolve(strict=True)),
            "--fold-contract", str(args.fold_contract.resolve(strict=True)),
            "--raw-history", str(args.raw_history.resolve(strict=True)),
            "--h5-preflight", str(args.h5_preflight.resolve(strict=True)),
            "--h5-admission", str(args.h5_admission.resolve(strict=True)),
            "--teacher-manifest", str(args.teacher_manifest.resolve(strict=True)),
            "--teacher-role-receipt", str(args.teacher_role_receipt.resolve(strict=True)),
            "--input-manifest", str(args.input_manifest.resolve(strict=True)),
            "--input-root", str(args.input_root.resolve(strict=True)),
            "--prepare-freeze", str(args.prepare_freeze.resolve(strict=True)),
            "--crosswalk-receipt", str(args.crosswalk_receipt.resolve(strict=True)),
            "--crosswalk", str(args.crosswalk.resolve(strict=True)),
        ],
    }
    receipt: dict[str, Any] = {
        "schema": "independent_phase_e_a5_source_admission_metadata_v1",
        "status": "METADATA_BOUND_FIRST_BATCH_NOT_RUN",
        "cpu_threads": 1,
        "input_payloads_read": 0,
        "teacher_payloads_read": 0,
        "model_constructed": False,
        "optimizer_updates": 0,
        "plan_sha256": plan["plan_sha256"],
        "run_contract_sha256": contract["contract_sha256"],
        "h5_admission_sha256": _sha256(args.h5_admission),
        "future_fit_output": str(args.fit_output.resolve()),
        "launch": launch,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    output.mkdir(parents=True)
    _write_new(output / "PLAN.json", plan)
    _write_new(output / "RUN_CONTRACT.json", contract)
    _write_new(output / "LAUNCH.json", launch)
    _write_new(output / "METADATA_RECEIPT.json", receipt)
    return receipt


def _first_batch(args: argparse.Namespace) -> dict[str, Any]:
    import pandas as pd
    import psutil

    from operational.independent_refit.lineage import load_rotation_role
    from operational.independent_refit.producer_contract import verify_producer_plan
    from operational.independent_refit.producer_inputs import open_event_producer_source
    from operational.independent_refit.producer_train import verify_run_contract
    from operational.independent_refit.teacher_cache import TeacherRoleCache

    evidence = args.evidence_dir.resolve(strict=True)
    plan = _json(evidence / "PLAN.json")
    contract = _json(evidence / "RUN_CONTRACT.json")
    verify_producer_plan(plan)
    verify_run_contract(contract)
    if contract.get("plan_sha256") != plan.get("plan_sha256"):
        raise ValueError("Frozen source-admission plan and run contract differ")
    source = open_event_producer_source(
        source_manifest_path=args.input_manifest,
        source_root=args.input_root,
        prepare_freeze_path=args.prepare_freeze,
        crosswalk_receipt_path=args.crosswalk_receipt,
        crosswalk_path=args.crosswalk,
        h5_admission_path=args.h5_admission,
        plan=plan,
        teacher_manifest_path=args.teacher_manifest,
        teacher_role_receipt_path=args.teacher_role_receipt,
        rotation=0,
        max_cached_shards=2,
    )
    if len(source) != 35_048 or len(source.history_invalid_tokens) != 68:
        raise ValueError("Real event source history-valid population differs")
    group = source.sampling_groups[0]
    if len(group) < 32:
        raise ValueError("First physical sampling group cannot provide one full batch")
    indices = list(group[:32])
    tokens = source.index.iloc[indices]["sample_token"].astype(str).tolist()

    manifest = _json(args.teacher_manifest)
    teacher_root = args.teacher_manifest.resolve(strict=True).parent
    index_path = (teacher_root / str(manifest["index_path"])).resolve(strict=True)
    if _sha256(index_path) != manifest.get("index_sha256"):
        raise ValueError("Teacher cache index changed")
    nominal = load_rotation_role(args.fold_contract, rotation=0, role="producer")
    allowed = frozenset(nominal["sample_token"].astype(str))
    teacher_index = pd.read_parquet(index_path)
    scoped = teacher_index.loc[
        teacher_index["sample_token"].astype(str).isin(list(allowed))
    ].reset_index(drop=True)
    if len(scoped) != 35_116 or frozenset(scoped["sample_token"].astype(str)) != allowed:
        raise ValueError("Teacher cache nominal producer role differs")
    references = {
        str(item["name"]): cast(dict[str, Any], item)
        for item in cast(list[dict[str, Any]], manifest["shards"])
    }
    selected_rows = scoped.loc[scoped["sample_token"].astype(str).isin(tokens)]
    selected_names = sorted(set(selected_rows["cache_shard"].astype(str)))
    selected_paths: list[tuple[Path, Path, Mapping[str, Any]]] = []
    for name in selected_names:
        reference = references[name]
        payload = (teacher_root / "shards" / name).resolve(strict=True)
        receipt_path = payload.with_suffix(".json")
        if (
            _sha256(payload) != reference["sha256"]
            or _sha256(receipt_path) != reference["receipt_sha256"]
        ):
            raise ValueError("Selected teacher shard changed before first batch")
        selected_paths.append((payload, receipt_path, reference))
    teacher = TeacherRoleCache(
        root=teacher_root,
        index=scoped,
        allowed_tokens=allowed,
        role_receipt_path=args.teacher_role_receipt.resolve(strict=True),
        role_receipt_sha256=_sha256(args.teacher_role_receipt),
    )
    process = psutil.Process()
    rss_before = int(process.memory_info().rss)
    batch = source.batch(indices, teacher)
    rss_after = int(process.memory_info().rss)
    for payload, receipt_path, reference in selected_paths:
        if (
            _sha256(payload) != reference["sha256"]
            or _sha256(receipt_path) != reference["receipt_sha256"]
        ):
            raise ValueError("Selected teacher shard changed during first batch")
    expected_shapes = {
        "events": [32, 3, 12, 128, 128],
        "delta_t_s": [32],
        "observable_motion": [32, 18],
        "visible_heights_px": [32, 2],
        "target_ttc_s": [32],
        "boxes_xyxy": [32, 3, 4],
        "common_square_xyxy": [32, 4],
        "dinov3_relation_targets": [32, 2, 6, 32, 32],
        "dinov3_relation_valid": [32, 2, 6, 32, 32],
    }
    tensors = {
        name: getattr(batch, name)
        for name in expected_shapes
    }
    if any(list(tensor.shape) != expected_shapes[name] for name, tensor in tensors.items()):
        raise ValueError("First E_A5 batch tensor shape differs")
    if (
        batch.sample_tokens != tokens
        or len(set(tokens)) != 32
        or batch.dinov3_relation_targets is None
        or batch.dinov3_relation_valid is None
        or str(batch.dinov3_relation_valid.dtype) != "torch.bool"
        or any(not bool(tensor.isfinite().all()) for tensor in tensors.values())
    ):
        raise ValueError("First E_A5 batch identity/dtype/finite contract differs")
    receipt: dict[str, Any] = {
        "schema": "independent_phase_e_a5_first_batch_admission_v1",
        "status": "PASS_REAL_CPU_FIRST_BATCH_NO_MODEL_NO_UPDATE",
        "plan_sha256": plan["plan_sha256"],
        "run_contract_sha256": contract["contract_sha256"],
        "batch_size": 32,
        "sample_tokens": tokens,
        "tensor_shapes": expected_shapes,
        "tensor_dtypes": {name: str(tensor.dtype) for name, tensor in tensors.items()},
        "input_shards_resident_after_batch": sorted(source.cached_shard_names),
        "input_cache_max_resident_shards": 2,
        "teacher_shards_read": selected_names,
        "teacher_payload_hashes_verified_before_and_after": True,
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "model_constructed": False,
        "optimizer_updates": 0,
        "cuda_used": False,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_new(evidence / "FIRST_BATCH_RECEIPT.json", receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--fit-output", type=Path, required=True)
    parser.add_argument("--train40-protocol", type=Path, required=True)
    parser.add_argument("--rgb-recipe", type=Path, required=True)
    parser.add_argument("--fold-contract", type=Path, required=True)
    parser.add_argument("--raw-history", type=Path, required=True)
    parser.add_argument("--h5-preflight", type=Path, required=True)
    parser.add_argument("--h5-admission", type=Path, required=True)
    parser.add_argument("--teacher-manifest", type=Path, required=True)
    parser.add_argument("--teacher-role-receipt", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--prepare-freeze", type=Path, required=True)
    parser.add_argument("--crosswalk-receipt", type=Path, required=True)
    parser.add_argument("--crosswalk", type=Path, required=True)
    parser.add_argument("--first-batch", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    args = _parser().parse_args(argv)
    result = _first_batch(args) if args.first_batch else _metadata(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
