"""Audit every D0 input/control on one complete real fold, without fitting or scoring."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256, write_new_json
from e_jepa_ttc.simplex_t.arms import resolve_arm
from e_jepa_ttc.simplex_t.context_sources import load_context_sources
from e_jepa_ttc.simplex_t.coordination import verified_ack
from e_jepa_ttc.simplex_t.lifecycle import admitted
from e_jepa_ttc.simplex_t.registry import registered_graph


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-paths", type=Path, required=True)
    parser.add_argument("--compiled", type=Path, required=True)
    parser.add_argument("--compiled-sha256", required=True)
    parser.add_argument("--outer", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve previous real source QA")
    paths = json.loads(args.local_paths.read_text(encoding="utf-8"))
    work = Path(paths["worktree"]).resolve(strict=True)
    if not args.output.resolve().is_relative_to(work):
        raise ValueError("QA output must remain in the companion")
    ack_path = Path(paths["shared_coordination"]) / "SIMPLEX_T_STAGE70_ACK.json"
    ack_hash = "3e55ab3c6e9a57eecd862ad05e999627ea90957e58e329b2eb3652120e953318"
    ack = verified_ack(ack_path, ack_hash)
    ancestry = ack["producers"]["authoritative_historical_manifest"]
    index = work / "artifacts/simplex_t/T1/query_context_index"
    if sha256(index / "INDEX_MANIFEST.json") != (
        "93a4f62e5025c5046fc82fcb1428a428f8a8df869b34486b92d5c753b0f68a3b"
    ):
        raise ValueError("OLD8192 context amendment changed")
    compiled_manifest = args.compiled / "COMPILED.json"
    if sha256(compiled_manifest) != args.compiled_sha256:
        raise ValueError("compiled fold bytes changed")
    if json.loads(compiled_manifest.read_text(encoding="utf-8"))["outer"] != args.outer:
        raise ValueError("compiled outer fold differs")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    graph = registered_graph(
        d1=False, density=False, t3=True, latent=True, replicate_scalar=True, replicate_latent=True
    )
    selected = [spec for spec in graph if spec.fold == args.outer and spec.seed == 7]
    records = []
    start = time.monotonic()
    source_root = work / "src/e_jepa_ttc/simplex_t"
    code = {
        name: sha256(source_root / name)
        for name in (
            "context_sources.py",
            "current_inputs.py",
            "cache.py",
            "controls.py",
            "arms.py",
            "registry.py",
        )
    }

    def validate() -> None:
        if not admitted([work])["has_headroom"]:
            raise InterruptedError("PAUSED_RESOURCE: source gather QA")
        if sha256(compiled_manifest) != args.compiled_sha256:
            raise ValueError("compiled fold changed during QA")

    try:
        for features in (17, 145):
            validate()
            sources = load_context_sources(
                args.compiled,
                index,
                work / "artifacts/simplex_t/T1/query_context_dedup",
                Path(ancestry["path"]).parent,
                compiled_manifest_sha256=args.compiled_sha256,
                ancestry_sha256=ancestry["sha256"],
                allowed_sequences=set(ack["interfaces"]["role_manifest"]["roles"]["original"]),
                feature_count=features,
            )
            assert sources["inner_oof"].normalizer is sources["outer_dev"].normalizer
            for spec in selected:
                binding = resolve_arm(spec, graph)
                if binding.model.feature_count != features:
                    continue
                for role, base in sources.items():
                    source = binding.source(base)
                    reference = replace(base, length=binding.history)
                    slots = 0
                    for offset in range(0, source.population, 128):
                        validate()
                        ids = torch.arange(offset, min(offset + 128, source.population))
                        batch = source.gather(ids)
                        plain = reference.gather(ids)
                        x, timing, mask, experts, target, mass = batch
                        if not all(torch.isfinite(value).all() for value in batch):
                            raise ValueError("nonfinite real head input")
                        if x.shape != (len(ids), binding.history, features):
                            raise ValueError("real feature shape differs from registered head")
                        if not mask[:, -1].all() or x[~mask].count_nonzero():
                            raise ValueError("real current/missing history mask changed")
                        if any(
                            not torch.equal(value, original)
                            for value, original in zip(batch[1:], plain[1:], strict=True)
                        ):
                            raise ValueError("control changed timing, experts, targets or mass")
                        if not torch.equal(x[:, -1, :17], plain[0][:, -1, :17]):
                            raise ValueError("control changed current scalar inputs")
                        if binding.zero_latent and x[:, :, 17:].count_nonzero():
                            raise ValueError("latent-zero control retains latent values")
                        slots += int(mask.sum())
                    records.append(
                        dict(
                            name=spec.name,
                            role=role,
                            queries=source.population,
                            valid_slots=slots,
                            source_sha256=source.identity_sha256,
                            normalizer_ids_sha256=source.normalizer.consumed_ids_sha256,
                        )
                    )
            del sources, source, reference, base
            gc.collect()
        if verified_ack(ack_path, ack_hash) != ack or any(
            sha256(source_root / name) != digest for name, digest in code.items()
        ):
            raise ValueError("source implementation or authority changed during QA")
        status = "REAL_SINGLE_FOLD_ALL_D0_INPUT_ARMS_CHECKED_NOT_SCIENTIFIC_FREEZE"
    except InterruptedError:
        status = "PAUSED_RESOURCE_PARTIAL_SOURCE_QA"
    result = dict(
        status=status,
        outer=args.outer,
        compiled_sha256=args.compiled_sha256,
        records=records,
        code_sha256=code,
        script_sha256=sha256(Path(__file__)),
        observed_seconds=time.monotonic() - start,
        optimizer_updates=0,
        model_inference=False,
        scientific_freeze=False,
        all_fold_qa=False,
    )
    write_new_json(args.output, result)
    print(json.dumps(dict(status=status, checked_arm_roles=len(records), optimizer_updates=0)))


if __name__ == "__main__":
    main()
