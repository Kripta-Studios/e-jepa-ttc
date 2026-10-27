"""Measure real OOF batches without updating scientific optimizer state."""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time
from pathlib import Path

import torch

from e_jepa_ttc.training.causal_scale_eap import _autocast
from operational.final_refit.fold_producers import _effective_batch
from operational.final_refit.fold_sources import load_fold_rgb_source
from operational.final_refit.producers import (
    ParallelRGBDecoder,
    configure_rgb_runtime,
    producer_objective,
)
from operational.final_refit.recipe import sha256
from operational.final_refit.scientific_authority import load_config
from operational.final_refit.scientific_fold_producers import scientific_recipe
from operational.rgb_port.accounting import atomic_write_json, read_json_shared
from operational.rgb_port.train_producers import build_producer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    config = load_config(args.run)
    fit = args.run / "rotations/0/fits/R_A5"
    receipt = read_json_shared(fit / "CHECKPOINT_RECEIPT.json")
    if receipt["status"] != "PAUSED_RESOURCE":
        raise ValueError("Requires a durable paused checkpoint")
    checkpoint = Path(receipt["checkpoint_path"])
    digest = sha256(checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    configure_rgb_runtime()
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = load_fold_rgb_source(
        Path(config["rgb_source"]), Path(config["teacher"]), Path(config["fold_contract"]), 0
    )
    recipe = scientific_recipe(source, "R_A5", Path(config["config"]))
    cursor = saved["cursor"]
    order, position = cursor["order"], int(cursor["position"])
    requests = [
        order[position + i * 32 : position + (i + 1) * 32].tolist() for i in range(args.steps)
    ]
    model = build_producer(recipe).cuda().train()
    model.load_state_dict(saved["model_state_dict"])
    rows = []
    profiler = cProfile.Profile()
    profiler.enable()
    host = _effective_batch(source, requests[0], recipe)
    profiler.disable()
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(30)
    (args.run / "OOF_CPU_PROFILE.txt").write_text(stream.getvalue(), encoding="utf-8")
    # Warm lazy CUDA libraries separately from the measured batches.
    batch = host.to(torch.device("cuda"))
    with _autocast(torch.device("cuda"), "bf16"):
        loss, _ = producer_objective(model, batch, recipe, int(cursor["epoch"]))
    loss.backward()
    torch.cuda.synchronize()
    for ids in requests:
        start = time.perf_counter()
        host = _effective_batch(source, ids, recipe)
        decode = time.perf_counter() - start
        start = time.perf_counter()
        batch = host.to(torch.device("cuda"))
        torch.cuda.synchronize()
        transfer = time.perf_counter() - start
        model.zero_grad(set_to_none=True)
        start = time.perf_counter()
        with _autocast(torch.device("cuda"), "bf16"):
            loss, _ = producer_objective(model, batch, recipe, int(cursor["epoch"]))
        torch.cuda.synchronize()
        forward = time.perf_counter() - start
        start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        backward = time.perf_counter() - start
        rows.append(
            dict(
                decode_s=decode,
                transfer_s=transfer,
                forward_s=forward,
                backward_s=backward,
                parts=len(host.parts),
                loss=float(loss.detach()),
            )
        )
    source.close()
    # A fresh source keeps decoded cache state comparable between reader modes.
    source = load_fold_rgb_source(
        Path(config["rgb_source"]), Path(config["teacher"]), Path(config["fold_contract"]), 0
    )
    decoder = ParallelRGBDecoder(source.base)
    parallel = []
    try:
        for ids in requests:
            start = time.perf_counter()
            _effective_batch(source, ids, recipe)
            parallel.append(time.perf_counter() - start)
        decoder_stats = decoder.snapshot()
    finally:
        decoder.close()
        source.close()
    if sha256(checkpoint) != digest:
        raise ValueError("Scientific checkpoint changed during profiling")
    result = dict(
        checkpoint_sha256=digest,
        completed_updates=saved["completed_updates"],
        scientific_optimizer_updates=0,
        technical_optimizer_updates=0,
        epoch=int(cursor["epoch"]),
        steps=rows,
        parallel_decode_s=parallel,
        parallel_decoder=decoder_stats,
        means={
            key: sum(r[key] for r in rows) / len(rows)
            for key in ("decode_s", "transfer_s", "forward_s", "backward_s")
        },
    )
    atomic_write_json(args.run / "OOF_RUNTIME_PROFILE.json", result)
    print(json.dumps(result["means"]))
    print("parallel_decode_s", sum(parallel) / len(parallel))


if __name__ == "__main__":
    main()
