"""Verify exact compact-cache parity on real admitted RGB frames and both storage tiers."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from e_jepa_ttc.detection.campaign_training import CampaignConfig  # noqa: E402
from e_jepa_ttc.detection.compact_training_cache import CompactTrainingCache  # noqa: E402
from e_jepa_ttc.detection.eap_rgb_data import EAPRGBDataset, RawRGBSample  # noqa: E402
from e_jepa_ttc.detection.monodgp_eap_adapter import PreparedSample, prepare_sample  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "manifest", "admission", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = json.loads(args.manifest.read_text())
    admission = json.loads(args.admission.read_text())
    population = manifest["roles"]["train"]["sample_tokens"]
    tokens = [population[int(i * (len(population) - 1) / 5)] for i in range(6)]
    hashes = {token: admission["final_media"][token]["member_sha256"] for token in tokens}
    dataset = EAPRGBDataset(args.root, tokens, max_open_shards=8, max_label_sequences=8)
    config = SimpleNamespace(model=SimpleNamespace(width=640, height=384))

    def prepare(raw: RawRGBSample) -> PreparedSample:
        return prepare_sample(raw, width=640, height=384, include_velocity=True)

    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    cache = CompactTrainingCache(
        dataset,
        cast(CampaignConfig, config),
        hashes,
        prepare=prepare,
        device="cuda",
        gpu_image_budget=3 * 640 * 384 * 3,
    )
    originals = [prepare(dataset[index]) for index in range(len(tokens))]
    cache.prefetch(list(range(len(tokens))))
    samples = []
    for index, token in enumerate(tokens):
        expected = originals[index]
        for _ in range(2):
            actual, identity = cache.get(index)
            assert identity.sample_token == token
            assert torch.equal(expected.image, actual.image.cpu())
            assert torch.equal(expected.calibration, actual.calibration)
            assert torch.equal(expected.image_size, actual.image_size)
            assert expected.native_ground_truth == actual.native_ground_truth
            assert expected.report == actual.report
            assert all(
                torch.equal(value, actual.target[key]) for key, value in expected.target.items()
            )
        samples.append(
            {
                "token": token,
                "member_sha256": hashes[token],
                "target_count": len(expected.target["labels"]),
            }
        )
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    torch.cuda.synchronize()
    started = time.perf_counter()
    for index in range(120):
        cache.get(index % len(tokens))
    torch.cuda.synchronize()
    hot_seconds = (time.perf_counter() - started) / 120
    cache.close()
    dataset.close()
    result = {
        "status": "PASS",
        "samples": samples,
        "all_inputs_targets_bitwise_equal": True,
        "cpu_and_cuda_rng_unchanged": True,
        "both_cpu_and_cuda_storage_tested": True,
        "cache": cache.statistics(),
        "hot_cache_seconds_per_sample": hot_seconds,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
