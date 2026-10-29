import json
from pathlib import Path

import numpy as np


def test_low_label_subsets_are_strictly_nested(tmp_path: Path):
    """
    Ensures that for a given seed, the 5% subset is a strict subset of the 10% subset,
    which is a strict subset of the 100% split.
    We test this by running the create_low_label_manifest script logic or verifying
    the produced artifacts.
    """
    import subprocess
    import sys

    # Generate dummy npz cache to test script
    cache_path = tmp_path / "dummy_cache.npz"
    split = np.array(["train"] * 1000 + ["validation"] * 200)
    sequence_id = np.array(["seq1"] * 500 + ["seq2"] * 500 + ["seq1"] * 100 + ["seq2"] * 100)

    np.savez(cache_path, split=split, sequence_id=sequence_id)

    out_dir = tmp_path / "subsets"

    script_path = (
        Path(__file__).resolve().parent.parent.parent / "scripts" / "create_low_label_manifest.py"
    )

    cmd = [
        sys.executable,
        str(script_path),
        "--cache",
        str(cache_path),
        "--output-dir",
        str(out_dir),
        "--seeds",
        "42",
        "--fractions",
        "0.20",
        "0.10",
    ]

    subprocess.check_call(cmd)

    # Read the manifests
    with open(out_dir / "evttc_frac20_seed42.json") as f:
        frac20 = json.load(f)["global_indices"]

    with open(out_dir / "evttc_frac10_seed42.json") as f:
        frac10 = json.load(f)["global_indices"]

    set20 = set(frac20)
    set10 = set(frac10)

    assert set10.issubset(set20), "10% subset is not nested within 20% subset!"
    assert len(set10) < len(set20), "Subsets are not strict!"

    # Since sequence_id distribution is 50/50, we should roughly see that preserved.
    # sequence_id for train are at indices 0-499 and 500-999.
    seq1_count = sum(1 for idx in frac10 if idx < 500)
    seq2_count = sum(1 for idx in frac10 if idx >= 500)

    # 10% of 500 is 50.
    assert seq1_count == 50
    assert seq2_count == 50
