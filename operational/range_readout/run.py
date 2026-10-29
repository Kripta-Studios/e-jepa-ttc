"""Local V15 closed stages; never invokes the closed V14 runner."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from operational.range_transfer.common import ownership

from .common import HISTORY, OWNER, ROOT, RUN, freeze, read, sha, utc, write


def main() -> None:
    """Admit and run the independent sampling stage; V requires all four heads."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["admit", "qa", "sampling"])
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"
    import torch

    from operational.range_transfer.admission import admit, revalidate
    from operational.range_transfer.cache import HeadCache, verify_h_population

    from .qa import run_qa
    from .train_heads import fit

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run = args.run.resolve()
    with ownership(run):
        if args.stage == "admit":
            old = read(HISTORY / "ADMISSION.json")
            revalidate(old)
            binding = admit(OWNER, run)
            if binding is None:
                raise FileNotFoundError("Required matched parents/cache absent")
            cache = HeadCache(Path(binding["cache_path"]))
            verify_h_population(cache, OWNER)
            freeze(run / "ADMISSION.json", binding)
            freeze(
                run / "PROTOCOL.json",
                {
                    "base": "1c251781ac845a5d661523e607e0f7c4a8d3af8a",
                    "policy_sha256": sha(
                        ROOT / "docs/range_readout_handoff_20261012/execution_policy.json"
                    ),
                    "V14": "NO_CANDIDATE",
                    "sampling_seed": 7,
                    "R_caps": [0.05, 0.20],
                    "R_slope": 0.05,
                    "parity_atol": 2e-5,
                    "parity_rtol": 2e-5,
                    "extraction": "CPU_FP32",
                    "V_evaluation_requires_four_frozen_heads": True,
                },
            )
            print("ADMITTED: P/H/V and matched parents, H unchanged", flush=True)
        elif args.stage == "qa":
            print(run_qa(run), flush=True)
        elif args.stage == "sampling":
            if read(run / "qa/RESUME_TEST.json")["status"] != "PASS":
                raise ValueError("Synthetic resume QA must pass before scientific fits")
            binding = read(run / "ADMISSION.json")
            cache = HeadCache(Path(binding["cache_path"]))
            for mass in (0, 1):
                result = fit(cache, run, binding, mass_id=mass, loss_id=0, seed=7)
                print(result, flush=True)
                if result["status"] != "COMPLETE":
                    break
            write(
                run / "NEXT_DECISION.json",
                {
                    "status": "WAITING_R_ENDPOINTS_BEFORE_V",
                    "utc": utc(),
                    "V_evaluated": False,
                    "new_fits_outside_protocol": False,
                },
            )


if __name__ == "__main__":
    main()
