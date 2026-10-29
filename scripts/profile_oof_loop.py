"""Profile an isolated checkpoint copy; never write scientific run state."""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import shutil
import tempfile
from pathlib import Path

from operational.final_refit import scientific_fold_producers as runner
from operational.final_refit.fold_sources import load_fold_rgb_source
from operational.final_refit.recipe import sha256
from operational.final_refit.resource_monitor import _windows_commit_headroom
from operational.rgb_port.accounting import atomic_write_json, read_json_shared


class TechnicalGuard:
    def guard(self, *args: object, **kwargs: object) -> tuple[bool, dict[str, int | None]]:
        import psutil

        free = psutil.virtual_memory().available
        commit = _windows_commit_headroom()
        ok = free > 2 * 1024**3 and commit is not None and commit > 2 * 1024**3
        return ok, dict(host_available_bytes=free, windows_commit_headroom_bytes=commit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12)
    args = parser.parse_args()
    config = runner.load_config(args.run)
    binding = runner.execution_binding(args.run)
    original = args.run / "rotations/0/fits/R_A5"
    receipt = read_json_shared(original / "CHECKPOINT_RECEIPT.json")
    if receipt["status"] != "PAUSED_RESOURCE":
        raise ValueError("Pause required")
    original_digest = sha256(original / "checkpoint_last.pt")
    source = load_fold_rgb_source(
        Path(config["rgb_source"]), Path(config["teacher"]), Path(config["fold_contract"]), 0
    )
    with tempfile.TemporaryDirectory(prefix="technical_profile_", dir=args.run) as tmp:
        scratch = Path(tmp)
        shutil.copytree(original, scratch / "rotations/0/fits/R_A5")
        # Preserve embedded scientific identity solely to restore copied state.
        # All producer output paths now resolve below the disposable scratch root.
        runner.load_config = lambda _: config
        runner.execution_binding = lambda _: binding
        prof = cProfile.Profile()
        prof.enable()
        result = runner.train(
            source=source,
            fit="R_A5",
            config=Path(config["config"]),
            contract=Path(config["fold_contract"]),
            run=scratch,
            monitor=TechnicalGuard(),
            max_updates=args.steps,
        )
        prof.disable()
        stream = io.StringIO()
        pstats.Stats(prof, stream=stream).sort_stats("cumulative").print_stats(65)
        (args.run / "OOF_LOOP_PROFILE.txt").write_text(stream.getvalue(), encoding="utf-8")
        atomic_write_json(
            args.run / "OOF_LOOP_PROFILE.json",
            dict(
                scientific_optimizer_updates=0,
                technical_optimizer_updates=result["completed_updates"]
                - receipt["completed_updates"],
                original_checkpoint_sha256=original_digest,
                original_checkpoint_unchanged=sha256(original / "checkpoint_last.pt")
                == original_digest,
                result=result,
            ),
        )
        print(stream.getvalue())
    source.close()


if __name__ == "__main__":
    main()
