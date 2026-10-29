"""Process-local bounded retries for Windows atomic replacement sharing failures.

The pinned trainers and scientific modules stay byte-for-byte unchanged. Only
os.replace retries the same already-fsynced temporary file, without rerunning
an optimizer update, inference fragment or bootstrap draw.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / "artifacts/simplex_t/nocturnal_20261003/IO_REPLACE_RETRIES.jsonl"
ORIGINAL_REPLACE = os.replace
DELAYS = (0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 0.8, 0.8)


def replace_with_retry(
    source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    target: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    *,
    src_dir_fd: int | None = None,
    dst_dir_fd: int | None = None,
) -> None:
    """Retry only Windows access/sharing errors on the exact same replacement."""
    for attempt in range(len(DELAYS) + 1):
        try:
            ORIGINAL_REPLACE(source, target, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (5, 32, 33) or attempt == len(DELAYS):
                raise
            LOG.parent.mkdir(parents=True, exist_ok=True)
            with LOG.open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            pid=os.getpid(),
                            target=str(target),
                            attempt=attempt + 1,
                            winerror=exc.winerror,
                            delay=DELAYS[attempt],
                        )
                    )
                    + "\n"
                )
            time.sleep(DELAYS[attempt])


def wrap_command(original: list[str]) -> list[str]:
    """Keep the pinned child invocation and arguments inside this I/O shim."""
    head = [original[0], "-B", str(Path(__file__).resolve())]
    tail = original[2:] if original[1] == "-B" else original[1:]
    if tail[0] == "-m":
        return head + ["--module", tail[1], *tail[2:]]
    return head + ["--script", *tail]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--script", type=Path)
    mode.add_argument("--module")
    mode.add_argument("--driver", action="store_true")
    args, remaining = parser.parse_known_args()
    os.replace = replace_with_retry
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
    recovery = ROOT / "artifacts/simplex_t/nocturnal_20261003/recovery"
    recovery.mkdir(parents=True, exist_ok=True)
    payload = Path(__file__).read_bytes()
    pin = hashlib.sha256(payload).hexdigest()
    snapshot = recovery / ("IO_SHIM_" + pin[:12] + ".py")
    if not snapshot.exists():
        snapshot.write_bytes(payload)
    receipt = recovery / ("LAUNCH_" + str(os.getpid()) + ".json")
    receipt.write_text(
        json.dumps(
            dict(
                pid=os.getpid(),
                source_sha256=pin,
                arguments=sys.argv,
                scientific_files_modified=False,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.driver:
        path = ROOT / "operational/simplex_t_cost_context/driver.py"
        sys.path.insert(0, str(ROOT / "operational/simplex_t_h16_replication"))
        state = runpy.run_path(str(path), run_name="recovered_nocturnal_driver")
        main_fn = state["main"]
        namespace = main_fn.__globals__
        original_command = namespace["command"]
        namespace["command"] = lambda *a, **kw: wrap_command(original_command(*a, **kw))
        sys.argv = [str(path), *remaining]
        raise SystemExit(main_fn())
    if args.script:
        path = args.script.resolve(strict=True)
        sys.path.insert(0, str(path.parent))
        sys.argv = [str(path), *remaining]
        runpy.run_path(str(path), run_name="__main__")
    elif args.module == "operational.simplex_t_cost_context.deliver":
        state = runpy.run_module(args.module, run_name="recovered_publisher")
        namespace = state["main"].__globals__
        select = namespace["selected_files"]
        namespace["selected_files"] = lambda: {
            name: path
            for name, path in select().items()
            if name not in {"RECOVERED_DRIVER.log", "IO_REPLACE_RETRIES.jsonl"}
        }
        sys.argv = [args.module, *remaining]
        state["main"]()
    else:
        sys.argv = [args.module, *remaining]
        runpy.run_module(args.module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
