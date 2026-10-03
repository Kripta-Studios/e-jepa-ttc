"""Resume the remaining pinned queue after a diagnosed, recovered resource pause."""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

from operational.simplex_t_io_recovery import run as io


def main() -> None:
    """Reuse completed H16/setup; keep the original queue and scientific engine."""
    root = io.ROOT
    night = root / "artifacts/simplex_t/nocturnal_20261003"
    sys.path[:0] = [str(root / "operational/simplex_t_h16_replication"), str(root / "src")]
    from common import atomic_json, digest, memory, record

    os.replace = io.replace_with_retry
    metrics = memory()
    if metrics["available"] < 3 * 1024**3 or metrics["commit_headroom"] < 2 * 1024**3:
        raise InterruptedError("Wait for startup margin: available RAM3GiB/commit2GiB")
    h16 = root / "artifacts/simplex_t/h16_replication_20261003"
    if record(h16 / "FINAL_DELIVERY.json")["status"] != "DELIVERED_REGENERATED_INDEPENDENT_H16":
        raise ValueError("H16 completion/regeneration receipt missing")
    protocol = night / "PROTOCOL_COST_CONTEXT.json"
    pin = (night / "PROTOCOL_COST_CONTEXT.sha256").read_text().split()[0]
    proof = record(night / "verification/ENGINE_RESUME.json")
    if (
        digest(protocol) != pin
        or proof["protocol_sha256"] != pin
        or proof["optimizer_updates"] != 80
    ):
        raise ValueError("existing protocol/resume proof differs")
    state = runpy.run_path(
        str(root / "operational/simplex_t_cost_context/driver.py"),
        run_name="remaining_pinned_driver",
    )
    namespace = state["main"].__globals__
    command, run = namespace["command"], namespace["run"]
    namespace["command"] = lambda *a, **kw: io.wrap_command(command(*a, **kw))

    def remaining_run(phase: str, argv: list[str], *, training: bool = False) -> int:
        if phase in {"N1_ANALYSIS", "N1_DELIVERY", "N2_QA", "N2_REGISTER", "N2_TECHNICAL"}:
            namespace["persist"](phase, "REUSED_VALID_COMPLETED_ARTIFACT_NO_WORK")
            return 0
        result = run(phase, argv, training=training)
        if training and result:
            path = night / "execution/RESOURCES.json"
            if path.exists():
                payload = path.read_bytes()
                receipt = night / "resource_pause_receipts" / (digest(path) + "_closed_worker.json")
                if not receipt.exists():
                    atomic_json(
                        receipt,
                        dict(
                            phase=phase,
                            returncode=result,
                            source_sha256=digest(path),
                            resource_snapshot=json.loads(payload),
                        ),
                    )
        return result

    namespace["run"] = remaining_run
    atomic_json(
        night / "RESOURCE_RECOVERY_ADMISSION.json",
        dict(
            resource_snapshot=metrics,
            protocol_sha256=pin,
            numeric_source_files_changed=False,
            h16_reexecuted=False,
            technical_updates_repeated=0,
            source_sha256=digest(Path(__file__)),
        ),
    )
    sys.argv = [
        str(root / "operational/simplex_t_cost_context/driver.py"),
        "--campaign",
        "SIMPLEX_T_NOCTURNAL_20261003",
        "--adopt-h16",
    ]
    raise SystemExit(state["main"]())


if __name__ == "__main__":
    main()
