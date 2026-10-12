"""Resume frozen inference with the validated scan covering shared admission too."""

from datetime import UTC, datetime
from unittest.mock import patch

import psutil

from operational.multimodal_validation.common import sha, write

from . import common, test12_run
from .guard_scan import python_first


def main() -> None:
    receipt = dict(
        utc_start=datetime.now(UTC).isoformat(),
        source_sha256=sha(__file__),
        frozen_entrypoint_sha256=sha(test12_run.__file__),
        model_input_freeze_sha256=sha(common.OUT / "test12_phase_only/MODEL_INPUT_FREEZE.json"),
        scope="Same frozen entrypoint, models, arithmetic and per-item guards. The already "
        "validated Python-first metadata scan also covers the shared-admission exception "
        "handler. No admission condition, threshold, PID check or accounting is skipped.",
        status="RUNNING",
    )
    path = common.OUT / "test12_phase_only/GUARDED_RESUME_EXECUTION.json"
    assert not path.exists(), "Preserve an earlier recovery receipt before rerunning"
    write(path, receipt)
    try:
        with patch.object(psutil, "process_iter", python_first):
            test12_run.main()
        receipt["status"] = "COMPLETE"
    except BaseException as error:
        receipt.update(status="STOPPED", error=str(error))
        raise
    finally:
        receipt["utc_end"] = datetime.now(UTC).isoformat()
        write(path, receipt)


if __name__ == "__main__":
    main()
