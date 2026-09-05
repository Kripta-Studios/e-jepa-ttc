"""Single writer ownership and failure closure shared by campaign CLIs."""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

import psutil

from e_jepa_ttc.artifacts.campaign_failure import close_campaign_failure


def append_transition(path: Path, state: str, **details: object) -> None:
    """Durably record a campaign/endpoint transition before performing its work."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {"time_ns": time.time_ns(), "state": state, **details},
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())


class CampaignLock:
    """Preserve stale locks; never mutate a campaign owned by a live writer."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __enter__(self) -> CampaignLock:
        value = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "created_ns": time.time_ns(),
            "process_created_epoch": psutil.Process().create_time(),
        }
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            prior = json.loads(self.path.read_text(encoding="utf-8"))
            # Unknown/remote owner is not evidence of a dead writer.
            if prior.get("host") != socket.gethostname():
                raise RuntimeError(
                    "campaign lock belongs to an unverified remote writer"
                ) from error
            if psutil.pid_exists(int(prior.get("pid", -1))):
                raise RuntimeError(f"campaign already has a live writer: {prior}") from error
            self.path.replace(self.path.with_name(f"STALE_CAMPAIGN_LOCK_{time.time_ns()}.json"))
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        return self

    def __exit__(self, _kind: object, error: BaseException | None, _traceback: object) -> None:
        if error is not None:
            close_campaign_failure(self.path.parent, error)
        if self.path.is_file():
            owner = json.loads(self.path.read_text(encoding="utf-8"))
            if owner.get("pid") != os.getpid() or owner.get("host") != socket.gethostname():
                raise RuntimeError("campaign writer lock changed while owned")
            self.path.unlink()
