"""Verify receipts, durable checkpoints, source identities and frozen A5 parameters."""

from __future__ import annotations

import importlib.metadata
import subprocess

import torch

from operational.range_transfer.admission import revalidate
from operational.range_transfer.common import digest, read, sha, utc, write
from operational.representation_audit.run import OLD, OWNER, ROOT, RUN, accounting


def main() -> None:
    revalidate(read(OLD / "ADMISSION.json"))
    original = torch.load(
        OWNER / "fits/E_A5_MATCHED/checkpoint_last.pt", map_location="cpu", weights_only=False
    )["model_state_dict"]
    rows = []
    total = 0
    for path in sorted((RUN / "fits").glob("*/CHECKPOINT_RECEIPT.json")):
        receipt = read(path)
        if receipt["status"] != "COMPLETE":
            raise ValueError("Finish or classify incomplete fit before endpoint audit")
        directory = path.parent
        checkpoint = directory / "checkpoint_last.pt"
        pointer = read(directory / "CHECKPOINT_POINTER.json")
        version = directory / "checkpoint_versions" / pointer["version"]
        expected = receipt["checkpoint_sha256"]
        if sha(checkpoint) != expected or sha(version) != expected:
            raise ValueError("Durable endpoint content differs")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        identity = payload["identity"]
        identity_sha = digest({k: v for k, v in identity.items() if k != "identity_sha256"})
        if identity_sha != receipt["identity_sha256"] or identity_sha != pointer["identity_sha256"]:
            raise ValueError("Identity receipt differs")
        if (
            payload["completed_updates"] != identity["endpoint"]
            or payload["completed_updates"] != receipt["completed_updates"]
        ):
            raise ValueError("Incorrect fixed endpoint")
        sources = {
            name: sha(ROOT / name) == expected_hash
            for name, expected_hash in identity["sources"].items()
        }
        if not all(sources.values()):
            raise ValueError("Scientific source changed after the fit")
        state = payload["model_state_dict"]
        if any(
            tensor.is_floating_point() and not torch.isfinite(tensor).all()
            for tensor in state.values()
        ):
            raise ValueError("Nonfinite model state")
        unchanged = None
        if directory.name.split("-S")[0] not in ("Z", "G", "D"):
            checks = {
                name: torch.equal(tensor, state["model." + name])
                for name, tensor in original.items()
                if not name.startswith("residual.scorer.")
            }
            if not all(checks.values()):
                raise ValueError("Unauthorized parent tensor changed")
            unchanged = len(checks)
        total += receipt["completed_updates"]
        rows.append(
            {
                "fit": directory.name,
                "updates": receipt["completed_updates"],
                "sha256": expected,
                "sources_unchanged": True,
                "durable_version_matches": True,
                "native_A5_tensors_exactly_unchanged": unchanged,
            }
        )
    counters = accounting()
    if total != counters["scientific"]:
        raise ValueError("Scientific journal count differs from complete endpoints")
    write(
        RUN / "ENDPOINT_AUDIT.json",
        {
            "status": "PASS",
            "utc": utc(),
            "fits": rows,
            "scientific_updates": total,
            "parents_revalidated": True,
            "reproduction_git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT
            )
            .decode()
            .strip(),
            "campaign_status": read(RUN / "CAMPAIGN_NEXT_DECISION.json")["status"],
        },
    )
    packages = {
        dist.metadata["Name"]: dist.version
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]
    }
    write(RUN / "ENVIRONMENT_PACKAGES.json", packages)
    print(
        f"{len(rows)} complete endpoints; {total} scientific updates; "
        "parent/source/durable checks pass"
    )


if __name__ == "__main__":
    main()
