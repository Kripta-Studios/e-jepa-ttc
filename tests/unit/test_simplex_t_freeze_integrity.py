"""Fail-closed byte and stage-source binding without scientific fits."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

from e_jepa_ttc.simplex_t.freeze_integrity import (
    REQUIRED_CATEGORIES,
    FrozenFile,
    verify_files,
    verify_stage_sources,
)
from e_jepa_ttc.simplex_t.phase_manifest import fit_key
from e_jepa_ttc.simplex_t.registry import registered_graph


def pins(root: Path) -> list[FrozenFile]:
    result = []
    for category in sorted(REQUIRED_CATEGORIES):
        payload = category.encode()
        (root / category).write_bytes(payload)
        result.append(FrozenFile(category, "work", category, sha256(payload).hexdigest()))
    return result


def test_files_rechecked_on_each_invocation(tmp_path: Path) -> None:
    files = pins(tmp_path)
    verify_files(files, {"work": tmp_path})
    (tmp_path / "normalizers").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes changed"):
        verify_files(files, {"work": tmp_path})


@pytest.mark.parametrize("failure", ["missing", "alias", "traversal", "root", "digest"])
def test_invalid_file_contract(tmp_path: Path, failure: str) -> None:
    files = pins(tmp_path)
    if failure == "missing":
        files.pop()
    elif failure == "alias":
        files[0] = replace(files[0], relative_path=files[1].relative_path)
    elif failure == "traversal":
        files[0] = replace(files[0], relative_path="../outside")
    elif failure == "root":
        files[0] = replace(files[0], root="unapproved")
    else:
        files[0] = replace(files[0], sha256="not-a-hash")
    with pytest.raises(ValueError):
        verify_files(files, {"work": tmp_path})


@pytest.mark.parametrize("failure", [None, "missing", "extra", "dev", "role", "digest"])
def test_complete_stage_sources(failure: str | None) -> None:
    availability = dict(
        d1=False,
        density=False,
        t3=False,
        latent=False,
        replicate_scalar=False,
        replicate_latent=False,
    )
    frozen = {
        fit_key(spec): {"inner_oof": "a" * 64, "outer_dev": "b" * 64}
        for spec in registered_graph(**availability)
    }
    observed = {key: dict(value) for key, value in frozen.items()}
    key = next(iter(observed))
    if failure == "missing":
        observed.pop(key)
    elif failure == "extra":
        observed["invented"] = dict(observed[key])
    elif failure == "dev":
        observed[key]["outer_dev"] = "c" * 64
    elif failure == "role":
        observed[key].pop("outer_dev")
    elif failure == "digest":
        observed[key]["inner_oof"] = "bad"
    if failure is None:
        verify_stage_sources(
            stage="T2", availability=availability, frozen=frozen, observed=observed
        )
    else:
        with pytest.raises(ValueError):
            verify_stage_sources(
                stage="T2", availability=availability, frozen=frozen, observed=observed
            )
