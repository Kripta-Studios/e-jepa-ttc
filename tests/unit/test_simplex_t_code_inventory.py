"""A real temporary Git tree proves omitted/uncommitted executables are refused."""

import subprocess

import pytest

from e_jepa_ttc.artifacts.simplex_t_preflight import sha256
from e_jepa_ttc.simplex_t.freeze_integrity import FrozenFile, verify_code_commit


@pytest.mark.parametrize(
    "change", ["", "omit_python", "omit_powershell", "untracked", "modified", "document"]
)
def test_complete_executable_inventory(tmp_path, change):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(tmp_path), *args], stderr=subprocess.DEVNULL
        )

    paths = []
    for relative in ("src/pkg/model.py", "scripts/launch.ps1"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# fixture\n")
        paths.append(path)
    git("init", "--quiet")
    git("add", "src", "scripts")
    git(
        "-c",
        "user.name=QA",
        "-c",
        "user.email=qa@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "-m",
        "fixture",
    )
    commit = git("rev-parse", "HEAD").decode().strip()
    pins = [
        FrozenFile("code", "work", p.relative_to(tmp_path).as_posix(), sha256(p)) for p in paths
    ]
    if change == "omit_python":
        pins = pins[1:]
    elif change == "omit_powershell":
        pins = pins[:1]
    elif change == "untracked":
        (tmp_path / "src/pkg/new.py").write_text("# uncommitted", encoding="utf-8")
    elif change == "modified":
        paths[0].write_text("# changed", encoding="utf-8")
    elif change == "document":
        (tmp_path / "proposal.md").write_text("unrelated", encoding="utf-8")
    if change not in {"", "document"}:
        message = "working source differs" if change == "modified" else "omits executable source"
        with pytest.raises(ValueError, match=message):
            verify_code_commit(pins, {"work": tmp_path}, commit)
    else:
        verify_code_commit(pins, {"work": tmp_path}, commit)
