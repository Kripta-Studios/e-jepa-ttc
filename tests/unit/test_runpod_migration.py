"""No CUDA, network, cloud resources, or optimizer updates are needed here."""

import json
import tarfile
from pathlib import Path

import pytest

from operational.runpod_migration.bundle import pack, safe_member, verify
from operational.runpod_migration.inventory import digest
from operational.runpod_migration.paths import PathMap, mapped_path_class
from operational.runpod_migration.resources import cgroup_headroom, parse_meminfo


def test_mapping_preserves_unicode_and_is_fail_closed():
    mapping = PathMap(
        {r"C:\Users\Álvaro\project": "workspace/project", "E:\\": "drives/E"}, "/workspace/bundle"
    )
    assert (
        mapping.translate(r"C:\Users\Álvaro\project\x.json")
        == "/workspace/bundle/workspace/project/x.json"
    )
    assert mapping.translate("/workspace/results/a.pt") == "/workspace/results/a.pt"
    assert mapping.translate(r"E:\cache\a.npz") == "/workspace/bundle/drives/E/cache/a.npz"
    with pytest.raises(ValueError, match="Unregistered"):
        mapping.translate(r"D:\secrets")
    with pytest.raises(ValueError, match="traversal"):
        mapping.translate(r"E:\cache\..\elsewhere")
    with pytest.raises(ValueError):
        PathMap({"E:\\": "../escape"}, "/workspace/bundle")


def test_path_subclass_keeps_normal_path_behaviour(tmp_path):
    mapping = PathMap(
        {"E:\\": "drives/E"},
        tmp_path.as_posix() if tmp_path.as_posix().startswith("/") else "/bundle",
    )
    mapped = mapped_path_class(mapping)
    assert isinstance(mapped("relative") / "file.txt", Path)
    assert (mapped("relative") / "file.txt").name == "file.txt"


@pytest.mark.parametrize("value", ["/etc/passwd", "../escape", "a/../escape", "a\\b"])
def test_bundle_rejects_unsafe_member(value):
    with pytest.raises(ValueError):
        safe_member(value)


def test_pack_and_verify_and_corruption(tmp_path):
    original = tmp_path / "original.txt"
    original.write_bytes(b"unchanged source")
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "profiles": {"rgb_pilot": {}},
                "files": [
                    {
                        "source": str(original),
                        "relative": "data/original.txt",
                        "bytes": original.stat().st_size,
                        "sha256": digest(original),
                        "profiles": ["rgb_pilot"],
                    }
                ],
            }
        )
    )
    output = tmp_path / "bundle.tar"
    receipt = pack(inventory, "rgb_pilot", output)
    assert receipt["sha256"] == digest(output)
    extracted = tmp_path / "extracted"
    with tarfile.open(output) as archive:
        archive.extractall(extracted, filter="data")
    assert verify(extracted)["profile"] == "rgb_pilot"
    (extracted / "data/original.txt").write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        verify(extracted)


def test_changed_input_does_not_publish_archive(tmp_path):
    original = tmp_path / "input"
    original.write_bytes(b"abc")
    inventory = tmp_path / "inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "profiles": {"rgb_pilot": {}},
                "files": [
                    {
                        "source": str(original),
                        "relative": "input",
                        "bytes": 3,
                        "sha256": "0" * 64,
                        "profiles": ["rgb_pilot"],
                    }
                ],
            }
        )
    )
    output = tmp_path / "bundle.tar"
    with pytest.raises(ValueError, match="Source changed"):
        pack(inventory, "rgb_pilot", output)
    assert not output.exists()
    assert not output.with_suffix(".tar.partial").exists()


def test_cgroup_uses_most_restrictive_ancestor_and_budget(tmp_path):
    (tmp_path / "memory.current").write_text("100")
    (tmp_path / "memory.max").write_text("1000")
    child = tmp_path / "pod"
    child.mkdir()
    (child / "memory.current").write_text("490")
    (child / "memory.max").write_text("500")
    remaining, records = cgroup_headroom(tmp_path, "0::/pod\n", 800)
    assert remaining == 10
    assert len(records) == 2
    (child / "memory.max").write_text("max")
    assert cgroup_headroom(tmp_path, "0::/pod\n", 600)[0] == 110
    (child / "memory.current").unlink()
    with pytest.raises(OSError):
        cgroup_headroom(tmp_path, "0::/pod\n", 600)


def test_meminfo_and_unsupported_cgroup_fail_closed(tmp_path):
    assert parse_meminfo("MemAvailable: 123 kB\nHugePages_Total: 0\n") == {
        "MemAvailable": 123 * 1024,
        "HugePages_Total": 0,
    }
    with pytest.raises(RuntimeError, match="v2"):
        cgroup_headroom(tmp_path, "5:memory:/foo\n", 1024)


def test_cgroup_does_not_charge_other_pods_against_our_budget(tmp_path):
    (tmp_path / "memory.current").write_text("50000")
    (tmp_path / "memory.max").write_text("100000")
    child = tmp_path / "pod"
    child.mkdir()
    (child / "memory.current").write_text("100")
    (child / "memory.max").write_text("1000")
    assert cgroup_headroom(tmp_path, "0::/pod\n", 500)[0] == 400
    (tmp_path / "memory.current").unlink()
    (tmp_path / "memory.max").unlink()
    assert cgroup_headroom(tmp_path, "0::/pod\n", 500)[0] == 400


def test_path_import_overlay_is_scoped():
    import types

    from operational.runpod_migration.runtime import ProjectFinder, ProjectLoader

    mapping = PathMap({"E:\\": "drives/E"}, "/bundle")
    path_type = mapped_path_class(mapping)

    class Loader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            module.Path = Path
            module.unchanged = "original"

    module = types.ModuleType("operational.fixture")
    ProjectLoader(Loader(), path_type).exec_module(module)
    assert module.Path is path_type
    assert module.unchanged == "original"
    finder = ProjectFinder(path_type)
    assert finder.find_spec("numpy", None) is None
    assert finder.find_spec("operational.runpod_migration.runner", None) is None


def test_return_paths_keep_receipt_bytes_and_reject_unknown_roots(tmp_path):
    from operational.runpod_migration.return_verify import ReturnPathMap

    mapper = ReturnPathMap(
        {
            "run_root": "/workspace/run4",
            "bundle_root": "/workspace/bundle",
            "roots": {str(tmp_path): "workspace"},
        },
        tmp_path,
    )
    assert mapper.translate("/workspace/run4/rotations/4/checkpoint.pt") == str(
        tmp_path / "rotations/4/checkpoint.pt"
    )
    assert mapper.translate("/workspace/bundle/workspace/source.json") == str(
        tmp_path / "source.json"
    )
    with pytest.raises(ValueError, match="Unregistered"):
        mapper.translate("/etc/passwd")
