"""Regression coverage for split-tree RGB paths and scientific native Path checks."""

from pathlib import Path

import pytest

from scripts.runpod_final_path_compat import mapped_path_class


class Identity:
    def translate(self, value):
        return str(value)


def setup_tree(tmp_path):
    roots = [tmp_path / "inputs", tmp_path / "bundle", tmp_path / "rgb"]
    for root in roots:
        root.mkdir()
    cls = mapped_path_class(Identity(), roots)
    return roots, cls


def test_relative_rgb_binding_resolves_from_another_input_tree(tmp_path):
    roots, cls = setup_tree(tmp_path)
    descriptor = roots[0] / "workspace/v18/artifacts/final/descriptor.json"
    descriptor.parent.mkdir(parents=True)
    descriptor.write_text("{}")
    binding = roots[2] / "workspace/v13/artifacts/rgb/P_RGB_CACHE_BINDING.json"
    binding.parent.mkdir(parents=True)
    binding.write_text("sealed")
    resolved = (
        cls(descriptor).parent / "../../../v13/artifacts/rgb/P_RGB_CACHE_BINDING.json"
    ).resolve(strict=True)
    assert resolved == binding
    assert resolved.read_text() == "sealed"


def test_joinpath_and_direct_missing_file_use_same_copy(tmp_path):
    roots, cls = setup_tree(tmp_path)
    target = roots[1] / "cache/data.npz"
    target.parent.mkdir()
    target.write_bytes(b"payload")
    assert cls(roots[0]).joinpath("cache", "data.npz").resolve(strict=True) == target
    assert cls(roots[0] / "cache/data.npz").resolve(strict=True) == target


def test_native_paths_remain_recognized(tmp_path):
    _, cls = setup_tree(tmp_path)
    assert isinstance(Path(tmp_path), cls)


def test_output_path_is_not_redirected(tmp_path):
    _, cls = setup_tree(tmp_path)
    output = tmp_path / "outputs/final/checkpoint.pt"
    assert cls(output) == output
    assert cls(output).resolve() == output.resolve()


def test_escape_from_registered_root_is_rejected(tmp_path):
    roots, cls = setup_tree(tmp_path)
    with pytest.raises(ValueError, match="escaped"):
        cls(roots[0]) / "../outside.json"


def test_missing_input_stays_fail_closed(tmp_path):
    roots, cls = setup_tree(tmp_path)
    with pytest.raises(FileNotFoundError):
        (cls(roots[0]) / "unknown/file.npz").resolve(strict=True)
