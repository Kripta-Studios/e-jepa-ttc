"""Native paths remain recognized without making strings or dictionaries paths."""

from pathlib import Path

from scripts.runpod_rgb_path_compat_v1 import compatible_path_class


class Mapping:
    def translate(self, value):
        return str(value).replace("ORIGINAL", "MAPPED")


def test_native_and_mapped_paths_are_recognized() -> None:
    mapped = compatible_path_class(Mapping())
    assert isinstance(Path("native"), mapped)
    assert isinstance(mapped("ORIGINAL"), mapped)
    assert str(mapped("ORIGINAL")) == "MAPPED"
    assert not isinstance("ORIGINAL", mapped)
    assert not isinstance({"path": "ORIGINAL"}, mapped)
    assert not isinstance(None, mapped)


def test_path_dispatch_reads_native_path_instead_of_coercing_dict(tmp_path: Path) -> None:
    path = tmp_path / "lineage.json"
    path.write_text('{"verified": true}')
    mapped = compatible_path_class(Mapping())
    assert (path.read_text() if isinstance(path, mapped) else dict(path)) == '{"verified": true}'
