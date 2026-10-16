"""Losslessly compress repeated evidence together, keeping all original files and hashes."""

from __future__ import annotations

import hashlib
import io
import json
import lzma
import tarfile
import tempfile
from pathlib import Path
from typing import BinaryIO

from .bundle import MANIFEST, safe_name, write_archive
from .unpack_solid import verify


class HashingReader:
    """Pass original bytes to TAR and retain the digest of exactly those bytes."""

    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        block = self.stream.read(size)
        self.digest.update(block)
        return block


def write_solid_archive(output: Path, files: dict[str, Path], context: dict) -> dict:
    """Write an XZ-compressed TAR inside a standard ZIP, without quantization or row removal."""
    names = [safe_name(name) for name in files]
    if len(names) != len({name.casefold() for name in names}) or MANIFEST.casefold() in {
        name.casefold() for name in names
    }:
        raise ValueError("Duplicate or reserved payload path")
    with tempfile.TemporaryDirectory(prefix="v18_solid_", dir=output.parent) as temporary:
        root = Path(temporary)
        payload = root / "PAYLOAD.tar.xz"
        manifest = {"schema": "v18_essential_bundle_v1", "context": context, "files": {}}
        filters = [{"id": lzma.FILTER_LZMA2, "preset": 6, "dict_size": 64 * 1024 * 1024}]
        with lzma.open(payload, "wb", filters=filters) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                for name in sorted(files, key=lambda n: (files[n].suffix, n)):
                    source = files[name]
                    before = source.stat()
                    info = tarfile.TarInfo(name)
                    info.size = before.st_size
                    info.mode = 0o644
                    with source.open("rb") as stream:
                        reader = HashingReader(stream)
                        tar.addfile(info, reader)
                    after = source.stat()
                    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                        raise RuntimeError(f"Source changed during lossless packing: {name}")
                    manifest["files"][name] = {
                        "bytes": info.size,
                        "sha256": reader.digest.hexdigest(),
                    }
                content = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
                info = tarfile.TarInfo(MANIFEST)
                info.size = len(content)
                tar.addfile(info, io.BytesIO(content))
        readme = root / "README.md"
        readme.write_text(
            "# V18: paquete completo con compresión sin pérdida\n\n"
            "Para respetar el máximo de 100 MB sin eliminar tablas ni pesos, todos los "
            "archivos originales están en PAYLOAD.tar.xz. No hay cuantización.\n\n"
            "Extrae este ZIP normalmente. Con Python 3.11 o posterior, ejecuta:\n\n"
            "```powershell\npython -B unpack.py RUTA_DEL_ZIP CARPETA_NUEVA\n```\n\n"
            "El script sólo usa la biblioteca estándar. Verifica CRC, SHA-256 y tamaños "
            "del ZIP y cada archivo original. No sobrescribe carpetas existentes. "
            "La carpeta nueva contiene source/, campaign/ y su README original. "
            "Para el replay científico sigue DELIVERY_REPRODUCTION.md dentro de source/docs/"
            "joint_measurement_20261012. No confundir integridad con reentrenamiento.\n",
            encoding="utf-8",
        )
        write_archive(
            output,
            {
                "README.md": readme,
                "unpack.py": Path(__file__).with_name("unpack_solid.py"),
                "PAYLOAD.tar.xz": payload,
            },
            {**context, "packaging": "lossless_solid_tar_xz_inside_zip_v1"},
        )
    return verify(output)
