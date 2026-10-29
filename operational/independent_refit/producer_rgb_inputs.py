"""Strict raw-RGB inputs for the independent rotation-0 R_A5 producer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .population import file_sha256

if TYPE_CHECKING:
    from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch

SOURCE_COMMIT = "2ad7a05797799caf1bcdc9202c307d244f9fe31d"
DESCRIPTOR_SCHEMA = "independent_phase_r_a5_rgb_source_v1"
ROLE_ROWS, ROLE_VALID, ROLE_INVALID = 35_116, 35_048, 68
SOURCE_SHA256 = {
    "operational/final_refit/producers.py": (
        "45717d9ca859461d403f938f80388dfcf3bb1ecf292b5f9beb439982397d8c14"
    ),
    "operational/final_refit/rgb.py": (
        "77dd54104d04095948d3ed06964978028277e25b3290342629c7c636fc2c60ab"
    ),
    "src/e_jepa_ttc/rgb_port/data.py": (
        "a33a0ddeb62b7a668d25e9d2e8f880261cc7e325d8bcfe953bcee18e4b0410ec"
    ),
    "operational/rgb_port/cache_prepare.py": (
        "352a20124fabc91f57670965c0c4eebde405fc3d473b01f85a13e1288bf3109b"
    ),
    "configs/rgb_port/producers.json": (
        "9ffb221ebe540ee5fe0013d69be07012954f008ffdd988a15acc2b2d2106f0aa"
    ),
}
ROLE_MANIFEST_SHA256 = {
    "P": "6f22b070c53c96aa851d90ae2c6836b604e8280965f4887ba6ef3a0cd6d3727c",
    "H": "ac26288c11583d859166737feb53491167f2f1d908b309fe555220cb9ee27a92",
    "V": "f73aaab45d868fcab406f8d90aea65dc97c7bd702647677e7cc77b0781f9d720",
}


def _canonical(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _verify_sources(repository: Path) -> None:
    for relative, digest in SOURCE_SHA256.items():
        if file_sha256(repository / relative) != digest:
            raise ValueError(f"R_A5 source differs from {SOURCE_COMMIT}: {relative}")


def _safe_tar(root: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts or len(posix.parts) < 5:
        raise ValueError("Unsafe RGB tar path")
    path = (root / Path(*posix.parts)).resolve(strict=True)
    if not path.is_relative_to(root) or path.suffix != ".tar":
        raise ValueError("RGB tar escapes the public dataset root")
    return path


def build_rgb_source_descriptor(
    *,
    population_path: Path,
    role_mask_path: Path,
    role_manifests: Mapping[str, Path],
    public_manifest_path: Path,
    teacher_manifest_path: Path,
    teacher_role_receipt_path: Path,
    output_dir: Path,
    p_cache_manifest_path: Path | None = None,
    p_cache_parity_path: Path | None = None,
) -> dict[str, Any]:
    """Freeze the exact independent RGB rows without opening a tar member."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    if set(role_manifests) != {"P", "H", "V"}:
        raise ValueError("R_A5 requires exact P/H/V historical manifests")
    repository = Path(__file__).resolve().parents[2]
    _verify_sources(repository)
    population = pd.read_parquet(population_path)
    mask = pd.read_parquet(role_mask_path)
    if (
        len(mask) != ROLE_ROWS
        or int(np.count_nonzero(mask["history_valid"].to_numpy(dtype=np.bool_))) != ROLE_VALID
        or int(np.count_nonzero(~mask["history_valid"].to_numpy(dtype=np.bool_))) != ROLE_INVALID
        or mask[["fit_ordinal", "sample_token"]].duplicated().any()
    ):
        raise ValueError("R_A5 independent role mask differs")
    independent = population.merge(
        mask[["fit_ordinal", "sample_token", "history_valid", "invalid_reason"]],
        on=["fit_ordinal", "sample_token"],
        how="inner",
        validate="one_to_one",
    )
    if len(independent) != ROLE_ROWS:
        raise ValueError("R_A5 role mask does not bind the population exactly")
    valid = independent.loc[independent["history_valid"]].copy()
    excluded = independent.loc[~independent["history_valid"]].copy()
    teacher_manifest = _json(teacher_manifest_path)
    teacher_manifest_sha256 = file_sha256(teacher_manifest_path)
    teacher_role = _json(teacher_role_receipt_path)
    if (
        teacher_manifest.get("schema") != "independent_phase_train24_teacher_cache_v2"
        or teacher_manifest.get("status") != "COMPLETE_VERIFIED_CONVERSION"
        or teacher_manifest.get("rows") != 53_047
        or teacher_role.get("schema") != "independent_phase_teacher_role_receipt_v1"
        or teacher_role.get("status") != "PASS_EXACT_ROLE_FILTER"
        or teacher_role.get("rotation") != 0
        or teacher_role.get("role") != "producer"
        or teacher_role.get("queries") != ROLE_ROWS
        or teacher_role.get("cache_manifest_sha256") != teacher_manifest_sha256
        or teacher_role.get("full_train24_exposed") is not False
        or teacher_role.get("generation_excluded") is not True
    ):
        raise ValueError("Independent nominal teacher role differs")
    historical: list[pd.DataFrame] = []
    manifest_refs: dict[str, Any] = {}
    eap_roots: set[Path] = set()
    for role in ("P", "H", "V"):
        manifest_path = role_manifests[role].resolve(strict=True)
        if file_sha256(manifest_path) != ROLE_MANIFEST_SHA256[role]:
            raise ValueError(f"Historical {role} manifest changed")
        manifest = _json(manifest_path)
        rows_path = Path(str(manifest["rows_path"])).resolve(strict=True)
        if (
            manifest.get("schema") != "rgb_port_role_manifest_v1"
            or manifest.get("status") != "COMPLETE"
            or manifest.get("role") != role
            or file_sha256(rows_path) != manifest.get("rows_sha256")
        ):
            raise ValueError(f"Historical {role} RGB rows changed")
        rows = pd.read_parquet(rows_path).reset_index(names="historical_role_ordinal")
        if len(rows) != manifest.get("population_size") or rows["sample_token"].duplicated().any():
            raise ValueError(f"Historical {role} RGB population differs")
        rows["historical_role"] = role
        historical.append(rows)
        eap_roots.add(Path(str(manifest["eap_root"])).resolve(strict=True))
        manifest_refs[role] = {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
            "rows_path": str(rows_path),
            "rows_sha256": str(manifest["rows_sha256"]),
        }
    if len(eap_roots) != 1:
        raise ValueError("Historical RGB roles disagree on eAP root")
    all_rgb = pd.concat(historical, ignore_index=True)
    if all_rgb["sample_token"].duplicated().any():
        raise ValueError("P/H/V RGB union has duplicate tokens")
    selected = valid.merge(
        all_rgb,
        on="sample_token",
        how="left",
        suffixes=("_independent", "_rgb"),
        validate="one_to_one",
    )
    required = ("historical_role", "historical_role_ordinal", "producer_frame_count")
    if len(selected) != ROLE_VALID or selected[list(required)].isna().any().any():
        raise ValueError("R_A5 valid tokens are not exactly covered by P/H/V")
    for left, right in (
        ("sequence_id_independent", "sequence_id_rgb"),
        ("track_id_independent", "track_id_rgb"),
        ("public_track_id_independent", "public_track_id_rgb"),
    ):
        if not selected[left].astype(str).equals(selected[right].astype(str)):
            raise ValueError("Independent and RGB row identity differs")
    if not np.array_equal(
        selected["timestamp_us"].to_numpy(dtype=np.int64),
        selected["query_time_us"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("Independent and RGB query clocks differ")
    if set(selected["producer_frame_count"].astype(int)) - {2, 3}:
        raise ValueError("R_A5 rows must be genuine T2/T3 RGB inputs")
    public = _json(public_manifest_path)
    public_files = public.get("files")
    if not isinstance(public_files, dict):
        raise ValueError("Public eAP file manifest differs")
    eap_root = next(iter(eap_roots))
    tar_paths = sorted({str(item) for values in selected["producer_shards"] for item in values})
    total_tar_bytes = 0
    for relative in tar_paths:
        entry = public_files.get(relative)
        path = _safe_tar(eap_root, relative)
        if (
            not isinstance(entry, dict)
            or not isinstance(entry.get("sha256"), str)
            or path.stat().st_size != entry.get("size_bytes")
        ):
            raise ValueError("Required RGB tar is not the pinned public file")
        total_tar_bytes += path.stat().st_size
    cache = {"P": "RAW_TAR_UNTIL_FRESH_BINDING", "H": "RAW_TAR", "V": "RAW_TAR"}
    cache_evidence: dict[str, Any] | None = None
    if p_cache_manifest_path is not None or p_cache_parity_path is not None:
        if p_cache_manifest_path is None or p_cache_parity_path is None:
            raise ValueError("P cache manifest and parity must be supplied together")
        cache_manifest = _json(p_cache_manifest_path)
        parity = _json(p_cache_parity_path)
        if (
            cache_manifest.get("schema") != "rgb_port_prepared_input_cache_v1"
            or cache_manifest.get("status") != "COMPLETE"
            or cache_manifest.get("role") != "P"
            or cache_manifest.get("role_manifest_sha256") != ROLE_MANIFEST_SHA256["P"]
            or cache_manifest.get("stored_pixels") != "uint8_RGB"
            or cache_manifest.get("decode_parity_verified") is not True
            or parity.get("status") != "COMPLETE"
            or parity.get("all_fields_bit_exact") is not True
        ):
            raise ValueError("Existing P cache lacks lossless parity evidence")
        cache_evidence = {
            "manifest_path": str(p_cache_manifest_path.resolve(strict=True)),
            "manifest_sha256": file_sha256(p_cache_manifest_path),
            "parity_path": str(p_cache_parity_path.resolve(strict=True)),
            "parity_sha256": file_sha256(p_cache_parity_path),
            "fresh_binding_present": False,
        }
    selected = selected.sort_values("fit_ordinal", kind="stable").reset_index(drop=True)
    output_dir.mkdir(parents=True)
    rows_path = output_dir / "rows.parquet"
    selected.to_parquet(rows_path, index=False)
    excluded_path = output_dir / "history_excluded.parquet"
    excluded[["fit_ordinal", "sample_token", "invalid_reason"]].to_parquet(
        excluded_path, index=False
    )
    role_counts = {
        role: int(np.count_nonzero(selected["historical_role"].astype(str) == role))
        for role in ("P", "H", "V")
    }
    frame_counts = {
        str(count): int(np.count_nonzero(selected["producer_frame_count"].astype(int) == count))
        for count in (2, 3)
    }
    value: dict[str, Any] = {
        "schema": DESCRIPTOR_SCHEMA,
        "status": "METADATA_FROZEN_RAW_RGB_NOT_FIRST_BATCH_ADMITTED",
        "source_commit": SOURCE_COMMIT,
        "source_sha256": SOURCE_SHA256,
        "rows_path": rows_path.name,
        "rows_sha256": file_sha256(rows_path),
        "history_excluded_path": excluded_path.name,
        "history_excluded_sha256": file_sha256(excluded_path),
        "population_path": str(population_path.resolve(strict=True)),
        "population_sha256": file_sha256(population_path),
        "role_mask_path": str(role_mask_path.resolve(strict=True)),
        "role_mask_sha256": file_sha256(role_mask_path),
        "rows": len(selected),
        "history_invalid_rows": ROLE_INVALID,
        "role_counts": role_counts,
        "frame_counts": frame_counts,
        "role_manifests": manifest_refs,
        "teacher_manifest_path": str(teacher_manifest_path.resolve(strict=True)),
        "teacher_manifest_sha256": teacher_manifest_sha256,
        "teacher_role_receipt_path": str(teacher_role_receipt_path.resolve(strict=True)),
        "teacher_role_receipt_sha256": file_sha256(teacher_role_receipt_path),
        "eap_root": str(eap_root),
        "public_manifest_path": str(public_manifest_path.resolve(strict=True)),
        "public_manifest_sha256": file_sha256(public_manifest_path),
        "required_tar_files": len(tar_paths),
        "required_tar_bytes": total_tar_bytes,
        "cache_routes": cache,
        "p_cache_evidence": cache_evidence,
        "pixel_contract": "raw_uint8_shared_square_crop_then_float32_div255_exactly_once",
        "learned_features_or_normalizers": False,
        "first_batch_admitted": False,
        "training_admitted": False,
    }
    value["descriptor_sha256"] = _canonical(value)
    (output_dir / "SOURCE.json").write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return value


class IndependentRGBProducerSource:
    """Lazy exact T2/T3 RGB source; tar members are opened only by ``record``."""

    def __init__(
        self, descriptor_path: Path, *, max_frame_cache_bytes: int = 256 * 1024**2
    ) -> None:
        value = _json(descriptor_path)
        unsigned = dict(value)
        claimed = unsigned.pop("descriptor_sha256", None)
        if claimed != _canonical(unsigned) or value.get("schema") != DESCRIPTOR_SCHEMA:
            raise ValueError("Independent RGB descriptor identity changed")
        if (
            value.get("status") != "METADATA_FROZEN_RAW_RGB_NOT_FIRST_BATCH_ADMITTED"
            or value.get("training_admitted") is not False
            or value.get("first_batch_admitted") is not False
            or value.get("learned_features_or_normalizers") is not False
            or value.get("pixel_contract")
            != "raw_uint8_shared_square_crop_then_float32_div255_exactly_once"
            or value.get("cache_routes")
            != {"P": "RAW_TAR_UNTIL_FRESH_BINDING", "H": "RAW_TAR", "V": "RAW_TAR"}
        ):
            raise ValueError("Unreviewed RGB descriptor claims admission")
        rows_path = descriptor_path.parent / str(value["rows_path"])
        if file_sha256(rows_path) != value.get("rows_sha256"):
            raise ValueError("Independent RGB rows changed")
        self.rows = pd.read_parquet(rows_path).to_dict(orient="records")
        if len(self.rows) != value.get("rows"):
            raise ValueError("Independent RGB row count differs")
        excluded_path = descriptor_path.parent / str(value["history_excluded_path"])
        if file_sha256(excluded_path) != value.get("history_excluded_sha256"):
            raise ValueError("Independent RGB excluded rows changed")
        excluded = pd.read_parquet(excluded_path)
        self.history_invalid_tokens = frozenset(excluded["sample_token"].astype(str))
        self.teacher_role_receipt_path = Path(value["teacher_role_receipt_path"])
        self.teacher_role_receipt_sha256 = str(value["teacher_role_receipt_sha256"])
        if (
            file_sha256(Path(value["teacher_manifest_path"])) != value["teacher_manifest_sha256"]
            or file_sha256(self.teacher_role_receipt_path) != self.teacher_role_receipt_sha256
        ):
            raise ValueError("Independent RGB teacher binding changed")
        from e_jepa_ttc.rgb_port.data import RGBPortDataset, TarFrameReader

        for role in ("P", "H", "V"):
            reference = value["role_manifests"][role]
            if (
                file_sha256(Path(reference["path"])) != reference["sha256"]
                or file_sha256(Path(reference["rows_path"])) != reference["rows_sha256"]
            ):
                raise ValueError("Historical RGB source changed after freeze")
        self.datasets = {
            role: RGBPortDataset(Path(value["role_manifests"][role]["path"]), role=role)
            for role in ("P", "H", "V")
        }
        # A descriptor cannot silently activate a historical cache binding.
        for dataset in self.datasets.values():
            dataset.prepared_cache = None
            dataset.reader = TarFrameReader(
                Path(value["eap_root"]), max_frame_cache_bytes=max_frame_cache_bytes
            )
        self.population_size = len(self.rows)
        self.batch_keys = tuple(
            (str(row["historical_role"]), int(row["producer_frame_count"])) for row in self.rows
        )

    def __len__(self) -> int:
        return self.population_size

    def record(self, index: int) -> dict[str, Any]:
        if not 0 <= int(index) < len(self.rows):
            raise IndexError(index)
        row = self.rows[int(index)]
        role = str(row["historical_role"])
        local = int(row["historical_role_ordinal"])
        sample = self.datasets[role][local]
        if str(sample["sample_token"]) != str(row["sample_token"]):
            raise ValueError("RGB physical row token differs")
        return sample

    def batch(self, indices: Sequence[int], teacher: object) -> ObjectEventV4Batch:
        if not indices or len(set(map(int, indices))) != len(indices):
            raise ValueError("R_A5 batch must contain unique rows")
        keys = {self.batch_keys[int(index)] for index in indices}
        if len(keys) != 1:
            raise ValueError("R_A5 physical batch must be homogeneous by role and T")
        from .teacher_cache import TeacherRoleCache

        if not isinstance(teacher, TeacherRoleCache):
            raise TypeError("R_A5 requires the independent nominal teacher")
        expected_tokens = frozenset(str(row["sample_token"]) for row in self.rows)
        if (
            teacher.role_receipt_sha256 != self.teacher_role_receipt_sha256
            or teacher.allowed_tokens != expected_tokens | self.history_invalid_tokens
        ):
            raise ValueError("R_A5 nominal teacher role identity differs")
        samples = [self.record(int(index)) for index in indices]
        tokens = [str(sample["sample_token"]) for sample in samples]
        targets, valid = zip(*(teacher.get(token) for token in tokens), strict=True)
        if any(value.dtype != np.float16 or value.shape != (2, 6, 32, 32) for value in targets):
            raise ValueError("R_A5 teacher target contract differs")
        if any(value.dtype != np.bool_ or value.shape != (2, 6, 32, 32) for value in valid):
            raise ValueError("R_A5 teacher validity contract differs")
        import torch

        from e_jepa_ttc.data.object_event_v4 import ObjectEventV4Batch

        def stack_field(name: str) -> Any:  # noqa: ANN401 - runtime torch is intentionally lazy
            return torch.stack(  # pyright: ignore[reportAttributeAccessIssue]
                [torch.as_tensor(sample[name]) for sample in samples]  # pyright: ignore[reportAttributeAccessIssue]
            )

        return ObjectEventV4Batch(
            events=stack_field("rgb").float(),
            delta_t_s=stack_field("delta_t_s").float(),
            observable_motion=torch.empty(  # pyright: ignore[reportAttributeAccessIssue]
                (len(samples), 0),
                dtype=torch.float32,  # pyright: ignore[reportAttributeAccessIssue]
            ),
            visible_heights_px=stack_field("visible_heights_px").float(),
            target_ttc_s=stack_field("target_ttc").float(),
            boxes_xyxy=stack_field("boxes_in_crop_xyxy").float(),
            common_square_xyxy=stack_field("roi_xyxy").float(),
            sequence_ids=[str(sample["sequence_id"]) for sample in samples],
            sample_tokens=tokens,
            track_ids=[str(sample["track_id"]) for sample in samples],
            dinov3_relation_targets=torch.from_numpy(  # pyright: ignore[reportAttributeAccessIssue]
                np.stack(targets).astype(np.float32)
            ),
            dinov3_relation_valid=torch.from_numpy(  # pyright: ignore[reportAttributeAccessIssue]
                np.stack(valid)
            ),
        )

    def close(self) -> None:
        for dataset in self.datasets.values():
            if dataset.reader is not None:
                dataset.reader.close()


__all__ = ["IndependentRGBProducerSource", "build_rgb_source_descriptor"]
