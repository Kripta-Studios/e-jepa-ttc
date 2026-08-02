"""Training-only foreground-mask support for the Garl-TTC B9 protocol gap.

This module deliberately has no dependency on dataset caches or training runners.
Masks produced or loaded here are supervision targets; none of the APIs accepts a
model input tensor or participates in inference.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch.nn.functional import grid_sample


class ForegroundMaskUnavailableError(RuntimeError):
    """Raised when a declared foreground target is not materially available."""


@dataclass(frozen=True)
class MaskPathTrace:
    """Auditable result of resolving one official ``mask_paths`` reference."""

    reference: str
    sequence_id: str | None
    candidates: tuple[Path, ...]
    matches: tuple[Path, ...]
    status: Literal["resolved", "missing", "ambiguous"]

    @property
    def resolved_path(self) -> Path | None:
        """Return the sole material match, if one exists."""

        return self.matches[0] if self.status == "resolved" else None


class OfficialMaskPathResolver:
    """Resolve official mask references against explicit, ordered local roots."""

    def __init__(self, roots: Sequence[str | Path]) -> None:
        resolved_roots = tuple(Path(root).expanduser().resolve() for root in roots)
        if not resolved_roots:
            raise ValueError("At least one explicit mask root is required.")
        self.roots = resolved_roots

    def trace(self, reference: str | Path, *, sequence_id: str | None = None) -> MaskPathTrace:
        """Trace all safe candidates without pretending a missing reference exists."""

        raw = Path(reference).expanduser()
        candidates: list[Path] = []
        if raw.is_absolute():
            candidates.append(raw.resolve())
        else:
            for root in self.roots:
                candidates.append(self._safe_candidate(root, raw))
                if sequence_id:
                    candidates.append(self._safe_candidate(root, Path(sequence_id) / raw))

        unique_candidates = tuple(dict.fromkeys(candidates))
        matches = tuple(candidate for candidate in unique_candidates if candidate.is_file())
        status: Literal["resolved", "missing", "ambiguous"]
        if len(matches) == 1:
            status = "resolved"
        elif not matches:
            status = "missing"
        else:
            status = "ambiguous"
        return MaskPathTrace(str(reference), sequence_id, unique_candidates, matches, status)

    def require(self, reference: str | Path, *, sequence_id: str | None = None) -> Path:
        """Resolve one path or fail with a material-availability guard."""

        trace = self.trace(reference, sequence_id=sequence_id)
        if trace.resolved_path is not None:
            return trace.resolved_path
        attempted = ", ".join(str(path) for path in trace.candidates)
        raise ForegroundMaskUnavailableError(
            f"Official foreground mask is {trace.status}: {trace.reference!r}; "
            f"attempted [{attempted}]. Foreground supervision must remain disabled."
        )

    @staticmethod
    def _safe_candidate(root: Path, relative: Path) -> Path:
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Mask reference escapes its declared root: {relative}") from exc
        return candidate


@dataclass(frozen=True)
class BinaryMaskRLE:
    """Row-major binary run-length encoding, starting with the zero run."""

    height: int
    width: int
    counts: tuple[int, ...]
    order: Literal["C"] = "C"


def encode_binary_mask(mask: np.ndarray) -> BinaryMaskRLE:
    """Compress a 2-D binary mask into deterministic row-major RLE."""

    binary = _as_binary_mask(mask)
    flat = binary.reshape(-1).astype(np.uint8, copy=False)
    counts: list[int] = []
    expected = 0
    run = 0
    for value in flat:
        current = int(value)
        if current == expected:
            run += 1
        else:
            counts.append(run)
            run = 1
            expected = current
    counts.append(run)
    return BinaryMaskRLE(binary.shape[0], binary.shape[1], tuple(counts))


def decode_binary_mask(encoded: BinaryMaskRLE) -> np.ndarray:
    """Decode and validate a :class:`BinaryMaskRLE`."""

    if encoded.height <= 0 or encoded.width <= 0:
        raise ValueError("RLE dimensions must be positive.")
    if not encoded.counts or any(count < 0 for count in encoded.counts):
        raise ValueError("RLE counts must be a non-empty sequence of non-negative integers.")
    expected_size = encoded.height * encoded.width
    if sum(encoded.counts) != expected_size:
        raise ValueError("RLE counts do not match the declared mask dimensions.")
    values = np.arange(len(encoded.counts), dtype=np.uint8) % 2
    return np.repeat(values, encoded.counts).reshape(encoded.height, encoded.width).astype(bool)


def square_crop_resize_mask(
    mask: np.ndarray,
    square_xyxy: tuple[int, int, int, int],
    *,
    output_size: tuple[int, int] = (256, 256),
    quantization: Literal["official_truncate", "threshold"] = "official_truncate",
    threshold: float = 0.5,
) -> np.ndarray:
    """Apply the release's square ROI/grid resize to a full-frame target mask.

    ``official_truncate`` reproduces the release's bilinear-resize then integer-cast
    behavior. ``threshold`` is explicit for derived teacher masks. Out-of-frame crop
    coordinates are zero padded by ``grid_sample``.
    """

    binary = _as_binary_mask(mask)
    target_height, target_width = output_size
    if target_height <= 0 or target_width <= 0:
        raise ValueError("output_size values must be positive.")
    x_min, y_min, x_max, y_max = square_xyxy
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("square_xyxy must have positive area.")
    if (x_max - x_min) != (y_max - y_min):
        raise ValueError("square_xyxy must describe a square crop.")

    image = torch.from_numpy(binary.astype(np.float32))[None, None]
    image_height, image_width = binary.shape
    xs = torch.linspace(x_min, x_max, steps=target_width, dtype=image.dtype)
    ys = torch.linspace(y_min, y_max, steps=target_height, dtype=image.dtype)
    x_grid, y_grid = torch.meshgrid(xs, ys, indexing="xy")
    coords = torch.stack(
        (x_grid / image_width * 2.0 - 1.0, y_grid / image_height * 2.0 - 1.0), dim=-1
    )
    resized = grid_sample(
        image, coords[None], mode="bilinear", padding_mode="zeros", align_corners=True
    )[0, 0]
    if quantization == "official_truncate":
        return resized.to(torch.int64).bool().numpy()
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")
    return (resized >= threshold).numpy()


@dataclass(frozen=True)
class TeacherMetadata:
    """Identity and provenance of a local mask teacher."""

    backend: str
    model_type: str
    checkpoint_path: Path
    checkpoint_sha256: str
    selection_policy: str
    uses_gt_boxes: Literal[False] = False
    target_only: Literal[True] = True


@dataclass(frozen=True)
class TeacherMaskTarget:
    """Compressed supervision target generated independently of inference."""

    sample_id: str
    mask: BinaryMaskRLE
    teacher: TeacherMetadata


class LocalTeacherMaskAdapter:
    """Adapt an image-only local teacher callable to traced binary targets."""

    def __init__(
        self,
        *,
        checkpoint: str | Path,
        backend: Callable[[np.ndarray], np.ndarray],
        backend_name: str,
        model_type: str,
        selection_policy: str,
    ) -> None:
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise ForegroundMaskUnavailableError(
                f"Local teacher checkpoint does not exist: {checkpoint_path}"
            )
        if not selection_policy.strip():
            raise ValueError("A traceable, box-free selection_policy is required.")
        self._backend = backend
        self.metadata = TeacherMetadata(
            backend=backend_name,
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=_sha256(checkpoint_path),
            selection_policy=selection_policy,
        )

    def generate(self, image: np.ndarray, *, sample_id: str) -> TeacherMaskTarget:
        """Generate one target from an image; GT boxes are absent by construction."""

        if image.ndim != 3 or image.shape[2] not in (1, 3, 4):
            raise ValueError("Teacher image must have shape [H,W,C].")
        mask = _as_binary_mask(self._backend(np.asarray(image)))
        if mask.shape != image.shape[:2]:
            raise ValueError("Teacher mask must retain the input image height and width.")
        return TeacherMaskTarget(sample_id, encode_binary_mask(mask), self.metadata)


def make_sam_automatic_teacher(
    *,
    checkpoint: str | Path,
    proposal_selector: Callable[[Sequence[Mapping[str, Any]], np.ndarray], np.ndarray],
    selection_policy: str,
    model_type: str = "vit_h",
    device: str = "cuda",
) -> LocalTeacherMaskAdapter:
    """Create a lazy local SAM automatic-mask teacher without box prompts.

    The large model and optional ``segment_anything`` package are loaded only on the
    first generation call. The caller-supplied selector may inspect automatic SAM
    proposals and the image, but the API intentionally has no GT-box argument.
    """

    generator: Any = None

    def backend(image: np.ndarray) -> np.ndarray:
        nonlocal generator
        if generator is None:
            try:
                sam_module = importlib.import_module("segment_anything")
            except ImportError as exc:
                raise ForegroundMaskUnavailableError(
                    "segment_anything is not installed; SAM targets cannot be generated locally."
                ) from exc
            registry = sam_module.sam_model_registry
            generator_type = sam_module.SamAutomaticMaskGenerator
            sam = registry[model_type](checkpoint=str(Path(checkpoint).expanduser().resolve()))
            generator = generator_type(sam.to(device=device))
        proposals = generator.generate(image)
        return proposal_selector(proposals, image)

    return LocalTeacherMaskAdapter(
        checkpoint=checkpoint,
        backend=backend,
        backend_name="segment_anything.SamAutomaticMaskGenerator",
        model_type=model_type,
        selection_policy=selection_policy,
    )


def _as_binary_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("Foreground mask must be a 2-D array.")
    if array.size == 0:
        raise ValueError("Foreground mask must not be empty.")
    if not np.isin(array, (0, 1, False, True)).all():
        raise ValueError("Foreground mask values must be binary.")
    return array.astype(bool, copy=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "BinaryMaskRLE",
    "ForegroundMaskUnavailableError",
    "LocalTeacherMaskAdapter",
    "MaskPathTrace",
    "OfficialMaskPathResolver",
    "TeacherMaskTarget",
    "TeacherMetadata",
    "decode_binary_mask",
    "encode_binary_mask",
    "make_sam_automatic_teacher",
    "square_crop_resize_mask",
]
