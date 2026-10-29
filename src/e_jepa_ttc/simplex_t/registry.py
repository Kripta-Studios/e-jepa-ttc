"""Finite registered fit graph; availability resolves before scientific scores."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class FitSpec:
    """One fixed 2500-update endpoint, never a best-validation checkpoint."""

    stage: str
    name: str
    fold: int
    seed: int = 7
    updates: int = 2500


def registered_graph(
    *,
    d1: bool,
    density: bool,
    t3: bool,
    latent: bool,
    replicate_scalar: bool,
    replicate_latent: bool,
) -> list[FitSpec]:
    """Generate only documented arms; gates are external validated evidence."""
    if density and not d1:
        raise ValueError("matched diversity requires a D1 pool")
    if replicate_latent and not latent:
        raise ValueError("latent replication requires technical latent availability")
    primary = "D1" if d1 else "D0"
    result: list[FitSpec] = []

    def add(stage: str, name: str, seeds: tuple[int, ...] = (7,)) -> None:
        for seed in seeds:
            for fold in range(3):
                result.append(FitSpec(stage, name, fold, seed))

    for data in ("D0", "D1") if d1 else ("D0",):
        for history in (1, 8):
            for capacity in (64, 160):
                add("T2", f"TPR-{data}-H{history}-C{capacity}")
    for control in ("PAST_REVERSED", "REPEAT_CURRENT", "SELECTOR", "FREE"):
        add("T2", f"{control}-{primary}-H8-C160")
    if density:
        for data in ("DENSE_OLD", "DIVERSE_MATCHED"):
            add("T2", f"TPR-{data}-H8-C160")
    if t3:
        for history in (4, 16):
            add("T3", f"TPR-{primary}-H{history}-C160")
        add("T3", f"TRANSFORMER-{primary}-H8-C128")
    if latent:
        for name in (
            f"LATENT-{primary}-H1-C160",
            f"LATENT-{primary}-H8-C160",
            f"LATENT_ZERO-{primary}-H8-C160",
        ):
            add("T4", name)
    for enabled, family in ((replicate_scalar, "TPR"), (replicate_latent, "LATENT")):
        if enabled:
            for history in (1, 8):
                add("T5", f"{family}-{primary}-H{history}-C160", (13, 23))
    if sum(fit.updates for fit in result) > 210000:
        raise ValueError("registered graph exceeds scientific update cap")
    return result


def serialized_graph(**availability: bool) -> list[dict[str, str | int]]:
    """Serialize canonical IDs for the pre-fit availability/freeze artifact."""
    return [asdict(fit) for fit in registered_graph(**availability)]
