"""V15 identities; reuse atomic IO and resource observations without owning V13."""

from pathlib import Path

from operational.range_transfer.common import (  # noqa: F401
    csv_write,
    digest,
    freeze,
    git,
    read,
    resources,
    sha,
    utc,
    write,
)

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "artifacts/range_readout_20261012"
OWNER = ROOT.parent / "e-jepa-ttc-v13-rgb-port/artifacts/rgb_port_20261008"
HISTORY = ROOT.parent / "e-jepa-ttc-v14-range-transfer/artifacts/range_transfer_20261010"


def code_binding() -> dict[str, str]:
    """Bind S sources independently of the later R integration."""
    paths = [
        "operational/range_readout/train_heads.py",
        "operational/range_readout/common.py",
        "operational/range_readout/ledger.py",
        "src/e_jepa_ttc/simplex_t/model.py",
        "src/e_jepa_ttc/simplex_t/phase.py",
        "operational/range_transfer/cache.py",
        "operational/rgb_port/train_heads.py",
        "operational/rgb_port/train_producers.py",
        "operational/rgb_port/accounting.py",
        "src/e_jepa_ttc/range_transfer/mass.py",
    ]
    return {p: sha(ROOT / p) for p in paths}


def fit_id(mass: int, loss: int, seed: int) -> str:
    """Admit only the two S recipes and registered seeds."""
    if mass not in (0, 1) or loss != 0 or seed not in (7, 13, 23):
        raise ValueError("Outside registered V15 S recipes")
    return f"S-M{mass}-L0-S{seed}"
