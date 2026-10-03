"""Prospective H8 heads; historical SIMPLEX-T sources and models stay immutable.

Masks are applied to independently normalized coordinates. Parent cache hashes
retain historical provenance, including excluded producers; excluded bytes are
never numeric model inputs or cost supervision. This is cached-context research,
not proof that upstream historical availability is producer independent online.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from e_jepa_ttc.simplex_t.model import TemporalConfig, TemporalRefiner
from e_jepa_ttc.simplex_t.phase import MAX_PHASE, MIN_PHASE, emitted_phase, pinball
from e_jepa_ttc.simplex_t.training import QuerySource, state_digest

FEATURE_NAMES = (
    "shared_event_count_log1p",
    "shared_event_rate_log1p",
    "a5_flow",
    "a5_margin",
    "a5_log_variance",
    "c2f_flow",
    "c2f_margin",
    "c2f_log_variance",
    "a5_benchmark_phase",
    "c2f_benchmark_phase",
    "pair_benchmark_phase",
    "pair_minus_a5_phase",
    "pair_minus_c2f_phase",
    "c2f_minus_a5_phase",
    "abs_pair_minus_a5_phase",
    "abs_pair_minus_c2f_phase",
    "abs_c2f_minus_a5_phase",
)
ARMS = (
    "FULL_C0",
    "A5_ONLY_C0",
    "C2F_ONLY_C0",
    "A5_PAIR_C0",
    "SET_AGE_C0",
    "SET_NOTIME_C0",
)
_ALLOWED_NAMES = {
    "FULL_C0": FEATURE_NAMES,
    "A5_ONLY_C0": FEATURE_NAMES[:5] + (FEATURE_NAMES[8],),
    "C2F_ONLY_C0": FEATURE_NAMES[:2] + FEATURE_NAMES[5:8] + (FEATURE_NAMES[9],),
    "A5_PAIR_C0": FEATURE_NAMES[:5]
    + (
        FEATURE_NAMES[8],
        FEATURE_NAMES[10],
        FEATURE_NAMES[11],
        FEATURE_NAMES[14],
    ),
    "SET_AGE_C0": FEATURE_NAMES,
    "SET_NOTIME_C0": FEATURE_NAMES,
}
ALLOWED_EXPERTS = {
    "FULL_C0": ("a5", "c2f", "pair"),
    "A5_ONLY_C0": ("a5",),
    "C2F_ONLY_C0": ("c2f",),
    "A5_PAIR_C0": ("a5", "pair"),
    "SET_AGE_C0": ("a5", "c2f", "pair"),
    "SET_NOTIME_C0": ("a5", "c2f", "pair"),
}


def validate_schema(names: tuple[str, ...]) -> None:
    """Require one-to-one named PHASE17 coordinates, never positional aliases."""
    if len(names) != 17 or set(names) != set(FEATURE_NAMES):
        raise ValueError("PHASE17 schema must be bijective with canonical names")


def mask_features(
    values: Tensor,
    arm: str,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
) -> Tensor:
    """Zero excluded slots AFTER the historical diagonal TRAIN normalization."""
    validate_schema(feature_names)
    if arm not in ARMS or values.ndim != 3 or values.shape[-1] != 17:
        raise ValueError("registered arm and [B,L,17] features required")
    keep = torch.tensor(
        [name in _ALLOWED_NAMES[arm] for name in feature_names],
        dtype=torch.bool,
        device=values.device,
    )
    return torch.where(keep, values, torch.zeros_like(values))


def anchor_from_allowed(experts: Tensor, arm: str) -> Tensor:
    """Use only allowed current raw phases; two-expert median is their midpoint."""
    if arm not in ARMS or experts.ndim != 2 or experts.shape[-1] != 3:
        raise ValueError("registered arm and current A5/C2F/PAIR phase columns required")
    if arm == "A5_ONLY_C0":
        anchor = experts[:, 0]
    elif arm == "C2F_ONLY_C0":
        anchor = experts[:, 1]
    elif arm == "A5_PAIR_C0":
        anchor = (experts[:, 0] + experts[:, 2]) * 0.5
    else:
        anchor = experts.median(-1).values
        if not bool(torch.isfinite(experts).all()):
            raise ValueError("nonfinite permitted expert")
    columns = {"a5": 0, "c2f": 1, "pair": 2}
    if any(
        not bool(torch.isfinite(experts[:, columns[name]]).all()) for name in ALLOWED_EXPERTS[arm]
    ):
        raise ValueError("nonfinite permitted expert")
    return anchor


class MaskedGRU(TemporalRefiner):
    """The canonical two-GRUCell160 model with a prospective information mask."""

    def __init__(self, arm: str) -> None:
        if arm not in ARMS[:4]:
            raise ValueError("registered GRU arm required")
        super().__init__(TemporalConfig(feature_count=17, hidden=160))
        self.arm = arm

    def forward(
        self,
        features: Tensor,
        times: Tensor,
        valid: Tensor,
        experts: Tensor,
    ) -> dict[str, Tensor]:
        if features.ndim != 3 or features.shape[1] != 8:
            raise ValueError("new study is fixed to H8")
        anchor = anchor_from_allowed(experts, self.arm)
        # Repeated allowed anchor lets the frozen forward perform its own finite
        # emission/quantile support without inspecting any excluded expert value.
        return super().forward(
            mask_features(features, self.arm),
            times,
            valid,
            # CPU Torch 2.11's median mishandles an expanded stride-zero view;
            # materialize the three values before calling the frozen forward.
            anchor[:, None].expand(-1, 3).contiguous(),
        )


class SetRefiner(nn.Module):
    """Fixed 289765-parameter reference pooling plus canonical TTC emission."""

    def __init__(self, *, use_timing: bool) -> None:
        super().__init__()
        self.use_timing = use_timing
        self.phi = nn.Sequential(
            nn.Linear(21, 160),
            nn.LayerNorm(160),
            nn.SiLU(),
            nn.Linear(160, 160),
            nn.LayerNorm(160),
            nn.SiLU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(342, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 160),
            nn.LayerNorm(160),
            nn.SiLU(),
        )
        self.location = nn.Linear(160, 1)
        self.widths = nn.Linear(160, 2)
        self.costs = nn.Linear(160, 2)
        for head in (self.location, self.widths, self.costs):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        features: Tensor,
        times: Tensor,
        valid: Tensor,
        experts: Tensor,
    ) -> dict[str, Tensor]:
        if features.ndim != 3 or features.shape[1:] != (8, 17) or len(features) == 0:
            raise ValueError("expected nonempty [B,8,17]")
        if times.shape != (*features.shape[:2], 4):
            raise ValueError("expected [B,8,4] raw-second time slots")
        if valid.shape != features.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("expected boolean [B,8] validity")
        if not bool(valid[:, -1].all()):
            raise ValueError("a current observation is required")
        anchor = anchor_from_allowed(experts, "FULL_C0")
        f = torch.where(valid[..., None], features, torch.zeros_like(features))
        t = torch.where(valid[..., None], times, torch.zeros_like(times))
        if not self.use_timing:
            t = torch.zeros_like(t)
        if not bool(torch.isfinite(f).all() and torch.isfinite(t).all()):
            raise ValueError("nonfinite valid input")
        x = torch.cat((f, t), dim=-1)
        past_valid = valid[:, :-1]
        z = self.phi(x[:, :-1])
        count = past_valid.sum(1, keepdim=True)
        mean = torch.where(past_valid[..., None], z, torch.zeros_like(z)).sum(1)
        mean = mean / count.clamp_min(1).to(z.dtype)
        maximum = z.masked_fill(~past_valid[..., None], -torch.inf).amax(1)
        maximum = torch.where(count > 0, maximum, torch.zeros_like(maximum))
        h = self.rho(torch.cat((mean, maximum, x[:, -1], count.to(z.dtype) / 7), -1))
        delta = 0.03 * self.location(h).squeeze(-1)
        location = anchor + delta
        widths = 0.03 * functional.softplus(self.widths(h))
        risk = torch.cat((experts.new_zeros((len(features), 1)), self.costs(h)), -1)
        return {
            "point_phase": emitted_phase(location),
            "raw_location": location,
            "raw_residual": delta,
            "q10": (location - widths[:, 0]).clamp(MIN_PHASE, MAX_PHASE),
            "q90": (location + widths[:, 1]).clamp(MIN_PHASE, MAX_PHASE),
            "relative_cost": risk,
            "expert_index": risk.argmin(-1),
        }


def build_model(arm: str) -> nn.Module:
    """Initialize one authorized prospective head without loading historical weights."""
    if arm in ARMS[:4]:
        return MaskedGRU(arm)
    if arm in ARMS[4:]:
        return SetRefiner(use_timing=arm == "SET_AGE_C0")
    raise ValueError("unregistered prospective arm")


def training_loss(
    output: dict[str, Tensor],
    truth: Tensor,
    experts: Tensor,
    mass: Tensor,
    population: int,
) -> Tensor:
    """Uniform-query estimate of L_phase+.1L_quant; NO expert cost targets read."""
    del experts  # Signature compatible with the historical driver, not supervision.
    if truth.ndim != 1 or mass.shape != truth.shape or population <= 0:
        raise ValueError("invalid target/mass/population")
    if not bool(torch.isfinite(truth).all() and torch.isfinite(mass).all()):
        raise ValueError("nonfinite supervision")
    if bool((mass < 0).any()):
        raise ValueError("negative population mass")
    l1 = (output["point_phase"] - truth).abs() / 0.03
    quant = (pinball(output["q10"], truth, 0.1) + pinball(output["q90"], truth, 0.9)) / 0.03
    return (population * mass * (l1 + 0.1 * quant)).mean()


@dataclass
class MaskedSource:
    """Read-only binding over a verified H8 source; no fitting or role mutation."""

    parent: QuerySource
    arm: str
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self) -> None:
        validate_schema(self.feature_names)
        if self.arm not in ARMS or len(self.parent.identity_sha256) != 64:
            raise ValueError("registered arm and sealed parent source required")
        self.identity_sha256 = state_digest(
            {
                "namespace": "SIMPLEX_T_PROSPECTIVE_COST_CONTEXT_20261003",
                "historical_parent_provenance": self.parent.identity_sha256,
                "arm": self.arm,
                "features": self.feature_names,
                "allowed_features": _ALLOWED_NAMES[self.arm],
                "allowed_experts": ALLOWED_EXPERTS[self.arm],
                "mask_after_diagonal_normalization": True,
                "history": 8,
                "lambda_cost": 0.0,
                "lambda_quant": 0.1,
                "availability": "UNCHANGED_HISTORICAL_COMMON_INDEX",
            }
        )

    @property
    def population(self) -> int:
        """Preserve every query and its original TRAIN/OLD_DEV global mass."""
        return self.parent.population

    def gather(self, query_ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Normalize in parent, mask by names, and erase excluded current phases."""
        x, times, valid, experts, truth, mass = self.parent.gather(query_ids)
        if x.shape[1:] != (8, 17):
            raise ValueError("prospective source must be canonical D1/H8 PHASE17")
        x = mask_features(x, self.arm, self.feature_names)
        # Canonical forward columns are name ordered even if a source schema is
        # supplied in a different (bijective) order.
        take = torch.tensor([self.feature_names.index(name) for name in FEATURE_NAMES])
        x = x.index_select(-1, take)
        keep = torch.tensor(
            [name in ALLOWED_EXPERTS[self.arm] for name in ("a5", "c2f", "pair")],
            device=experts.device,
        )
        experts = torch.where(keep, experts, torch.zeros_like(experts))
        if self.arm == "SET_NOTIME_C0":
            times = torch.zeros_like(times)
        return x, times, valid, experts, truth, mass
