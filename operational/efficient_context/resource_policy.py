"""Explicit user resource amendments, separate from frozen scientific protocols."""

from pathlib import Path

from . import common
from .common import Campaign, atomic_json, digest, read


class AuthorizedCampaign(Campaign):
    """Apply only the explicitly authorized RAM increase to the unchanged resource guard."""

    def __init__(self, protocol: Path) -> None:
        super().__init__(protocol)
        amendment = self.out / "RESOURCE_AUTHORIZATION.json"
        if not amendment.exists():
            return
        value = read(amendment)
        if (
            value["parent_policy_sha256"] != digest(self.policy_path)
            or value["parent_protocol_sha256"] != digest(self.out / "PROTOCOL.json")
            or value["max_tree_rss_bytes"] != 12_000_000_000
            or value["only_change"] != "RAM_limit"
        ):
            raise ValueError("RAM amendment is not bound to the original campaign authorization")
        self.policy = dict(self.policy)
        self.policy["max_tree_rss_gib"] = value["max_tree_rss_bytes"] / 1024**3
        atomic_json(
            self.out / "ACTIVE_RESOURCE_POLICY.json",
            {
                "amendment_sha256": digest(amendment),
                "max_tree_rss_bytes": value["max_tree_rss_bytes"],
                "active_policy": self.policy,
                "scientific_protocol_unchanged": True,
                "global_environment_modified": False,
            },
        )


def install() -> None:
    """Bind new workers without editing the frozen historical/scientific source files."""
    common.Campaign = AuthorizedCampaign
