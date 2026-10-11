"""Run the unchanged FCWD campaign with missing-public-GT compatible scoring."""

from unittest.mock import patch

from operational.fcwd_rgb_registration import campaign
from operational.fcwd_rgb_score_fix import score


def main() -> None:
    """Preserve registration/inference bindings and install only the post-seal decoder."""
    with patch.object(campaign, "score", score):
        campaign.main()


if __name__ == "__main__":
    main()
