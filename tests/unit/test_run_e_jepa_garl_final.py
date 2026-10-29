from pathlib import Path

import pytest

from scripts.run_e_jepa_garl_final import PROFILE_SEEDS, resolve_stages, training_command


def test_screen_command_is_cache_free_and_bounded_by_config() -> None:
    command = training_command(
        profile="screen",
        seed=7,
        eap_root=Path("eap"),
        garlttc_root=Path("garl"),
        split=Path("split.json"),
        output_root=Path("runs"),
        device="cpu",
        resume=False,
    )

    assert "--cache-manifest" not in command
    assert "e_jepa_garl_event_screen_v1.yaml" in " ".join(command)
    assert command[-2:] == ["--device", "cpu"]


def test_full_command_uses_predeclared_three_seed_profile() -> None:
    assert PROFILE_SEEDS["full"] == (7, 13, 23)
    command = training_command(
        profile="full",
        seed=13,
        eap_root=Path("eap"),
        garlttc_root=Path("garl"),
        split=Path("split.json"),
        output_root=Path("runs"),
        device="auto",
        resume=True,
    )

    assert "e_jepa_garl_event_full_v1.yaml" in " ".join(command)
    assert "--max-samples-per-split" not in command
    assert command[-1] == "--resume"


def test_stage_order_is_explicit() -> None:
    assert resolve_stages(["all"])[0] == "train"
    assert resolve_stages(["all"])[-1] == "submission-validate"
    with pytest.raises(ValueError, match="order"):
        resolve_stages(["freeze", "train"])
