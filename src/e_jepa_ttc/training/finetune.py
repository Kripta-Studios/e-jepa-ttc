"""Supervised fine-tuning entry points."""

from e_jepa_ttc.training.eap_lhr_jepa_ttc import (
    EAPLHRTrainerConfig,
    train_eap_lhr_jepa_ttc,
)
from e_jepa_ttc.training.supervised import train_tiny_cnn

__all__ = ["EAPLHRTrainerConfig", "train_eap_lhr_jepa_ttc", "train_tiny_cnn"]
