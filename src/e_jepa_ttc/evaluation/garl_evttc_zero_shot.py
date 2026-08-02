"""Predict/score separation for the EvTTC zero-shot protocol."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from e_jepa_ttc.data.evttc_garl_adapter import reject_labels_from_predict_payload
from e_jepa_ttc.evaluation.garl_ttc_protocol import signed_garl_metrics


def score_zero_shot_predictions(
    y_true_ttc_s: Iterable[float],
    y_pred_ttc_s: Iterable[float],
) -> dict[str, object]:
    """Score predictions only after the predict stage has completed."""

    return signed_garl_metrics(np.asarray(list(y_true_ttc_s)), np.asarray(list(y_pred_ttc_s)))


__all__ = ["reject_labels_from_predict_payload", "score_zero_shot_predictions"]
