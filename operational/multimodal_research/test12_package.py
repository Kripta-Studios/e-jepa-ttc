"""Validate every frozen prediction and prepare one local official-format candidate ZIP."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from operational.multimodal_validation.common import read, sha, write
from operational.sota_evidence.codabench import package_predictions

from .test12_infer import DEST, restore


def main() -> None:
    freeze = DEST / "MODEL_INPUT_FREEZE.json"
    binding = sha(freeze)
    seal = read(DEST / "PREDICTIONS_SEALED.json")
    assert seal["binding_sha256"] == binding and seal["population"] == 6762
    inputs = read(DEST / "INPUTS.json")
    assert sha(DEST / "INPUTS.json") == read(freeze)["inputs_sha256"]
    tokens = [r["sample_token"] for r in inputs["queries"]]
    assert len(set(tokens)) == len(tokens) == 6762 and set(tokens) == set(seal["queries"])
    rows = []
    for query in inputs["queries"]:
        token = query["sample_token"]
        receipt = DEST / "predictions" / (hashlib.sha256(token.encode()).hexdigest() + ".json")
        assert read(receipt) == seal["queries"][token]
        value = restore(DEST / "predictions", token, binding)
        assert value is not None
        row = dict(
            sample_token=token,
            sequence_id=query["sequence_id"],
            track_id=query["track_id"],
            timestamp_us=query["query_time_us"],
            rgb_available=bool(query["rgb_history"]),
        )
        for name in ("EVENT", "ZERO", "E_CTX", "R_CTX", "F_TRUE"):
            array = np.asarray(value[name])
            assert array.shape == (1,)
            row[name] = float(array[0])
            if name != "R_CTX" or row["rgb_available"]:
                assert np.isfinite(row[name])
        if not row["rgb_available"]:
            assert row["EVENT"] == row["ZERO"] == row["F_TRUE"] == row["E_CTX"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["prediction"] = frame.EVENT
    path = DEST / "TEST_PREDICTIONS.csv"
    frame.to_csv(path, index=False)
    package = DEST / "PHASE_ONLY_S7_GarlTTC_submission.zip"
    receipt = package_predictions(tokens, frame, freeze, package)
    receipt.update(
        utc=datetime.now(UTC).isoformat(),
        candidate="PHASE_ONLY_EVENT_S7",
        predictions_sha256=sha(path),
        prediction_seal_sha256=sha(DEST / "PREDICTIONS_SEALED.json"),
        rgb_available=int(frame.rgb_available.sum()),
        fallback_count=int((~frame.rgb_available).sum()),
        checkpoint_identity="MODEL_INPUT_FREEZE.json binds the full trained graph and normalizers",
        private_labels_read=False,
        optimizer_updates=0,
        external_submission=False,
        official_score_available=False,
        official_SOTA=False,
        source_sha256=sha(__file__),
    )
    write(DEST / "PACKAGE_VERIFICATION.json", receipt)
    print(receipt, flush=True)


if __name__ == "__main__":
    main()
