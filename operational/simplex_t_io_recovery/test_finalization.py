"""Zero-update contracts for final authority and exact cached inference scope."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import finalize_continuation as final


class FinalizationContracts(unittest.TestCase):
    """Do not import Torch or execute an optimizer in these metadata tests."""

    def test_authority_keeps_scores_and_closes_training(self) -> None:
        authority: dict[str, object] = dict.fromkeys(
            (
                "accepted_at_utc",
                "deadline_utc",
                "own_artifact_budget_bytes",
                "original_window_sha256",
                "scientific_protocol_sha256",
                "operational_source_sha256",
                "startup_margin_waiver_by_user",
            ),
            "binding",
        )
        authority["scientific_saved_update_cap"] = 60000
        decision = {"score": 119.24135567986619, "no_push": True}
        result = final.amended_decision(
            decision, {"endpoints": 24, "scientific_saved_updates": 60000}, authority
        )
        self.assertEqual(result["score"], decision["score"])
        self.assertEqual(result["future_optimizer_updates_authorized"], 0)
        self.assertIsNone(result["resume_command"])
        self.assertTrue(decision["no_push"])
        self.assertFalse(result["no_push"])

    def test_exact_scope_rejects_missing_export_and_wrong_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            night = Path(temporary)
            execution = night / "execution"
            execution.mkdir()
            payload = execution / "input.npz"
            payload.write_bytes(b"cached bytes")
            digest = final.sha(payload)
            fits, exports, sealed = [], {}, []
            for fold in range(3):
                key = f"COST_CONTEXT_20261003/FULL_C0/fold{fold}/seed7"
                fits.append(
                    dict(
                        id=key,
                        arm="FULL_C0",
                        family="N2",
                        saved_updates=2500,
                        checkpoint="checkpoint.pt",
                        checkpoint_sha256=digest,
                    )
                )
                exports[key] = dict(checkpoint_sha256=digest, fragments=[])
                for name in ("weights", "predictions", "publication"):
                    exports[key][name + "_path"] = payload.name
                    exports[key][name + "_sha256"] = digest
                sealed.append(dict(key=key, checkpoint_sha256=digest))
            final.save(
                execution / "N2_ENDPOINTS.json",
                dict(fits=sealed, all_family_frozen_before_evaluation=True),
            )
            index = execution / "ANALYSIS_EXPORT_INDEX.json"
            families = {
                "N2": dict(
                    seal_path=str(execution / "N2_ENDPOINTS.json"),
                    seal_sha256=final.sha(execution / "N2_ENDPOINTS.json"),
                )
            }
            final.save(index, {"fits": exports, "families": families})
            inventory = dict(fits=fits, endpoints=3)
            with patch.object(final, "NIGHT", night):
                self.assertEqual(len(final.export_scope(inventory)["cached_head_ids"]), 3)
                with self.assertRaisesRegex(ValueError, "18 C0"):
                    final.export_scope(dict(inventory, endpoints=24))
                removed = exports.pop(next(iter(exports)))
                final.save(index, {"fits": exports, "families": families})
                with self.assertRaisesRegex(ValueError, "scope differs"):
                    final.export_scope(inventory)
                exports[fits[0]["id"]] = removed
                final.save(index, {"fits": exports, "families": families})
                payload.write_bytes(b"changed input")
                with self.assertRaisesRegex(ValueError, "payload or binding"):
                    final.export_scope(inventory)


if __name__ == "__main__":
    unittest.main()
