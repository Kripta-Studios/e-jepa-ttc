"""Include the reviewed campaign test sources alongside immutable QA receipts."""

from .common import ROOT, Campaign, atomic_bytes, atomic_json, digest


def snapshot(c: Campaign) -> None:
    """Copy reviewed tests; integration tests still require the recorded base checkout."""
    names = (
        "test_efficient_context_contracts.py",
        "test_parallel_resource_adapter.py",
        "test_garl_recovery.py",
        "test_fast_owned_scan.py",
        "test_parallel_decode.py",
        "test_garl_feature_inputs.py",
        "test_garl_parallel_preflight.py",
        "test_train40_pilot.py",
    )
    rows = []
    for name in names:
        source = ROOT / "tests/unit" / name
        relative = "tests/unit/" + name
        destination = c.out / "source" / relative
        atomic_bytes(destination, source.read_bytes())
        rows.append({"path": relative, "sha256": digest(source), "bytes": source.stat().st_size})
    atomic_json(
        c.out / "TEST_SOURCE_MANIFEST.json",
        {
            "files": rows,
            "compatible_environment_and_recorded_base_checkout_required": True,
            "fixture_data_is_generated_by_tests": True,
            "raw_TRAIN_parity_QA_is_separate_and_receipted": True,
        },
    )
