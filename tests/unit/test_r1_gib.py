"""Capacity wrappers must actually select one GiB and restore frozen bindings."""

import argparse
import sys

from operational.efficient_context import (
    r1_gib_measure,
    r1_gib_parity,
    r1_gib_supervisor,
    r1_measure,
    r1_parity,
    r1_profile,
)


def test_r1_gib_parity_capacity_and_restore(monkeypatch, tmp_path):
    original = r1_parity.ResidentReplay
    seen = []

    def admit(output):
        seen.append((output, r1_parity.ResidentReplay(None).capacity_bytes))

    monkeypatch.setattr(r1_parity, "run", admit)
    monkeypatch.setattr(sys, "argv", ["parity", "--output", str(tmp_path)])
    assert r1_gib_parity.main() == 0
    assert seen == [(tmp_path, 1024**3)]
    assert r1_parity.ResidentReplay is original


def test_r1_gib_measure_capacity_and_restore(monkeypatch, tmp_path):
    seen = []

    def measure(args):
        seen.append(args.capacity_mib)

    def main():
        r1_profile.run(argparse.Namespace(capacity_mib=64))
        return 0

    monkeypatch.setattr(r1_profile, "run", measure)
    monkeypatch.setattr(r1_measure, "main", main)
    argv = ["measure", "--output", str(tmp_path)]
    monkeypatch.setattr(sys, "argv", argv)
    assert r1_gib_measure.main() == 0
    assert seen == [1024]
    assert r1_profile.run is measure
    assert sys.argv is argv


def test_r1_gib_recovery_ignores_old_transient():
    assert r1_gib_supervisor.failure_kind("InterruptedError: old\nValueError: parity") == (
        "INTEGRITY_OR_CONTRACT_FAILURE"
    )


def test_r1_gib_excludes_shell_text_and_detects_protected_entrypoint():
    assert not r1_gib_supervisor.heavy_entrypoint(
        "powershell.exe", ["powershell", "python -m operational.stage76.evaluate"]
    )
    assert r1_gib_supervisor.heavy_entrypoint(
        "python.exe", ["python", "-m", "operational.stage76.evaluate"]
    )
