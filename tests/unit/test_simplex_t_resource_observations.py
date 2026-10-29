"""Observation uses supplied samples only, preserving RNG and admission decisions."""

import random

from e_jepa_ttc.simplex_t.resource_observations import ResourceObservations


def test_sampled_extrema_and_missing_counters():
    ticks = iter((10.0, 15.0))
    rng = random.getstate()
    observer = ResourceObservations(clock=lambda: next(ticks))
    observer.observe(
        dict(
            process_tree_rss_bytes=100, host_available_bytes=900, written_volume_free_bytes=[2000]
        ),
        allowed=True,
    )
    observer.observe(
        dict(
            process_tree_rss_bytes=200, host_available_bytes=800, written_volume_free_bytes=[1900]
        ),
        allowed=False,
    )
    observer.observe(dict(has_headroom=False), allowed=False)
    result = observer.summary()
    assert result["elapsed_seconds"] == 5
    assert result["admission_samples"] == 3
    assert result["denied_admission_samples"] == 2
    assert result["missing_counter_samples"] == 1
    assert result["sampled_process_tree_rss_max_bytes"] == 200
    assert result["sampled_host_available_min_bytes"] == 800
    assert result["sampled_written_volume_free_min_bytes"] == 1900
    assert result["continuous_peak_measurement"] is False
    assert result["vram_measurement"] is None
    assert random.getstate() == rng


def test_no_sample_is_not_zero_resource_use():
    observer = ResourceObservations(clock=lambda: 0.0)
    result = observer.summary()
    assert result["admission_samples"] == 0
    assert result["sampled_process_tree_rss_max_bytes"] is None
