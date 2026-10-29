"""Recorded inventory count gates do not establish technical cache availability."""

import pytest

from e_jepa_ttc.simplex_t.pools import density_size


@pytest.mark.parametrize("original,dense", [(5461, 14520), (5461, 13473), (5462, 14949)])
def test_actual_inventory_counts_pass_registered_upper_bound(original, dense):
    assert (
        density_size(d0_count=original, dense_old_available=dense, diverse_available=32768) == dense
    )


def test_further_input_rejections_can_change_availability():
    assert density_size(d0_count=5461, dense_old_available=10921, diverse_available=32768) is None
    assert density_size(d0_count=5461, dense_old_available=14520, diverse_available=10921) is None
