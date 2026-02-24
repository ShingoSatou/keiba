from __future__ import annotations

import numpy as np

from scripts_v2.diagnose_ranker_generalization_v2 import (
    _bucket_distance,
    _bucket_field_size,
    _psi_numeric,
    _two_sample_ks_statistic,
)


def test_distance_bucket_boundaries():
    assert _bucket_distance(1200) == "sprint(<1400m)"
    assert _bucket_distance(1400) == "mile(1400-1799m)"
    assert _bucket_distance(1799) == "mile(1400-1799m)"
    assert _bucket_distance(1800) == "middle(1800-2199m)"
    assert _bucket_distance(2200) == "long(>=2200m)"


def test_field_size_bucket_boundaries():
    assert _bucket_field_size(8) == "small(<=10)"
    assert _bucket_field_size(10) == "small(<=10)"
    assert _bucket_field_size(11) == "medium(11-14)"
    assert _bucket_field_size(14) == "medium(11-14)"
    assert _bucket_field_size(16) == "large(>=15)"


def test_ks_is_zero_for_same_distribution():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    ks = _two_sample_ks_statistic(x, x.copy())
    assert ks == 0.0


def test_ks_detects_shifted_distribution():
    train = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    valid = np.array([0.6, 0.7, 0.8, 0.9, 1.0])
    ks = _two_sample_ks_statistic(train, valid)
    assert ks > 0.9


def test_psi_increases_when_distribution_shifts():
    rng = np.random.default_rng(42)
    train = rng.normal(loc=0.0, scale=1.0, size=4000)
    valid_same = rng.normal(loc=0.0, scale=1.0, size=4000)
    valid_shift = rng.normal(loc=1.2, scale=1.0, size=4000)

    psi_same = _psi_numeric(train, valid_same)
    psi_shift = _psi_numeric(train, valid_shift)

    assert psi_same >= 0.0
    assert psi_shift > psi_same
    assert psi_shift > 0.1
