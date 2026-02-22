from __future__ import annotations

import math

from scripts.build_dataset import going_to_bucket


def test_going_to_bucket_treats_nan_as_missing() -> None:
    assert going_to_bucket(None) == 1
    assert going_to_bucket(math.nan) == 1
    assert going_to_bucket(1) == 1
    assert going_to_bucket(2) == 1
    assert going_to_bucket(3) == 2
