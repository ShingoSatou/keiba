from __future__ import annotations

import math
import re

import pandas as pd

from scripts_v3.pl_v3_common import estimate_p_wide_by_race


def test_estimate_p_wide_by_race_invariants() -> None:
    race_id = 202601010101
    frame = pd.DataFrame(
        {
            "race_id": [race_id] * 6,
            "horse_no": [1, 2, 3, 4, 5, 6],
            "pl_score": [0.3, 0.1, -0.2, -0.5, -1.0, -1.3],
        }
    )

    out = estimate_p_wide_by_race(
        frame,
        score_col="pl_score",
        mc_samples=3000,
        seed=42,
        top_k=3,
    )

    expected_pairs = math.comb(6, 2)
    assert len(out) == expected_pairs
    assert out["p_wide"].between(0.0, 1.0).all()
    assert out["p_wide"].notna().all()
    assert out["horse_no_1"].lt(out["horse_no_2"]).all()
    assert out["kumiban"].map(lambda value: bool(re.fullmatch(r"\d{4}", str(value)))).all()
