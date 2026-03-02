from __future__ import annotations

import pandas as pd

from scripts_v2.wide_prob_v2_common import (
    estimate_wide_probabilities_for_race,
    kumiban_from_horse_nos,
    make_race_rng,
)


def test_pl_sampler_invariants():
    race_id = 202301010101
    race_df = pd.DataFrame(
        {
            "race_id": [race_id] * 6,
            "horse_no": [1, 2, 3, 4, 5, 6],
            "p_top3": [0.42, 0.31, 0.27, 0.19, 0.14, 0.08],
        }
    )

    out = estimate_wide_probabilities_for_race(
        race_df,
        mc_samples=3000,
        top_k=3,
        eps=1e-6,
        rng=make_race_rng(42, race_id),
    )

    assert not out.empty
    assert len(out) == 15  # 6C2
    assert abs(float(out["p_wide"].sum()) - 3.0) < 1e-6  # C(3,2) pairs per sample
    assert out["p_wide"].between(0.0, 1.0).all()
    assert out["p_wide"].notna().all()
    assert (out["horse_no_1"] < out["horse_no_2"]).all()
    assert (
        out.apply(
            lambda row: row["kumiban"]
            == kumiban_from_horse_nos(int(row["horse_no_1"]), int(row["horse_no_2"])),
            axis=1,
        )
    ).all()


def test_pl_sampler_is_reproducible_with_same_seed():
    race_id = 202401010101
    race_df = pd.DataFrame(
        {
            "race_id": [race_id] * 5,
            "horse_no": [1, 2, 3, 4, 5],
            "p_top3": [0.51, 0.33, 0.26, 0.17, 0.10],
        }
    )

    out1 = estimate_wide_probabilities_for_race(
        race_df,
        mc_samples=2000,
        top_k=3,
        eps=1e-6,
        rng=make_race_rng(77, race_id),
    )
    out2 = estimate_wide_probabilities_for_race(
        race_df,
        mc_samples=2000,
        top_k=3,
        eps=1e-6,
        rng=make_race_rng(77, race_id),
    )

    out1 = out1.sort_values(["horse_no_1", "horse_no_2"], kind="mergesort").reset_index(drop=True)
    out2 = out2.sort_values(["horse_no_1", "horse_no_2"], kind="mergesort").reset_index(drop=True)
    pd.testing.assert_frame_equal(out1, out2)
