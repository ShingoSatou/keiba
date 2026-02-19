from __future__ import annotations

import pandas as pd

from scripts.t5_modeling import apply_tau_rule, choose_tau


def test_choose_tau_respects_min_bet_ratio():
    rows = []
    for race_id in range(1, 21):
        rows.append(
            {
                "race_id": race_id,
                "horse_no": 1,
                "EV_hat": 0.30 - race_id * 0.005,
                "is_win": 1 if race_id % 4 == 0 else 0,
                "odds_win_final": 5.0,
            }
        )
        rows.append(
            {
                "race_id": race_id,
                "horse_no": 2,
                "EV_hat": -0.10,
                "is_win": 0,
                "odds_win_final": 8.0,
            }
        )
    calib = pd.DataFrame(rows)

    tau_info = choose_tau(calib, min_bet_ratio=0.10, score_col="EV_hat")

    assert tau_info["tau"] is not None
    assert tau_info["stats"]["bet_ratio"] >= 0.10


def test_apply_tau_rule_keeps_one_candidate_per_race():
    frame = pd.DataFrame(
        [
            {"race_id": 1, "horse_no": 1, "EV_hat": 0.20},
            {"race_id": 1, "horse_no": 2, "EV_hat": 0.35},
            {"race_id": 2, "horse_no": 1, "EV_hat": 0.05},
            {"race_id": 2, "horse_no": 2, "EV_hat": 0.04},
        ]
    )
    out = apply_tau_rule(frame, tau=0.1, score_col="EV_hat")
    assert int(out["buy_flag"].sum()) == 1
    assert int(out.loc[(out["race_id"] == 1) & (out["horse_no"] == 2), "buy_flag"].iloc[0]) == 1
