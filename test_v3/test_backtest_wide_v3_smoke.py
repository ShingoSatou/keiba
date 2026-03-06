from __future__ import annotations

import pandas as pd
import pytest

from scripts_v3.backtest_wide_v3 import _build_backtest_report, load_backtest_input, parse_args
from scripts_v3.v3_common import BankrollConfig


def test_parse_args_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as exc_info:
        parse_args(["--help"])
    assert exc_info.value.code == 0


def test_horse_input_requires_pl_score(tmp_path) -> None:
    path = tmp_path / "horse_missing_pl_score.parquet"
    pd.DataFrame(
        {
            "race_id": [1, 1],
            "horse_no": [1, 2],
        }
    ).to_parquet(path, index=False)

    with pytest.raises(SystemExit, match="Missing required columns in horse-level input"):
        load_backtest_input(path)


def test_pair_input_requires_kumiban(tmp_path) -> None:
    path = tmp_path / "pair_missing_kumiban.parquet"
    pd.DataFrame(
        {
            "race_id": [1],
            "horse_no_1": [1],
            "horse_no_2": [2],
            "p_wide": [0.1],
        }
    ).to_parquet(path, index=False)

    with pytest.raises(SystemExit, match="Missing required columns in pair-level input"):
        load_backtest_input(path)


def test_backtest_preserves_cv_policy_metadata_from_input(tmp_path) -> None:
    path = tmp_path / "horse_input.parquet"
    pd.DataFrame(
        {
            "race_id": [1, 1],
            "horse_no": [1, 2],
            "pl_score": [0.1, 0.2],
            "race_date": ["2024-01-10", "2024-01-10"],
            "cv_window_policy": ["fixed_sliding", "fixed_sliding"],
            "train_window_years": [4, 4],
            "holdout_year": [2025, 2025],
            "window_definition": [
                "train = previous 4 years only, valid = current year",
                "train = previous 4 years only, valid = current year",
            ],
        }
    ).to_parquet(path, index=False)

    loaded, input_mode, p_wide_source = load_backtest_input(path)

    assert input_mode == "horse"
    assert p_wide_source == "v3_pl_score_mc"
    assert loaded["valid_year"].tolist() == [2024, 2024]

    args = parse_args([])
    summary, meta = _build_backtest_report(
        args=args,
        summary={
            "n_races": 1,
            "n_bets": 0,
            "n_hits": 0,
            "total_bet": 0,
            "total_return": 0,
            "roi": 0.0,
            "max_drawdown": 0,
        },
        monthly_rows=[],
        bet_records=[],
        bankroll_config=BankrollConfig(),
        input_mode=input_mode,
        p_wide_source=p_wide_source,
        selected_years=[2024],
        available_years=[2024],
        loaded_input_len=int(len(loaded)),
        pair_probs=pd.DataFrame(
            {
                "race_id": [1],
                "horse_no_1": [1],
                "horse_no_2": [2],
                "kumiban": ["0102"],
                "p_wide": [0.1],
                "valid_year": [2024],
                "cv_window_policy": ["fixed_sliding"],
                "train_window_years": [4],
                "holdout_year": [2025],
                "window_definition": ["train = previous 4 years only, valid = current year"],
            }
        ),
        metric_frame=loaded,
    )

    assert meta["cv_policy"]["cv_window_policy"] == "fixed_sliding"
    assert meta["cv_policy"]["train_window_years"] == 4
    assert meta["cv_policy"]["valid_years"] == [2024]
    assert meta["cv_policy"]["holdout_year"] == 2025
    assert meta["input"]["input_filter_holdout_year"] == 2025
