from __future__ import annotations

import pandas as pd
import pytest

from scripts_v3.backtest_wide_v3 import load_backtest_input, parse_args


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
