from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.check_dataset_quality import main
from scripts.train import FEATURE_COLS


def _build_dataset(rows_per_race: int = 5, races: int = 10) -> pd.DataFrame:
    records: list[dict] = []
    race_base = 202601010100

    for race_index in range(races):
        race_id = race_base + race_index
        race_date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=race_index)
        for horse_no in range(1, rows_per_race + 1):
            row: dict = {feature: 1.0 for feature in FEATURE_COLS}
            row.update(
                {
                    "race_id": race_id,
                    "race_date": race_date,
                    "horse_id": f"h{race_index:03d}{horse_no:02d}",
                    "horse_no": horse_no,
                    "track_code": 5,
                    "surface": 1,
                    "distance_m": 1600,
                    "going": 1,
                    "class_code": 0,
                    "field_size": rows_per_race,
                    "gate": horse_no,
                    "distance_change_m": horse_no,
                    "n_runs_5": 3,
                    "n_sim_runs_5": 3,
                    "is_win": 1 if horse_no == 1 else 0,
                }
            )
            records.append(row)

    return pd.DataFrame(records)


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def test_quality_gate_passes_for_clean_dataset(tmp_path, monkeypatch):
    dataset_path = tmp_path / "train.parquet"
    _write_parquet(dataset_path, _build_dataset())

    monkeypatch.setattr(
        sys,
        "argv",
        ["check_dataset_quality.py", "--input", str(dataset_path), "--gate"],
    )

    main()


def test_quality_gate_fails_for_placeholder_rows(tmp_path, monkeypatch):
    dataset_path = tmp_path / "train_bad.parquet"
    df = _build_dataset()
    df.loc[df.index[0], "surface"] = 0
    _write_parquet(dataset_path, df)

    monkeypatch.setattr(
        sys,
        "argv",
        ["check_dataset_quality.py", "--input", str(dataset_path), "--gate"],
    )

    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
