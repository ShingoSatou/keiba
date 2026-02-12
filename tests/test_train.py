"""train.py の分割ロジックのテスト"""

import pandas as pd
import pytest

from scripts.train import _split_by_race_id


class TestSplitByRaceId:
    """race_id 境界分割のテスト"""

    @pytest.fixture
    def sample_df(self):
        """テスト用データフレーム (20レース分)"""
        # 20レースを作成 (各レース2-3頭) → 6:1:1:2 = 12:2:2:4 レース
        data = []
        for i in range(20):
            race_id = f"R{i:03d}"
            race_date = f"2024-01-{i + 1:02d}"
            n_horses = 3 if i % 2 == 0 else 2
            for j in range(n_horses):
                data.append(
                    {
                        "race_id": race_id,
                        "race_date": race_date,
                        "feature1": i * 10 + j,
                        "is_win": 1 if j == 0 else 0,
                    }
                )
        return pd.DataFrame(data)

    def test_no_race_leakage(self, sample_df):
        """同一レースが複数分割に跨がらないこと"""
        X = sample_df[["feature1"]]
        y = sample_df["is_win"]

        result = _split_by_race_id(sample_df, X, y)
        X_train, _, X_es_val, _, X_test, _ = result

        train_races = set(sample_df.loc[X_train.index, "race_id"])
        es_val_races = set(sample_df.loc[X_es_val.index, "race_id"])
        test_races = set(sample_df.loc[X_test.index, "race_id"])

        # 各セット間で重複がないこと
        assert len(train_races & es_val_races) == 0, "TrainとES-Valに重複"
        assert len(train_races & test_races) == 0, "TrainとTestに重複"
        assert len(es_val_races & test_races) == 0, "ES-ValとTestに重複"

    def test_temporal_order(self, sample_df):
        """時系列順になっていること (Train < ES-Val < Test)"""
        X = sample_df[["feature1"]]
        y = sample_df["is_win"]

        result = _split_by_race_id(sample_df, X, y)
        X_train, _, X_es_val, _, X_test, _ = result

        train_max = sample_df.loc[X_train.index, "race_date"].max()
        es_val_min = sample_df.loc[X_es_val.index, "race_date"].min()
        es_val_max = sample_df.loc[X_es_val.index, "race_date"].max()
        test_min = sample_df.loc[X_test.index, "race_date"].min()

        assert train_max < es_val_min, "TrainがES-Valより後"
        assert es_val_max < test_min, "ES-ValがTestより後"

    def test_all_rows_assigned(self, sample_df):
        """全行がいずれかのセットに割り当てられること"""
        X = sample_df[["feature1"]]
        y = sample_df["is_win"]

        result = _split_by_race_id(sample_df, X, y)
        X_train, _, X_es_val, _, X_test, _ = result

        total = len(X_train) + len(X_es_val) + len(X_test)
        assert total == len(sample_df), f"行数が合わない: {total} != {len(sample_df)}"

    def test_production_mode(self, sample_df):
        """本番モード (test_size=0.0) の動作確認"""
        X = sample_df[["feature1"]]
        y = sample_df["is_win"]

        # test_size=0.0 で呼び出し
        result = _split_by_race_id(sample_df, X, y, test_size=0.0, es_val_size=0.1)
        X_train, _, X_es_val, _, X_test, _ = result

        assert len(X_test) == 0, "Testセットが空であること"
        assert len(X_es_val) > 0, "ES-Valセットが存在すること"
        assert len(X_train) > 0, "Trainセットが存在すること"

        # Train + ES-Val = 全体
        total = len(X_train) + len(X_es_val)
        assert total == len(sample_df), "全データがTrainとES-Valに使われていること"

    def test_missing_race_id_raises(self):
        """race_id がないとエラー"""
        df = pd.DataFrame({"race_date": ["2024-01-01"], "feature1": [1], "is_win": [0]})
        X = df[["feature1"]]
        y = df["is_win"]

        with pytest.raises(ValueError, match="race_id"):
            _split_by_race_id(df, X, y)

    def test_missing_race_date_raises(self):
        """race_date がないとエラー"""
        df = pd.DataFrame({"race_id": ["R001"], "feature1": [1], "is_win": [0]})
        X = df[["feature1"]]
        y = df["is_win"]

        with pytest.raises(ValueError, match="race_date"):
            _split_by_race_id(df, X, y)
