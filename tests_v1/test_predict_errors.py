from __future__ import annotations

import pytest

from scripts import predict


class _DummyModel:
    def predict(self, matrix):
        return matrix


def test_predict_race_handles_only_obstacle_error(monkeypatch, capsys):
    monkeypatch.setattr(
        predict,
        "load_model",
        lambda: (_DummyModel(), None, ["track_code"]),
    )
    monkeypatch.setattr(
        predict,
        "get_race_features",
        lambda db, race_id: (_ for _ in ()).throw(
            predict.ObstacleRaceNotSupportedError(
                "障害レース(surface=3)は今回の予測対象外です: 123"
            )
        ),
    )

    predict.predict_race(db=None, race_id=123, odds_dict={})
    out = capsys.readouterr().out
    assert "障害レース(surface=3)は今回の予測対象外です: 123" in out


def test_predict_race_reraises_non_obstacle_value_error(monkeypatch):
    monkeypatch.setattr(
        predict,
        "load_model",
        lambda: (_DummyModel(), None, ["track_code"]),
    )
    monkeypatch.setattr(
        predict,
        "get_race_features",
        lambda db, race_id: (_ for _ in ()).throw(ValueError("レースが見つかりません: 123")),
    )

    with pytest.raises(ValueError, match="レースが見つかりません: 123"):
        predict.predict_race(db=None, race_id=123, odds_dict={})
