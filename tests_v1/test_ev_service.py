"""
EV計算サービスのテスト

EVService の計算ロジックと推奨ロジックをテストする。
"""

import pytest

from app.services.ev_service import EVResult, EVService


class TestEVService:
    """EVService のテスト"""

    def test_calculate_ev_positive(self):
        """EV > 0 のケース: 買い推奨になる"""
        service = EVService(slippage=0.15, min_prob=0.03)

        # p_win=0.20, odds=6.0 → Oeff=5.1, EVprofit=0.02
        result = service.calculate_ev(p_win=0.20, odds_10min=6.0)

        assert isinstance(result, EVResult)
        assert result.odds_effective == pytest.approx(5.1, rel=0.01)
        assert result.ev_profit == pytest.approx(0.02, rel=0.01)
        assert result.is_buy is True

    def test_calculate_ev_negative(self):
        """EV < 0 のケース: 見送りになる"""
        service = EVService(slippage=0.15, min_prob=0.03)

        # p_win=0.10, odds=5.0 → Oeff=4.25, EVprofit=-0.575
        result = service.calculate_ev(p_win=0.10, odds_10min=5.0)

        assert result.odds_effective == pytest.approx(4.25, rel=0.01)
        assert result.ev_profit == pytest.approx(-0.575, rel=0.01)
        assert result.is_buy is False

    def test_calculate_ev_low_probability(self):
        """確率が低すぎる場合: EVが正でも見送り"""
        service = EVService(slippage=0.15, min_prob=0.03)

        # p_win=0.02 (< min_prob=0.03), odds=100.0 → EVは正だが見送り
        result = service.calculate_ev(p_win=0.02, odds_10min=100.0)

        # EV自体は正
        assert result.ev_profit > 0
        # しかし min_prob 未満なので買わない
        assert result.is_buy is False

    def test_calculate_ev_boundary(self):
        """EV=0 の境界ケース: 見送りになる"""
        service = EVService(slippage=0.0, min_prob=0.0)

        # スリッページ0で p_win=0.5, odds=2.0 → 期待値ちょうど0
        result = service.calculate_ev(p_win=0.5, odds_10min=2.0)

        assert result.ev_profit == pytest.approx(0.0, abs=0.001)
        assert result.is_buy is False  # EV > 0 ではないので

    def test_recommend_single_buy(self):
        """買い候補が1つある場合"""
        service = EVService(slippage=0.15, min_prob=0.03)

        candidates = [
            {"horse_no": 1, "horse_name": "Horse A", "p_win": 0.20, "odds_10min": 6.0},
            {"horse_no": 2, "horse_name": "Horse B", "p_win": 0.05, "odds_10min": 5.0},
        ]

        result = service.recommend(candidates)

        assert result is not None
        assert result["horse_no"] == 1
        assert result["bet_amount"] == 500

    def test_recommend_multiple_buy_selects_best_ev(self):
        """複数の買い候補がある場合: EV最大を選択"""
        service = EVService(slippage=0.10, min_prob=0.03)

        candidates = [
            {
                "horse_no": 1,
                "horse_name": "Horse A",
                "p_win": 0.15,
                "odds_10min": 10.0,
            },  # EV = 0.35
            {"horse_no": 2, "horse_name": "Horse B", "p_win": 0.20, "odds_10min": 8.0},  # EV = 0.44
            {
                "horse_no": 3,
                "horse_name": "Horse C",
                "p_win": 0.10,
                "odds_10min": 15.0,
            },  # EV = 0.35
        ]

        result = service.recommend(candidates)

        assert result is not None
        assert result["horse_no"] == 2  # EV最大

    def test_recommend_no_buy(self):
        """買い候補がない場合: None を返す"""
        service = EVService(slippage=0.15, min_prob=0.03)

        candidates = [
            {"horse_no": 1, "horse_name": "Horse A", "p_win": 0.05, "odds_10min": 5.0},  # EV < 0
            {"horse_no": 2, "horse_name": "Horse B", "p_win": 0.02, "odds_10min": 50.0},  # p < min
        ]

        result = service.recommend(candidates)

        assert result is None

    def test_recommend_empty_candidates(self):
        """候補がない場合: None を返す"""
        service = EVService()

        result = service.recommend([])

        assert result is None


class TestEVServiceParameters:
    """EVService のパラメータ設定テスト"""

    def test_default_parameters(self):
        """デフォルトパラメータの確認"""
        service = EVService()

        assert service.slippage == 0.15
        assert service.min_prob == 0.03
        assert service.bet_amount == 500

    def test_custom_parameters(self):
        """カスタムパラメータの確認"""
        service = EVService(slippage=0.20, min_prob=0.05, bet_amount=1000)

        assert service.slippage == 0.20
        assert service.min_prob == 0.05
        assert service.bet_amount == 1000

    def test_slippage_affects_calculation(self):
        """スリッページ率が計算に影響することを確認"""
        service_low = EVService(slippage=0.10)
        service_high = EVService(slippage=0.30)

        result_low = service_low.calculate_ev(p_win=0.20, odds_10min=6.0)
        result_high = service_high.calculate_ev(p_win=0.20, odds_10min=6.0)

        # スリッページが高いほど有効オッズが低い
        assert result_low.odds_effective > result_high.odds_effective
        # スリッページが高いほどEVが低い
        assert result_low.ev_profit > result_high.ev_profit
