from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class BetDecision:
    """ベット判断の結果"""

    should_bet: bool
    amount: int
    reason: str = ""


class BettingStrategy(ABC):
    """ベット戦略の基底クラス"""

    @abstractmethod
    def decide_bet(self, prob: float, odds: float, **kwargs) -> BetDecision:
        """
        ベット判断を行う

        Args:
            prob: 予測勝率 (0.0 - 1.0)
            odds: オッズ (例: 2.5)
            kwargs: その他の情報 (bankrollなど)

        Returns:
            BetDecision
        """
        pass


class FixedBetStrategy(BettingStrategy):
    """固定額ベット戦略"""

    def __init__(self, bet_amount: int = 500, min_prob: float = 0.0, min_ev: float = 0.0):
        self.bet_amount = bet_amount
        self.min_prob = min_prob
        self.min_ev = min_ev

    def decide_bet(self, prob: float, odds: float, **kwargs) -> BetDecision:
        # 期待値計算
        ev_profit = prob * odds - 1.0

        if prob < self.min_prob:
            return BetDecision(False, 0, "Low probability")

        if ev_profit <= self.min_ev:
            return BetDecision(False, 0, "Low EV")

        return BetDecision(True, self.bet_amount, "Fixed bet")


class KellyStrategy(BettingStrategy):
    """ケリー基準ベット戦略"""

    def __init__(
        self,
        kelly_fraction: float = 1.0,
        min_bet: int = 500,
        max_bet: int = 3000,
        fixed_bet_mode: bool = False,
        fixed_bet_amount: int = 500,
    ):
        """
        Args:
            kelly_fraction: ケリー係数 (例: 0.5 でハーフケリー)
            min_bet: 最小ベット額 (これ未満は見送り)
            max_bet: 最大ベット額 (これより多くは賭けない)
            fixed_bet_mode: Trueなら、amount決定にケリー値を使わず、判定のみに使う (戦略A用)
            fixed_bet_amount: fixed_bet_mode時のベット額
        """
        self.kelly_fraction = kelly_fraction
        self.min_bet = min_bet
        self.max_bet = max_bet
        self.fixed_bet_mode = fixed_bet_mode
        self.fixed_bet_amount = fixed_bet_amount

    def calculate_kelly_f(self, prob: float, odds: float) -> float:
        """
        ケリー基準値 f* を計算
        f* = (p * (o - 1) - (1 - p)) / (o - 1)
           = (p * o - 1) / (o - 1)
        """
        if odds <= 1.0:
            return 0.0

        p = prob
        b = odds - 1.0

        f_star = (p * b - (1 - p)) / b
        return max(0.0, f_star)

    def decide_bet(self, prob: float, odds: float, **kwargs) -> BetDecision:
        bankroll = kwargs.get("bankroll", 0)

        # 1. f* の計算
        f_star = self.calculate_kelly_f(prob, odds)

        # 2. 係数の適用
        # f* が 0以下なら即見送り
        if f_star <= 0:
            return BetDecision(False, 0, "Negative Kelly")

        # 戦略A: ケリー基準 > 0 なら固定額
        if self.fixed_bet_mode:
            # 係数を考慮しても正なら購入対象 (通常 f* > 0 なら fraction * f* も > 0)
            return BetDecision(True, self.fixed_bet_amount, "Kelly Filter Passed")

        # 戦略B: 変動ベット
        # Bet = Bankroll * Fraction * f*
        raw_amount = bankroll * self.kelly_fraction * f_star

        # 3. 制約の適用
        if raw_amount < self.min_bet:
            return BetDecision(False, 0, f"Below min_bet ({int(raw_amount)} < {self.min_bet})")

        bet_amount = min(int(raw_amount), self.max_bet)

        # 100円単位に切り捨て (JRA仕様)
        bet_amount = (bet_amount // 100) * 100

        if bet_amount == 0:
            return BetDecision(False, 0, "Round down to zero")

        return BetDecision(True, bet_amount, f"Kelly {f_star:.3f} -> {bet_amount}")
