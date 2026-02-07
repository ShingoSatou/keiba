"""
EV計算・推奨ロジック

期待値（EV）を計算し、買い/見送りを推奨する。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EVResult:
    """EV計算結果"""

    odds_effective: float  # スリッページ考慮後オッズ
    ev_profit: float  # 期待利益 (p * Oeff - 1)
    is_buy: bool  # 買い推奨かどうか


class EVService:
    """期待値計算サービス"""

    def __init__(
        self,
        slippage: float = 0.15,
        min_prob: float = 0.03,
        bet_amount: int = 500,
    ):
        """
        Args:
            slippage: スリッページ率 (10分前→最終オッズの想定減少率)
            min_prob: 最低確率閾値 (これ以下は見送り)
            bet_amount: 賭け金 (円)
        """
        self.slippage = slippage
        self.min_prob = min_prob
        self.bet_amount = bet_amount

    def calculate_ev(self, p_win: float, odds_10min: float) -> EVResult:
        """
        EVを計算

        Args:
            p_win: 予測勝率
            odds_10min: 10分前オッズ

        Returns:
            EVResult
        """
        # スリッページ考慮後オッズ
        odds_effective = odds_10min * (1 - self.slippage)

        # 期待利益 = 勝率 × 有効オッズ - 1
        ev_profit = p_win * odds_effective - 1

        # 買い条件: EV > 0 かつ 最低確率以上
        is_buy = ev_profit > 0 and p_win > self.min_prob

        return EVResult(
            odds_effective=odds_effective,
            ev_profit=ev_profit,
            is_buy=is_buy,
        )

    def recommend(self, candidates: list[dict]) -> dict | None:
        """
        候補から最良の買い目を選択

        Args:
            candidates: 各馬の情報リスト
                - horse_no: 馬番
                - horse_name: 馬名
                - p_win: 予測勝率
                - odds_10min: 10分前オッズ

        Returns:
            推奨馬の情報 (なければ None)
        """
        evaluated = []

        for c in candidates:
            ev_result = self.calculate_ev(c["p_win"], c["odds_10min"])

            evaluated.append(
                {
                    **c,
                    "odds_effective": ev_result.odds_effective,
                    "ev_profit": ev_result.ev_profit,
                    "is_buy": ev_result.is_buy,
                }
            )

        # 買い候補のみ抽出
        buy_candidates = [e for e in evaluated if e["is_buy"]]

        if not buy_candidates:
            return None

        # EV最大を選択
        best = max(buy_candidates, key=lambda x: x["ev_profit"])
        best["bet_amount"] = self.bet_amount

        return best

    def format_recommendation(self, candidates: list[dict]) -> str:
        """
        推奨結果を整形済み文字列で返す

        Args:
            candidates: 候補リスト

        Returns:
            整形済み文字列
        """
        evaluated = []
        for c in candidates:
            ev_result = self.calculate_ev(c["p_win"], c["odds_10min"])
            evaluated.append(
                {
                    **c,
                    "odds_effective": ev_result.odds_effective,
                    "ev_profit": ev_result.ev_profit,
                    "is_buy": ev_result.is_buy,
                }
            )

        # EV順でソート
        evaluated.sort(key=lambda x: x["ev_profit"], reverse=True)

        lines = []
        lines.append("=" * 80)
        lines.append(
            f"{'馬番':>4}  {'馬名':<16}  {'p(win)':>7}  {'O10':>6}  {'EV利益':>8}  {'推奨':>6}"
        )
        lines.append("-" * 80)

        for e in evaluated:
            recommendation = "★ BUY" if e["is_buy"] else "SKIP"
            lines.append(
                f"{e['horse_no']:>4}  {e['horse_name']:<16}  {e['p_win']:>7.3f}  "
                f"{e['odds_10min']:>6.1f}  {e['ev_profit']:>+8.3f}  {recommendation:>6}"
            )

        lines.append("=" * 80)

        # 推奨
        best = self.recommend(candidates)
        if best:
            lines.append(
                f"推奨: {best['horse_no']}番 {best['horse_name']} を {self.bet_amount}円 で購入"
            )
        else:
            lines.append("推奨: なし（EV > 0 の馬がいません）")

        lines.append("=" * 80)

        return "\n".join(lines)
