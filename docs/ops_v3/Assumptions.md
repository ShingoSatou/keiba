# Assumptions（v3実装時の推定前提）

1. Rolling年次CVのロジックは v2 `build_rolling_year_folds()` をそのまま利用し、デフォルト `train_window_years=5` は可変パラメータとして扱う。
   - ただし PL は OOF行だけを使うため有効年が短くなるケースがあり、実行デフォルトを `3` にしている。
2. Benter R* の `beta` 最適化は fold train 内の in-sample 予測（trainでfitした分類器の train_pred）で実施する。
3. `core.o1_win` の `win_odds_x10` が `NULL/0/負` の場合は欠損として扱う。
4. final odds は検証用途とし、運用推論（`predict_race_v3.py`）は t10 特徴のみを許可する。
   - ただし現状の win/place モデルは `features_v3` の final/t10 odds列を特徴として含むため、厳密な「運用=t10」には再設計が必要。
5. PL学習で使う後段入力は OOF 予測のみとし、OOFが欠ける行は PL 学習対象から除外する。
6. PLは `u`（馬ID固定効果）を持たず、線形スコア `w^T x` のみを学習する。
7. PLの `p_top3` / `p_wide` は Monte Carlo 推定を採用し、乱数シードは race_id とグローバルseedから決定する。
8. ROI算出は v2の `scripts_v2/backtest_wide_v2.py` を利用し、`p_top3` から `p_wide` を推定している（`pl_score` を直接は使っていない）。
