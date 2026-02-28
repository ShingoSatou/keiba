# Phase 5（v2）Wideバックテスト：`min_p_wide` 閾値スイープ記録（2026-03-01）

## 目的

- `p_top3`（校正済み）→ PL+MC による `p_wide` 推定 → EV → 資金配分 → バックテスト、までの PoC 実装を前提に、
  **確率足切り `min_p_wide`** の最適化（OOF期間でのグリッドサーチ）を記録する。
- 注意: `min_p_wide` は「低確率だからダメ」ではなく、**低確率帯での過大推定がEV選抜に集中しやすい**ため、
  買い方の安全装置として導入する（校正器を設ける前段の暫定策）。

## 用語（このメモ内）

- `p_top3`: 各馬が3着以内に入る単独確率（Phase 4の校正器出力）
- `p_wide`: 2頭がともに3着以内に入る同時確率（Phase 5で PL+MC 推定）
- `min_p_wide`: `p_wide` の購入下限（確率足切り）
- `min_p_wide_stage`:
  - `candidate`: 候補段階で除外（点数上限の「置き換え」が起きる）
  - `selected`: 選抜後に除外（置き換えない）
- ROI: `total_return / total_bet`

## OOF（2023–2024）での閾値スイープ（最適化）

- スイープ出力:
  - `data/oof/min_p_wide_sweep_2023_2024_selected.json`
  - `data/oof/min_p_wide_sweep_2023_2024_selected_meta.json`
- 設定:
  - 対象: `valid_year in {2023, 2024}`（`data/oof/top3_convex_oof_cw_none.parquet`）
  - `min_p_wide_stage=selected`
  - グリッド: `min_p_wide in [0.00, 0.15]`（step=0.01）
- 結果（best）:
  - `min_p_wide=0.11` が最大（ROI=0.8696, n_bets=420）
  - 年別ROI: 2023=0.7572 / 2024=1.0138

再現コマンド例:

```bash
uv run python scripts_v2/sweep_min_p_wide_threshold_v2.py \
  --input data/oof/top3_convex_oof_cw_none.parquet \
  --years 2023,2024 \
  --require-years 2023,2024 \
  --holdout-year 2025 \
  --min-p-wide-stage selected \
  --grid-start 0.00 --grid-stop 0.15 --grid-step 0.01 \
  --output data/oof/min_p_wide_sweep_2023_2024_selected.json \
  --meta-output data/oof/min_p_wide_sweep_2023_2024_selected_meta.json
```

## Holdout（2025）での評価（固定閾値の適用）

- 2025評価に使用した閾値: `min_p_wide=0.11`（OOF最適値を固定して適用）
- 出力:
  - `data/holdout/backtest_wide_2025_convex_cw_none_reval_fixed_minp011_selected.json`
  - `data/holdout/backtest_wide_2025_convex_cw_none_reval_fixed_minp011_selected_meta.json`
- 結果（2025）:
  - ROI=0.8238, n_bets=245, n_hits=21, max_drawdown=1,666,150

比較（足切りなし）:

- `data/holdout/backtest_wide_2025_convex_cw_none_reval_fixed.json`（`min_p_wide=0.0`）: ROI=0.7041

## 注意（one-shot運用）

- `docs/specs_v2/時系列交差検証およびモデル評価仕様書.md` の通り、
  holdout年（例: 2025）で閾値・買い方を最適化してはいけない。
- 本リポジトリの作業履歴上、2025は既に複数回参照しているため、今後「厳密な封印」を継続する場合は
  **封印期間を 2026 以降へ更新**して運用する。

## 次の打ち手（暫定足切り→本命対応）

- 暫定: `min_p_wide_stage=selected` の確率足切りで「低確率帯の過大推定」影響を抑える
- 本命: `p_wide` 自体を OOF で校正（選抜されやすい領域での過大を潰す）し、`min_p_wide` 依存を下げる

