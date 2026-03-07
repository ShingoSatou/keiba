# Assumptions（v3実装時の推定前提）

1. v3 の標準評価条件は stage ごとに固定する。
   - binary: `cv_window_policy=fixed_sliding` / `train_window_years=4`
   - stacker: `cv_window_policy=capped_expanding` / `min_train_years=2` / `max_train_years=4`
   - PL: `cv_window_policy=fixed_sliding` / `train_window_years=3`
2. `core.o1_win` / `core.o1_place` の snapshot は as-of 契約を厳守し、`announce <= as_of` を満たす最新値だけを採用する。
3. stacker の市場入力は固定 snapshot のみを使う。
   - `stack_win`: `p_win_odds_t20/t15/t10_norm` と `d_logit_win_*`
   - `stack_place`: `place_mid_prob_t20/t15/t10`, `place_width_log_ratio_t20/t15/t10`, `d_place_*`
4. `place_width_log_ratio` は `log(high / low)`、`place_mid_prob_t*` は `1 / sqrt(low * high)` とする。
5. binary / stacker / PL ともに `feature_registry_v3.py` の whitelist / contract を使う。
   - binary default は市場オッズを含めない
6. PL の default feature profile は `stack_default` とする。
   - `z_win_stack`
   - `z_place_stack`
   - `place_width_log_ratio`
   - interaction / race-relative block
7. grouped meta (`p_win_meta`, `p_place_meta`) は comparison 用に残すが、default main path では使わない。
8. stacker 予測は strict temporal OOF / holdout / inference のみを downstream に渡し、同年 fitted 値は使わない。
9. current repo の年範囲では `base_oof_years=2020-2024`, `stacker_oof_years=2022-2024` となるため、`holdout_year=2025` の PL fixed3 OOF fold は 0 件になる。
10. 上記 9 の場合でも `train_pl_v3.py` は失敗せず、空 OOF と `v3_pipeline_year_coverage.json` を出力しつつ holdout/final 学習を継続する。
11. final odds は検証用途とし、default stacker / PL main path には入れない。
12. PL は馬 ID 固定効果 `u` を持たず、線形スコア `w^T x` のみを学習する。
13. `p_top3` / `p_wide` は Monte Carlo 推定を採用し、乱数 seed は `race_id` と global seed から決定する。
14. v3 の標準 ROI 評価経路は `scripts_v3/backtest_wide_v3.py` を使う。
15. `features_v3_te.parquet` / `features_v3_te_{holdout_year}.parquet` は `scripts_v3/build_features_v3_te.py` で生成し、current `features_v3*` の列集合に safe TE extra 列だけを追加したものとする。
