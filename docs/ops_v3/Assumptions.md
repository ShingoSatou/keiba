# Assumptions（v3実装時の推定前提）

1. v3 の標準評価条件は `cv_window_policy=fixed_sliding` / `train_window_years=4` とする。
   - valid year ごとに直前4年のみを train に使う。
   - 4年未満しか履歴がない初期年は fold を作らずスキップする。
   - holdout year（例: `2025`）は将来年として従来どおり使ってよい。
   - この条件は今後のモデル改善比較の基準であり、過去 run report の窓条件とは直接比較しない場合がある。
2. Benter R* の `beta` 最適化は fold train 内の in-sample 予測（trainでfitした分類器の train_pred）で実施する。
3. `core.o1_win` の `win_odds_x10` が `NULL/0/負` の場合は欠損として扱う。
4. binary / PL ともに `scripts_v3/feature_registry_v3.py` の whitelist / contract を使う。
   - default operational profile は `t10_only`
   - `features_v3` に final/t10 odds 列が存在しても、学習投入可否は registry 側で制御する
5. binary の entity raw ID（`jockey_key`, `trainer_key`）は default で OFF とし、`--include-entity-id-features` 指定時のみ投入する。
6. final odds は検証用途とし、opt-in がない限り学習投入しない。
   - binary: `--operational-mode includes_final`
   - PL: `--include-final-odds-features`
   - 運用推論（`predict_race_v3.py`）は引き続き t10 特徴のみを許可する
7. PL の default feature profile は `meta_default` とする。
   - `p_win_meta`
   - `p_place_meta`
   - `p_win_odds_t10_norm`
   - `PL_CONTEXT_FEATURES_SMALL`
   - 比較用に `raw_legacy` profile を残す。
8. `p_win_meta` / `p_place_meta` は base 予測の結合器として `race_id` grouped の通常CVで生成する convenience OOF とする。
   - `StratifiedGroupKFold` を第一候補とし、難しい場合は `GroupKFold` にフォールバックする。
   - `cv_is_temporal = false`
   - `meta_oof_is_strict_temporal = false`
   - `meta_metrics_are_reference_only = true`
9. PL学習で使う後段入力は contract で要求された予測列のみとし、必要列が欠ける行は PL 学習対象から除外する。
10. PL の context 特徴は small registry に固定し、frame 全 numeric の自動収集は行わない。
11. PLは `u`（馬ID固定効果）を持たず、線形スコア `w^T x` のみを学習する。
12. PLの `p_top3` / `p_wide` は Monte Carlo 推定を採用し、乱数シードは race_id とグローバルseedから決定する。
13. ROI算出のデフォルトは v3の `scripts_v3/backtest_wide_v3.py` を利用し、`pl_score -> MC -> p_wide`（または `train_pl_v3.py --emit-wide-oof` が出力した `p_wide`）で評価する。
14. 旧方式の v2近似（`scripts_v2/backtest_wide_v2.py`: `p_top3 -> p/(1-p) -> p_wide`）は比較用の参考経路として扱い、v3の標準評価経路にはしない。
15. binary では bundle meta に加えて feature manifest を保存し、feature contract は unit test で固定する。
