# 2026-03-07_v3-te-leakage-audit-fix

## タイトル

- v3 TE リーク監査と必要時の修正・xgb 再学習

## ステータス

- `done`

## 対象範囲

- `scripts_v2/build_features_v2.py`
- `scripts_v3/build_features_v3.py`
- `test_v2/test_features_v2_no_leakage.py`
- `data/features_v2_te*.parquet`
- `data/features_v3_te*.parquet`
- `scripts_v3/train_win_xgb_v3.py`

## 対象バージョン

- `v3`

## 前提・仮定

- v3 の TE 入力は `data/features_v2_te*.parquet` を元に `build_features_v3.py` で生成されている。
- リークが見つかった場合、まず source の TE 生成ロジックを修正する。
- xgb の再学習対象は TE 依存の `win_xgb` を最優先で確認する。

## チェックリスト

- [x] TE のリーク有無を仕様と実装で確認する
- [x] リークがあれば修正とテスト追加を行う
- [x] 実データ TE 成果物を再生成する
- [x] xgb 再学習後のスコアを算出する

## 確認結果

- `scripts_v2/build_features_v2.py` の TE は、entity rolling 自体は `closed="left"` で同日を除外していた。
- 一方で `with_te` 時の `prior_label_mean` が build 対象全期間の `target_label` 平均で算出されており、future leakage になっていた。
- 修正前は同じ 2016-01-05 行でも `to_date` を延ばすと TE 値が変わっていたが、修正後は short horizon build と full build で完全一致した。
- 修正後の `features_v2_te*.parquet` / `features_v3_te*.parquet` を再生成済み。
- `win_xgb` を tuned params のまま再学習し、mean CV logloss は `0.2086504017556902` だった。
- 旧 tuned best の mean CV logloss `0.2085696064918138` に対して差分は `+0.00008079526387638758`。
- `models/win_xgb_bundle_meta_v3.json` で input が `data/features_v3_te.parquet` のまま再学習されていることを確認済み。

## 実行コマンド

- `uv run pytest -q -s test_v2/test_features_v2_no_leakage.py`
- `uv run python - <<'PY' ... build_features_dataframe(... to_date=2016-01-10) と build_features_dataframe(... to_date=2024-12-31) の 2016-01-05 TE を比較 ... PY`
- `uv run python scripts_v2/build_features_v2.py --from-date 2016-01-01 --to-date 2024-12-31 --with-te --output data/features_v2_te.parquet --meta-output data/features_v2_te_meta.json --log-level INFO`
- `uv run python scripts_v2/build_features_v2.py --from-date 2025-01-01 --to-date 2025-12-31 --with-te --output data/features_v2_te_2025.parquet --meta-output data/features_v2_te_2025_meta.json --log-level INFO`
- `uv run python scripts_v3/build_features_v3.py --input data/features_v2_te.parquet --output data/features_v3_te.parquet --meta-output data/features_v3_te_meta.json --log-level INFO`
- `uv run python scripts_v3/build_features_v3.py --input data/features_v2_te_2025.parquet --output data/features_v3_te_2025.parquet --meta-output data/features_v3_te_2025_meta.json --log-level INFO`
- `V3_MODEL_THREADS=4 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 uv run python scripts_v3/train_win_xgb_v3.py --log-level INFO`
- `uv run ruff check scripts_v2/build_features_v2.py test_v2/test_features_v2_no_leakage.py`

## 関連ドキュメント

- `docs/specs_v3/v3_03_二値分類と校正仕様.md`
- `docs/specs_v2/特徴量設計・データパイプライン仕様書.md`

## 残リスク

- `binary_v3_win_xgb` の tuned best はリーク修正前の study から選ばれているため、厳密には `win_xgb` の再 tuning をやり直した方がよい。
- `place_xgb` の tuned default は `feature_set=base` のため今回の TE 修正影響は受けない前提で、再学習は未実施。
