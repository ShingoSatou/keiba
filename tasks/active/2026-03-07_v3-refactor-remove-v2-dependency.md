# v3 リファクタリング: v2 依存排除とドキュメント整備

| 項目 | 値 |
|---|---|
| version | v3 |
| 対象範囲 | `scripts_v3/`, `test_v3/`, `docs/specs_v3/`, `docs/ops_v3/`, `models/`, `data/` |
| 完了日 | 2026-03-07 |

---

## チェックリスト

### Phase 1: legacy profile (`raw_legacy`) の削除

- [x] `feature_registry_v3.py`: `raw_legacy` プロファイルを `PL_FEATURE_PROFILE_CHOICES` から除去
- [x] `feature_registry_v3.py`: `PL_REQUIRED_PRED_FEATURES_RAW_LEGACY` 定数と関連分岐を削除
- [x] `feature_registry_v3.py`: `PL_REQUIRED_PRED_FEATURES` エイリアスを削除
- [x] `feature_registry_v3.py`: `__all__` を整理
- [x] `train_pl_v3.py`: `raw_legacy` 分岐を削除
- [x] `predict_race_v3.py`: `_infer_pl_feature_profile` の fallback を `meta_default` に変更
- [x] テスト更新: 3ファイルから `raw_legacy` テスト4件を削除/書き換え

### Phase 2: v2 変数名・コメントのクリーンアップ

- [x] `build_features_v3.py`: デフォルト・変数名・description を修正
- [x] `v3_common.py`: v2 由来コメントを削除

### Phase 3: v2 モデルファイル削除 + データリネーム

- [x] `models/` から v2 専用ファイル 35件を削除（ranker_*, calibrator_*, t5_bundle_*, lgb_model）
- [x] `models/` から raw_legacy PL artifacts 3件を削除
- [x] `data/features_v2.parquet` → `data/features_base.parquet` にリネーム

### Phase 4: ドキュメント整備

- [x] `v3_01_全体アーキテクチャ.md`: legacy path 削除、データフロー更新
- [x] `v3_04_PL推論とワイドバックテスト仕様.md`: `raw_legacy` 削除
- [x] `v3_05_共通基盤と付録.md`: CLI choices 更新
- [x] `スクリプトリファレンス.md`: v2入力パス更新、raw_legacy コマンド削除

### Phase 5: 検証

- [x] `uv run ruff format .` — clean
- [x] `uv run ruff check .` — All checks passed
- [x] `uv run pytest -q` — 135 passed
