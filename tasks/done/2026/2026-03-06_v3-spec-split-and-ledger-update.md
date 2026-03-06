# 2026-03-06_v3-spec-split-and-ledger-update

## タイトル

- v3仕様書分割と AGENTS の仕様書台帳化

## ステータス

- `done`

## 対象範囲

- `docs/specs_v3/v3_実装仕様.md` 削除
- `docs/specs_v3/v3_システム仕様書.md` を index 化
- `docs/specs_v3/v3_01_全体アーキテクチャ.md` 新規追加
- `docs/specs_v3/v3_02_特徴量生成とオッズ仕様.md` 新規追加
- `docs/specs_v3/v3_03_二値分類と校正仕様.md` 新規追加
- `docs/specs_v3/v3_04_PL推論とワイドバックテスト仕様.md` 新規追加
- `docs/specs_v3/v3_05_共通基盤と付録.md` 新規追加
- `AGENTS.md` 更新
- `README.md` 更新
- `tasks/todo.md` 更新

## 対象バージョン

- `v3`

## 前提・仮定

- monolithic な仕様書は削除せず、`v3_システム仕様書.md` を入口兼台帳として残す
- 分割単位は実装責務と日常の作業単位を優先する
- current contract に合わせて分割時に記述の古い箇所も補正する

## チェックリスト

- [x] `v3_実装仕様.md` を削除
- [x] `v3_システム仕様書.md` を index / 台帳へ置き換え
- [x] 分割仕様書を 5 本追加
- [x] `AGENTS.md` に作業別の仕様書台帳を追加
- [x] `README.md` の v3 導線を更新
- [x] `tasks/todo.md` に完了履歴を記録
- [x] 残参照と markdown 差分整合を確認

## 確認結果

- `docs/specs_v3/v3_実装仕様.md` を通常参照先から外し、index / README / AGENTS の導線を新構成へ更新
- `docs/specs_v3/v3_システム仕様書.md` は短い index / source-of-truth 案内として再構成
- old monolithic 仕様を `全体アーキテクチャ / 特徴量生成 / binary+校正 / PL+backtest / 共通基盤` に分割
- `AGENTS.md` に、作業内容ごとにどの仕様書を確認すべきかの台帳を追加

## 実行コマンド

- `rg -n '^## |^### ' docs/specs_v3/v3_システム仕様書.md`
- `wc -l docs/specs_v3/v3_システム仕様書.md docs/specs_v3/v3_実装仕様.md`
- `rg -n 'v3_実装仕様.md|v3_システム仕様書.md|docs/specs_v3/' -g '!tasks/**' -g '!data/**' -g '!models/**' .`
- `git diff --check`
- `rg --files tasks`

## 関連ドキュメント

- `docs/specs_v3/v3_システム仕様書.md`
- `docs/specs_v3/v3_01_全体アーキテクチャ.md`
- `docs/specs_v3/v3_02_特徴量生成とオッズ仕様.md`
- `docs/specs_v3/v3_03_二値分類と校正仕様.md`
- `docs/specs_v3/v3_04_PL推論とワイドバックテスト仕様.md`
- `docs/specs_v3/v3_05_共通基盤と付録.md`

## 残リスク

- 既存の外部メモや口頭運用で `v3_実装仕様.md` を参照している場合は、新しい index への認識合わせが必要
- 今回は docs 構造の再編であり、コード変更は伴わないため、テスト実行はしていない
