## 目的 / 背景
- バックテスト結果（`data/backtest_result.json`）をダミーデータから実データに置き換え、Web UIで正しく表示・分析できるようにするため
- バックテスト実行ロジックを app/services 配下に分離し、UI との繋ぎ込みをスムーズに行うため

## 変更内容
- `app/services/backtest_runner.py` を新規追加（モデルロード、DBからの情報取得、予想確率計算、JSONファイル生成までを一貫して実行）
- バックテストでの馬名・馬番取得処理を追加（`core.horse`, `core.runner` テーブルとの JOIN）
- `app/routers/ui.py` に `POST /ui/reload` エンドポイントを追加（FastAPIの再起動なしで最新のJSONを読み込み可能に）
- `test_backtest_runner.py` を新規追加し、ロジックとJSON生成構造のユニットテストを12件実装

## 影響範囲
- API: `POST /ui/reload` が新たに追加された。既存の GET エンドポイントへの影響はなし。
- 挙動: バックテスト実行スクリプトを利用することで、最新形式の `backtest_result.json` が出力されるようになった。
- 設定: 特になし。

## 動作確認（実施済み）
- [x] uv run ruff format .
- [x] uv run ruff check .
- [x] uv run pytest -q

## リスク / 注意点
- 現状の戦略設定でのバックテスト結果（ROI 約76.2%）は回収率が100%を下回っているため、パラメータ調整（ev_threshold等）の見直しが必要です。
- キャッシュリロード機能は追加しましたが、バックテスト自体のオンデマンド実行APIは今回見送っています（現状はCLIから実行）。

## メモ / TODO
- パラメータ調整による戦略の改善
- 実行時間短縮のためのチューニング
