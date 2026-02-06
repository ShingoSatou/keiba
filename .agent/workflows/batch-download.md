---
description: 過去10年分のJV-Linkデータを2年刻みで取得・ロードする
---

# バッチダウンロード ワークフロー

過去10年分 (2016-2025) のJV-Linkデータを2年刻みで取得し、DBにロードする手順。

## 前提条件
- JV-Link がインストール済み
- 32bit Python環境 (`.venv32`) が構築済み
- PostgreSQL DB が起動済み

---

## Step 1: マスタデータ取得 (1回のみ)

// turbo
```powershell
.\.venv32\Scripts\python.exe scripts/extract_jvlink.py `
    --dataspec DIFF --from-date 20160101 --option 4
```

これで `data/DIFF_*.jsonl` が生成される（馬・騎手・調教師マスタ）。

// turbo
```powershell
uv run python scripts/load_to_db.py --input data/DIFF_*.jsonl
```

---

## Step 2: RACEデータ取得 (2年刻み)

// turbo
```powershell
# 2016-2017
.\.venv32\Scripts\python.exe scripts/extract_jvlink.py `
    --from-date 20160101 --to-date 20171231 --option 4

# 2018-2019
.\.venv32\Scripts\python.exe scripts/extract_jvlink.py `
    --from-date 20180101 --to-date 20191231 --option 4

# 2020-2021
.\.venv32\Scripts\python.exe scripts/extract_jvlink.py `
    --from-date 20200101 --to-date 20211231 --option 4

# 2022-2023
.\.venv32\Scripts\python.exe scripts/extract_jvlink.py `
    --from-date 20220101 --to-date 20231231 --option 4

# 2024-2025
.\.venv32\Scripts\python.exe scripts/extract_jvlink.py `
    --from-date 20240101 --to-date 20251231 --option 4
```

---

## Step 3: DBロード

各ファイルごと、またはまとめてロード：

// turbo
```powershell
uv run python scripts/load_to_db.py --input data/RACE_20160101-20171231_*.jsonl
uv run python scripts/load_to_db.py --input data/RACE_20180101-20191231_*.jsonl
# ... 繰り返し
```

または一括：

// turbo
```powershell
uv run python scripts/load_to_db.py --input data/RACE_*.jsonl
```

---

## 検証

// turbo
```powershell
uv run python scripts/analyze_db_stats.py
```

期待: 2016年〜2025年のデータが均等に分布していること。
