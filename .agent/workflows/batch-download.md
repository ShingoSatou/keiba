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

## Step 3: 追加データ取得（学習・特徴量で使用）

RACE以外にも、学習・特徴量で使用するdataspecを取得します（初回は `option 4` 推奨）。

// turbo
```powershell
# MING: DM/TM（学習はMINGを正）
.\.venv32\Scripts\python.exe scripts/extract_jvlink.py `
    --dataspec MING --from-date 20160101 --option 4

# SNPN: CK（SNAPは使わずSNPNで統一）
.\.venv32\Scripts\python.exe scripts/extract_jvlink.py `
    --dataspec SNPN --from-date 20160101 --option 4

# SLOP: 坂路調教 (HC)
.\.venv32\Scripts\python.exe scripts/extract_jvlink.py `
    --dataspec SLOP --from-date 20160101 --option 4

# WOOD: ウッド調教 (WC)
.\.venv32\Scripts\python.exe scripts/extract_jvlink.py `
    --dataspec WOOD --from-date 20160101 --option 4

# COMM: コース情報など (CS)
.\.venv32\Scripts\python.exe scripts/extract_jvlink.py `
    --dataspec COMM --from-date 20160101 --option 4

# YSCH: 開催スケジュール (YS)（任意）
.\.venv32\Scripts\python.exe scripts/extract_jvlink.py `
    --dataspec YSCH --from-date 20160101 --option 4
```

---

## Step 3.5: 0B41 バックフィル（時系列オッズ）

JVRTOpen 経由で 0B41（時系列オッズ）をバックフィル取得。WSL から実行可（subprocess で Win32 Python を呼び出し）。

// turbo
```bash
# DB に RACE データが入っている前提
uv run python scripts/backfill_rt.py \
    --from-date 20160101 --to-date 20261231 \
    --resume
```

---

## Step 4: DBロード

各ファイルごと、またはまとめてロード：

// turbo
```powershell
uv run python scripts/load_to_db.py --input data/DIFF_*.jsonl
uv run python scripts/load_to_db.py --input data/RACE_20160101-20171231_*.jsonl
uv run python scripts/load_to_db.py --input data/RACE_20180101-20191231_*.jsonl
# ... 繰り返し
uv run python scripts/load_to_db.py --input data/MING_*.jsonl
uv run python scripts/load_to_db.py --input data/SNPN_*.jsonl
uv run python scripts/load_to_db.py --input data/SLOP_*.jsonl
uv run python scripts/load_to_db.py --input data/WOOD_*.jsonl
uv run python scripts/load_to_db.py --input data/COMM_*.jsonl
uv run python scripts/load_to_db.py --input data/YSCH_*.jsonl
```

または一括：

// turbo
```powershell
uv run python scripts/load_to_db.py --input-dir data/
```

---

## Step 5: 検証（最低限）

// turbo
```powershell
uv run pytest -q
uv run python scripts/verify_parsers.py
```

期待: パーサのスモークが通ること（DBの件数/年別分布の確認はSQLで別途）。
