# JV-Link / WSL連携ルール

## WSL → Windows subprocess 連携

- WSL から Windows 32bit Python を呼ぶ場合、**スクリプトパス・出力パスは Windows 形式に変換が必要**
  - `/home/...` → `\\wsl.localhost\Ubuntu\home\...`（UNC パス）
  - `/mnt/c/...` → `C:\...`
  - `wslpath -w` コマンドが最も信頼性が高い。フォールバックで手動変換。
- `PYTHONIOENCODING=utf-8` は **WSL interop 経由の subprocess では効かない**
  - 根本対策: `io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')` をスクリプト先頭で適用
- subprocess の `text=True` は **Windows 出力の cp932 で壊れる** → バイナリモード + `decode('utf-8', errors='replace')` を使う

## JVRTOpen (リアルタイム系API)

- **中央競馬（track_code 1-10）のみ対応**。地方・海外（0, 30-55）は `Bad file descriptor` エラーになる
  - `core.race` クエリ時に `AND track_code BETWEEN 1 AND 10` フィルタ必須
- dataspec ごとのバックフィル可否:
  - **0B41/0B42**: バックフィル可能（2003年〜）
  - **0B11/0B16/0B13/0B17**: 当日データのみ提供。過去データは `rc_open = -1` で取得不可

## 環境パス

- 32bit Python (.venv32): `C:\Users\sato\Desktop\projects\keiba\.venv32\Scripts\python.exe`
  - WSL パス: `/mnt/c/Users/sato/Desktop/projects/keiba/.venv32/Scripts/python.exe`
- データ保存先: `/home/sato/projects/keibas/data`（リポジトリ外）
