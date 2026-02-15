"""
JVRTOpen オーケストレーション共通ユーティリティ

backfill_rt.py / ops_rt.py から共通で使われる関数群。
- racekey 生成
- subprocess 経由の extract_rt_jvlink.py 呼び出し
- 進捗ファイル管理
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# プロジェクトルート
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
# データ保存先（リポジトリ外: /home/sato/projects/keibas/data）
DATA_DIR = PROJECT_ROOT.parent / "data"

# デフォルトの dataspecs（ops 用）
# NOTE:
# - 変更系は 0B14 (速報開催情報(一括), 開催日単位) を使用する。
# - 0B16 (速報開催情報(指定)) は JVWatchEvent で得たイベント key 前提のため、
#   本リポジトリのポーリング運用（開催日/レースkey）では通常使わない。
OPS_DEFAULT_DATASPECS = ["0B41", "0B14", "0B11", "0B13", "0B17"]

# Windows 32bit Python パス（環境変数 or デフォルト）
DEFAULT_PYTHON32 = os.getenv(
    "PYTHON32_PATH",
    str(PROJECT_ROOT / ".venv32" / "Scripts" / "python.exe"),
)


def race_id_to_racekey(race_id: int) -> str:
    """race_id (YYYYMMDDTTRR) → racekey (YYYYMMDDJJRR) 変換

    race_id = YYYYMMDD * 10000 + track_code * 100 + race_no
    racekey = YYYYMMDDJJRR（12文字の短縮形式、JVRTOpen で使用可能）

    >>> race_id_to_racekey(202602030501)
    '202602030501'
    """
    date_part = race_id // 10000  # YYYYMMDD
    track_code = (race_id // 100) % 100  # JJ
    race_no = race_id % 100  # RR
    return f"{date_part:08d}{track_code:02d}{race_no:02d}"


def generate_racekeys_from_db(
    db,
    from_date: str,
    to_date: str,
) -> list[str]:
    """core.race から racekey 一覧を生成

    Args:
        db: Database インスタンス
        from_date: 開始日 (YYYYMMDD)
        to_date: 終了日 (YYYYMMDD)

    Returns:
        racekey のリスト（昇順ソート済み）
    """
    from_iso = f"{from_date[:4]}-{from_date[4:6]}-{from_date[6:8]}"
    to_iso = f"{to_date[:4]}-{to_date[4:6]}-{to_date[6:8]}"

    sql = """
        SELECT race_id FROM core.race
        WHERE race_date >= %s AND race_date <= %s
          AND track_code BETWEEN 1 AND 10  -- 中央競馬のみ（JVRTOpen対応）
        ORDER BY race_id
    """
    rows = db.fetch_all(sql, (from_iso, to_iso))
    return [race_id_to_racekey(row["race_id"]) for row in rows]


def build_output_path(
    output_dir: Path,
    dataspec: str,
    racekey: str,
    timestamp: str | None = None,
) -> Path:
    """出力 JSONL ファイルパスを生成

    Args:
        output_dir: 出力ディレクトリ
        dataspec: データ種別 (0B41 等)
        racekey: レースキー
        timestamp: タイムスタンプ文字列（省略時は自動生成）

    Returns:
        出力パス (例: data/0B41_202602030501_20260214_123456.jsonl)
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{dataspec}_{racekey}_{timestamp}.jsonl"


def find_existing_output(output_dir: Path, dataspec: str, racekey: str) -> Path | None:
    """既存の出力ファイルを検索（重複排除用）

    Args:
        output_dir: 出力ディレクトリ
        dataspec: データ種別
        racekey: レースキー

    Returns:
        既存ファイルのパス（なければ None）
    """
    pattern = f"{dataspec}_{racekey}_*.jsonl"
    matches = list(output_dir.glob(pattern))
    return matches[0] if matches else None


def call_extract_rt(
    python32_path: str,
    dataspec: str,
    request_key: str,
    output_dir: Path,
    dry_run: bool = False,
) -> tuple[bool, str]:
    """subprocess で extract_rt_jvlink.py を呼び出し

    Args:
        python32_path: Windows 32bit Python の実行パス
        dataspec: データ種別
        request_key: 要求キー（レースキー/開催日/イベントキー）
        output_dir: 出力ディレクトリ
        dry_run: DRY-RUN モード

    Returns:
        (成功フラグ, メッセージ)
    """
    # Windows パスに変換（WSL interop 用）
    script_path = SCRIPTS_DIR / "extract_rt_jvlink.py"

    # WSL から Windows exe を呼ぶ場合、スクリプトパスは Windows 形式が必要
    # ただし WSL interop は /mnt/c/ 以下のパスも自動変換してくれる場合がある
    # 安全のため wslpath で変換を試みる
    win_script_path = _to_windows_path(str(script_path))

    cmd = [
        python32_path,
        win_script_path,
        "--dataspec",
        dataspec,
        "--key",
        request_key,
        "--output-dir",
        _to_windows_path(str(output_dir)),
    ]
    if dry_run:
        cmd.append("--dry-run")

    logger.debug("subprocess: %s", " ".join(cmd))

    # Windows 側の Python で絵文字/日本語を正しく扱うため
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            env=env,
            timeout=120,  # 2分タイムアウト
        )
        # Windows 32bit Python の出力は cp932（Shift_JIS）の場合がある
        stdout = result.stdout.decode("utf-8", errors="replace")
        stderr = result.stderr.decode("utf-8", errors="replace")
        if result.returncode == 0:
            return True, stdout.strip()
        else:
            msg = stderr.strip() or stdout.strip() or f"exit code {result.returncode}"
            return False, msg
    except subprocess.TimeoutExpired:
        return False, "タイムアウト (120秒)"
    except FileNotFoundError:
        return False, f"Python32 が見つかりません: {python32_path}"
    except Exception as e:
        return False, str(e)


def _to_windows_path(posix_path: str) -> str:
    """WSL パスを Windows パスに変換

    変換パターン:
    - /mnt/c/foo/bar  → C:\\foo\\bar
    - /home/user/foo  → \\\\wsl.localhost\\Ubuntu\\home\\user\\foo

    wslpath コマンドが利用可能なら最優先で使う。
    """
    # まず wslpath を試行（最も信頼性が高い）
    try:
        result = subprocess.run(
            ["wslpath", "-w", posix_path],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # フォールバック: 手動変換
    if posix_path.startswith("/mnt/"):
        # /mnt/c/foo/bar → C:\foo\bar
        parts = posix_path.split("/")
        if len(parts) >= 4 and parts[1] == "mnt":
            drive = parts[2].upper()
            rest = "\\".join(parts[3:])
            return f"{drive}:\\{rest}"
    elif posix_path.startswith("/"):
        # WSL ネイティブパス → UNC パス
        # /home/user/foo → \\wsl.localhost\Ubuntu\home\user\foo
        distro = os.getenv("WSL_DISTRO_NAME", "Ubuntu")
        win_path = posix_path.replace("/", "\\")
        return f"\\\\wsl.localhost\\{distro}{win_path}"

    return posix_path


def load_progress(path: Path) -> dict:
    """進捗ファイルを読み込み

    Returns:
        進捗データ（存在しなければ空dict）
    """
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_progress(path: Path, data: dict) -> None:
    """進捗ファイルを書き出し"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def group_racekeys_by_date(racekeys: list[str]) -> dict[str, list[str]]:
    """racekey をYYYYMMDD（開催日）でグルーピング

    Args:
        racekeys: racekey のリスト (YYYYMMDDJJRR 形式)

    Returns:
        {YYYYMMDD: [racekeys...]} の辞書
    """
    grouped: dict[str, list[str]] = {}
    for key in racekeys:
        date_str = key[:8]  # YYYYMMDD
        grouped.setdefault(date_str, []).append(key)
    return grouped


def detect_python32() -> str:
    """Windows 32bit Python のパスを自動検出

    優先順位:
    1. 環境変数 PYTHON32_PATH
    2. プロジェクト内 .venv32
    3. Windows 側の既知パス (.venv32)
    4. /mnt/c/Python311-32/python.exe (win32com なし)
    """
    # 環境変数
    env_path = os.getenv("PYTHON32_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    # プロジェクト内 .venv32（WSL 上）
    venv32 = PROJECT_ROOT / ".venv32" / "Scripts" / "python.exe"
    if venv32.exists():
        return str(venv32)

    # Windows 側の .venv32（win32com インストール済み）
    win_venv32 = Path("/mnt/c/Users/sato/Desktop/projects/keiba/.venv32/Scripts/python.exe")
    if win_venv32.exists():
        return str(win_venv32)

    # グローバルインストール（win32com がない可能性あり）
    global_py = Path("/mnt/c/Python311-32/python.exe")
    if global_py.exists():
        return str(global_py)

    # 見つからなければデフォルトを返す（実行時にエラーになる）
    logger.warning("Python32 が自動検出できませんでした。--python32 を指定してください。")
    return DEFAULT_PYTHON32


if __name__ == "__main__":
    # 簡易テスト用
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"Python32 検出: {detect_python32()}")
    print(f"race_id_to_racekey(202602030501) = {race_id_to_racekey(202602030501)}")
    print(f"OPS_DEFAULT_DATASPECS = {OPS_DEFAULT_DATASPECS}")

    if "--test-db" in sys.argv:
        from app.infrastructure.database import Database

        with Database() as db:
            keys = generate_racekeys_from_db(db, "20260201", "20260228")
            print(f"2026/02 racekeys: {len(keys)} 件")
            if keys:
                print(f"  先頭: {keys[0]}")
                print(f"  末尾: {keys[-1]}")
