"""
0B41 バックフィルスクリプト（JVRTOpen 経由）

core.race から racekey 一覧を取得し、0B41（時系列オッズ）を一括取得する。
WSL 上で実行し、JVRTOpen 呼び出しは subprocess で Windows 32bit Python に委譲。

使用方法:
    # 2016〜2026年の全レースの 0B41 をバックフィル
    uv run python scripts/backfill_rt.py \\
        --from-date 20160101 --to-date 20261231 \\
        --python32 /mnt/c/path/to/.venv32/Scripts/python.exe

    # 途中再開（前回の進捗から）
    uv run python scripts/backfill_rt.py \\
        --from-date 20160101 --to-date 20261231 --resume

    # DRY-RUN: racekey 一覧の確認のみ
    uv run python scripts/backfill_rt.py \\
        --from-date 20160101 --to-date 20261231 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rt_common import (  # noqa: E402
    DATA_DIR,
    call_extract_rt,
    detect_python32,
    find_existing_output,
    generate_racekeys_from_db,
    group_racekeys_by_date,
    load_progress,
    save_progress,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# 進捗ファイルパス
PROGRESS_FILE = DATA_DIR / "backfill_0B41_progress.json"

# バックフィル対象 dataspec
BACKFILL_DATASPEC = "0B41"


def _save_progress(path: Path, completed: set[str], failed: set[str]) -> None:
    """完了・失敗 racekey を進捗ファイルに保存"""
    save_progress(path, {"completed": sorted(completed), "failed": sorted(failed)})


def run_backfill(
    from_date: str,
    to_date: str,
    python32: str,
    output_dir: Path,
    resume: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    """0B41 バックフィルを実行

    JV-Link 側のデータ欠損（Bad file descriptor 等）は改善不可のため、
    エラーはログに記録してスキップし、最後まで走りきる。

    Args:
        from_date: 開始日 (YYYYMMDD)
        to_date: 終了日 (YYYYMMDD)
        python32: Windows 32bit Python パス
        output_dir: 出力ディレクトリ
        resume: 途中再開モード
        dry_run: DRY-RUN モード

    Returns:
        実行統計 dict
    """
    from app.infrastructure.database import Database

    stats = {"total": 0, "success": 0, "skip": 0, "error": 0, "dates": 0}

    # 1. racekey 一覧を DB から取得
    logger.info("📋 racekey 一覧を生成中...")
    with Database() as db:
        racekeys = generate_racekeys_from_db(db, from_date, to_date)

    if not racekeys:
        logger.warning("対象レースがありません (from=%s, to=%s)", from_date, to_date)
        return stats

    # 開催日でグルーピング
    grouped = group_racekeys_by_date(racekeys)
    stats["total"] = len(racekeys)
    stats["dates"] = len(grouped)

    logger.info("  対象レース: %d 件 (%d 開催日)", len(racekeys), len(grouped))

    if dry_run:
        logger.info("[DRY-RUN] racekey 一覧:")
        for date_str, keys in sorted(grouped.items()):
            logger.info(
                "  %s: %d レース (%s ... %s)",
                date_str,
                len(keys),
                keys[0],
                keys[-1],
            )
        return stats

    # 2. 進捗読み込み（resume時）
    completed: set[str] = set()
    failed: set[str] = set()
    if resume:
        progress = load_progress(PROGRESS_FILE)
        completed = set(progress.get("completed", []))
        failed = set(progress.get("failed", []))
        skip_count = len(completed) + len(failed)
        if skip_count:
            logger.info(
                "  途中再開: %d 件完了済み, %d 件エラー済み",
                len(completed),
                len(failed),
            )

    # 3. 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. 開催日ごとにループ（エラーは記録してスキップ、停止しない）
    processed_dates = 0

    for date_str in sorted(grouped.keys()):
        keys_for_date = grouped[date_str]
        processed_dates += 1

        # この開催日のすべてのレースが完了/失敗済みならスキップ
        done_keys = completed | failed
        pending_keys = [k for k in keys_for_date if k not in done_keys]
        if not pending_keys:
            stats["skip"] += len(keys_for_date)
            continue

        logger.info(
            "📅 [%d/%d] %s: %d レース (%d 件スキップ)",
            processed_dates,
            len(grouped),
            date_str,
            len(pending_keys),
            len(keys_for_date) - len(pending_keys),
        )

        for _i, racekey in enumerate(pending_keys):
            # 既存ファイルチェック
            existing = find_existing_output(output_dir, BACKFILL_DATASPEC, racekey)
            if existing:
                logger.debug("  スキップ (既存): %s", existing.name)
                stats["skip"] += 1
                completed.add(racekey)
                continue

            # subprocess で取得
            ok, msg = call_extract_rt(
                python32, BACKFILL_DATASPEC, racekey, output_dir, dry_run=False
            )

            if ok:
                stats["success"] += 1
                completed.add(racekey)
            else:
                stats["error"] += 1
                failed.add(racekey)
                logger.warning("  ❌ %s: %s", racekey, msg)

            # 進捗保存（10件ごと）
            if (stats["success"] + stats["error"]) % 10 == 0:
                _save_progress(PROGRESS_FILE, completed, failed)

            # API レート制御（連続呼び出しを避ける）
            time.sleep(0.5)

    # 5. 最終進捗保存
    _save_progress(PROGRESS_FILE, completed, failed)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="0B41 バックフィル (JVRTOpen → JSONL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # 2016〜2026年の全レースをバックフィル
  uv run python scripts/backfill_rt.py \\
      --from-date 20160101 --to-date 20261231

  # 途中再開
  uv run python scripts/backfill_rt.py \\
      --from-date 20160101 --to-date 20261231 --resume

  # DRY-RUN: racekey 一覧の確認のみ
  uv run python scripts/backfill_rt.py \\
      --from-date 20160101 --to-date 20261231 --dry-run
""",
    )
    parser.add_argument(
        "--from-date",
        required=True,
        help="開始日 (YYYYMMDD)",
    )
    parser.add_argument(
        "--to-date",
        required=True,
        help="終了日 (YYYYMMDD)",
    )
    parser.add_argument(
        "--python32",
        default=None,
        help="Windows 32bit Python パス (未指定時は自動検出)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help=f"出力ディレクトリ (デフォルト: {DATA_DIR})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="途中再開モード（前回の進捗から）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="DRY-RUN: racekey 一覧の確認のみ",
    )
    args = parser.parse_args()

    # Python32 パスの決定
    python32 = args.python32 or detect_python32()

    print("=" * 60)
    print("0B41 バックフィル (JVRTOpen)")
    print("=" * 60)
    print(f"期間      : {args.from_date} 〜 {args.to_date}")
    print(f"Python32  : {python32}")
    print(f"出力先    : {args.output_dir}")
    print(f"再開モード: {'ON' if args.resume else 'OFF'}")
    if args.dry_run:
        print("モード    : DRY-RUN")
    print("=" * 60)

    stats = run_backfill(
        from_date=args.from_date,
        to_date=args.to_date,
        python32=python32,
        output_dir=args.output_dir,
        resume=args.resume,
        dry_run=args.dry_run,
    )

    print("")
    print("=" * 60)
    print("✅ バックフィル完了")
    print(f"  対象レース : {stats['total']:,} 件 ({stats['dates']} 開催日)")
    print(f"  成功       : {stats['success']:,} 件")
    print(f"  スキップ   : {stats['skip']:,} 件")
    print(f"  エラー     : {stats['error']:,} 件")
    print("=" * 60)

    if stats["success"] > 0:
        print("")
        print("次のステップ:")
        print(f"  uv run python scripts/load_to_db.py --input '{args.output_dir}/0B41_*.jsonl'")


if __name__ == "__main__":
    main()
