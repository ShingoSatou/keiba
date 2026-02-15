"""
当日運用 リアルタイムデータ一括取得スクリプト（JVRTOpen 経由）

当日レース一覧を元に 0B41/0B14/0B11/0B13/0B17 を一括取得する。
WSL 上で実行し、JVRTOpen 呼び出しは subprocess で Windows 32bit Python に委譲。

使用方法:
    # 当日の全 dataspec を一括取得
    uv run python scripts/ops_rt.py --race-date 20260214

    # 特定 dataspec のみ
    uv run python scripts/ops_rt.py \\
        --race-date 20260214 --dataspecs 0B41,0B14

    # DRY-RUN
    uv run python scripts/ops_rt.py --race-date 20260214 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rt_common import (  # noqa: E402
    DATA_DIR,
    OPS_DEFAULT_DATASPECS,
    call_extract_rt,
    detect_python32,
    find_existing_output,
    generate_racekeys_from_db,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATE_KEY_DATASPECS = {"0B11", "0B13", "0B14", "0B17"}


def _build_request_keys(dataspec: str, race_date: str, racekeys: list[str]) -> list[str]:
    if dataspec.upper() in DATE_KEY_DATASPECS:
        return [race_date]
    return racekeys


def run_ops(
    race_date: str,
    dataspecs: list[str],
    python32: str,
    output_dir: Path,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, int]:
    """当日運用の一括取得

    Args:
        race_date: 対象日 (YYYYMMDD)
        dataspecs: 取得する dataspec リスト
        python32: Windows 32bit Python パス
        output_dir: 出力ディレクトリ
        force: 既存ファイルを再取得
        dry_run: DRY-RUN モード

    Returns:
        実行統計 dict
    """
    from app.infrastructure.database import Database

    stats = {"total": 0, "success": 0, "skip": 0, "error": 0}

    # 1. 当日の racekey 一覧を取得
    logger.info("📋 %s の racekey 一覧を取得中...", race_date)
    with Database() as db:
        racekeys = generate_racekeys_from_db(db, race_date, race_date)

    if not racekeys:
        logger.warning("対象レースがありません (date=%s)", race_date)
        logger.info("  → core.race にデータが入っていない可能性があります。")
        logger.info("    先に RACE データをロードしてください。")
        return stats

    request_keys_by_dataspec = {
        dataspec: _build_request_keys(dataspec, race_date, racekeys) for dataspec in dataspecs
    }
    stats["total"] = sum(len(keys) for keys in request_keys_by_dataspec.values())
    logger.info("  対象レース: %d 件", len(racekeys))
    logger.info("  取得要求  : %d 件", stats["total"])

    if dry_run:
        logger.info("[DRY-RUN] 取得対象:")
        for ds in dataspecs:
            request_keys = request_keys_by_dataspec[ds]
            key_unit = "開催日単位" if ds.upper() in DATE_KEY_DATASPECS else "レース単位"
            logger.info("  dataspec=%s:", ds)
            logger.info("    key unit: %s (%d件)", key_unit, len(request_keys))
            for key in request_keys:
                logger.info("    %s", key)
        return stats

    # 2. 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)

    # 3. racekey × dataspec のループ
    for ds_idx, dataspec in enumerate(dataspecs):
        request_keys = request_keys_by_dataspec[dataspec]
        key_unit = "date" if dataspec.upper() in DATE_KEY_DATASPECS else "race"
        logger.info(
            "📡 [%d/%d] dataspec=%s key_unit=%s keys=%d",
            ds_idx + 1,
            len(dataspecs),
            dataspec,
            key_unit,
            len(request_keys),
        )

        for key_idx, request_key in enumerate(request_keys):
            # 既存ファイルチェック
            if not force:
                existing = find_existing_output(output_dir, dataspec, request_key)
                if existing:
                    logger.debug("  スキップ (既存): %s", existing.name)
                    stats["skip"] += 1
                    continue

            # subprocess で取得
            ok, msg = call_extract_rt(
                python32,
                dataspec,
                request_key,
                output_dir,
                dry_run=False,
            )

            if ok:
                stats["success"] += 1
                logger.info(
                    "  ✅ [%d/%d] %s × %s",
                    key_idx + 1,
                    len(request_keys),
                    dataspec,
                    request_key,
                )
            else:
                stats["error"] += 1
                logger.warning(
                    "  ❌ [%d/%d] %s × %s: %s",
                    key_idx + 1,
                    len(request_keys),
                    dataspec,
                    request_key,
                    msg,
                )

            # API レート制御
            time.sleep(0.3)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="当日運用 リアルタイムデータ一括取得 (JVRTOpen)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # 当日の全 dataspec を一括取得
  uv run python scripts/ops_rt.py --race-date 20260214

  # 特定 dataspecs のみ
  uv run python scripts/ops_rt.py \\
      --race-date 20260214 --dataspecs 0B41,0B11

  # DRY-RUN
  uv run python scripts/ops_rt.py --race-date 20260214 --dry-run
""",
    )
    parser.add_argument(
        "--race-date",
        default=None,
        help="対象日 (YYYYMMDD, デフォルト: 当日)",
    )
    parser.add_argument(
        "--dataspecs",
        default=None,
        help=f"取得する dataspec (カンマ区切り, デフォルト: {','.join(OPS_DEFAULT_DATASPECS)})",
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
        "--force",
        action="store_true",
        help="既存ファイルを再取得",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="DRY-RUN: 取得対象の確認のみ",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="取得エラーが1件でもあれば非0終了する",
    )
    args = parser.parse_args()

    # 引数の解決
    race_date = args.race_date or datetime.now().strftime("%Y%m%d")
    dataspecs = args.dataspecs.split(",") if args.dataspecs else OPS_DEFAULT_DATASPECS
    python32 = args.python32 or detect_python32()

    print("=" * 60)
    print("当日運用 リアルタイムデータ一括取得 (JVRTOpen)")
    print("=" * 60)
    print(f"対象日    : {race_date}")
    print(f"dataspecs : {', '.join(dataspecs)}")
    print(f"Python32  : {python32}")
    print(f"出力先    : {args.output_dir}")
    if args.dry_run:
        print("モード    : DRY-RUN")
    print("=" * 60)

    stats = run_ops(
        race_date=race_date,
        dataspecs=dataspecs,
        python32=python32,
        output_dir=args.output_dir,
        force=args.force,
        dry_run=args.dry_run,
    )

    print("")
    print("=" * 60)
    print("✅ 取得完了")
    print(f"  対象     : {stats['total']:,} 件")
    print(f"  成功     : {stats['success']:,} 件")
    print(f"  スキップ : {stats['skip']:,} 件")
    print(f"  エラー   : {stats['error']:,} 件")
    print("=" * 60)

    if args.fail_on_error and stats["error"] > 0:
        print("")
        print("❌ fail-on-error により終了します（取得エラーあり）")
        sys.exit(1)

    if stats["success"] > 0:
        print("")
        print("次のステップ:")
        print(f"  uv run python scripts/load_to_db.py --input-dir {args.output_dir}")


if __name__ == "__main__":
    main()
