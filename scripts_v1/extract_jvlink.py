"""
JV-Link データ抽出スクリプト (32bit Python用)

JV-Linkからデータを取得し、JSONLファイルに出力します。
32bit Python環境 (.venv32) で実行する必要があります。

使用方法:
    # 初回セットアップ (過去10年分を取得)
    .venv32\\Scripts\\python.exe scripts/extract_jvlink.py --from-date 20160101 --option 4

    # 日々の更新 (過去1年分)
    .venv32\\Scripts\\python.exe scripts/extract_jvlink.py \\
        --from-date 20250101 --option 1

    # 馬マスタ(UM)のみ取得 (DIFFを開いてUMでフィルタ)
    .venv32\\Scripts\\python.exe scripts/extract_jvlink.py \\
        --dataspec DIFF --from-date 20160101 \\
        --option 4 --record-filter UM

    # 調教データ(坂路)
    .venv32\\Scripts\\python.exe scripts/extract_jvlink.py \\
        --dataspec SLOP --from-date 20160101 --option 4

    # 調教データ(ウッド)
    .venv32\\Scripts\\python.exe scripts/extract_jvlink.py \\
        --dataspec WOOD --from-date 20160101 --option 4

    # マイニング
    .venv32\\Scripts\\python.exe scripts/extract_jvlink.py \\
        --dataspec MING --from-date 20160101 --option 4

Option説明:
    1: 通常データ (過去1年分、日々の更新用)
    2: 非蓄積系 (速報データ用)
    3: セットアップ (過去データ一括取得、ダイアログあり)
    4: セットアップ (過去データ一括取得、ダイアログなし) ★初回推奨

dataspec一覧:
    RACE: レース基本情報 (RA/SE/HR/O1/UM/KS/CH/JG)
    DIFF: 更新差分
    SLOP: 坂路調教 (HC)
    WOOD: ウッド調教 (WC)
    SNAP: 出走別着度数 (CK)
    MING: マイニング (DM/TM)
    YSCH: 開催スケジュール (YS)
    COMM: コース情報 (CS)

出力:
    data/{dataspec}_{timestamp}.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Windows cp932 環境で Unicode 出力が落ちるケースの回避（WSL interop経由でも安全側に倒す）
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"

# Option定数
OPTION_NORMAL = 1  # 通常データ (過去1年)
OPTION_REALTIME = 2  # 非蓄積系
OPTION_SETUP = 3  # セットアップ (ダイアログあり)
OPTION_SETUP_NO_DIALOG = 4  # セットアップ (ダイアログなし)


def ensure_32bit():
    """32bit Python であることを確認"""
    import struct

    bits = struct.calcsize("P") * 8
    if bits != 32:
        print(f"[ERR] 32bit Python が必要です (現在: {bits}bit)")
        print("   .venv32\\Scripts\\python.exe を使用してください")
        sys.exit(1)


def get_jvlink():
    """JV-Link COMオブジェクトを取得"""
    import win32com.client

    jv = win32com.client.Dispatch("JVDTLab.JVLink")
    ret = jv.JVInit("UNKNOWN")
    if ret not in (0, -102):  # 0=成功, -102=初期化済み
        raise RuntimeError(f"JVInit failed: {ret}")
    print(f"[OK] JVInit success (ret={ret})")
    return jv


def jv_open_with_logging(jv, dataspec: str, from_date: str, option: int):
    """
    JVOpenを詳細ログ付きで実行

    Returns:
        (rc_open, readcount, downloadcount, lastfiletimestamp)
    """
    print("\n[INFO] JVOpen:")
    print(f"   dataspec  = {dataspec}")
    print(f"   fromtime  = {from_date}")
    print(f"   option    = {option}")

    # JVOpen(dataspec, fromtime, option, readcount, downloadcount, lastfiletimestamp)
    # JVOpen(dataspec, fromtime, option, readcount, downloadcount, lastfiletimestamp)
    # win32com では ダミー引数 (0, 0, "") を渡し、戻り値タプルで out パラメータを受け取る
    ret = jv.JVOpen(dataspec, from_date, option, 0, 0, "")

    # 戻り値の解析: (rc_open, readcount, downloadcount, lastfiletimestamp)
    if isinstance(ret, tuple):
        rc_open = ret[0]
        readcount = ret[1] if len(ret) > 1 else 0
        downloadcount = ret[2] if len(ret) > 2 else 0
        lastfiletimestamp = ret[3] if len(ret) > 3 else ""
    else:
        rc_open = ret
        readcount = 0
        downloadcount = 0
        lastfiletimestamp = ""

    print("\n[INFO] JVOpen result:")
    print(f"   rc_open           = {rc_open}")
    print(f"   readcount         = {readcount}")
    print(f"   downloadcount     = {downloadcount}")
    print(f"   lastfiletimestamp = {lastfiletimestamp}")

    # 判定と説明
    if rc_open < 0:
        error_messages = {
            -1: "該当データ無し",
            -2: "fromtime が不正",
            -3: "option が不正",
            -100: "JVInit が実行されていない",
            -101: "前回のダウンロードが異常終了",
            -102: "JVRead が未完了",
            -103: "サーバー接続エラー",
            -111: "dataspec が不正 (UM等はレコード種別であり、RACE/DIFF/DIFN等を使用)",
            -112: "fromtime の書式エラー",
            -114: "データ種別が未購入",
            -115: "option と dataspec の組合せが不正",
            -116: "JVOpen 重複呼出",
            -201: "JVInit されていない (JVRead用)",
            -203: "JVOpen されていない (JVRead用)",
            -301: "利用キーが無効、または複数PCで同時使用",
            -503: "ファイルがない",
        }
        print(f"\n[ERR] {error_messages.get(rc_open, 'unknown error')}")
        return (rc_open, readcount, downloadcount, lastfiletimestamp)

    # rc_open=0 でも downloadcount>0 ならダウンロード待ちデータあり
    if rc_open == 0 and downloadcount == 0 and readcount == 0:
        print("\n[WARN] no data (rc_open=0, downloadcount=0)")
        print("   考えられる原因:")
        print("   1. option=1 は過去1年分のみ -> option=4 でセットアップが必要")
        print("   2. 指定した fromtime 以降のデータが存在しない")
        print("   3. dataspec と option の組合せが不正")
        return (rc_open, readcount, downloadcount, lastfiletimestamp)

    # データあり
    if downloadcount > 0:
        print(f"\n[INFO] download pending: {downloadcount} files")
        print("   -> JVStatus でダウンロード完了を待機します")
    if readcount > 0 or rc_open > 0:
        print(f"\n[OK] data available: readcount={readcount}, rc_open={rc_open}")

    return (rc_open, readcount, downloadcount, lastfiletimestamp)


def wait_for_download(jv, downloadcount: int, timeout_sec: int = 3600):
    """
    JVStatusでダウンロード完了を待機

    Args:
        jv: JV-Link COM object
        downloadcount: JVOpenで返されたダウンロードファイル数
        timeout_sec: タイムアウト秒数

    Returns:
        True if download completed, False if timeout
    """
    print(f"\n[INFO] waiting download... (target: {downloadcount} files)")
    start_time = time.time()
    check_count = 0

    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout_sec:
            print(f"\n[ERR] timeout ({timeout_sec}s)")
            return False

        # JVStatus でダウンロード状況を確認
        # 戻り値: ダウンロード済みファイル数
        ret = jv.JVStatus()
        check_count += 1

        if isinstance(ret, tuple):
            downloaded = ret[0]
        else:
            downloaded = ret

        # JVStatus戻り値の解釈:
        # 正数 = ダウンロード済みファイル数
        # downloadcount と一致 = 完了
        # 負数 = エラー

        if downloaded < 0:
            print(f"\n[ERR] JVStatus error: {downloaded}")
            return False

        if downloaded >= downloadcount:
            print(f"\n[OK] download completed ({downloaded}/{downloadcount})")
            return True

        # 進行中
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)
        pct = int(downloaded * 100 / downloadcount) if downloadcount > 0 else 0
        print(f"   downloading... {downloaded}/{downloadcount} ({pct}%) [{mins}m{secs}s]", end="\r")
        time.sleep(3)


def extract_records(
    jv, max_records: int = 0, record_filter: str | None = None, to_date: str | None = None
):
    """
    JVReadでレコードを抽出

    Args:
        jv: JV-Link COM object
        max_records: 最大レコード数 (0=無制限)
        record_filter: フィルタするレコード種別 (例: "UM", "RA", None=全て)
        to_date: 終了日 (YYYYMMDD形式)。RAレコードの日付がこれを超えたらスキップ

    Yields:
        dict: レコード情報
    """
    count = 0
    file_switches = 0
    read_attempts = 0

    while True:
        if max_records > 0 and count >= max_records:
            print(f"\n   最大レコード数 ({max_records}) に達しました")
            break

        try:
            # JVRead(buff, size, filename)
            # win32com では ダミー引数 ("", 110000, "") を渡し、戻り値タプルで受け取る
            # 110000 は公式サンプルのバッファサイズ
            ret = jv.JVRead("", 110000, "")
        except Exception as e:
            print(f"\n[ERR] JVRead exception: {e}")
            traceback.print_exc()
            break

        read_attempts += 1

        if isinstance(ret, tuple):
            res = ret[0]
            buff = ret[1] if len(ret) > 1 else ""
            fname = ret[2] if len(ret) > 2 else ""
        else:
            res = ret
            buff = ""
            fname = ""

        if res > 0:
            actual_buff = buff if isinstance(buff, str) else str(buff)
            rec_id = actual_buff[:2] if len(actual_buff) >= 2 else "??"

            # レコードフィルタ
            if record_filter and rec_id != record_filter:
                continue

            # 日付フィルタ (RAレコードのみ)
            if to_date and rec_id == "RA":
                # RAレコードの開催年月日: 位置12-19 (1-indexed) = 11-18 (0-indexed)
                try:
                    race_date_str = actual_buff[11:19]  # YYYYMMDD
                    if race_date_str > to_date:
                        continue  # 終了日を超えたレコードはスキップ
                except (IndexError, ValueError):
                    pass  # パース失敗時はフィルタしない

            yield {
                "rec_id": rec_id,
                "filename": fname,
                "payload": actual_buff,
                "size": res,
            }
            count += 1

            if count % 1000 == 0:
                print(f"   {count} 件処理...")

        elif res == 0:
            print(f"\n   読み込み完了: {count} 件 (ファイル切替: {file_switches} 回)")
            break
        elif res == -1:
            # ファイル切替
            file_switches += 1
            continue
        elif res == -3:
            # ダウンロード中 - 待機
            download_waits = getattr(extract_records, "_download_waits", 0) + 1
            extract_records._download_waits = download_waits

            if download_waits == 1:
                print("\n   [INFO] waiting download (JVRead=-3)...")
            elif download_waits % 10 == 0:
                print(f"   [INFO] waiting download... ({download_waits * 3}s elapsed)")

            # 長時間待機の場合はタイムアウト
            if download_waits > 600:  # 30分
                print("\n[ERR] download timeout (30min)")
                break

            time.sleep(3)
            continue
        else:
            print(f"\n[ERR] JVRead error: {res}")
            break

    jv.JVClose()
    return count


def save_jsonl(records, output_path: Path):
    """レコードをJSONL形式で保存"""
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(
        description="JV-Link データ抽出",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Option説明:
  1: 通常データ (過去1年分、日々の更新用)
  2: 非蓄積系 (速報データ用)
  3: セットアップ (過去データ一括取得、ダイアログあり)
  4: セットアップ (過去データ一括取得、ダイアログなし) ★初回推奨

例:
  # 初回セットアップ (過去10年分)
  .venv32\\Scripts\\python.exe scripts/extract_jvlink.py --from-date 20160101 --option 4

  # 2年刻みでの取得
  .venv32\\Scripts\\python.exe scripts/extract_jvlink.py \\
      --from-date 20160101 --to-date 20171231 --option 4

  # 日々の更新
  .venv32\\Scripts\\python.exe scripts/extract_jvlink.py --from-date 20250101 --option 1
""",
    )
    parser.add_argument(
        "--from-date",
        required=True,
        help="開始日 (YYYYMMDD形式)",
    )
    parser.add_argument(
        "--dataspec",
        default="RACE",
        help="データ種別 (デフォルト: RACE)",
    )
    parser.add_argument(
        "--option",
        type=int,
        default=4,
        choices=[1, 2, 3, 4],
        help="JVOpen option (デフォルト: 4=セットアップ)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="最大レコード数 (0=無制限)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help=f"出力ディレクトリ (デフォルト: {DATA_DIR})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="JVOpenまで実行して戻り値を確認 (データは読まない)",
    )
    parser.add_argument(
        "--record-filter",
        type=str,
        default=None,
        help="フィルタするレコード種別 (例: UM, RA, SE)。DIFFからUM取得時に使用",
    )
    parser.add_argument(
        "--to-date",
        type=str,
        default=None,
        help="終了日 (YYYYMMDD形式)。この日付を超えるRAレコードはスキップ",
    )
    args = parser.parse_args()

    # 環境チェック
    if sys.platform != "win32":
        print("[ERR] Windows environment is required")
        sys.exit(1)

    ensure_32bit()

    # fromtime を14桁に
    from_date = args.from_date
    if len(from_date) == 8:
        from_date = from_date + "000000"

    # 出力ディレクトリ作成
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("JV-Link データ抽出")
    print("=" * 60)
    print(f"開始日    : {from_date}")
    print(f"データ種別: {args.dataspec}")
    print(f"Option    : {args.option}")
    if args.dry_run:
        print("モード    : DRY-RUN (JVOpenのみ)")
    print("=" * 60)

    # JV-Link接続
    jv = get_jvlink()

    # JVOpen
    rc_open, readcount, downloadcount, lastfiletimestamp = jv_open_with_logging(
        jv, args.dataspec, from_date, args.option
    )

    # エラーチェック: rc_open<0 はエラー、downloadcount=0かつrc_open=0はデータなし
    if rc_open < 0:
        jv.JVClose()
        sys.exit(1)

    if rc_open == 0 and downloadcount == 0 and readcount == 0:
        print("\n-> 取得対象データがありません")
        jv.JVClose()
        sys.exit(0)

    if args.dry_run:
        print("\n[DRY-RUN] JVOpen成功。データ読み込みはスキップします。")
        print(f"   lastfiletimestamp = {lastfiletimestamp}")
        jv.JVClose()
        return

    # ダウンロード待機 (必要な場合)
    if downloadcount > 0:
        if not wait_for_download(jv, downloadcount):
            jv.JVClose()
            sys.exit(1)

    # 出力ファイル名 (期間を含む)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.to_date:
        # RACE_20160101-20171231_20260206_123456.jsonl
        fname = f"{args.dataspec}_{args.from_date}-{args.to_date}_{timestamp}.jsonl"
        output_file = args.output_dir / fname
    else:
        output_file = args.output_dir / f"{args.dataspec}_{timestamp}.jsonl"
    print(f"\n[INFO] output: {output_file}")

    # 抽出＆保存
    print("\n[INFO] JVRead start...")
    if args.record_filter:
        print(f"   フィルタ: rec_id={args.record_filter}")
    if args.to_date:
        print(f"   日付フィルタ: ~{args.to_date}")
    records = extract_records(jv, args.max_records, args.record_filter, args.to_date)
    count = save_jsonl(records, output_file)

    print("")
    print("=" * 60)
    print(f"[OK] saved: {count} records -> {output_file.name}")
    print("=" * 60)
    print("")
    print("次のステップ:")
    print(f"  uv run python scripts/load_to_db.py --input {output_file}")


if __name__ == "__main__":
    main()
