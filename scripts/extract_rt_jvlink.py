"""
JV-Link リアルタイムデータ抽出スクリプト (32bit Python用)

JVRTOpenを使用してリアルタイム系データ（時系列オッズ等）を取得します。
32bit Python環境 (.venv32) で実行する必要があります。

使用方法:
    # 時系列オッズ（単複枠）の取得テスト
    .venv32\\Scripts\\python.exe scripts/extract_rt_jvlink.py \\
        --dataspec 0B41 --racekey 2016010506010101 --dry-run

    # 時系列オッズ（馬連）の取得テスト
    .venv32\\Scripts\\python.exe scripts/extract_rt_jvlink.py \\
        --dataspec 0B42 --racekey 2016010506010101 --dry-run

レースキー形式:
    YYYYMMDDJJKKHHRR または YYYYMMDDJJRR
    - YYYYMMDD: 開催年月日
    - JJ: 競馬場コード (01=札幌, 02=函館, 03=福島, 04=新潟, 05=東京,
          06=中山, 07=中京, 08=京都, 09=阪神, 10=小倉)
    - KK: 開催回 (第N回)
    - HH: 開催日目 (N日目)
    - RR: レース番号 (01-12)
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

# Windows cp932 環境で絵文字/日本語を正しく出力するためのワークアラウンド
# WSL interop 経由の subprocess では PYTHONIOENCODING が効かない場合がある
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"


def ensure_32bit():
    """32bit Python であることを確認"""
    import struct

    bits = struct.calcsize("P") * 8
    if bits != 32:
        print(f"❌ エラー: 32bit Python が必要です (現在: {bits}bit)")
        print("   .venv32\\Scripts\\python.exe を使用してください")
        sys.exit(1)


def get_jvlink():
    """JV-Link COMオブジェクトを取得"""
    import win32com.client

    jv = win32com.client.Dispatch("JVDTLab.JVLink")
    ret = jv.JVInit("UNKNOWN")
    if ret not in (0, -102):  # 0=成功, -102=初期化済み
        raise RuntimeError(f"JVInit failed: {ret}")
    print(f"✅ JVInit: 成功 (戻り値={ret})")
    return jv


def jv_rt_open_with_logging(jv, dataspec: str, key: str):
    """
    JVRTOpenを詳細ログ付きで実行

    Returns:
        rc_open: 戻り値
    """
    print("\n📡 JVRTOpen 呼び出し:")
    print(f"   dataspec = {dataspec}")
    print(f"   key      = {key}")

    # JVRTOpen(dataspec, key) -> rc
    try:
        ret = jv.JVRTOpen(dataspec, key)
    except Exception as e:
        print(f"\n❌ JVRTOpen 例外発生: {e}")
        traceback.print_exc()
        return -999

    # 戻り値の解析
    if isinstance(ret, tuple):
        rc_open = ret[0]
    else:
        rc_open = ret

    print("\n📊 JVRTOpen 戻り値:")
    print(f"   rc_open = {rc_open}")

    # 判定と説明
    if rc_open < 0:
        error_messages = {
            -1: "該当データ無し",
            -2: "key が不正",
            -100: "JVInit が実行されていない",
            -101: "前回のダウンロードが異常終了",
            -102: "JVRead が未完了",
            -103: "サーバー接続エラー",
            -111: "dataspec が不正",
            -112: "key の書式エラー",
            -114: "データ種別が未購入",
            -201: "JVInit されていない (JVRead用)",
            -203: "JVOpen/JVRTOpen されていない (JVRead用)",
            -301: "利用キーが無効、または複数PCで同時使用",
            -401: "JV-Link 内部エラー",
            -411: "サーバーエラー (HTTP 404)",
            -412: "サーバーエラー (HTTP 403)",
            -502: "ダウンロード失敗",
            -503: "ファイルがない（データ未提供または保持期間外）",
            -504: "サーバーメンテナンス中",
        }
        print(f"\n❌ エラー: {error_messages.get(rc_open, f'不明なエラー ({rc_open})')}")
        return rc_open

    # 0 = 成功（データの有無は JVRead で確認）
    print("\n✅ JVRTOpen 成功")
    return rc_open


def extract_records(jv, max_records: int = 0):
    """
    JVReadでレコードを抽出

    Yields:
        dict: レコード情報
    """
    count = 0
    read_attempts = 0

    while True:
        if max_records > 0 and count >= max_records:
            print(f"\n   最大レコード数 ({max_records}) に達しました")
            break

        try:
            ret = jv.JVRead("", 110000, "")
        except Exception as e:
            print(f"\n❌ JVRead 例外発生: {e}")
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

            yield {
                "rec_id": rec_id,
                "filename": fname,
                "payload": actual_buff,
                "size": res,
            }
            count += 1

            if count % 100 == 0:
                print(f"   {count} 件処理...")

        elif res == 0:
            print(f"\n   読み込み完了: {count} 件")
            break
        elif res == -1:
            # ファイル切替 (リアルタイム系では通常1ファイルのみ)
            continue
        elif res == -3:
            # ダウンロード中 - 待機
            print("   ⏳ ダウンロード待機中...")
            time.sleep(1)
            continue
        else:
            print(f"\n❌ JVRead エラー: {res}")
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
        description="JV-Link リアルタイムデータ抽出 (JVRTOpen)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # 2016年1月5日 中山1回1日目1Rの時系列オッズを確認
  .venv32\\Scripts\\python.exe scripts/extract_rt_jvlink.py \\
      --dataspec 0B41 --racekey 2016010506010101 --dry-run

  # データを実際に取得して保存
  .venv32\\Scripts\\python.exe scripts/extract_rt_jvlink.py \\
      --dataspec 0B41 --racekey 2016010506010101
""",
    )
    parser.add_argument(
        "--dataspec",
        required=True,
        help="データ種別 (0B41=時系列オッズ単複枠, 0B42=時系列オッズ馬連等, "
        "0B11=馬体重, 0B14=開催情報(一括), 0B16=開催情報(指定, event key前提))",
    )
    parser.add_argument(
        "--key",
        default="",
        help="要求キー (YYYYMMDDJJKKHHRR / YYYYMMDDJJRR / YYYYMMDD / イベントkey)",
    )
    parser.add_argument(
        "--racekey",
        default="",
        help="レースキー (YYYYMMDDJJKKHHRR形式)",
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
        help="JVRTOpenまで実行して戻り値を確認 (データは読まない)",
    )
    args = parser.parse_args()

    # 環境チェック
    if sys.platform != "win32":
        print("❌ エラー: Windows環境が必要です")
        sys.exit(1)

    ensure_32bit()

    key = args.key or args.racekey
    if not key:
        parser.error("--key または --racekey のいずれかを指定してください")

    # 出力ディレクトリ作成
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("JV-Link リアルタイムデータ抽出 (JVRTOpen)")
    print("=" * 60)
    print(f"データ種別: {args.dataspec}")
    print(f"要求キー  : {key}")
    if args.dry_run:
        print("モード    : DRY-RUN (JVRTOpenのみ)")
    print("=" * 60)

    # JV-Link接続
    jv = get_jvlink()

    # JVRTOpen
    rc_open = jv_rt_open_with_logging(jv, args.dataspec, key)

    # エラーチェック (負の値のみがエラー、0は成功)
    if rc_open < 0:
        jv.JVClose()
        sys.exit(1)

    if args.dry_run:
        # dry-run でも JVRead を1回呼んでデータ有無を確認
        print("\n[DRY-RUN] JVRead でデータ有無を確認...")
        try:
            ret = jv.JVRead("", 110000, "")
            if isinstance(ret, tuple):
                res = ret[0]
                buff = ret[1] if len(ret) > 1 else ""
            else:
                res = ret
                buff = ""

            if res > 0:
                rec_id = buff[:2] if len(buff) >= 2 else "??"
                print(f"✅ データあり: 最初のレコード種別={rec_id}, サイズ={res}")
                print("   (これ以上の読み込みはスキップ)")
            elif res == 0:
                print("⚠️  データなし (EOF)")
                print("   考えられる原因:")
                print("   1. 指定したレースキーにデータが存在しない")
                print("   2. データ保持期間（1年）を超えている")
            elif res == -1:
                print("⚠️  ファイル切り替わり (データなしの可能性)")
            elif res == -3:
                print("⏳ ダウンロード中...")
            else:
                print(f"❌ JVRead エラー: {res}")
        except Exception as e:
            print(f"❌ JVRead 例外: {e}")
        jv.JVClose()
        return

    # 出力ファイル名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = args.output_dir / f"{args.dataspec}_{key}_{timestamp}.jsonl"
    print(f"\n📁 出力先: {output_file}")

    # 抽出＆保存
    print("\n📖 データ読み込み開始...")
    records = extract_records(jv, args.max_records)
    count = save_jsonl(records, output_file)

    print("")
    print("=" * 60)
    print(f"✅ 完了: {count} 件を {output_file.name} に保存")
    print("=" * 60)


if __name__ == "__main__":
    main()
