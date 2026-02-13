"""
パーサー動作確認スクリプト

ダウンロード済みJSONLファイルから少量データを読み取り、
新規追加したパーサーの動作を検証します。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.parsers import (  # noqa: E402
    CKRecord,
    DMRecord,
    HCRecord,
    OddsTimeSeriesRecord,
    TMRecord,
    WCRecord,
)

DATA_DIR = PROJECT_ROOT / "data"


def load_first_n(file_path: Path, n: int = 5):
    """JSONLファイルから最初のN件を読み込み"""
    records = []
    with file_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            if line.strip():
                records.append(json.loads(line))
    return records


def verify_slop():
    """SLOP（坂路調教 HC）パーサー検証"""
    files = sorted(DATA_DIR.glob("SLOP_*.jsonl"))
    if not files:
        print("⚠️  SLOP ファイルなし → スキップ")
        return

    print("\n📋 SLOP (HC) パーサー検証:")
    records = load_first_n(files[0], 5)
    success = 0
    for r in records:
        if r["rec_id"] != "HC":
            continue
        try:
            parsed = HCRecord.parse(r["payload"])
            print(
                f"  ✅ HC: horse_id={parsed.horse_id}, "
                f"date={parsed.training_date}, "
                f"center={parsed.training_center}, "
                f"time={parsed.training_time}, "
                f"Total4F={parsed.total_4f}s, "
                f"Lap1F={parsed.lap_1f}s"
            )
            success += 1
        except Exception as e:
            print(f"  ❌ HC parse error: {e}")

    print(f"  結果: {success}/{len(records)} 成功")


def verify_wood():
    """WOOD（ウッド調教 WC）パーサー検証"""
    files = sorted(DATA_DIR.glob("WOOD_*.jsonl"))
    if not files:
        print("⚠️  WOOD ファイルなし → スキップ")
        return

    print("\n📋 WOOD (WC) パーサー検証:")
    records = load_first_n(files[0], 5)
    success = 0
    for r in records:
        if r["rec_id"] != "WC":
            continue
        try:
            parsed = WCRecord.parse(r["payload"])
            print(
                f"  ✅ WC: horse_id={parsed.horse_id}, "
                f"date={parsed.training_date}, "
                f"center={parsed.center if hasattr(parsed, 'center') else parsed.training_center}, "
                f"course={parsed.course}, "
                f"dir={parsed.direction}, "
                f"Total6F={parsed.total_6f}s, "
                f"Lap1F={parsed.lap_1f}s"
            )
            success += 1
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"  ❌ WC parse error: {e}")

    print(f"  結果: {success}/{len(records)} 成功")


def verify_ming():
    """MING（マイニング DM/TM）パーサー検証"""
    dm_files = sorted(DATA_DIR.glob("MING_*.jsonl"))
    if not dm_files:
        print("⚠️  MING (DM) ファイルなし → スキップ")
        return

    print("\n📋 MING (DM/TM) パーサー検証:")
    # DM/TMは同一ファイルに含まれる可能性があるが、今回はDMファイルとして取得
    records = load_first_n(dm_files[0], 20)
    dm_count = 0
    tm_count = 0
    for r in records:
        try:
            if r["rec_id"] == "DM":
                parsed_list = DMRecord.parse(r["payload"])
                if parsed_list:
                    first = parsed_list[0]
                    print(
                        f"  ✅ DM: race_id={first.race_id}, "
                        f"data_kbn={first.data_kbn}, "
                        f"horses={len(parsed_list)}"
                    )
                else:
                    print("  ⚠️  DM: パース結果が空")
                dm_count += len(parsed_list)
            elif r["rec_id"] == "TM":
                parsed_list = TMRecord.parse(r["payload"])
                if parsed_list:
                    first = parsed_list[0]
                    print(
                        f"  ✅ TM: race_id={first.race_id}, "
                        f"data_kbn={first.data_kbn}, "
                        f"horses={len(parsed_list)}"
                    )
                else:
                    print("  ⚠️  TM: パース結果が空")
                tm_count += len(parsed_list)
        except Exception as e:
            print(f"  ❌ MING parse error ({r['rec_id']}): {e}")

    print(f"  結果: DM={dm_count}, TM={tm_count}")


def verify_o1():
    """O1時系列オッズパーサー検証"""
    files = sorted(DATA_DIR.glob("0B41_*.jsonl"))
    if not files:
        print("⚠️  0B41 ファイルなし → スキップ")
        return

    print("\n📋 O1 時系列オッズ パーサー検証:")
    records = load_first_n(files[0], 3)
    success = 0
    for r in records:
        if r["rec_id"] != "O1":
            continue
        try:
            parsed_list = OddsTimeSeriesRecord.parse(r["payload"])
            first = parsed_list[0] if parsed_list else None
            if first:
                print(
                    f"  ✅ O1: race_id={first.race_id}, "
                    f"data_kbn={first.data_kbn}, "
                    f"announce={first.announce_mmddhhmi}, "
                    f"horses={len(parsed_list)}, "
                    f"pool={first.win_pool_total_100yen}"
                )
                # 最初の3頭だけ詳細表示
                for h in parsed_list[:3]:
                    print(
                        f"       馬番{h.horse_no}: "
                        f"odds={h.win_odds_x10 / 10:.1f}倍, "
                        f"pop={h.win_popularity}"
                    )
                success += 1
            else:
                print("  ⚠️  O1: パース結果が空")
        except Exception as e:
            print(f"  ❌ O1 parse error: {e}")

    print(f"  結果: {success}/{len(records)} 成功")


def verify_snpn():
    """SNPN（出走別着度数 CK）パーサー検証"""
    files = sorted(DATA_DIR.glob("SNPN_*.jsonl"))
    if not files:
        print("⚠️  SNPN ファイルなし → スキップ")
        return

    print("\n📋 SNPN (CK) パーサー検証:")
    # 最新ファイルを使用 (2020年抽出ファイルがあるはず)
    records = load_first_n(files[-1], 5)
    success = 0
    for r in records:
        if r["rec_id"] != "CK":
            continue
        try:
            parsed = CKRecord.parse(r["payload"])
            try:
                track_int = int(parsed.track_cd) if parsed.track_cd.isdigit() else 0
                date_int = int(f"{parsed.kaisai_year:04d}{parsed.kaisai_md}")
                race_id = date_int * 10000 + track_int * 100 + parsed.race_no
            except Exception:
                race_id = 0
            print(
                f"  ✅ CK: race_id={race_id}, "
                f"make={parsed.make_date}, "
                f"horse={parsed.horse_name.strip()} ({parsed.horse_id}), "
                f"h_total={parsed.counts_total}, "  # Horse Total Stats
                f"jockey={parsed.jockey_code}"
            )
            success += 1
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"  ❌ CK parse error: {e}")

    print(f"  結果: {success}/{len(records)} 成功")


def main():
    print("=" * 60)
    print("パーサー動作確認")
    print("=" * 60)

    verify_o1()
    verify_slop()
    verify_wood()
    verify_ming()
    verify_snpn()

    print("\n" + "=" * 60)
    print("検証完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
