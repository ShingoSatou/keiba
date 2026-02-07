"""
JSONL → PostgreSQL ロードスクリプト (64bit Python用)

extract_jvlink.py で出力したJSONLファイルをPostgreSQLに投入します。
64bit Python環境 (.venv) で実行する必要があります。

使用方法:
    uv run python scripts/load_to_db.py --input data/RACE_20260203_123456.jsonl

    # 全ファイルを一括処理
    uv run python scripts/load_to_db.py --input-dir data/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# プロジェクトルート設定
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.infrastructure.database import Database  # noqa: E402
from app.infrastructure.parsers import (  # noqa: E402
    HorseExclusionRecord,
    HorseMasterRecord,
    JockeyRecord,
    OddsRecord,
    PayoutRecord,
    RaceRecord,
    RunnerRecord,
    TrainerRecord,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_jsonl(file_path: Path):
    """JSONLファイルを読み込み"""
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def insert_raw_record(db: Database, dataspec: str, record: dict) -> None:
    """raw.jv_rawにレコードを挿入（単一レコード版）"""
    sql = """
        INSERT INTO raw.jv_raw (dataspec, rec_id, filename, payload)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """
    db.execute(sql, (dataspec, record["rec_id"], record["filename"], record["payload"]))


def insert_raw_records_batch(db: Database, dataspec: str, records: list[dict]) -> int:
    """raw.jv_rawにレコードをバッチ挿入（高速版）"""
    if not records:
        return 0
    sql = """
        INSERT INTO raw.jv_raw (dataspec, rec_id, filename, payload)
        VALUES (%s, %s, %s, %s)
        ON CONFLICT DO NOTHING
    """
    values = [(dataspec, r["rec_id"], r["filename"], r["payload"]) for r in records]
    cursor = db.connect().cursor()
    cursor.executemany(sql, values)
    return len(records)


def upsert_race(db: Database, race: RaceRecord) -> None:
    """core.raceにレースを挿入/更新"""
    sql = """
        INSERT INTO core.race (
            race_id, race_date, track_code, race_no, surface,
            distance_m, going, weather, class_code, field_size, start_time
        ) VALUES (
            %(race_id)s, %(race_date)s, %(track_code)s, %(race_no)s, %(surface)s,
            %(distance_m)s, %(going)s, %(weather)s,
            %(class_code)s, %(field_size)s, %(start_time)s
        )
        ON CONFLICT (race_id) DO UPDATE SET
            going = EXCLUDED.going,
            weather = EXCLUDED.weather,
            field_size = EXCLUDED.field_size,
            start_time = EXCLUDED.start_time,
            updated_at = now()
    """
    db.execute(
        sql,
        {
            "race_id": race.race_id,
            "race_date": race.race_date,
            "track_code": race.track_code,
            "race_no": race.race_no,
            "surface": race.surface,
            "distance_m": race.distance_m,
            "going": race.going,
            "weather": race.weather,
            "class_code": race.class_code,
            "field_size": race.field_size,
            "start_time": race.start_time,
        },
    )


def prepare_master_data_cache(db: Database) -> tuple[set[int], set[int]]:
    """マスタデータのIDセットをキャッシュとして取得"""
    jockeys = {row["jockey_id"] for row in db.fetch_all("SELECT jockey_id FROM core.jockey")}
    trainers = {row["trainer_id"] for row in db.fetch_all("SELECT trainer_id FROM core.trainer")}
    return jockeys, trainers


def upsert_runner(
    db: Database,
    runner: RunnerRecord,
    master_jockeys: set[int] = None,
    master_trainers: set[int] = None,
) -> None:
    """core.runner と core.result にデータを挿入/更新"""

    # FK制御ロジック
    # 1. 地方競馬(A)や海外(B2)はマスタ参照できないためNULL
    # 2. それ以外でもマスタに存在しない場合はNULL (エラー回避)

    # マスタキャッシュが渡されていない場合は空集合として扱う（安全策）
    if master_jockeys is None:
        master_jockeys = set()
    if master_trainers is None:
        master_trainers = set()

    safe_jockey_id = runner.jockey_id
    safe_trainer_id = runner.trainer_id

    # データ区分によるフィルタ
    # 'A': 地方, 'B2': 海外
    # これらは仕様上コードが入っていてもマスタとリンクしない
    if runner.data_kubun in ("A", "B2"):
        safe_jockey_id = None
        safe_trainer_id = None
    else:
        # マスタ存在チェック (中央系でもマスタ欠けがある場合への対処)
        if safe_jockey_id is not None and safe_jockey_id not in master_jockeys:
            # logger.warning(f"Missing Jockey Master: {safe_jockey_id} (Race: {runner.race_id})")
            safe_jockey_id = None

        if safe_trainer_id is not None and safe_trainer_id not in master_trainers:
            # logger.warning(f"Missing Trainer Master: {safe_trainer_id} (Race: {runner.race_id})")
            safe_trainer_id = None

    # Missing Horse対策: 馬マスタになければ、SEレコードの情報で自動登録する
    # (地方馬や、UMレコード未着のケースに対応)
    horse_sql = """
        INSERT INTO core.horse (horse_id, horse_name)
        VALUES (%(horse_id)s, %(horse_name)s)
        ON CONFLICT (horse_id) DO NOTHING
    """
    db.execute(horse_sql, {"horse_id": runner.horse_id, "horse_name": runner.horse_name})

    # race_idから最低限の情報を復元して入れたいところだが、とりあえずrace_idのみ試みる。
    # スキーマのNOT NULL制約に引っかかる場合はデフォルト値を入れる必要がある。
    # ここでは一般的な必須項目（日付、場、回、R）をrace_idから分解して入れたほうが安全。
    # parsers.pyのロジック: race_id = date_int * 10000 + track * 100 + race_no
    # つまり YYYYMMDD + TT + RR = YYYYMMDDTTRR

    try:
        r_id = runner.race_id
        race_no = r_id % 100
        track_code = (r_id // 100) % 100
        date_int = int(r_id // 10000)
        y = date_int // 10000
        m = (date_int // 100) % 100
        d = date_int % 100
        race_date = f"{y:04d}-{m:02d}-{d:02d}"
    except Exception:
        race_no = 0
        track_code = 0
        race_date = "2000-01-01"

    # 簡易的に race_id のみでInsertしようとするとNOT NULL制約等で失敗し、
    # その際のRollbackで直前のHorse登録まで消えてしまうリスクがあるため、
    # 最初からデフォルト値付きの完全なSQLで実行する。
    race_sql = """
        INSERT INTO core.race (
            race_id, race_date, track_code, race_no, surface, distance_m, 
            going, weather, class_code, field_size
        ) VALUES (
            %(race_id)s, %(race_date)s, %(track_code)s, %(race_no)s, 0, 0, 0, 0, 0, 0
        ) 
        ON CONFLICT (race_id) DO NOTHING
    """
    db.execute(
        race_sql,
        {
            "race_id": runner.race_id,
            "race_date": race_date,
            "track_code": track_code,
            "race_no": race_no,
        },
    )

    # runner (出走表)
    runner_sql = """
        INSERT INTO core.runner (
            race_id, horse_id, horse_no, gate, jockey_id,
            trainer_id, carried_weight, body_weight, body_weight_diff,
            data_kubun, trainer_code_raw, trainer_name_abbr,
            jockey_code_raw, jockey_name_abbr
        ) VALUES (
            %(race_id)s, %(horse_id)s, %(horse_no)s, %(gate)s, %(jockey_id)s,
            %(trainer_id)s, %(carried_weight)s, %(body_weight)s, %(body_weight_diff)s,
            %(data_kubun)s, %(trainer_code_raw)s, %(trainer_name_abbr)s,
            %(jockey_code_raw)s, %(jockey_name_abbr)s
        )
        ON CONFLICT (race_id, horse_id) DO UPDATE SET
            horse_no = EXCLUDED.horse_no,
            gate = EXCLUDED.gate,
            jockey_id = EXCLUDED.jockey_id,
            trainer_id = EXCLUDED.trainer_id,
            carried_weight = EXCLUDED.carried_weight,
            body_weight = EXCLUDED.body_weight,
            body_weight_diff = EXCLUDED.body_weight_diff,
            data_kubun = EXCLUDED.data_kubun,
            trainer_code_raw = EXCLUDED.trainer_code_raw,
            trainer_name_abbr = EXCLUDED.trainer_name_abbr,
            jockey_code_raw = EXCLUDED.jockey_code_raw,
            jockey_name_abbr = EXCLUDED.jockey_name_abbr,
            updated_at = now()
    """
    db.execute(
        runner_sql,
        {
            "race_id": runner.race_id,
            "horse_id": runner.horse_id,
            "horse_no": runner.horse_no,
            "gate": runner.gate,
            "jockey_id": safe_jockey_id,
            "trainer_id": safe_trainer_id,
            "carried_weight": runner.carried_weight,
            "body_weight": runner.body_weight,
            "body_weight_diff": runner.body_weight_diff,
            "data_kubun": runner.data_kubun,
            "trainer_code_raw": runner.trainer_code_raw,
            "trainer_name_abbr": runner.trainer_name_abbr,
            "jockey_code_raw": runner.jockey_code_raw,
            "jockey_name_abbr": runner.jockey_name_abbr,
        },
    )

    # result (結果) - 着順がある場合のみ
    if runner.finish_pos and runner.finish_pos > 0:
        result_sql = """
            INSERT INTO core.result (
                race_id, horse_id, finish_pos, time_sec,
                margin, final3f_sec, corner1_pos, corner2_pos, corner3_pos, corner4_pos
            ) VALUES (
                %(race_id)s, %(horse_id)s, %(finish_pos)s, %(time_sec)s,
                %(margin)s, %(final3f_sec)s, %(corner1_pos)s, %(corner2_pos)s,
                %(corner3_pos)s, %(corner4_pos)s
            )
            ON CONFLICT (race_id, horse_id) DO UPDATE SET
                finish_pos = EXCLUDED.finish_pos,
                time_sec = EXCLUDED.time_sec,
                margin = EXCLUDED.margin,
                final3f_sec = EXCLUDED.final3f_sec,
                corner1_pos = EXCLUDED.corner1_pos,
                corner2_pos = EXCLUDED.corner2_pos,
                corner3_pos = EXCLUDED.corner3_pos,
                corner4_pos = EXCLUDED.corner4_pos,
                updated_at = now()
        """
        db.execute(
            result_sql,
            {
                "race_id": runner.race_id,
                "horse_id": runner.horse_id,
                "finish_pos": runner.finish_pos,
                "time_sec": runner.time_sec,
                "margin": runner.margin,
                "final3f_sec": runner.final3f_sec,
                "corner1_pos": runner.corner1_pos,
                "corner2_pos": runner.corner2_pos,
                "corner3_pos": runner.corner3_pos,
                "corner4_pos": runner.corner4_pos,
            },
        )


def upsert_payout(db: Database, payout: PayoutRecord) -> None:
    """core.payoutにデータを挿入/更新"""
    sql = """
        INSERT INTO core.payout (race_id, bet_type, selection, payout_yen, popularity)
        VALUES (%(race_id)s, %(bet_type)s, %(selection)s, %(payout_yen)s, %(popularity)s)
        ON CONFLICT (race_id, bet_type, selection) DO UPDATE SET
            payout_yen = EXCLUDED.payout_yen,
            popularity = EXCLUDED.popularity
    """
    db.execute(
        sql,
        {
            "race_id": payout.race_id,
            "bet_type": payout.bet_type,
            "selection": payout.selection,
            "payout_yen": payout.payout_yen,
            "popularity": payout.popularity,
        },
    )


def upsert_odds(db: Database, odds: OddsRecord) -> None:
    """core.odds_finalにデータを挿入/更新

    Note: O1レコードにはhorse_idがないため、core.runnerからhorse_noで引いて解決する
    bet_type=1 (Win), bet_type=2 (Place) のみ処理
    """
    if odds.bet_type == 1:  # Win
        sql = """
            INSERT INTO core.odds_final (race_id, horse_id, odds_win, pop_win)
            SELECT %(race_id)s, horse_id, %(odds_win)s, %(pop_win)s
            FROM core.runner
            WHERE race_id = %(race_id)s AND horse_no = %(horse_no)s
            ON CONFLICT (race_id, horse_id) DO UPDATE SET
                odds_win = EXCLUDED.odds_win,
                pop_win = EXCLUDED.pop_win,
                updated_at = now()
        """
        db.execute(
            sql,
            {
                "race_id": odds.race_id,
                "horse_no": int(odds.horse_no) if isinstance(odds.horse_no, str) else odds.horse_no,
                "odds_win": odds.odds_1,
                "pop_win": odds.popularity,
            },
        )
    elif odds.bet_type == 2:  # Place
        sql = """
            INSERT INTO core.odds_final
                (race_id, horse_id, odds_place_low, odds_place_high, pop_place)
            SELECT %(race_id)s, horse_id, %(odds_place_low)s,
                %(odds_place_high)s, %(pop_place)s
            FROM core.runner
            WHERE race_id = %(race_id)s AND horse_no = %(horse_no)s
            ON CONFLICT (race_id, horse_id) DO UPDATE SET
                odds_place_low = EXCLUDED.odds_place_low,
                odds_place_high = EXCLUDED.odds_place_high,
                pop_place = EXCLUDED.pop_place,
                updated_at = now()
        """
        db.execute(
            sql,
            {
                "race_id": odds.race_id,
                "horse_no": int(odds.horse_no) if isinstance(odds.horse_no, str) else odds.horse_no,
                "odds_place_low": odds.odds_1,
                "odds_place_high": odds.odds_2,
                "pop_place": odds.popularity,
            },
        )
    # bet_type=3 (Bracket) は現状スキップ（horse_idへの変換が困難）


def upsert_horse(db: Database, horse: HorseExclusionRecord) -> None:
    """core.horseにデータを挿入/更新

    JGレコードは競走馬除外情報なので、馬名のみ更新。
    """
    sql = """
        INSERT INTO core.horse (horse_id, horse_name)
        VALUES (%(horse_id)s, %(horse_name)s)
        ON CONFLICT (horse_id) DO UPDATE SET
            horse_name = EXCLUDED.horse_name,
            updated_at = now()
    """
    db.execute(
        sql,
        {
            "horse_id": horse.horse_id,
            "horse_name": horse.horse_name,
        },
    )


def upsert_horse_master(db: Database, horse: HorseMasterRecord) -> None:
    """core.horseにデータを挿入/更新

    UMレコード（馬マスタ）から馬情報を登録。
    """
    sql = """
        INSERT INTO core.horse (horse_id, horse_name)
        VALUES (%(horse_id)s, %(horse_name)s)
        ON CONFLICT (horse_id) DO UPDATE SET
            horse_name = EXCLUDED.horse_name,
            updated_at = now()
    """
    db.execute(
        sql,
        {
            "horse_id": horse.horse_id,
            "horse_name": horse.horse_name,
        },
    )


def upsert_jockey(db: Database, jockey: JockeyRecord) -> None:
    """core.jockeyにデータを挿入/更新"""
    sql = """
        INSERT INTO core.jockey (jockey_id, jockey_name)
        VALUES (%(jockey_id)s, %(jockey_name)s)
        ON CONFLICT (jockey_id) DO UPDATE SET
            jockey_name = EXCLUDED.jockey_name,
            updated_at = now()
    """
    db.execute(
        sql,
        {
            "jockey_id": jockey.jockey_id,
            "jockey_name": jockey.jockey_name,
        },
    )


def upsert_trainer(db: Database, trainer: TrainerRecord) -> None:
    """core.trainerにデータを挿入/更新"""
    sql = """
        INSERT INTO core.trainer (trainer_id, trainer_name)
        VALUES (%(trainer_id)s, %(trainer_name)s)
        ON CONFLICT (trainer_id) DO UPDATE SET
            trainer_name = EXCLUDED.trainer_name,
            updated_at = now()
    """
    db.execute(
        sql,
        {
            "trainer_id": trainer.trainer_id,
            "trainer_name": trainer.trainer_name,
        },
    )


def process_file(db: Database, file_path: Path) -> dict[str, int]:
    """JSONLファイルを処理してDBに投入

    RAレコード（レース情報）を最初に処理してコミットすることで、
    後続のSE/HR/O1レコードでのFK違反（race_id参照エラー）を防ぐ。
    """
    stats = {"raw": 0, "race": 0, "runner": 0, "payout": 0, "odds": 0, "horse": 0, "errors": 0}
    BATCH_SIZE = 1000

    # マスタデータをメモリにロード (FK検証用)
    master_jockeys, master_trainers = prepare_master_data_cache(db)

    # ファイル名からdataspecを推定
    dataspec = file_path.stem.split("_")[0]

    # Pass 1: Race (RA) のみ処理 + rawバッチ挿入
    logger.info("  Pass 1: Processing Race records...")
    raw_batch = []
    for record in load_jsonl(file_path):
        try:
            # rawレコードをバッチに追加
            raw_batch.append(record)
            stats["raw"] += 1

            rec_id = record["rec_id"]
            payload = record["payload"]

            if rec_id == "RA":
                race = RaceRecord.parse(payload)
                upsert_race(db, race)
                stats["race"] += 1

            # バッチサイズに達したらINSERT
            if len(raw_batch) >= BATCH_SIZE:
                insert_raw_records_batch(db, dataspec, raw_batch)
                raw_batch = []
                db.connect().commit()
                logger.info(f"    {stats['raw']:,} 件処理...")

        except Exception:
            db.connect().rollback()
            raw_batch = []  # バッチをクリア
            if stats["errors"] < 5:
                pass
            stats["errors"] += 1

    # 残りのバッチを処理
    if raw_batch:
        insert_raw_records_batch(db, dataspec, raw_batch)
    db.connect().commit()  # Pass 1完了コミット
    logger.info(f"  Pass 1 完了: {stats['raw']:,} raw, {stats['race']:,} race")

    # Pass 2: Skip RA, process others
    logger.info("  Pass 2: Processing other records...")
    # rawのカウントはPass 1で済ませているのでリセットしない
    # (insert_raw_recordがON CONFLICT DO NOTHINGなので再実行しても安全)
    # ただし stats["raw"] が2倍になるのを防ぐため、Pass 2では raw はスキップ

    pass2_count = 0
    for record in load_jsonl(file_path):
        pass2_count += 1
        try:
            # rawはPass 1で入れているのでスキップ

            rec_id = record["rec_id"]
            payload = record["payload"]

            if rec_id == "RA":
                continue  # 処理済み

            elif rec_id == "SE":
                runner = RunnerRecord.parse(payload)
                upsert_runner(db, runner, master_jockeys, master_trainers)
                stats["runner"] += 1

            elif rec_id == "HR":
                payouts = PayoutRecord.parse(payload)
                for p in payouts:
                    upsert_payout(db, p)
                    stats["payout"] += 1

            elif rec_id == "O1":
                odds_list = OddsRecord.parse(payload)
                for o in odds_list:
                    upsert_odds(db, o)
                    stats["odds"] += 1

            elif rec_id == "JG":
                horse = HorseExclusionRecord.parse(payload)
                upsert_horse(db, horse)
                stats["horse"] += 1

            elif rec_id == "UM":
                horse = HorseMasterRecord.parse(payload)
                upsert_horse_master(db, horse)
                stats["horse"] += 1

            elif rec_id == "KS":
                jockey = JockeyRecord.parse(payload)
                upsert_jockey(db, jockey)
                if "jockey" not in stats:
                    stats["jockey"] = 0
                stats["jockey"] += 1

            elif rec_id == "CH":
                trainer = TrainerRecord.parse(payload)
                upsert_trainer(db, trainer)
                if "trainer" not in stats:
                    stats["trainer"] = 0
                stats["trainer"] += 1

        except Exception as e:
            # ロールバックしてトランザクションを復旧
            db.connect().rollback()
            if stats["errors"] < 20:
                logger.warning(f"Pass 2 Error [{rec_id}]: {e}")
                # logger.warning(traceback.format_exc())
            stats["errors"] += 1

        if pass2_count % 1000 == 0:
            db.connect().commit()
            logger.info(f"  {pass2_count} 件スキャン完了...")

    db.connect().commit()
    return stats


def main():
    parser = argparse.ArgumentParser(description="JSONL → PostgreSQL ロード")
    parser.add_argument(
        "--input",
        type=str,
        help="入力JSONLファイル (ワイルドカード対応: data/RACE_*.jsonl)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="入力ディレクトリ (全JSONLファイルを処理)",
    )
    args = parser.parse_args()

    if not args.input and not args.input_dir:
        parser.error("--input または --input-dir を指定してください")

    files = []
    if args.input:
        # ワイルドカードサポート
        input_path = Path(args.input)
        if "*" in args.input:
            # glob展開
            parent = input_path.parent if input_path.parent.exists() else Path(".")
            pattern = input_path.name
            files.extend(sorted(parent.glob(pattern)))
        else:
            files.append(input_path)
    if args.input_dir:
        files.extend(sorted(args.input_dir.glob("*.jsonl")))

    if not files:
        print("❌ 処理対象ファイルがありません")
        sys.exit(1)

    print("=" * 60)
    print("JSONL → PostgreSQL ロード")
    print("=" * 60)
    print(f"対象ファイル: {len(files)} 件")
    print("=" * 60)

    total_stats = {"raw": 0, "race": 0, "runner": 0, "payout": 0, "odds": 0, "errors": 0}

    with Database() as db:
        for file_path in files:
            logger.info(f"📂 処理中: {file_path.name}")
            stats = process_file(db, file_path)
            for key in total_stats:
                total_stats[key] += stats[key]
            logger.info(f"  → {stats}")

    print("")
    print("=" * 60)
    print("✅ 完了")
    print(f"   Raw: {total_stats['raw']}")
    print(f"   Race: {total_stats['race']}")
    print(f"   Runner: {total_stats['runner']}")
    print(f"   Payout: {total_stats['payout']}")
    print(f"   Odds: {total_stats['odds']}")
    print(f"   Horse: {total_stats.get('horse', 0)}")
    print(f"   Jockey: {total_stats.get('jockey', 0)}")
    print(f"   Errors: {total_stats['errors']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
