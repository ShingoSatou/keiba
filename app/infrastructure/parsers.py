"""
JV-Data レコードパーサー

JRA-VAN の固定長レコードをパースしてPythonオブジェクトに変換するモジュール。
仕様書: JV-Data仕様書_4.9.0.1.pdf / xlsx 参照

バイト位置はJV-Data仕様書に基づいています。
文字列はShift_JIS (CP932) でエンコードされています。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, time


def _slice_decode(payload: str, start: int, length: int) -> str:
    """
    固定長文字列から指定位置の文字列を取得

    Args:
        payload: 固定長文字列 (UTF-8変換済み)
        start: 開始位置 (0-indexed)
        length: 文字長

    Returns:
        取得した文字列 (strip済み、全角スペースも除去)
    """
    result = payload[start : start + length].strip()
    # 全角スペース（\u3000）も除去
    return result.strip("\u3000")


def _slice_int(payload: str, start: int, length: int, default: int = 0) -> int:
    """固定長文字列から整数を取得"""
    val = _slice_decode(payload, start, length)
    if not val:
        return default
    # 数字以外の文字を除去
    digits = "".join(c for c in val if c.isdigit())
    if not digits:
        return default
    return int(digits)


def _slice_date(payload: str, start: int) -> date | None:
    """固定長文字列から日付を取得 (YYYYMMDD形式, 8文字)"""
    val = _slice_decode(payload, start, 8)
    if len(val) != 8 or not val.isdigit():
        return None
    try:
        year = int(val[:4])
        month = int(val[4:6])
        day = int(val[6:8])
        if year == 0 or month == 0 or day == 0:
            return None
        return date(year, month, day)
    except ValueError:
        return None


def _slice_time(payload: str, start: int) -> time | None:
    """固定長文字列から時刻を取得 (HHMM形式, 4文字)"""
    val = _slice_decode(payload, start, 4)
    if len(val) != 4 or not val.isdigit():
        return None
    try:
        return time(int(val[:2]), int(val[2:4]))
    except ValueError:
        return None


def _time_to_seconds(time_str: str) -> float | None:
    """
    走破タイム文字列を秒に変換

    Args:
        time_str: "MSSF" 形式 (M:分, SS:秒, F:1/10秒) または "MMSSF"

    Returns:
        秒数 (float) または None
    """
    time_str = time_str.strip()
    if not time_str or not time_str.replace(".", "").isdigit():
        return None

    try:
        if len(time_str) == 4:
            # MSSF形式: 1分23秒4 = "1234"
            minutes = int(time_str[0])
            seconds = int(time_str[1:3])
            tenths = int(time_str[3])
            return minutes * 60 + seconds + tenths / 10.0
        elif len(time_str) == 5:
            # MMSSF形式: 12分34秒5 = "12345"
            minutes = int(time_str[0:2])
            seconds = int(time_str[2:4])
            tenths = int(time_str[4])
            return minutes * 60 + seconds + tenths / 10.0
    except ValueError:
        pass
    return None


def _slice_byte_decode(data: bytes, start: int, length: int) -> str:
    """バイト列から指定位置の文字列をShift_JISデコードして取得"""
    chunk = data[start : start + length]
    try:
        return chunk.decode("cp932").strip().strip("\u3000")
    except UnicodeDecodeError:
        return ""


def _slice_byte_int(data: bytes, start: int, length: int, default: int = 0) -> int:
    """バイト列から整数を取得"""
    s = _slice_byte_decode(data, start, length)
    if not s:
        return default
    digits = "".join(c for c in s if c.isdigit())
    if not digits:
        return default
    return int(digits)


def _slice_byte_maskable_int(data: bytes, start: int, length: int) -> int | None:
    """バイト列から整数を取得（マスク/欠損はNone）"""
    s = _slice_byte_decode(data, start, length)
    if not s:
        return None
    if "*" in s:
        return None
    digits = "".join(c for c in s if c.isdigit())
    if not digits:
        return None
    return int(digits)


def _normalize_distance_m(distance_m: int | None) -> int | None:
    """距離(m)の異常スケールを補正する"""
    if distance_m is None or distance_m <= 0:
        return distance_m
    if distance_m < 100:
        return distance_m * 100
    return distance_m


def _slice_byte_time(data: bytes, start: int, length: int = 4) -> time | None:
    """バイト列から時刻(HHMM)を取得"""
    s = _slice_byte_decode(data, start, length)
    if len(s) != 4 or not s.isdigit():
        return None
    try:
        return time(int(s[:2]), int(s[2:4]))
    except ValueError:
        return None


# =============================================================================
# 共通ヘッダー (全レコード共通)
# =============================================================================
# JV-Dataの全レコードは以下の共通ヘッダーで始まる:
# 位置 0-1 (2文字): レコード種別ID (RA, SE, HR, etc.)
# 位置 2 (1文字): データ区分 (1=新規, 2=更新, 3=削除)
HEADER_REC_ID_START = 0
HEADER_REC_ID_LEN = 2
HEADER_DATA_DIV_START = 2
HEADER_DATA_DIV_LEN = 1


# =============================================================================
# RA: レース詳細 (Race)
# =============================================================================
# RAレコードの構造 (JV-Data仕様書 4.9.0.1 準拠 / 文字数ベース)

RA_CREATED_AT_START = 3
RA_CREATED_AT_LEN = 8

# レースキー
RA_YEAR_START = 11
RA_YEAR_LEN = 4
RA_MONTHDAY_START = 15
RA_MONTHDAY_LEN = 4
RA_TRACK_CODE_START = 19
RA_TRACK_CODE_LEN = 2
RA_KAI_START = 21
RA_KAI_LEN = 2
RA_NICHI_START = 23
RA_NICHI_LEN = 2
RA_RACE_NO_START = 25
RA_RACE_NO_LEN = 2

# 基本情報
RA_DAY_OF_WEEK_START = 27
RA_DAY_OF_WEEK_LEN = 1
RA_SPECIAL_NO_START = 28
RA_SPECIAL_NO_LEN = 4
RA_RACE_NAME_START = 32
RA_RACE_NAME_LEN = 30  # 全角30文字

# 詳細条件 (JV-Data 4.9.0.1 準拠: 位置は仕様書の1-index → 0-indexに変換)
RA_DISTANCE_START = 697  # 仕様書: 698
RA_DISTANCE_LEN = 4
RA_TRACK_TYPE_START = 705  # 仕様書: 706 (トラックコード)
RA_TRACK_TYPE_LEN = 2
RA_COURSE_START = 709  # 仕様書: 710
RA_COURSE_LEN = 2

# 競走条件・グレード（仕様書 1-origin -> 0-origin）
RA_GRADE_CODE_START = 614  # 仕様書: 615
RA_GRADE_CODE_LEN = 1
RA_RACE_TYPE_CODE_START = 616  # 実データ検証: 仕様位置617
RA_RACE_TYPE_CODE_LEN = 2
RA_WEIGHT_TYPE_CODE_START = 621  # 実データ検証: 仕様位置622
RA_WEIGHT_TYPE_CODE_LEN = 1
RA_COND_CODE_2YO_START = 622  # 実データ検証: 仕様位置623
RA_COND_CODE_2YO_LEN = 3
RA_COND_CODE_3YO_START = 625  # 実データ検証: 仕様位置626
RA_COND_CODE_3YO_LEN = 3
RA_COND_CODE_4YO_START = 628  # 実データ検証: 仕様位置629
RA_COND_CODE_4YO_LEN = 3
RA_COND_CODE_5UP_START = 631  # 実データ検証: 仕様位置632
RA_COND_CODE_5UP_LEN = 3
RA_COND_CODE_MIN_AGE_START = 634  # 実データ検証: 仕様位置635
RA_COND_CODE_MIN_AGE_LEN = 3

# 賞金 (本賞金 1着)
RA_PRIZE1_START = 713  # 仕様書: 714
RA_PRIZE1_LEN = 8

# 発走・頭数・条件
RA_START_TIME_START = 873  # 仕様書: 874
RA_START_TIME_LEN = 4
RA_FIELD_SIZE_START = 881  # 仕様書: 882 (登録頭数)
RA_FIELD_SIZE_LEN = 2
RA_STARTERS_START = 883  # 仕様書: 884 (出走頭数)
RA_STARTERS_LEN = 2
RA_WEATHER_START = 887  # 仕様書: 888
RA_WEATHER_LEN = 1
RA_TURF_GOING_START = 888  # 仕様書: 889
RA_TURF_GOING_LEN = 1
RA_DIRT_GOING_START = 889  # 仕様書: 890
RA_DIRT_GOING_LEN = 1


GRADE_CODE_MAP: dict[str, int] = {
    "A": 1,  # G1
    "B": 2,  # G2
    "C": 3,  # G3
    "D": 4,  # 非グレード重賞
    "E": 5,  # 特別競走
    "F": 6,  # JG1
    "G": 7,  # JG2
    "H": 8,  # JG3
    "L": 9,  # Listed
}


def _grade_code_to_int(code: str) -> int | None:
    code_norm = (code or "").strip().upper()
    if not code_norm or code_norm == "_":
        return None
    return GRADE_CODE_MAP.get(code_norm)


def _choose_condition_code(
    race_type_code: int | None,
    cond_2yo: int | None,
    cond_3yo: int | None,
    cond_4yo: int | None,
    cond_5up: int | None,
    cond_min_age: int | None,
) -> int:
    if race_type_code == 11 and cond_2yo and cond_2yo > 0:
        return cond_2yo
    if race_type_code in (12, 13) and cond_3yo and cond_3yo > 0:
        return cond_3yo
    if race_type_code == 14:
        if cond_4yo and cond_4yo > 0:
            return cond_4yo
        if cond_5up and cond_5up > 0:
            return cond_5up
    if cond_min_age and cond_min_age > 0:
        return cond_min_age
    for value in (cond_5up, cond_4yo, cond_3yo, cond_2yo):
        if value and value > 0:
            return value
    return 0


@dataclass
class RaceRecord:
    """RAレコード: レース詳細情報"""

    race_id: int  # 計算で生成
    race_date: date | None
    track_code: int  # 競馬場コード (01-10)
    race_no: int  # レース番号 (1-12)
    surface: int  # 1:芝, 2:ダート, 3:障害
    distance_m: int  # 距離 (m)
    going: int | None  # 馬場状態
    weather: int | None  # 天候
    class_code: int  # クラス
    field_size: int | None  # 頭数
    start_time: time | None  # 発走時刻
    turn_dir: int | None  # 回り (1:右, 2:左, 3:直線)
    course_inout: int | None  # コース区分
    grade_code: int | None  # グレード
    race_type_code: int | None  # 競走種別
    weight_type_code: int | None  # 重量種別
    condition_code_min_age: int | None  # 最若年条件コード

    @classmethod
    def parse(cls, payload: str) -> RaceRecord:
        """RAレコードをパース (JV-Data 4.9.0.1 準拠)"""
        try:
            b_payload = payload.encode("cp932")
        except UnicodeEncodeError:
            b_payload = payload.encode("cp932", errors="replace")

        if len(b_payload) < RA_WEATHER_START + RA_WEATHER_LEN:
            b_payload = b_payload.ljust(RA_WEATHER_START + RA_WEATHER_LEN, b" ")

        # 年月日からrace_dateを構築
        try:
            year = _slice_byte_int(b_payload, RA_YEAR_START, RA_YEAR_LEN)
            monthday = _slice_byte_decode(b_payload, RA_MONTHDAY_START, RA_MONTHDAY_LEN)
            month = int(monthday[:2])
            day = int(monthday[2:4])
            race_date = date(year, month, day)
        except (ValueError, IndexError):
            race_date = None

        track_code = _slice_byte_int(b_payload, RA_TRACK_CODE_START, RA_TRACK_CODE_LEN)
        race_no = _slice_byte_int(b_payload, RA_RACE_NO_START, RA_RACE_NO_LEN)

        # トラックコードからサーフェス判定 (コード表2009参照)
        # 10-22: 芝, 23-29: ダート(含サンド), 51-59: 障害
        # 0: 不明(主に地方競馬・海外)
        track_type_code = _slice_byte_int(b_payload, RA_TRACK_TYPE_START, RA_TRACK_TYPE_LEN)
        surface = 1  # デフォルト: 芝
        if 23 <= track_type_code <= 29:
            surface = 2  # ダート (27,28はサンドだが同等扱い)
        elif 51 <= track_type_code <= 59:
            surface = 3  # 障害

        distance_m = _normalize_distance_m(
            _slice_byte_int(b_payload, RA_DISTANCE_START, RA_DISTANCE_LEN)
        )

        # 馬場状態 (芝 or ダート)
        if surface == 1:
            going = _slice_byte_int(b_payload, RA_TURF_GOING_START, RA_TURF_GOING_LEN)
        else:
            going = _slice_byte_int(b_payload, RA_DIRT_GOING_START, RA_DIRT_GOING_LEN)
        if going <= 0:
            going = None

        weather = _slice_byte_int(b_payload, RA_WEATHER_START, RA_WEATHER_LEN)
        if weather <= 0:
            weather = None

        grade_code = _grade_code_to_int(
            _slice_byte_decode(b_payload, RA_GRADE_CODE_START, RA_GRADE_CODE_LEN)
        )
        race_type_raw = _slice_byte_int(b_payload, RA_RACE_TYPE_CODE_START, RA_RACE_TYPE_CODE_LEN)
        race_type_code = race_type_raw if race_type_raw > 0 else None
        weight_type_raw = _slice_byte_int(
            b_payload, RA_WEIGHT_TYPE_CODE_START, RA_WEIGHT_TYPE_CODE_LEN
        )
        weight_type_code = weight_type_raw if weight_type_raw > 0 else None
        cond_2yo = _slice_byte_int(b_payload, RA_COND_CODE_2YO_START, RA_COND_CODE_2YO_LEN)
        cond_3yo = _slice_byte_int(b_payload, RA_COND_CODE_3YO_START, RA_COND_CODE_3YO_LEN)
        cond_4yo = _slice_byte_int(b_payload, RA_COND_CODE_4YO_START, RA_COND_CODE_4YO_LEN)
        cond_5up = _slice_byte_int(b_payload, RA_COND_CODE_5UP_START, RA_COND_CODE_5UP_LEN)
        condition_min_raw = _slice_byte_int(
            b_payload, RA_COND_CODE_MIN_AGE_START, RA_COND_CODE_MIN_AGE_LEN
        )
        condition_code_min_age = condition_min_raw if condition_min_raw > 0 else None
        class_code = _choose_condition_code(
            race_type_code,
            cond_2yo if cond_2yo > 0 else None,
            cond_3yo if cond_3yo > 0 else None,
            cond_4yo if cond_4yo > 0 else None,
            cond_5up if cond_5up > 0 else None,
            condition_code_min_age,
        )
        turn_dir = None

        # コース区分 (数値変換できればする)
        course_str = _slice_byte_decode(b_payload, RA_COURSE_START, RA_COURSE_LEN)
        course_inout = int(course_str) if course_str.isdigit() else 0

        registered_field_size = _slice_byte_int(b_payload, RA_FIELD_SIZE_START, RA_FIELD_SIZE_LEN)
        starters = _slice_byte_int(b_payload, RA_STARTERS_START, RA_STARTERS_LEN)
        if starters > 0:
            field_size = starters
        elif registered_field_size > 0:
            field_size = registered_field_size
        else:
            field_size = None
        start_time = _slice_byte_time(b_payload, RA_START_TIME_START, RA_START_TIME_LEN)

        # race_id を生成
        if race_date:
            date_int = race_date.year * 10000 + race_date.month * 100 + race_date.day
            race_id = date_int * 10000 + track_code * 100 + race_no
        else:
            race_id = 0

        return cls(
            race_id=race_id,
            race_date=race_date,
            track_code=track_code,
            race_no=race_no,
            surface=surface,
            distance_m=distance_m,
            going=going,
            weather=weather,
            class_code=class_code,
            field_size=field_size,
            start_time=start_time,
            turn_dir=turn_dir,
            course_inout=course_inout,
            grade_code=grade_code,
            race_type_code=race_type_code,
            weight_type_code=weight_type_code,
            condition_code_min_age=condition_code_min_age,
        )


# =============================================================================
# SE: 馬毎レース情報 (Runner/Result)
# =============================================================================
# JV-Data 4.9.0.1 仕様 (555 byte)
# ★重要: 仕様書のバイト位置(1-origin) - 1 を定数とする (0-origin)
# パース時は payload.encode("cp932") してバイト列として扱う

# レースキー: 開催年(4)+開催月日(4)+場コード(2)+回(2)+日目(2)+R番号(2) = 16バイト
# 位置12-27 -> 0-origin: 11-26
# データ区分: 位置3 (1バイト)
SE_DATA_KUBUN_START = 2  # 位置3
SE_DATA_KUBUN_LEN = 1

SE_RACE_KEY_START = 11  # 位置12
SE_RACE_KEY_LEN = 16

# 枠番: 位置28 (1バイト)
SE_GATE_START = 27  # 位置28
SE_GATE_LEN = 1

# 馬番: 位置29 (2バイト)
SE_HORSE_NO_START = 28  # 位置29
SE_HORSE_NO_LEN = 2

# 血統登録番号: 位置31 (10バイト)
SE_HORSE_ID_START = 30  # 位置31
SE_HORSE_ID_LEN = 10

# 馬名: 位置41 (36バイト = 全角18文字)
SE_HORSE_NAME_START = 40  # 位置41
SE_HORSE_NAME_LEN = 36

# 性別コード: 位置79 (1バイト)
SE_SEX_CODE_START = 78  # 位置79
SE_SEX_CODE_LEN = 1

# 調教師コード: 位置86 (5バイト) ★修正: 旧84→86
SE_TRAINER_ID_START = 85  # 位置86
SE_TRAINER_ID_LEN = 5

# 調教師名略称: 位置91 (8バイト = 全角4文字) ★修正: 旧89→91, 長さ24→8
SE_TRAINER_NAME_START = 90  # 位置91
SE_TRAINER_NAME_LEN = 8

# 馬主コード: 位置99 (6バイト)
SE_OWNER_ID_START = 98  # 位置99
SE_OWNER_ID_LEN = 6

# 負担重量: 位置289 (3バイト, 単位0.1kg)
SE_WEIGHT_START = 288  # 位置289
SE_WEIGHT_LEN = 3

# ブリンカー使用区分: 位置295 (1バイト)
SE_BLINKER_START = 294  # 位置295
SE_BLINKER_LEN = 1

# 騎手コード: 位置297 (5バイト) ★修正: 旧246→297
SE_JOCKEY_ID_START = 296  # 位置297
SE_JOCKEY_ID_LEN = 5

# 騎手名略称: 位置307 (8バイト = 全角4文字) ★修正: 旧251→307, 長さ24→8
SE_JOCKEY_NAME_START = 306  # 位置307
SE_JOCKEY_NAME_LEN = 8

# 馬体重: 位置325 (3バイト, kg)
SE_BODY_WEIGHT_START = 324  # 位置325
SE_BODY_WEIGHT_LEN = 3

# 増減差: 位置329 (3バイト)
SE_WEIGHT_DIFF_START = 328  # 位置329
SE_WEIGHT_DIFF_LEN = 3

# 確定着順: 位置335 (2バイト)
SE_FINISH_POS_START = 334  # 位置335
SE_FINISH_POS_LEN = 2

# 走破タイム: 位置339 (4バイト, msss形式)
SE_TIME_START = 338  # 位置339
SE_TIME_LEN = 4

# 着差コード: 位置343 (3バイト)
SE_MARGIN_START = 342  # 位置343
SE_MARGIN_LEN = 3

# コーナー通過順位: 各2バイト
SE_CORNER1_START = 351  # 位置352
SE_CORNER2_START = 353  # 位置354
SE_CORNER3_START = 355  # 位置356
SE_CORNER4_START = 357  # 位置358
SE_CORNER_LEN = 2

# 後3ハロンタイム: 位置391 (3バイト, 0.1秒単位)
SE_FINAL3F_START = 390  # 位置391
SE_FINAL3F_LEN = 3


@dataclass
class RunnerRecord:
    """SEレコード: 出走馬情報"""

    race_id: int
    horse_id: str
    horse_name: str
    horse_no: int
    gate: int
    jockey_id: int | None
    jockey_name: str
    trainer_id: int | None
    trainer_name: str
    carried_weight: float
    body_weight: int | None
    body_weight_diff: int | None
    finish_pos: int | None
    time_sec: float | None
    final3f_sec: float | None
    corner1_pos: int | None
    corner2_pos: int | None
    corner3_pos: int | None
    corner4_pos: int | None
    margin: str | None
    # 新規追加フィールド
    data_kubun: str
    trainer_code_raw: str
    trainer_name_abbr: str
    jockey_code_raw: str
    jockey_name_abbr: str
    sex: int | None = None

    @classmethod
    def parse(cls, payload: str, race_id: int = 0) -> RunnerRecord:
        """SEレコードをパース (バイト列変換版)"""
        # Note: payload is unicode string. Convert to CP932 bytes.
        try:
            b_payload = payload.encode("cp932")
        except UnicodeEncodeError:
            b_payload = payload.encode("cp932", errors="replace")

        # Ensure length for safety (SE record is 555 bytes per spec)
        if len(b_payload) < 555:
            b_payload = b_payload.ljust(555, b" ")

        if race_id == 0:
            race_key = _slice_byte_decode(b_payload, SE_RACE_KEY_START, SE_RACE_KEY_LEN)
            if len(race_key) >= 16:
                try:
                    year = int(race_key[0:4])
                    month = int(race_key[4:6])
                    day = int(race_key[6:8])
                    track = int(race_key[8:10])
                    race_no = int(race_key[14:16])
                    date_int = year * 10000 + month * 100 + day
                    race_id = date_int * 10000 + track * 100 + race_no
                except ValueError:
                    pass

        data_kubun = _slice_byte_decode(b_payload, SE_DATA_KUBUN_START, SE_DATA_KUBUN_LEN)

        horse_id = _slice_byte_decode(b_payload, SE_HORSE_ID_START, SE_HORSE_ID_LEN)
        horse_name = _slice_byte_decode(b_payload, SE_HORSE_NAME_START, SE_HORSE_NAME_LEN)
        horse_no = _slice_byte_int(b_payload, SE_HORSE_NO_START, SE_HORSE_NO_LEN)
        gate = _slice_byte_int(b_payload, SE_GATE_START, SE_GATE_LEN)

        # Jockey
        jockey_code_raw = _slice_byte_decode(b_payload, SE_JOCKEY_ID_START, SE_JOCKEY_ID_LEN)
        jockey_id_int = _slice_byte_int(b_payload, SE_JOCKEY_ID_START, SE_JOCKEY_ID_LEN)
        jockey_id = jockey_id_int if jockey_id_int > 0 else None

        jockey_name_abbr = _slice_byte_decode(b_payload, SE_JOCKEY_NAME_START, SE_JOCKEY_NAME_LEN)
        # 後方互換性のため jockey_name には略称を入れる
        jockey_name = jockey_name_abbr

        # Trainer
        trainer_code_raw = _slice_byte_decode(b_payload, SE_TRAINER_ID_START, SE_TRAINER_ID_LEN)
        trainer_id_int = _slice_byte_int(b_payload, SE_TRAINER_ID_START, SE_TRAINER_ID_LEN)
        trainer_id = trainer_id_int if trainer_id_int > 0 else None

        trainer_name_abbr = _slice_byte_decode(
            b_payload, SE_TRAINER_NAME_START, SE_TRAINER_NAME_LEN
        )
        # 後方互換性のため trainer_name には略称を入れる
        trainer_name = trainer_name_abbr

        weight_raw = _slice_byte_int(b_payload, SE_WEIGHT_START, SE_WEIGHT_LEN)
        carried_weight = weight_raw / 10.0 if weight_raw else 0.0

        body_weight_raw = _slice_byte_int(b_payload, SE_BODY_WEIGHT_START, SE_BODY_WEIGHT_LEN)
        body_weight = body_weight_raw if body_weight_raw > 0 else None
        sex_raw = _slice_byte_int(b_payload, SE_SEX_CODE_START, SE_SEX_CODE_LEN)
        sex = sex_raw if sex_raw in (1, 2, 3) else None

        weight_diff_str = _slice_byte_decode(b_payload, SE_WEIGHT_DIFF_START, SE_WEIGHT_DIFF_LEN)
        try:
            body_weight_diff = int(weight_diff_str) if weight_diff_str else None
        except ValueError:
            body_weight_diff = None

        finish_pos_raw = _slice_byte_int(b_payload, SE_FINISH_POS_START, SE_FINISH_POS_LEN)
        finish_pos = finish_pos_raw if finish_pos_raw > 0 else None

        time_str = _slice_byte_decode(b_payload, SE_TIME_START, SE_TIME_LEN)
        try:
            time_sec = _time_to_seconds(time_str) if time_str else None
        except Exception:
            time_sec = None

        final3f_raw = _slice_byte_int(b_payload, SE_FINAL3F_START, SE_FINAL3F_LEN)
        final3f_sec = final3f_raw / 10.0 if final3f_raw else None

        corner1_pos = _slice_byte_int(b_payload, SE_CORNER1_START, SE_CORNER_LEN) or None
        corner2_pos = _slice_byte_int(b_payload, SE_CORNER2_START, SE_CORNER_LEN) or None
        corner3_pos = _slice_byte_int(b_payload, SE_CORNER3_START, SE_CORNER_LEN) or None
        corner4_pos = _slice_byte_int(b_payload, SE_CORNER4_START, SE_CORNER_LEN) or None

        margin = _slice_byte_decode(b_payload, SE_MARGIN_START, SE_MARGIN_LEN) or None

        return cls(
            race_id=race_id,
            horse_id=horse_id,
            horse_name=horse_name,
            horse_no=horse_no,
            gate=gate,
            jockey_id=jockey_id,
            jockey_name=jockey_name,
            trainer_id=trainer_id,
            trainer_name=trainer_name,
            carried_weight=carried_weight,
            body_weight=body_weight,
            body_weight_diff=body_weight_diff,
            finish_pos=finish_pos,
            time_sec=time_sec,
            final3f_sec=final3f_sec,
            corner1_pos=corner1_pos,
            corner2_pos=corner2_pos,
            corner3_pos=corner3_pos,
            corner4_pos=corner4_pos,
            margin=margin,
            # 新規フィールド
            data_kubun=data_kubun,
            trainer_code_raw=trainer_code_raw,
            trainer_name_abbr=trainer_name_abbr,
            jockey_code_raw=jockey_code_raw,
            jockey_name_abbr=jockey_name_abbr,
            sex=sex,
        )


# =============================================================================
# HR: 払戻金 (Payout)
# =============================================================================
# Key: 11-27
HR_RACE_KEY_START = 11
HR_RACE_KEY_LEN = 16

# Offsets (0-indexed based on Spec)
# Win ( 単勝 ): 53-1 = 52. Len 10 (2+8). Count 3.
HR_WIN_START = 52
HR_WIN_BLOCK_LEN = 10
HR_WIN_COUNT = 3

# Place ( 複勝 ): 83-1 = 82. Len 10 (2+8). Count 5.
HR_PLACE_START = 82
HR_PLACE_BLOCK_LEN = 10
HR_PLACE_COUNT = 5

# Bracket ( 枠連 ): 133-1 = 132. Len 10 (2+8). Count 3.
HR_BRACKET_START = 132
HR_BRACKET_BLOCK_LEN = 10
HR_BRACKET_COUNT = 3

# Quinella ( 馬連 ): 163-1 = 162. Len 12 (4+8). Count 3.
HR_QUINELLA_START = 162
HR_QUINELLA_BLOCK_LEN = 12
HR_QUINELLA_COUNT = 3

# Wide ( ワイド ): 199-1 = 198. Len 12 (4+8). Count 7.
HR_WIDE_START = 198
HR_WIDE_BLOCK_LEN = 12
HR_WIDE_COUNT = 7

# Exacta ( 馬単 ): 283-1 = 282. Len 12 (4+8). Count 6.
HR_EXACTA_START = 282
HR_EXACTA_BLOCK_LEN = 12
HR_EXACTA_COUNT = 6

# Trio ( 3連複 ): 355-1 = 354. Len 14 (6+8). Count 3.
HR_TRIO_START = 354
HR_TRIO_BLOCK_LEN = 14
HR_TRIO_COUNT = 3

# Trifecta ( 3連単 ): 397-1 = 396. Len 14 (6+8). Count 6.
HR_TRIFECTA_START = 396
HR_TRIFECTA_BLOCK_LEN = 14
HR_TRIFECTA_COUNT = 6


# =============================================================================
# KS: 騎手マスタ (Jockey Master)
# =============================================================================
# 騎手コード: 12-16 (5 bytes) -> Index 11-16
KS_JOCKEY_ID_START = 11
KS_JOCKEY_ID_LEN = 5

# 騎手名: 42-75 (34 bytes) -> Index 41-75
KS_JOCKEY_NAME_START = 41
KS_JOCKEY_NAME_LEN = 34


@dataclass
class JockeyRecord:
    """KSレコード: 騎手マスタ"""

    jockey_id: int
    jockey_name: str

    @classmethod
    def parse(cls, payload: str) -> JockeyRecord:
        """KSレコードをパース"""
        try:
            b_payload = payload.encode("cp932")
        except UnicodeEncodeError:
            b_payload = payload.encode("cp932", errors="replace")

        # レコード長チェック (最低限ヘッダ情報があればOKとする)
        if len(b_payload) < 80:
            b_payload = b_payload.ljust(80, b" ")

        jockey_id = _slice_byte_int(b_payload, KS_JOCKEY_ID_START, KS_JOCKEY_ID_LEN)
        jockey_name = _slice_byte_decode(b_payload, KS_JOCKEY_NAME_START, KS_JOCKEY_NAME_LEN)

        return cls(
            jockey_id=jockey_id,
            jockey_name=jockey_name,
        )


# =============================================================================
# CH: 調教師マスタ (Trainer Master)
# =============================================================================
# 調教師コード: 12-16 (5 bytes) -> Index 11-16
CH_TRAINER_ID_START = 11
CH_TRAINER_ID_LEN = 5

# 調教師名: 42-75 (34 bytes) -> Index 41-75
CH_TRAINER_NAME_START = 41
CH_TRAINER_NAME_LEN = 34


@dataclass
class TrainerRecord:
    """CHレコード: 調教師マスタ"""

    trainer_id: int
    trainer_name: str

    @classmethod
    def parse(cls, payload: str) -> TrainerRecord:
        """CHレコードをパース"""
        try:
            b_payload = payload.encode("cp932")
        except UnicodeEncodeError:
            b_payload = payload.encode("cp932", errors="replace")

        # レコード長チェック (最低限ヘッダ情報があればOKとする)
        if len(b_payload) < 80:
            b_payload = b_payload.ljust(80, b" ")

        trainer_id = _slice_byte_int(b_payload, CH_TRAINER_ID_START, CH_TRAINER_ID_LEN)
        trainer_name = _slice_byte_decode(b_payload, CH_TRAINER_NAME_START, CH_TRAINER_NAME_LEN)

        return cls(
            trainer_id=trainer_id,
            trainer_name=trainer_name,
        )


@dataclass
class PayoutRecord:
    """HRレコード: 払戻情報"""

    race_id: int
    bet_type: int
    selection: str
    payout_yen: int
    popularity: int | None

    @classmethod
    def parse(cls, payload: str, race_id: int = 0) -> list[PayoutRecord]:
        """HRレコードをパースして全ての払戻情報をリストで返す"""
        # Note: Input payload is unicode. Convert to CP932 bytes for strict slicing.
        try:
            b_payload = payload.encode("cp932")
        except UnicodeEncodeError:
            b_payload = payload.encode("cp932", errors="replace")

        # Ensure sufficient length (Spec 964 bytes)
        if len(b_payload) < 964:
            b_payload = b_payload.ljust(964, b" ")

        results = []

        if race_id == 0:
            race_key = _slice_byte_decode(b_payload, HR_RACE_KEY_START, HR_RACE_KEY_LEN)
            if len(race_key) >= 16:
                try:
                    year = int(race_key[0:4])
                    month = int(race_key[4:6])
                    day = int(race_key[6:8])
                    track = int(race_key[8:10])
                    race_no = int(race_key[14:16])
                    date_int = year * 10000 + month * 100 + day
                    race_id = date_int * 10000 + track * 100 + race_no
                except ValueError:
                    pass

        # Helper to extract blocks
        def extract(start, count, block_len, key_len, bet_type):
            for i in range(count):
                offset = start + i * block_len
                # key part (HorseNo or Kumiban)
                key_part = _slice_byte_decode(b_payload, offset, key_len)
                # yen part (always 8 bytes at end of block)
                yen_offset = offset + key_len
                yen_val = _slice_byte_int(b_payload, yen_offset, 8)

                if not key_part or not key_part.strip():
                    continue
                if yen_val is None or yen_val == 0:
                    continue

                # Special Check for "0" or "00" which means empty in repeated fields
                if int(key_part) == 0:
                    continue

                results.append(
                    cls(
                        race_id=race_id,
                        bet_type=bet_type,
                        selection=key_part.replace(
                            " ", "0"
                        ),  # Fill spaces with 0 for standardization
                        payout_yen=yen_val,
                        popularity=None,  # Popularity is not in HR record (only in O1/O2)
                    )
                )

        # 1: Win (単勝)
        extract(HR_WIN_START, HR_WIN_COUNT, HR_WIN_BLOCK_LEN, 2, 1)
        # 2: Place (複勝)
        extract(HR_PLACE_START, HR_PLACE_COUNT, HR_PLACE_BLOCK_LEN, 2, 2)
        # 3: Bracket (枠連)
        extract(HR_BRACKET_START, HR_BRACKET_COUNT, HR_BRACKET_BLOCK_LEN, 2, 3)
        # 4: Quinella (馬連)
        extract(HR_QUINELLA_START, HR_QUINELLA_COUNT, HR_QUINELLA_BLOCK_LEN, 4, 4)
        # 5: Wide (ワイド)
        extract(HR_WIDE_START, HR_WIDE_COUNT, HR_WIDE_BLOCK_LEN, 4, 5)
        # 6: Exacta (馬単)
        extract(HR_EXACTA_START, HR_EXACTA_COUNT, HR_EXACTA_BLOCK_LEN, 4, 6)
        # 7: Trio (3連複)
        extract(HR_TRIO_START, HR_TRIO_COUNT, HR_TRIO_BLOCK_LEN, 6, 7)
        # 8: Trifecta (3連単)
        extract(HR_TRIFECTA_START, HR_TRIFECTA_COUNT, HR_TRIFECTA_BLOCK_LEN, 6, 8)

        return results


# =============================================================================
# O1: 単勝オッズ (Win Odds), 複勝 (Place), 枠連 (Bracket)
# =============================================================================
# Key: 11-27
O1_RACE_KEY_START = 11
O1_RACE_KEY_LEN = 16

# データ区分: 位置3 (1バイト) - 時系列対応
O1_DATA_KBN_START = 2
O1_DATA_KBN_LEN = 1

# 発表月日時分: 位置28-35 (8バイト) - 時系列キー
O1_ANNOUNCE_MMDDHHMI_START = 27
O1_ANNOUNCE_MMDDHHMI_LEN = 8

# 登録頭数: 位置36-37 (2バイト)
O1_FIELD_SIZE_START = 35
O1_FIELD_SIZE_LEN = 2

# 単勝票数合計: 位置38-43 (6バイト) - 百円単位
O1_WIN_POOL_START = 37
O1_WIN_POOL_LEN = 11

# Win Odds ( 単勝 ): 44-1 = 43. Len 8 (2+4+2). Count 28.
O1_WIN_START = 43
O1_WIN_BLOCK_LEN = 8
O1_WIN_COUNT = 28

# Place Odds ( 複勝 ): 268-1 = 267. Len 12 (2+4+4+2). Count 28.
O1_PLACE_START = 267
O1_PLACE_BLOCK_LEN = 12
O1_PLACE_COUNT = 28

# Bracket Odds ( 枠連 ): 604-1 = 603. Len 9 (2+5+2). Count 36.
O1_BRACKET_START = 603
O1_BRACKET_BLOCK_LEN = 9
O1_BRACKET_COUNT = 36


O3_RACE_KEY_START = 11
O3_RACE_KEY_LEN = 16
O3_DATA_KBN_START = 2
O3_DATA_KBN_LEN = 1
O3_DATA_CREATE_YMD_START = 3
O3_DATA_CREATE_YMD_LEN = 8
O3_ANNOUNCE_MMDDHHMI_START = 27
O3_ANNOUNCE_MMDDHHMI_LEN = 8
O3_STARTERS_START = 37
O3_STARTERS_LEN = 2
O3_SALE_FLAG_WIDE_START = 39
O3_SALE_FLAG_WIDE_LEN = 1
O3_WIDE_START = 40
O3_WIDE_BLOCK_LEN = 17
O3_WIDE_COUNT = 153
O3_WIDE_POOL_START = 2641
O3_WIDE_POOL_LEN = 11


@dataclass
class O3HeaderRecord:
    """O3レコードヘッダー（ワイド）"""

    race_id: int
    data_kbn: int
    announce_mmddhhmi: str
    data_create_ymd: str
    wide_pool_total_100yen: int | None
    starters: int | None
    sale_flag_wide: int | None

    @classmethod
    def parse(cls, payload: str, race_id: int = 0) -> O3HeaderRecord:
        try:
            b_payload = payload.encode("cp932")
        except UnicodeEncodeError:
            b_payload = payload.encode("cp932", errors="replace")

        if len(b_payload) < 2654:
            b_payload = b_payload.ljust(2654, b" ")

        data_kbn = _slice_byte_int(b_payload, O3_DATA_KBN_START, O3_DATA_KBN_LEN)
        data_create_ymd = _slice_byte_decode(
            b_payload, O3_DATA_CREATE_YMD_START, O3_DATA_CREATE_YMD_LEN
        )
        if not data_create_ymd:
            data_create_ymd = "00000000"

        announce_mmddhhmi = _slice_byte_decode(
            b_payload, O3_ANNOUNCE_MMDDHHMI_START, O3_ANNOUNCE_MMDDHHMI_LEN
        )
        if not announce_mmddhhmi:
            announce_mmddhhmi = "00000000"

        starters_raw = _slice_byte_int(b_payload, O3_STARTERS_START, O3_STARTERS_LEN)
        starters = starters_raw if starters_raw > 0 else None

        sale_flag_raw = _slice_byte_int(b_payload, O3_SALE_FLAG_WIDE_START, O3_SALE_FLAG_WIDE_LEN)
        sale_flag_wide = sale_flag_raw if sale_flag_raw >= 0 else None

        wide_pool_total_raw = _slice_byte_int(b_payload, O3_WIDE_POOL_START, O3_WIDE_POOL_LEN)
        wide_pool_total_100yen = wide_pool_total_raw if wide_pool_total_raw > 0 else None

        if race_id == 0:
            race_key = _slice_byte_decode(b_payload, O3_RACE_KEY_START, O3_RACE_KEY_LEN)
            if len(race_key) >= 16:
                try:
                    year = int(race_key[0:4])
                    month = int(race_key[4:6])
                    day = int(race_key[6:8])
                    track = int(race_key[8:10])
                    race_no = int(race_key[14:16])
                    date_int = year * 10000 + month * 100 + day
                    race_id = date_int * 10000 + track * 100 + race_no
                except ValueError:
                    pass

        return cls(
            race_id=race_id,
            data_kbn=data_kbn,
            announce_mmddhhmi=announce_mmddhhmi,
            data_create_ymd=data_create_ymd,
            wide_pool_total_100yen=wide_pool_total_100yen,
            starters=starters,
            sale_flag_wide=sale_flag_wide,
        )


@dataclass
class O3WideRecord:
    """O3レコード詳細（ワイド組番別）"""

    race_id: int
    data_kbn: int
    announce_mmddhhmi: str
    data_create_ymd: str
    kumiban: str
    min_odds_x10: int | None
    max_odds_x10: int | None
    popularity: int | None
    wide_pool_total_100yen: int | None
    starters: int | None
    sale_flag_wide: int | None

    @classmethod
    def parse(cls, payload: str, race_id: int = 0) -> list[O3WideRecord]:
        try:
            b_payload = payload.encode("cp932")
        except UnicodeEncodeError:
            b_payload = payload.encode("cp932", errors="replace")

        if len(b_payload) < 2654:
            b_payload = b_payload.ljust(2654, b" ")

        header = O3HeaderRecord.parse(payload, race_id=race_id)
        rows: list[O3WideRecord] = []

        for i in range(O3_WIDE_COUNT):
            offset = O3_WIDE_START + i * O3_WIDE_BLOCK_LEN
            kumiban_raw = _slice_byte_decode(b_payload, offset, 4)
            kumiban = kumiban_raw.replace(" ", "0")
            if len(kumiban) != 4 or not kumiban.isdigit() or kumiban == "0000":
                continue

            min_odds_x10 = _slice_byte_maskable_int(b_payload, offset + 4, 5)
            max_odds_x10 = _slice_byte_maskable_int(b_payload, offset + 9, 5)
            popularity_raw = _slice_byte_maskable_int(b_payload, offset + 14, 3)
            popularity = popularity_raw if popularity_raw and popularity_raw > 0 else None

            rows.append(
                cls(
                    race_id=header.race_id,
                    data_kbn=header.data_kbn,
                    announce_mmddhhmi=header.announce_mmddhhmi,
                    data_create_ymd=header.data_create_ymd,
                    kumiban=kumiban,
                    min_odds_x10=min_odds_x10,
                    max_odds_x10=max_odds_x10,
                    popularity=popularity,
                    wide_pool_total_100yen=header.wide_pool_total_100yen,
                    starters=header.starters,
                    sale_flag_wide=header.sale_flag_wide,
                )
            )

        return rows


@dataclass
class OddsTimeSeriesRecord:
    """O1レコード: 時系列オッズ対応版"""

    race_id: int
    data_kbn: int  # 1=中間, 2=前日最終, 3=最終, 4=確定
    announce_mmddhhmi: str  # 発表月日時分 (MMDDhhmm)
    horse_no: int
    win_odds_x10: int | None  # オッズ×10 (12.3倍 → 123)
    win_popularity: int | None
    # ヘッダー情報
    win_pool_total_100yen: int | None

    @classmethod
    def parse(cls, payload: str, race_id: int = 0) -> list[OddsTimeSeriesRecord]:
        """O1レコードをパース（時系列対応版）"""
        try:
            b_payload = payload.encode("cp932")
        except UnicodeEncodeError:
            b_payload = payload.encode("cp932", errors="replace")

        if len(b_payload) < 962:
            b_payload = b_payload.ljust(962, b" ")

        results = []

        # データ区分
        data_kbn = _slice_byte_int(b_payload, O1_DATA_KBN_START, O1_DATA_KBN_LEN)

        # 発表月日時分
        announce_mmddhhmi = _slice_byte_decode(
            b_payload, O1_ANNOUNCE_MMDDHHMI_START, O1_ANNOUNCE_MMDDHHMI_LEN
        )
        if not announce_mmddhhmi:
            announce_mmddhhmi = "00000000"

        # 票数合計
        win_pool_total = _slice_byte_int(b_payload, O1_WIN_POOL_START, O1_WIN_POOL_LEN)

        if race_id == 0:
            race_key = _slice_byte_decode(b_payload, O1_RACE_KEY_START, O1_RACE_KEY_LEN)
            if len(race_key) >= 16:
                try:
                    year = int(race_key[0:4])
                    month = int(race_key[4:6])
                    day = int(race_key[6:8])
                    track = int(race_key[8:10])
                    race_no = int(race_key[14:16])
                    date_int = year * 10000 + month * 100 + day
                    race_id = date_int * 10000 + track * 100 + race_no
                except ValueError:
                    pass

        # Win Odds (28 horses)
        for i in range(O1_WIN_COUNT):
            offset = O1_WIN_START + i * O1_WIN_BLOCK_LEN
            h_no = _slice_byte_int(b_payload, offset, 2)
            odds_raw = _slice_byte_maskable_int(b_payload, offset + 2, 4)
            pop = _slice_byte_maskable_int(b_payload, offset + 6, 2)

            if h_no and h_no > 0:
                results.append(
                    cls(
                        race_id=race_id,
                        data_kbn=data_kbn,
                        announce_mmddhhmi=announce_mmddhhmi,
                        horse_no=h_no,
                        win_odds_x10=odds_raw,
                        win_popularity=pop if pop and pop > 0 else None,
                        win_pool_total_100yen=win_pool_total if win_pool_total > 0 else None,
                    )
                )

        return results


@dataclass
class OddsRecord:
    """O1レコード: オッズ (Win, Place, Bracket)"""

    # Note: Currently focused on Win Odds for MVP

    race_id: int
    bet_type: int  # 1:Win, 2:Place, 3:Bracket
    horse_no: int | str  # Win/Place=int(HorseNo), Bracket=str(Kumiban)
    odds_1: float | None  # Win:Odds, Place:Min, Bracket:Odds
    odds_2: float | None  # Place:Max
    popularity: int | None

    @classmethod
    def parse(cls, payload: str, race_id: int = 0) -> list[OddsRecord]:
        """O1レコードをパース"""
        # Note: Input payload is unicode. Convert to CP932 bytes.
        try:
            b_payload = payload.encode("cp932")
        except UnicodeEncodeError:
            b_payload = payload.encode("cp932", errors="replace")

        # Ensure sufficient length (Spec 962 bytes)
        if len(b_payload) < 962:
            b_payload = b_payload.ljust(962, b" ")

        results = []

        if race_id == 0:
            race_key = _slice_byte_decode(b_payload, O1_RACE_KEY_START, O1_RACE_KEY_LEN)
            if len(race_key) >= 16:
                try:
                    year = int(race_key[0:4])
                    month = int(race_key[4:6])
                    day = int(race_key[6:8])
                    track = int(race_key[8:10])
                    race_no = int(race_key[14:16])
                    date_int = year * 10000 + month * 100 + day
                    race_id = date_int * 10000 + track * 100 + race_no
                except ValueError:
                    pass

        # 1. Win Odds (28 horses)
        for i in range(O1_WIN_COUNT):
            offset = O1_WIN_START + i * O1_WIN_BLOCK_LEN
            # HorseNo (2)
            h_no = _slice_byte_int(b_payload, offset, 2)
            # Odds (4) 99.9
            odds_raw = _slice_byte_int(b_payload, offset + 2, 4)
            # Pop (2)
            pop = _slice_byte_int(b_payload, offset + 6, 2)

            if h_no and h_no > 0 and odds_raw is not None:
                results.append(
                    cls(
                        race_id=race_id,
                        bet_type=1,
                        horse_no=h_no,
                        odds_1=odds_raw / 10.0,
                        odds_2=None,
                        popularity=pop,
                    )
                )

        # 2. Place Odds (28 horses)
        for i in range(O1_PLACE_COUNT):
            offset = O1_PLACE_START + i * O1_PLACE_BLOCK_LEN
            h_no = _slice_byte_int(b_payload, offset, 2)
            min_odds = _slice_byte_int(b_payload, offset + 2, 4)
            max_odds = _slice_byte_int(b_payload, offset + 6, 4)
            pop = _slice_byte_int(b_payload, offset + 10, 2)

            if h_no and h_no > 0 and min_odds is not None:
                results.append(
                    cls(
                        race_id=race_id,
                        bet_type=2,
                        horse_no=h_no,
                        odds_1=min_odds / 10.0,
                        odds_2=max_odds / 10.0 if max_odds else None,
                        popularity=pop,
                    )
                )

        # 3. Bracket Odds (36 combinations)
        for i in range(O1_BRACKET_COUNT):
            # 1-1 to 8-8 ordered.
            # But spec says "Kumiban" is at offset.
            offset = O1_BRACKET_START + i * O1_BRACKET_BLOCK_LEN
            k_no_str = _slice_byte_decode(b_payload, offset, 2)
            odds_raw = _slice_byte_int(b_payload, offset + 2, 5)  # 5 digits! 999.9
            pop = _slice_byte_int(b_payload, offset + 7, 2)

            if k_no_str and odds_raw is not None:
                results.append(
                    cls(
                        race_id=race_id,
                        bet_type=3,
                        horse_no=k_no_str.replace(" ", "0"),  # String "12"
                        odds_1=odds_raw / 10.0,
                        odds_2=None,
                        popularity=pop,
                    )
                )

        return results


# =============================================================================
# JG: 競走馬除外情報 (Horse Exclusion)
# =============================================================================
# JGレコードの構造 (サンプルデータからの分析)
# レコード長: 80バイト (CR/LF含む)
#
# サンプル:
# JG12026010320260104060101012023100239ファストワーカー　　　　　　　　　　00110
#
# 位置は0-indexed

JG_DATA_DIV_START = 2  # データ区分 (1文字)
JG_DATA_DIV_LEN = 1
JG_CREATED_DATE_START = 3  # データ作成年月日 (8文字)
JG_CREATED_DATE_LEN = 8
JG_RACE_DATE_START = 11  # 開催年月日 (8文字)
JG_RACE_DATE_LEN = 8
JG_TRACK_CODE_START = 19  # 競馬場コード (2文字)
JG_TRACK_CODE_LEN = 2
JG_KAI_START = 21  # 開催回 (2文字)
JG_KAI_LEN = 2
JG_NICHI_START = 23  # 開催日目 (2文字)
JG_NICHI_LEN = 2
JG_RACE_NO_START = 25  # レース番号 (2文字)
JG_RACE_NO_LEN = 2
JG_HORSE_ID_START = 27  # 血統登録番号 (10文字)
JG_HORSE_ID_LEN = 10
JG_HORSE_NAME_START = 37  # 馬名 (20バイト = 全角10文字)
JG_HORSE_NAME_LEN = 18  # 全角9文字 (UTF-8ではカタカナ + 全角スペース)
JG_FLAGS_START = 55  # フラグ情報 (5文字)
JG_FLAGS_LEN = 5


@dataclass
class HorseExclusionRecord:
    """JGレコード: 競走馬除外情報"""

    horse_id: str  # 血統登録番号 (10桁)
    horse_name: str  # 馬名
    data_div: int  # データ区分 (1=新規, 2=更新, 0=削除)
    race_date: date | None  # 開催年月日
    track_code: int  # 競馬場コード
    race_no: int  # レース番号
    flags: str  # フラグ情報

    @classmethod
    def parse(cls, payload: str) -> HorseExclusionRecord:
        """JGレコードをパース"""
        data_div = _slice_int(payload, JG_DATA_DIV_START, JG_DATA_DIV_LEN)
        race_date = _slice_date(payload, JG_RACE_DATE_START)
        track_code = _slice_int(payload, JG_TRACK_CODE_START, JG_TRACK_CODE_LEN)
        race_no = _slice_int(payload, JG_RACE_NO_START, JG_RACE_NO_LEN)
        horse_id = _slice_decode(payload, JG_HORSE_ID_START, JG_HORSE_ID_LEN)
        horse_name = _slice_decode(payload, JG_HORSE_NAME_START, JG_HORSE_NAME_LEN)
        flags = _slice_decode(payload, JG_FLAGS_START, JG_FLAGS_LEN)

        return cls(
            horse_id=horse_id,
            horse_name=horse_name,
            data_div=data_div,
            race_date=race_date,
            track_code=track_code,
            race_no=race_no,
            flags=flags,
        )


# =============================================================================
# UM: 競走馬マスタ (Horse Master)
# =============================================================================
# UMレコードの構造 (JV-Data仕様書 4.9.0.1 準拠)
# レコード長: 3020 byte
#
# 位置は0-indexed (仕様書は1-indexed)
# データ区分: 3-1 = 2 (1byte)
# 血統登録番号: 11-1 = 10 (10byte)
# 馬名: 21-1 = 20 (36byte, 全角18文字)

UM_DATA_DIV_START = 2  # データ区分 (1文字)
UM_DATA_DIV_LEN = 1
UM_HORSE_ID_START = 10  # 血統登録番号 (11-1 = 10)
UM_HORSE_ID_LEN = 10
UM_HORSE_NAME_START = 20  # 馬名 (21-1 = 20, 36byte)
UM_HORSE_NAME_LEN = 36


@dataclass
class HorseMasterRecord:
    """UMレコード: 競走馬マスタ"""

    horse_id: str  # 血統登録番号 (10桁)
    horse_name: str  # 馬名
    data_div: int  # データ区分 (1=新規, 2=更新, 0=削除)

    @classmethod
    def parse(cls, payload: str) -> HorseMasterRecord:
        """UMレコードをパース"""
        # Note: payload is unicode string. Convert to CP932 bytes for strict slicing.
        try:
            b_payload = payload.encode("cp932")
        except UnicodeEncodeError:
            b_payload = payload.encode("cp932", errors="replace")

        # Ensure sufficient length
        if len(b_payload) < 100:
            b_payload = b_payload.ljust(100, b" ")

        data_div = _slice_byte_int(b_payload, UM_DATA_DIV_START, UM_DATA_DIV_LEN)
        horse_id = _slice_byte_decode(b_payload, UM_HORSE_ID_START, UM_HORSE_ID_LEN)
        horse_name = _slice_byte_decode(b_payload, UM_HORSE_NAME_START, UM_HORSE_NAME_LEN)

        return cls(
            horse_id=horse_id,
            horse_name=horse_name,
            data_div=data_div,
        )


# =============================================================================
# WH: 馬体重速報 (Horse Weight)
# =============================================================================
WH_DATA_KBN_START = 2
WH_DATA_KBN_LEN = 1
WH_RACE_KEY_START = 11
WH_RACE_KEY_LEN = 16
WH_ANNOUNCE_MMDDHHMI_START = 27
WH_ANNOUNCE_MMDDHHMI_LEN = 8
WH_DETAIL_START = 35
# 仕様書(101.馬体重/WH): 馬番(2) + 馬名(36) + 馬体重(3) + 増減符号(1) + 増減差(3) = 45 byte
WH_DETAIL_BLOCK_LEN = 45
WH_DETAIL_COUNT = 18


@dataclass
class WHRecord:
    """WHレコード: 馬体重速報"""

    race_id: int
    data_kbn: int
    announce_mmddhhmi: str
    horse_no: int
    body_weight_kg: int | None
    diff_sign: str  # '+', '-', ' '
    diff_kg: int | None

    @classmethod
    def parse(cls, payload: str, race_id: int = 0) -> list[WHRecord]:
        """WHレコードをパース"""
        try:
            b_payload = payload.encode("cp932")
        except UnicodeEncodeError:
            b_payload = payload.encode("cp932", errors="replace")

        required_len = WH_DETAIL_START + WH_DETAIL_BLOCK_LEN * WH_DETAIL_COUNT + 2
        if len(b_payload) < required_len:
            b_payload = b_payload.ljust(required_len, b" ")

        results = []

        data_kbn = _slice_byte_int(b_payload, WH_DATA_KBN_START, WH_DATA_KBN_LEN)
        announce_mmddhhmi = _slice_byte_decode(
            b_payload, WH_ANNOUNCE_MMDDHHMI_START, WH_ANNOUNCE_MMDDHHMI_LEN
        )
        if not announce_mmddhhmi:
            announce_mmddhhmi = "00000000"

        if race_id == 0:
            race_key = _slice_byte_decode(b_payload, WH_RACE_KEY_START, WH_RACE_KEY_LEN)
            if len(race_key) >= 16:
                try:
                    year = int(race_key[0:4])
                    month = int(race_key[4:6])
                    day = int(race_key[6:8])
                    track = int(race_key[8:10])
                    race_no = int(race_key[14:16])
                    date_int = year * 10000 + month * 100 + day
                    race_id = date_int * 10000 + track * 100 + race_no
                except ValueError:
                    pass

        for i in range(WH_DETAIL_COUNT):
            offset = WH_DETAIL_START + i * WH_DETAIL_BLOCK_LEN
            h_no = _slice_byte_int(b_payload, offset, 2)
            weight = _slice_byte_int(b_payload, offset + 38, 3)
            sign = _slice_byte_decode(b_payload, offset + 41, 1)
            diff = _slice_byte_int(b_payload, offset + 42, 3)

            if sign not in {"+", "-"}:
                sign = " "

            if h_no and h_no > 0:
                results.append(
                    cls(
                        race_id=race_id,
                        data_kbn=data_kbn,
                        announce_mmddhhmi=announce_mmddhhmi,
                        horse_no=h_no,
                        body_weight_kg=weight if 2 <= weight <= 998 else None,
                        diff_sign=sign,
                        diff_kg=diff if 1 <= diff <= 998 else None,
                    )
                )

        return results


# =============================================================================
# HC: 坂路調教 (SLOP)  ※レコード長60バイト (JV-Data仕様書 4.9.0.1)
# =============================================================================
HC_DATA_KBN = 2  # pos3, len1
HC_MAKE_DATE = 3  # pos4, len8
HC_TRAINING_CENTER = 11  # pos12, len1 (0:美浦 1:栗東) ※他とコード体系が異なる
HC_TRAINING_DATE = 12  # pos13, len8
HC_TRAINING_TIME = 20  # pos21, len4 (hhmm)
HC_HORSE_ID = 24  # pos25, len10
HC_TOTAL_4F = 34  # pos35, len4 (999.9秒)
HC_LAP_4F = 38  # pos39, len3 (99.9秒)
HC_TOTAL_3F = 41  # pos42, len4
HC_LAP_3F = 45  # pos46, len3
HC_TOTAL_2F = 48  # pos49, len4
HC_LAP_2F = 52  # pos53, len3
HC_LAP_1F = 55  # pos56, len3

# メモ: ラップタイム計算
# 4F合計 - 3F合計 = ラップ4F (800m-600m)
# 3F合計 - 2F合計 = ラップ3F (600m-400m)
# 2F合計 - 1F(Lap) = ラップ2F (400m-200m) ※仕様書の項目名注意
# 1F(Lap) = ラップ1F (200m-0m)


def _parse_time_tenth(b: bytes, start: int, length: int = 3) -> float | None:
    """0.1秒単位の数値文字列を秒に変換 (例: 520→52.0, 0635→63.5)"""
    # 60バイト固定長はもともとASCIIしか含まないはずだが、
    # 念のためバイト列で処理
    raw = b[start : start + length]
    digits = raw.strip()
    # "000" や "0000" は欠損値または計測不能
    try:
        val = int(digits)
        if val == 0:
            return None
        return val / 10.0
    except (ValueError, TypeError):
        return None


@dataclass
class HCRecord:
    """HCレコード: 坂路調教 (60バイト固定長)"""

    horse_id: str
    training_date: date | None
    data_kbn: int
    training_center: str  # 0:美浦 1:栗東
    training_time: str  # hhmm
    total_4f: float | None
    lap_4f: float | None
    total_3f: float | None
    lap_3f: float | None
    total_2f: float | None
    lap_2f: float | None
    lap_1f: float | None
    payload_raw: str

    @classmethod
    def parse(cls, payload: str) -> HCRecord:
        """HCレコードをパース"""
        # 60バイト固定長はASCII前提だが、一応CP932エンコードしてバイト列で扱う
        try:
            b = payload.encode("cp932")
        except UnicodeEncodeError:
            b = payload.encode("cp932", errors="replace")

        # 長さが足りない場合はスペース埋め
        if len(b) < 60:
            b = b.ljust(60, b" ")

        data_kbn = _slice_byte_int(b, HC_DATA_KBN, 1)
        training_center = _slice_byte_decode(b, HC_TRAINING_CENTER, 1)
        training_date = _slice_date(payload, HC_TRAINING_DATE)
        training_time = _slice_byte_decode(b, HC_TRAINING_TIME, 4)
        horse_id = _slice_byte_decode(b, HC_HORSE_ID, 10)

        return cls(
            horse_id=horse_id,
            training_date=training_date,
            data_kbn=data_kbn,
            training_center=training_center,
            training_time=training_time,
            total_4f=_parse_time_tenth(b, HC_TOTAL_4F, 4),
            lap_4f=_parse_time_tenth(b, HC_LAP_4F, 3),
            total_3f=_parse_time_tenth(b, HC_TOTAL_3F, 4),
            lap_3f=_parse_time_tenth(b, HC_LAP_3F, 3),
            total_2f=_parse_time_tenth(b, HC_TOTAL_2F, 4),
            lap_2f=_parse_time_tenth(b, HC_LAP_2F, 3),
            lap_1f=_parse_time_tenth(b, HC_LAP_1F, 3),
            payload_raw=payload,
        )


# =============================================================================
# WC: ウッド調教 (WOOD) ※レコード長105バイト (JV-Data仕様書 4.9.0.1)
# =============================================================================
WC_DATA_KBN = 2  # pos3, len1
WC_TRAINING_CENTER = 11  # pos12, len1 (0:美浦 1:栗東)
WC_TRAINING_DATE = 12  # pos13, len8
WC_TRAINING_TIME = 20  # pos21, len4 (hhmm)
WC_HORSE_ID = 24  # pos25, len10
WC_COURSE = 34  # pos35, len1 (0:A 1:B 2:C 3:D 4:E)
WC_DIRECTION = 35  # pos36, len1 (0:右 1:左)
# pos37は予備(1byte)
WC_Unused = 36  # skipping pos 37? 0-based index 36.

# タイムデータ開始位置 (10Fから2Fまでのペア + 1Fラップ)
# 10F: Total(4) Lap(3)
# 9F: Total(4) Lap(3)
# ...
WC_TIME_START = 37  # pos38


@dataclass
class WCRecord:
    """WCレコード: ウッド調教 (105バイト固定長)"""

    horse_id: str
    training_date: date | None
    data_kbn: int
    training_center: str  # 0:美浦 1:栗東
    training_time: str  # hhmm
    course: str  # 0-4
    direction: str  # 0:右 1:左

    # 10F-3FまではOptional (計測されないハロンもあるため)
    total_10f: float | None
    lap_10f: float | None
    total_9f: float | None
    lap_9f: float | None
    total_8f: float | None
    lap_8f: float | None
    total_7f: float | None
    lap_7f: float | None
    total_6f: float | None
    lap_6f: float | None
    total_5f: float | None
    lap_5f: float | None
    total_4f: float | None
    lap_4f: float | None
    total_3f: float | None
    lap_3f: float | None
    total_2f: float | None
    lap_2f: float | None
    lap_1f: float | None
    payload_raw: str

    @classmethod
    def parse(cls, payload: str) -> WCRecord:
        """WCレコードをパース"""
        # 105バイト固定長、ASCII前提
        try:
            b = payload.encode("cp932")
        except UnicodeEncodeError:
            b = payload.encode("cp932", errors="replace")

        if len(b) < 105:
            b = b.ljust(105, b" ")

        data_kbn = _slice_byte_int(b, WC_DATA_KBN, 1)
        training_center = _slice_byte_decode(b, WC_TRAINING_CENTER, 1)
        training_date = _slice_date(payload, WC_TRAINING_DATE)
        training_time = _slice_byte_decode(b, WC_TRAINING_TIME, 4)
        horse_id = _slice_byte_decode(b, WC_HORSE_ID, 10)
        course = _slice_byte_decode(b, WC_COURSE, 1)
        direction = _slice_byte_decode(b, WC_DIRECTION, 1)

        # タイムパース (pos 37から開始)
        # 10F Total(4) Lap(3) -> 7 bytes
        vals = []
        curr = WC_TIME_START
        # 10F down to 2F (9 iterations)
        for _ in range(9):
            # Total
            vals.append(_parse_time_tenth(b, curr, 4))
            curr += 4
            # Lap
            vals.append(_parse_time_tenth(b, curr, 3))
            curr += 3

        # 1F Lap (3 bytes)
        vals.append(_parse_time_tenth(b, curr, 3))

        return cls(
            horse_id=horse_id,
            training_date=training_date,
            data_kbn=data_kbn,
            training_center=training_center,
            training_time=training_time,
            course=course,
            direction=direction,
            total_10f=vals[0],
            lap_10f=vals[1],
            total_9f=vals[2],
            lap_9f=vals[3],
            total_8f=vals[4],
            lap_8f=vals[5],
            total_7f=vals[6],
            lap_7f=vals[7],
            total_6f=vals[8],
            lap_6f=vals[9],
            total_5f=vals[10],
            lap_5f=vals[11],
            total_4f=vals[12],
            lap_4f=vals[13],
            total_3f=vals[14],
            lap_3f=vals[15],
            total_2f=vals[16],
            lap_2f=vals[17],
            lap_1f=vals[18],
            payload_raw=payload,
        )


# =============================================================================
# CK: 出走別着度数 (SNPN) ※レコード長6870バイト (JV-Data仕様書 4.9.0.1)
# =============================================================================
CK_DATA_KBN = 2  # pos3, len1
CK_MAKE_DATE = 3  # pos4, len8
CK_YEAR = 11  # pos12, len4
CK_MONTHDAY = 15  # pos16, len4
CK_COURSE = 19  # pos20, len2
CK_KAI = 21  # pos22, len2
CK_NICHI = 23  # pos24, len2
CK_RACE_NO = 25  # pos26, len2
CK_HORSE_ID = 27  # pos28, len10
CK_HORSE_NAME = 37  # pos38, len36
CK_PRIZE_START = 73  # pos74 - Prize fields
# Stats start
CK_STATS_HORSE_START = 127  # pos128

# Jockey
CK_JOCKEY_CODE = 1384  # pos1385, len5
CK_JOCKEY_STATS_START = 1423  # pos1424

# Trainer
CK_TRAINER_CODE = 3863  # pos3864, len5
CK_TRAINER_STATS_START = 3902  # pos3903

# Owner
CK_OWNER_CODE = 6342  # pos6343, len6
CK_OWNER_STATS_START = 6476  # pos6477

# Breeder
CK_BREEDER_CODE = 6596  # pos6597, len8
CK_BREEDER_STATS_START = 6748  # pos6749


@dataclass
class CKRecord:
    """CKレコード: 出走別着度数 (SNPN 6870バイト)"""

    data_kbn: int
    make_date: date | None

    # CKキー
    kaisai_year: int
    kaisai_md: str
    track_cd: str
    kaisai_kai: int
    kaisai_nichi: int
    race_no: int
    horse_id: str
    horse_name: str

    # Human Keys
    jockey_code: str
    trainer_code: str
    owner_code: str
    breeder_code: str

    # Horse Stats (Cumulative)
    # [1着, 2着, 3着, 4着, 5着, 6着以下]
    counts_total: list[int]  # 総合
    counts_central: list[int]  # 中央

    payload_raw: str

    @classmethod
    def parse(cls, payload: str) -> CKRecord:
        try:
            b = payload.encode("cp932")
        except UnicodeEncodeError:
            b = payload.encode("cp932", errors="replace")

        # 6870バイトだが、短い場合パディング
        if len(b) < 6870:
            b = b.ljust(6870, b" ")

        data_kbn = _slice_byte_int(b, CK_DATA_KBN, 1)
        make_date = _slice_date(payload, CK_MAKE_DATE)

        kaisai_year = _slice_byte_int(b, CK_YEAR, 4)
        kaisai_md = _slice_byte_decode(b, CK_MONTHDAY, 4)
        track_cd = _slice_byte_decode(b, CK_COURSE, 2)
        kaisai_kai = _slice_byte_int(b, CK_KAI, 2)
        kaisai_nichi = _slice_byte_int(b, CK_NICHI, 2)
        race_no = _slice_byte_int(b, CK_RACE_NO, 2)

        horse_id = _slice_byte_decode(b, CK_HORSE_ID, 10)
        horse_name = _slice_byte_decode(b, CK_HORSE_NAME, 36)

        jockey_code = _slice_byte_decode(b, CK_JOCKEY_CODE, 5)
        trainer_code = _slice_byte_decode(b, CK_TRAINER_CODE, 5)
        owner_code = _slice_byte_decode(b, CK_OWNER_CODE, 6)
        breeder_code = _slice_byte_decode(b, CK_BREEDER_CODE, 8)

        # Helper for counts
        def _get_counts(offset, item_len, buckets=6):
            res = []
            curr = offset
            for _ in range(buckets):
                v = _slice_byte_int(b, curr, item_len) or 0
                res.append(v)
                curr += item_len
            return res

        # Horse Stats: 6 buckets * 3 bytes
        counts_total = _get_counts(CK_STATS_HORSE_START, 3)  # Pos 128
        counts_central = _get_counts(CK_STATS_HORSE_START + 18, 3)  # Pos 146

        return cls(
            data_kbn=data_kbn,
            make_date=make_date,
            kaisai_year=kaisai_year,
            kaisai_md=kaisai_md,
            track_cd=track_cd,
            kaisai_kai=kaisai_kai,
            kaisai_nichi=kaisai_nichi,
            horse_id=horse_id,
            horse_name=horse_name,
            race_no=race_no,
            jockey_code=jockey_code,
            trainer_code=trainer_code,
            owner_code=owner_code,
            breeder_code=breeder_code,
            counts_total=counts_total,
            counts_central=counts_central,
            payload_raw=payload,
        )

    def get_full_stats(self) -> dict:
        """Extract all detailed stats for Core/Mart tables"""
        # Re-encode payload to bytes safely
        try:
            b = self.payload_raw.encode("cp932")
        except UnicodeEncodeError:
            b = self.payload_raw.encode("cp932", errors="replace")
        if len(b) < 6870:
            b = b.ljust(6870, b" ")

        def _get_counts(offset, item_len, buckets=6):
            res = []
            curr = offset
            for _ in range(buckets):
                v = _slice_byte_int(b, curr, item_len) or 0
                res.append(v)
                curr += item_len
            return res

        # Horse Stats Offsets (from PDF)
        # Total: 128 (idx 127) -> Already parsed
        # Central: 146 (idx 145) -> Already parsed
        # 20. TurfStr: 164 (idx 163)
        # 21. TurfRight: 182
        # ...
        # Distance: 39 (idx 505)

        # We need specific blocks for Mart 45 cols
        # Turf Right/Left/Straight
        turf_str = _get_counts(163, 3)
        turf_right = _get_counts(181, 3)
        turf_left = _get_counts(199, 3)

        dirt_str = _get_counts(217, 3)
        dirt_right = _get_counts(235, 3)
        dirt_left = _get_counts(253, 3)

        # Condition (27-38) Start 290 (idx 289)
        # 27 芝良
        turf_good = _get_counts(289, 3)
        turf_soft = _get_counts(307, 3)
        turf_heavy = _get_counts(325, 3)
        turf_bad = _get_counts(343, 3)

        dirt_good = _get_counts(361, 3)
        dirt_soft = _get_counts(379, 3)
        dirt_heavy = _get_counts(397, 3)
        dirt_bad = _get_counts(415, 3)

        # Distance (39-56)
        # 仕様書(1-index): 芝 506/524/542/560/578/596/614/632/650
        #                 ダ 668/686/704/722/740/758/776/794/812
        # → 0-index:      芝 505/523/541/559/577/595/613/631/649
        #                ダ 667/685/703/721/739/757/775/793/811
        turf_1200_down = _get_counts(505, 3)
        turf_1201_1400 = _get_counts(523, 3)
        turf_1401_1600 = _get_counts(541, 3)
        turf_1601_1800 = _get_counts(559, 3)
        turf_1801_2000 = _get_counts(577, 3)
        turf_2001_2200 = _get_counts(595, 3)
        turf_2201_2400 = _get_counts(613, 3)
        turf_2401_2800 = _get_counts(631, 3)
        turf_2801_up = _get_counts(649, 3)

        dirt_1200_down = _get_counts(667, 3)
        dirt_1201_1400 = _get_counts(685, 3)
        dirt_1401_1600 = _get_counts(703, 3)
        dirt_1601_1800 = _get_counts(721, 3)
        dirt_1801_2000 = _get_counts(739, 3)
        dirt_2001_2200 = _get_counts(757, 3)
        dirt_2201_2400 = _get_counts(775, 3)
        dirt_2401_2800 = _get_counts(793, 3)
        dirt_2801_up = _get_counts(811, 3)

        # Style (87) Start 1370 (idx 1369). 4 items * 3 bytes = 12 bytes.
        # Nige, Senko, Sashi, Oikomi
        style_nige = _slice_byte_int(b, 1369, 3)
        style_senko = _slice_byte_int(b, 1372, 3)
        style_sashi = _slice_byte_int(b, 1375, 3)
        style_oikomi = _slice_byte_int(b, 1378, 3)

        # Registered Races
        registered_races = _slice_byte_int(b, 1381, 3)

        # Prize Info (Jockey/Trainer/Owner/Breeder)
        # Jockey: 91b-e. Pos 1424 (idx 1423). Year/Cum repeats.
        # Year: offset 1423 -> [Year(4), Flat(10), Obs(10), FlatAdd(10), ObsAdd(10)]
        # We need FlatPrizeTotal = Flat + FlatAdd?
        # Item 91b "平地本賞金"
        # Item 91d "平地付加賞金"

        def extract_human_prize(start_offset):
            # Returns {year: {flat: X, obs: Y}, cum: {flat: X, obs: Y}}
            # Block 1 (Year)
            off = start_offset
            y_flat = (_slice_byte_int(b, off + 4, 10) or 0) + (
                _slice_byte_int(b, off + 24, 10) or 0
            )
            y_obs = (_slice_byte_int(b, off + 14, 10) or 0) + (
                _slice_byte_int(b, off + 34, 10) or 0
            )

            # Block 2 (Cum) - Offset + 1220 bytes ??
            # WAIT! Jockey/Trainer block is 1220 bytes.
            # 91 <騎手本年･累計成績情報> 1424 2 1220 2440
            # So Year block is 1424. Cum block is 1424 + 1220 = 2644.

            off2 = start_offset + 1220
            c_flat = (_slice_byte_int(b, off2 + 4, 10) or 0) + (
                _slice_byte_int(b, off2 + 24, 10) or 0
            )
            c_obs = (_slice_byte_int(b, off2 + 14, 10) or 0) + (
                _slice_byte_int(b, off2 + 34, 10) or 0
            )

            return {"year_flat": y_flat, "year_obs": y_obs, "cum_flat": c_flat, "cum_obs": c_obs}

        # Owner/Breeder Block is smaller (60 bytes)
        # 98 <本年･累計> 6477 2 60 120
        # a, b, c, d. (Year(4), Main(10), Add(10), Counts(36))

        def extract_ob_prize(start_offset):
            # Year
            off = start_offset
            y_total = (_slice_byte_int(b, off + 4, 10) or 0) + (
                _slice_byte_int(b, off + 14, 10) or 0
            )

            # Cum (+60)
            off2 = start_offset + 60
            c_total = (_slice_byte_int(b, off2 + 4, 10) or 0) + (
                _slice_byte_int(b, off2 + 14, 10) or 0
            )

            return {"year": y_total, "cum": c_total}

        j_prize = extract_human_prize(CK_JOCKEY_STATS_START)
        t_prize = extract_human_prize(CK_TRAINER_STATS_START)
        o_prize = extract_ob_prize(CK_OWNER_STATS_START)
        b_prize = extract_ob_prize(CK_BREEDER_STATS_START)

        return {
            # Core specific structures
            "finish_counts": {
                "overall": self.counts_total,
                "central": self.counts_central,
                "turf_good": turf_good,
                "turf_soft": turf_soft,
                "turf_heavy": turf_heavy,
                "turf_bad": turf_bad,
                "dirt_good": dirt_good,
                "dirt_soft": dirt_soft,
                "dirt_heavy": dirt_heavy,
                "dirt_bad": dirt_bad,
                "turf_right": turf_right,
                "turf_left": turf_left,
                "turf_str": turf_str,
                "dirt_right": dirt_right,
                "dirt_left": dirt_left,
                "dirt_str": dirt_str,
                "turf_1200_down": turf_1200_down,
                "turf_1201_1400": turf_1201_1400,
                "turf_1401_1600": turf_1401_1600,
                "turf_1601_1800": turf_1601_1800,
                "turf_1801_2000": turf_1801_2000,
                "turf_2001_2200": turf_2001_2200,
                "turf_2201_2400": turf_2201_2400,
                "turf_2401_2800": turf_2401_2800,
                "turf_2801_up": turf_2801_up,
                "dirt_1200_down": dirt_1200_down,
                "dirt_1201_1400": dirt_1201_1400,
                "dirt_1401_1600": dirt_1401_1600,
                "dirt_1601_1800": dirt_1601_1800,
                "dirt_1801_2000": dirt_1801_2000,
                "dirt_2001_2200": dirt_2001_2200,
                "dirt_2201_2400": dirt_2201_2400,
                "dirt_2401_2800": dirt_2401_2800,
                "dirt_2801_up": dirt_2801_up,
            },
            "style_counts": {
                "nige": style_nige,
                "senko": style_senko,
                "sashi": style_sashi,
                "oikomi": style_oikomi,
            },
            "registered_races": registered_races,
            "entity_prize": {
                "jockey": j_prize,
                "trainer": t_prize,
                "owner": o_prize,
                "breeder": b_prize,
            },
        }


# =============================================================================
# DM: タイム型マイニング (MING)
# =============================================================================
# 仕様: 28.タイム型データマイニング予想 レコード長303バイト
# 位置は1-indexed → 0-indexed に変換して使用
DM_DATA_KBN_START = 2  # 仕様:位置3, 1バイト
DM_DATA_KBN_LEN = 1
DM_DATA_CREATE_YMD_START = 3  # 仕様:位置4, 8バイト (yyyymmdd)
DM_DATA_CREATE_YMD_LEN = 8
DM_RACE_KEY_START = 11  # 仕様:位置12(開催年)〜位置27(レース番号末尾), 16バイト
DM_RACE_KEY_LEN = 16
DM_DATA_CREATE_HM_START = 27  # 仕様:位置28, 4バイト (hhmm)
DM_DATA_CREATE_HM_LEN = 4
DM_MINING_START = 31  # 仕様:位置32(1-indexed), 繰返18回, 各15バイト
DM_MINING_REPEAT = 18
DM_MINING_ITEM_LEN = 15
# 繰返内の相対オフセット (0-indexed)
DM_HORSE_NO_OFF = 0  # 馬番 2バイト
DM_HORSE_NO_LEN = 2
DM_TIME_OFF = 2  # 予想走破タイム 5バイト (9分99秒99)
DM_TIME_LEN = 5
DM_ERR_PLUS_OFF = 7  # 予想誤差+ 4バイト
DM_ERR_PLUS_LEN = 4
DM_ERR_MINUS_OFF = 11  # 予想誤差- 4バイト
DM_ERR_MINUS_LEN = 4


@dataclass
class DMRecord:
    """DMレコード: タイム型マイニング（馬ごと）"""

    race_id: int
    horse_no: int
    data_kbn: int
    data_create_ymd: str
    data_create_hm: str
    dm_time_x10: int | None
    dm_rank: int | None
    payload_raw: str

    @classmethod
    def parse(cls, payload: str, race_id: int = 0) -> list[DMRecord]:
        """DMレコードをパースし、馬ごとにリストで返す（最大18頭）"""
        try:
            b_payload = payload.encode("cp932")
        except UnicodeEncodeError:
            b_payload = payload.encode("cp932", errors="replace")

        if len(b_payload) < 303:
            b_payload = b_payload.ljust(303, b" ")

        data_kbn = _slice_byte_int(b_payload, DM_DATA_KBN_START, DM_DATA_KBN_LEN)
        data_create_ymd = _slice_byte_decode(
            b_payload, DM_DATA_CREATE_YMD_START, DM_DATA_CREATE_YMD_LEN
        )
        data_create_hm = _slice_byte_decode(
            b_payload, DM_DATA_CREATE_HM_START, DM_DATA_CREATE_HM_LEN
        )
        if not data_create_ymd:
            data_create_ymd = "00000000"
        if not data_create_hm:
            data_create_hm = "0000"

        # race_id を構築
        if race_id == 0:
            race_key = _slice_byte_decode(b_payload, DM_RACE_KEY_START, DM_RACE_KEY_LEN)
            if len(race_key) >= 16:
                try:
                    year = int(race_key[0:4])
                    month = int(race_key[4:6])
                    day = int(race_key[6:8])
                    track = int(race_key[8:10])
                    race_no = int(race_key[14:16])
                    date_int = year * 10000 + month * 100 + day
                    race_id = date_int * 10000 + track * 100 + race_no
                except ValueError:
                    pass

        def _dm_time_to_x10(time_str: str) -> int | None:
            s = time_str.strip()
            if len(s) != 5 or not s.isdigit():
                return None
            # "9分99秒99" は欠損扱い
            if s == "99999":
                return None
            minutes = int(s[0])
            seconds = int(s[1:3])
            centisec = int(s[3:5])
            if minutes == 0 and seconds == 0 and centisec == 0:
                return None
            return minutes * 600 + seconds * 10 + (centisec // 10)

        # 繰返し構造から馬ごとのレコードを抽出
        results: list[DMRecord] = []
        for i in range(DM_MINING_REPEAT):
            base = DM_MINING_START + i * DM_MINING_ITEM_LEN
            horse_no = _slice_byte_int(b_payload, base + DM_HORSE_NO_OFF, DM_HORSE_NO_LEN)
            # 馬番が0またはNoneはスキップ（空エントリ）
            if not horse_no:
                continue

            time_str = _slice_byte_decode(b_payload, base + DM_TIME_OFF, DM_TIME_LEN)
            dm_time_x10 = _dm_time_to_x10(time_str)

            results.append(
                cls(
                    race_id=race_id,
                    horse_no=horse_no,
                    data_kbn=data_kbn,
                    data_create_ymd=data_create_ymd,
                    data_create_hm=data_create_hm,
                    dm_time_x10=dm_time_x10,
                    dm_rank=None,
                    payload_raw=payload,
                )
            )

        # 同一レコード内の相対順位（小さいほど上位）
        valid = sorted(
            (r for r in results if r.dm_time_x10 is not None),
            key=lambda r: (r.dm_time_x10, r.horse_no),
        )
        for idx, r in enumerate(valid, start=1):
            r.dm_rank = idx
        return results


# =============================================================================
# TM: 対戦型マイニング (MING)
# =============================================================================
# 仕様: 29.対戦型データマイニング予想 レコード長141バイト
TM_DATA_KBN_START = 2  # 仕様:位置3, 1バイト
TM_DATA_KBN_LEN = 1
TM_DATA_CREATE_YMD_START = 3  # 仕様:位置4, 8バイト (yyyymmdd)
TM_DATA_CREATE_YMD_LEN = 8
TM_RACE_KEY_START = 11  # 仕様:位置12〜27, 16バイト
TM_RACE_KEY_LEN = 16
TM_DATA_CREATE_HM_START = 27  # 仕様:位置28, 4バイト (hhmm)
TM_DATA_CREATE_HM_LEN = 4
TM_MINING_START = 31  # 仕様:位置32(1-indexed), 繰返18回, 各6バイト
TM_MINING_REPEAT = 18
TM_MINING_ITEM_LEN = 6
# 繰返内の相対オフセット (0-indexed)
TM_HORSE_NO_OFF = 0  # 馬番 2バイト
TM_HORSE_NO_LEN = 2
TM_SCORE_OFF = 2  # 予測スコア 4バイト (000.0〜100.0)
TM_SCORE_LEN = 4


@dataclass
class TMRecord:
    """TMレコード: 対戦型マイニング（馬ごと）"""

    race_id: int
    horse_no: int
    data_kbn: int
    data_create_ymd: str
    data_create_hm: str
    tm_score: int | None
    tm_rank: int | None
    payload_raw: str

    @classmethod
    def parse(cls, payload: str, race_id: int = 0) -> list[TMRecord]:
        """TMレコードをパースし、馬ごとにリストで返す（最大18頭）"""
        try:
            b_payload = payload.encode("cp932")
        except UnicodeEncodeError:
            b_payload = payload.encode("cp932", errors="replace")

        if len(b_payload) < 141:
            b_payload = b_payload.ljust(141, b" ")

        data_kbn = _slice_byte_int(b_payload, TM_DATA_KBN_START, TM_DATA_KBN_LEN)
        data_create_ymd = _slice_byte_decode(
            b_payload, TM_DATA_CREATE_YMD_START, TM_DATA_CREATE_YMD_LEN
        )
        data_create_hm = _slice_byte_decode(
            b_payload, TM_DATA_CREATE_HM_START, TM_DATA_CREATE_HM_LEN
        )
        if not data_create_ymd:
            data_create_ymd = "00000000"
        if not data_create_hm:
            data_create_hm = "0000"

        # race_id を構築
        if race_id == 0:
            race_key = _slice_byte_decode(b_payload, TM_RACE_KEY_START, TM_RACE_KEY_LEN)
            if len(race_key) >= 16:
                try:
                    year = int(race_key[0:4])
                    month = int(race_key[4:6])
                    day = int(race_key[6:8])
                    track = int(race_key[8:10])
                    race_no = int(race_key[14:16])
                    date_int = year * 10000 + month * 100 + day
                    race_id = date_int * 10000 + track * 100 + race_no
                except ValueError:
                    pass

        # 繰返し構造から馬ごとのレコードを抽出
        results: list[TMRecord] = []
        for i in range(TM_MINING_REPEAT):
            base = TM_MINING_START + i * TM_MINING_ITEM_LEN
            horse_no = _slice_byte_int(b_payload, base + TM_HORSE_NO_OFF, TM_HORSE_NO_LEN)
            # 馬番が0またはNoneはスキップ（空エントリ）
            if not horse_no:
                continue

            # 予測スコア: 4桁 (例 "0853" → 85.3)
            score_str = _slice_byte_decode(b_payload, base + TM_SCORE_OFF, TM_SCORE_LEN)
            tm_score_x10 = int(score_str) if score_str.isdigit() else None

            results.append(
                cls(
                    race_id=race_id,
                    horse_no=horse_no,
                    data_kbn=data_kbn,
                    data_create_ymd=data_create_ymd,
                    data_create_hm=data_create_hm,
                    tm_score=tm_score_x10,
                    tm_rank=None,
                    payload_raw=payload,
                )
            )

        # 同一レコード内の相対順位（大きいほど上位）
        valid = sorted(
            (r for r in results if r.tm_score is not None),
            key=lambda r: (-r.tm_score, r.horse_no),
        )
        for idx, r in enumerate(valid, start=1):
            r.tm_rank = idx
        return results


# =============================================================================
# イベント変更 (WE/AV/JC/TC/CC)
# =============================================================================
EVENT_DATA_KBN_START = 2
EVENT_DATA_KBN_LEN = 1
EVENT_DATA_CREATE_YMD_START = 3  # pos4, len8
EVENT_DATA_CREATE_YMD_LEN = 8

# Common key starts at pos12 (0-index 11)
EVENT_KEY_START = 11
EVENT_KEY_LEN = 16

# WE does not include race_no. Layout differs from others.
WE_KAISAI_YEAR_START = 11  # pos12, len4
WE_KAISAI_MD_START = 15  # pos16, len4
WE_TRACK_CD_START = 19  # pos20, len2
WE_KAISAI_KAI_START = 21  # pos22, len2
WE_KAISAI_NICHI_START = 23  # pos24, len2
WE_ANNOUNCE_START = 25  # pos26, len8
WE_CHANGE_ID_START = 33  # pos34, len1
WE_WEATHER_NOW_START = 34  # pos35, len1
WE_GOING_TURF_NOW_START = 35  # pos36, len1
WE_GOING_DIRT_NOW_START = 36  # pos37, len1
WE_WEATHER_PREV_START = 37  # pos38, len1
WE_GOING_TURF_PREV_START = 38  # pos39, len1
WE_GOING_DIRT_PREV_START = 39  # pos40, len1

# Other events include race_no.
EVENT_ANNOUNCE_START = 27  # pos28, len8
EVENT_ANNOUNCE_LEN = 8


@dataclass
class EventChangeRecord:
    """WE/AV/JC/TC/CCレコード: 当日変更"""

    record_type: str  # 'WE', 'AV', 'JC', 'TC', 'CC'
    race_id: int
    data_kbn: int
    data_create_ymd: str
    announce_mmddhhmi: str
    payload_parsed: dict
    payload_raw: str

    @classmethod
    def parse(cls, payload: str, race_id: int = 0) -> EventChangeRecord:
        """イベント変更レコードをパース（監査キー+最低限構造化）"""
        rec_type = payload[:2] if len(payload) >= 2 else ""

        try:
            b_payload = payload.encode("cp932")
        except UnicodeEncodeError:
            b_payload = payload.encode("cp932", errors="replace")

        expected_len = {"WE": 42, "AV": 78, "JC": 161, "TC": 45, "CC": 50}.get(rec_type, 50)
        if len(b_payload) < expected_len:
            b_payload = b_payload.ljust(expected_len, b" ")

        data_kbn = _slice_byte_int(b_payload, EVENT_DATA_KBN_START, EVENT_DATA_KBN_LEN)
        data_create_ymd = _slice_byte_decode(
            b_payload, EVENT_DATA_CREATE_YMD_START, EVENT_DATA_CREATE_YMD_LEN
        )
        if not data_create_ymd:
            data_create_ymd = "00000000"

        payload_parsed: dict = {"record_type": rec_type}

        # WE has a different key layout (no race_no)
        if rec_type == "WE":
            kaisai_year = _slice_byte_int(b_payload, WE_KAISAI_YEAR_START, 4)
            kaisai_md = _slice_byte_decode(b_payload, WE_KAISAI_MD_START, 4)
            track_cd_str = _slice_byte_decode(b_payload, WE_TRACK_CD_START, 2)

            try:
                date_int = int(f"{kaisai_year:04d}{kaisai_md}")
                track_int = int(track_cd_str) if track_cd_str.isdigit() else 0
                if race_id == 0:
                    race_id = date_int * 10000 + track_int * 100 + 0  # pseudo race_no=00
            except Exception:
                if race_id == 0:
                    race_id = 0

            announce_mmddhhmi = _slice_byte_decode(b_payload, WE_ANNOUNCE_START, 8) or "00000000"

            payload_parsed.update(
                {
                    "kaisai_year": kaisai_year,
                    "kaisai_md": kaisai_md,
                    "track_cd": track_cd_str,
                    "kaisai_kai": _slice_byte_int(b_payload, WE_KAISAI_KAI_START, 2),
                    "kaisai_nichi": _slice_byte_int(b_payload, WE_KAISAI_NICHI_START, 2),
                    "announce_mmddhhmi": announce_mmddhhmi,
                    "change_id": _slice_byte_int(b_payload, WE_CHANGE_ID_START, 1),
                    "weather_now": _slice_byte_int(b_payload, WE_WEATHER_NOW_START, 1),
                    "going_turf_now": _slice_byte_int(b_payload, WE_GOING_TURF_NOW_START, 1),
                    "going_dirt_now": _slice_byte_int(b_payload, WE_GOING_DIRT_NOW_START, 1),
                    "weather_prev": _slice_byte_int(b_payload, WE_WEATHER_PREV_START, 1),
                    "going_turf_prev": _slice_byte_int(b_payload, WE_GOING_TURF_PREV_START, 1),
                    "going_dirt_prev": _slice_byte_int(b_payload, WE_GOING_DIRT_PREV_START, 1),
                }
            )

            return cls(
                record_type=rec_type,
                race_id=race_id,
                data_kbn=data_kbn,
                data_create_ymd=data_create_ymd,
                announce_mmddhhmi=announce_mmddhhmi,
                payload_parsed=payload_parsed,
                payload_raw=payload,
            )

        if race_id == 0:
            race_key = _slice_byte_decode(b_payload, EVENT_KEY_START, EVENT_KEY_LEN)
            if len(race_key) >= 16:
                try:
                    year = int(race_key[0:4])
                    month = int(race_key[4:6])
                    day = int(race_key[6:8])
                    track = int(race_key[8:10])
                    race_no = int(race_key[14:16])
                    date_int = year * 10000 + month * 100 + day
                    race_id = date_int * 10000 + track * 100 + race_no
                except ValueError:
                    pass

        announce_mmddhhmi = (
            _slice_byte_decode(b_payload, EVENT_ANNOUNCE_START, EVENT_ANNOUNCE_LEN) or "00000000"
        )

        if rec_type == "AV":
            payload_parsed.update(
                {
                    "announce_mmddhhmi": announce_mmddhhmi,
                    "horse_no": _slice_byte_int(b_payload, 35, 2),  # pos36
                    "reason_kbn": _slice_byte_decode(b_payload, 73, 3),  # pos74
                }
            )
        elif rec_type == "JC":
            payload_parsed.update(
                {
                    "announce_mmddhhmi": announce_mmddhhmi,
                    "horse_no": _slice_byte_int(b_payload, 35, 2),  # pos36
                    "carried_weight_x10_after": _slice_byte_int(b_payload, 73, 3),  # pos74
                    "jockey_code_after": _slice_byte_decode(b_payload, 76, 5),  # pos77
                    "carried_weight_x10_before": _slice_byte_int(b_payload, 116, 3),  # pos117
                    "jockey_code_before": _slice_byte_decode(b_payload, 119, 5),  # pos120
                }
            )
        elif rec_type == "TC":
            payload_parsed.update(
                {
                    "announce_mmddhhmi": announce_mmddhhmi,
                    "post_time_after": _slice_byte_decode(b_payload, 35, 4),  # pos36
                    "post_time_before": _slice_byte_decode(b_payload, 39, 4),  # pos40
                }
            )
        elif rec_type == "CC":
            distance_after = _normalize_distance_m(_slice_byte_int(b_payload, 35, 4))
            distance_before = _normalize_distance_m(_slice_byte_int(b_payload, 41, 4))
            payload_parsed.update(
                {
                    "announce_mmddhhmi": announce_mmddhhmi,
                    "distance_m_after": distance_after,  # pos36
                    "track_type_after": _slice_byte_int(b_payload, 39, 2),  # pos40
                    "distance_m_before": distance_before,  # pos42
                    "track_type_before": _slice_byte_int(b_payload, 45, 2),  # pos46
                    "reason_kbn": _slice_byte_int(b_payload, 47, 1),  # pos48
                }
            )

        return cls(
            record_type=rec_type,
            race_id=race_id,
            data_kbn=data_kbn,
            data_create_ymd=data_create_ymd,
            announce_mmddhhmi=announce_mmddhhmi,
            payload_parsed=payload_parsed,
            payload_raw=payload,
        )


# =============================================================================
# パーサー ディスパッチャ
# =============================================================================
PARSERS: dict[str, Callable] = {
    "RA": RaceRecord.parse,
    "SE": RunnerRecord.parse,
    "HR": PayoutRecord.parse,
    "O1": OddsRecord.parse,
    "O3": O3WideRecord.parse,
    "JG": HorseExclusionRecord.parse,
    "UM": HorseMasterRecord.parse,
    # 新規追加
    "WH": WHRecord.parse,
    "HC": HCRecord.parse,
    "WC": WCRecord.parse,
    "CK": CKRecord.parse,
    "DM": DMRecord.parse,
    "TM": TMRecord.parse,
    "WE": EventChangeRecord.parse,
    "AV": EventChangeRecord.parse,
    "JC": EventChangeRecord.parse,
    "TC": EventChangeRecord.parse,
    "CC": EventChangeRecord.parse,
}

# 時系列オッズ用（別途使用）
PARSERS_TIMESERIES: dict[str, Callable] = {
    "O1": OddsTimeSeriesRecord.parse,
    "O3": O3WideRecord.parse,
}


def parse_record(rec_id: str, payload: str, **kwargs):
    """
    レコード種別に応じたパーサーを呼び出す

    Args:
        rec_id: レコード種別 (2文字)
        payload: 固定長文字列
        **kwargs: パーサーに渡す追加引数 (race_id等)

    Returns:
        パース結果のデータクラスインスタンス
    """
    parser = PARSERS.get(rec_id)
    if parser:
        return parser(payload, **kwargs)
    return None
