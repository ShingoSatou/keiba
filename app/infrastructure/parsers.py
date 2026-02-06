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


@dataclass
class RaceRecord:
    """RAレコード: レース詳細情報"""

    race_id: int  # 計算で生成
    race_date: date | None
    track_code: int  # 競馬場コード (01-10)
    race_no: int  # レース番号 (1-12)
    surface: int  # 1:芝, 2:ダート, 3:障害
    distance_m: int  # 距離 (m)
    going: int  # 馬場状態
    weather: int  # 天候
    class_code: int  # クラス
    field_size: int  # 頭数
    start_time: time | None  # 発走時刻
    turn_dir: int | None  # 回り (1:右, 2:左, 3:直線)
    course_inout: int | None  # コース区分

    @classmethod
    def parse(cls, payload: str) -> RaceRecord:
        """RAレコードをパース (JV-Data 4.9.0.1 準拠)"""
        # 年月日からrace_dateを構築
        try:
            year = _slice_int(payload, RA_YEAR_START, RA_YEAR_LEN)
            monthday = _slice_decode(payload, RA_MONTHDAY_START, RA_MONTHDAY_LEN)
            month = int(monthday[:2])
            day = int(monthday[2:4])
            race_date = date(year, month, day)
        except (ValueError, IndexError):
            race_date = None

        track_code = _slice_int(payload, RA_TRACK_CODE_START, RA_TRACK_CODE_LEN)
        race_no = _slice_int(payload, RA_RACE_NO_START, RA_RACE_NO_LEN)

        # トラックコードからサーフェス判定 (コード表2009参照)
        # 10-22: 芝, 23-29: ダート(含サンド), 51-59: 障害
        # 0: 不明(主に地方競馬・海外)
        track_type_code = _slice_int(payload, RA_TRACK_TYPE_START, RA_TRACK_TYPE_LEN)
        surface = 1  # デフォルト: 芝
        if 23 <= track_type_code <= 29:
            surface = 2  # ダート (27,28はサンドだが同等扱い)
        elif 51 <= track_type_code <= 59:
            surface = 3  # 障害

        distance_m = _slice_int(payload, RA_DISTANCE_START, RA_DISTANCE_LEN)

        # 馬場状態 (芝 or ダート)
        if surface == 1:
            going = _slice_int(payload, RA_TURF_GOING_START, RA_TURF_GOING_LEN)
        else:
            going = _slice_int(payload, RA_DIRT_GOING_START, RA_DIRT_GOING_LEN)

        weather = _slice_int(payload, RA_WEATHER_START, RA_WEATHER_LEN)

        # クラスコード・回りは一旦未取得（必要なら定義追加）
        class_code = 0
        turn_dir = None

        # コース区分 (数値変換できればする)
        course_str = _slice_decode(payload, RA_COURSE_START, RA_COURSE_LEN)
        course_inout = int(course_str) if course_str.isdigit() else 0

        field_size = _slice_int(payload, RA_FIELD_SIZE_START, RA_FIELD_SIZE_LEN)
        start_time = _slice_time(payload, RA_START_TIME_START)

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
# パーサー ディスパッチャ
# =============================================================================
PARSERS: dict[str, Callable] = {
    "RA": RaceRecord.parse,
    "SE": RunnerRecord.parse,
    "HR": PayoutRecord.parse,
    "O1": OddsRecord.parse,
    "JG": HorseExclusionRecord.parse,
    "UM": HorseMasterRecord.parse,
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
