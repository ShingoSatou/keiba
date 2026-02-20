"""
パーサーのテスト
"""

from app.infrastructure.parsers import (
    RA_COURSE_START,
    RA_DISTANCE_START,
    RA_FIELD_SIZE_START,
    RA_KAI_START,
    RA_MONTHDAY_START,
    RA_NICHI_START,
    RA_RACE_NAME_START,
    RA_RACE_NO_START,
    RA_START_TIME_START,
    RA_STARTERS_START,
    RA_TRACK_CODE_START,
    RA_TRACK_TYPE_START,
    RA_TURF_GOING_START,
    RA_WEATHER_START,
    RA_YEAR_START,
    CKRecord,
    DMRecord,
    EventChangeRecord,
    HorseExclusionRecord,
    OddsRecord,
    OddsTimeSeriesRecord,
    PayoutRecord,
    RaceRecord,
    RunnerRecord,
    TMRecord,
    _slice_date,
    _slice_decode,
    _slice_int,
)


class TestSliceFunctions:
    """スライス関数のテスト"""

    def test_slice_decode_basic(self):
        payload = "RACE12345678"
        assert _slice_decode(payload, 0, 4) == "RACE"
        assert _slice_decode(payload, 4, 8) == "12345678"

    def test_slice_decode_with_spaces(self):
        payload = "RA  TEST  "
        assert _slice_decode(payload, 0, 2) == "RA"
        assert _slice_decode(payload, 2, 6) == "TEST"

    def test_slice_int(self):
        payload = "XX1234567890"
        assert _slice_int(payload, 2, 4) == 1234
        assert _slice_int(payload, 6, 6) == 567890

    def test_slice_int_invalid(self):
        payload = "XXABCD1234"
        assert _slice_int(payload, 2, 4) == 0  # default
        assert _slice_int(payload, 2, 4, default=-1) == -1

    def test_slice_date(self):
        payload = "XX20260203XXXX"
        result = _slice_date(payload, 2)
        assert result is not None
        assert result.year == 2026
        assert result.month == 2
        assert result.day == 3

    def test_slice_date_invalid(self):
        payload = "XX00000000XXXX"
        assert _slice_date(payload, 2) is None


class TestRaceRecordParser:
    """RaceRecordパーサーのテスト"""

    def test_parse_simple(self):
        # RAレコード (JV-Data 4.9.0.1準拠: 距離=698(1-idx)/697(0-idx))
        # Header(3) + Date(8) + Key(16) -> Start 27

        # 簡易的に空文字で埋める
        payload = "RA" + " " + "20260203"  # 11 chars
        payload += "2026020305051112"  # Key 16 chars (Tokyo, 5th Kai, 11th Day, 12R)

        # Distance (697 start, 0-indexed)
        # Current 27. Need 697. Diff 670.
        payload += " " * 670
        payload += "1600"  # Dist at 697
        payload += "    "  # OldDist at 701 (4 bytes)
        payload += "10"  # TrackType at 705 (Turf=10)

        payload = payload.ljust(1200)

        record = RaceRecord.parse(payload)
        assert record is not None
        assert record.race_date is not None
        assert record.race_date.year == 2026
        assert record.race_date.month == 2
        assert record.race_date.day == 3
        assert record.track_code == 5
        assert record.race_no == 12
        assert record.distance_m == 1600
        assert record.surface == 1  # 芝 (Code 10)

    def test_parse_with_multibyte_fields_keeps_byte_offsets(self):
        b_payload = bytearray(b" " * 1200)

        def put(offset: int, text: str):
            b = text.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        # RAヘッダー/キー
        put(0, "RA1")
        put(RA_YEAR_START, "2026")
        put(RA_MONTHDAY_START, "0203")
        put(RA_TRACK_CODE_START, "05")
        put(RA_KAI_START, "05")
        put(RA_NICHI_START, "11")
        put(RA_RACE_NO_START, "12")
        # マルチバイト文字を含む項目
        put(RA_RACE_NAME_START, "テストレース")
        # 条件/時刻
        put(RA_DISTANCE_START, "1600")
        put(RA_TRACK_TYPE_START, "10")
        put(RA_COURSE_START, "01")
        put(RA_START_TIME_START, "1540")
        put(RA_FIELD_SIZE_START, "18")
        put(RA_WEATHER_START, "1")
        put(RA_TURF_GOING_START, "2")

        payload = bytes(b_payload).decode("cp932", errors="replace")
        record = RaceRecord.parse(payload)

        assert record.race_id == 202602030512
        assert record.race_date is not None
        assert record.track_code == 5
        assert record.race_no == 12
        assert record.distance_m == 1600
        assert record.field_size == 18
        assert record.start_time is not None
        assert record.start_time.hour == 15
        assert record.start_time.minute == 40

    def test_parse_normalizes_distance_and_zero_values(self):
        b_payload = bytearray(b" " * 1200)

        def put(offset: int, text: str):
            b = text.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        put(0, "RA1")
        put(RA_YEAR_START, "2026")
        put(RA_MONTHDAY_START, "0203")
        put(RA_TRACK_CODE_START, "05")
        put(RA_KAI_START, "05")
        put(RA_NICHI_START, "11")
        put(RA_RACE_NO_START, "12")
        put(RA_DISTANCE_START, "0014")
        put(RA_TRACK_TYPE_START, "10")
        put(RA_WEATHER_START, "0")
        put(RA_TURF_GOING_START, "0")
        put(RA_FIELD_SIZE_START, "00")
        put(RA_STARTERS_START, "16")

        payload = bytes(b_payload).decode("cp932", errors="replace")
        record = RaceRecord.parse(payload)

        assert record.distance_m == 1400
        assert record.weather is None
        assert record.going is None
        assert record.field_size == 16

    def test_parse_uses_registered_field_size_when_starters_missing(self):
        b_payload = bytearray(b" " * 1200)

        def put(offset: int, text: str):
            b = text.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        put(0, "RA1")
        put(RA_YEAR_START, "2026")
        put(RA_MONTHDAY_START, "0203")
        put(RA_TRACK_CODE_START, "05")
        put(RA_KAI_START, "05")
        put(RA_NICHI_START, "11")
        put(RA_RACE_NO_START, "12")
        put(RA_DISTANCE_START, "1600")
        put(RA_TRACK_TYPE_START, "10")
        put(RA_FIELD_SIZE_START, "18")
        put(RA_STARTERS_START, "00")

        payload = bytes(b_payload).decode("cp932", errors="replace")
        record = RaceRecord.parse(payload)

        assert record.field_size == 18


class TestRunnerRecordParser:
    """RunnerRecordパーサーの基本テスト"""

    def test_parse_starter(self):
        # SEレコードテスト (仕様書 4.9.0.1 - 555バイト仕様)
        b_payload = bytearray(b" " * 555)  # スペースで初期化

        def put(offset: int, s: str):
            """0-origin offsetに文字列をセット"""
            b = s.encode("cp932")
            for i, byte in enumerate(b):
                if offset + i < len(b_payload):
                    b_payload[offset + i] = byte

        # Header (位置1-11)
        put(0, "SE120260203")  # rec_id + data_cat + date
        # Key (位置12-27 = 16バイト)
        put(11, "2026020305010101")

        # 枠番 (位置28, 1バイト)
        put(27, "5")
        # 馬番 (位置29, 2バイト)
        put(28, "10")
        # 血統登録番号 (位置31, 10バイト)
        put(30, "2021101234")
        # 馬名 (位置41, 36バイト)
        put(40, "テストホース")

        # 性別コード (位置79, 1バイト)
        put(78, "1")

        # 調教師コード (位置86, 5バイト) ★
        put(85, "54321")
        # 調教師名略称 (位置91, 8バイト)
        put(90, "調教師名")

        # 馬主コード (位置99, 6バイト)
        put(98, "999999")

        # 負担重量 (位置289, 3バイト) = 50.0kg -> 500
        put(288, "500")

        # ブリンカー (位置295, 1バイト)
        put(294, "0")

        # 騎手コード (位置297, 5バイト) ★
        put(296, "12345")
        # 騎手名略称 (位置307, 8バイト)
        put(306, "騎手名略")

        # 馬体重 (位置325, 3バイト)
        put(324, "480")
        # 増減差 (位置329, 3バイト)
        put(328, "+02")

        # 確定着順 (位置335, 2バイト)
        put(334, "01")

        # 走破タイム (位置339, 4バイト) = 1:30.0 -> 1300
        put(338, "1300")

        s_payload = b_payload.decode("cp932", errors="replace")
        record = RunnerRecord.parse(s_payload)

        assert "テストホース" in record.horse_name
        assert record.horse_no == 10
        assert record.jockey_id == 12345
        assert record.trainer_id == 54321
        assert record.carried_weight == 50.0
        assert record.body_weight == 480
        assert record.body_weight_diff == 2
        assert record.finish_pos == 1
        assert record.time_sec == 90.0

        # 新フィールドの検証
        # "SE1..." -> S(0), E(1), 1(2) -> data_kubun="1"
        assert record.data_kubun == "1"

        assert record.trainer_code_raw == "54321"
        assert "調教師名" in record.trainer_name_abbr
        assert record.jockey_code_raw == "12345"
        assert "騎手名略" in record.jockey_name_abbr


class TestPayoutRecordParser:
    """PayoutRecordパーサーのテスト"""

    def test_parse_hr(self):
        # HR Spec: 964 bytes.
        b_payload = bytearray()

        def add(s: str, length: int):
            b = s.encode("cp932")
            if len(b) > length:
                b = b[:length]
            else:
                b += b" " * (length - len(b))
            b_payload.extend(b)

        # 1. Header (1-2: "HR", 3: "1", 4-11: Date) -> 11 bytes
        add("HR120260203", 11)

        # 2. Key (12-27: 16 bytes)
        add("2026020306010101", 16)

        # Current 27.
        # 3. Gap to Win (Start 52). 52-27=25 bytes.
        add("", 25)

        # 4. Win (52, 10 bytes * 3)
        # Win 1: No "01", Yen "150".
        add("01", 2)
        add("00000150", 8)
        # Win 2: Empty
        add("", 10)
        # Win 3: Empty
        add("", 10)

        # Current: 52 + 30 = 82. Correct.

        # 5. Place (82, 10 bytes * 5)
        # Place 1: No "01", Yen "110"
        add("01", 2)
        add("00000110", 8)
        # Place 2: No "02", Yen "120"
        add("02", 2)
        add("00000120", 8)
        # Place 3-5: Empty
        add("", 30)

        # Current: 82 + 50 = 132. Correct.

        # 6. Bracket (132, 10 bytes * 3). Empty for now.
        add("", 30)

        # 7. Quinella (162, 12 bytes * 3).
        # Q 1: 01-02, 200 yen.
        add("0102", 4)
        add("00000200", 8)
        add("", 24)

        # 8. Wide (198, 12 bytes * 7). Empty.
        add("", 84)

        # 9. Exacta (282, 12 bytes * 6).
        # E 1: 01-02, 300 yen.
        add("0102", 4)
        add("00000300", 8)
        add("", 60)

        # 10. Trio (354, 14 bytes * 3).
        add("", 42)

        # 11. Trifecta (396, 14 bytes * 6).
        add("", 84)

        # Pad to end
        curr = len(b_payload)
        if curr < 964:
            b_payload.extend(b" " * (964 - curr))

        s_payload = b_payload.decode("cp932")
        records = PayoutRecord.parse(s_payload)

        assert len(records) > 0

        # Win
        wins = [r for r in records if r.bet_type == 1]
        assert len(wins) == 1
        assert wins[0].selection == "01"
        assert wins[0].payout_yen == 150

        # Place
        places = [r for r in records if r.bet_type == 2]
        assert len(places) == 2
        assert places[0].selection == "01"
        assert places[0].payout_yen == 110

        # Quinella
        quins = [r for r in records if r.bet_type == 4]
        assert len(quins) == 1
        assert quins[0].selection == "0102"
        assert quins[0].payout_yen == 200

        # Exacta
        exactas = [r for r in records if r.bet_type == 6]
        assert len(exactas) == 1
        assert exactas[0].selection == "0102"
        assert exactas[0].payout_yen == 300


class TestOddsRecordParser:
    """OddsRecordパーサーのテスト"""

    def test_parse_odds(self):
        # O1 Spec: 962 bytes.
        b_payload = bytearray()

        def add(s: str, length: int):
            b = s.encode("cp932")
            if len(b) > length:
                b = b[:length]
            else:
                b += b" " * (length - len(b))
            b_payload.extend(b)

        # 1. Header (1-2: "O1", 3: "1", 4-11: Date) -> 11 bytes
        add("O1120260203", 11)

        # 2. Key (12-27: 16 bytes)
        add("2026020306010101", 16)

        # Current 27.
        # 3. Gap to Win (Start 43). 43-27=16 bytes.
        add("", 16)

        # 4. Win Odds (43, 8 bytes * 28).
        # Horse 1 (Offset 43): #01, 2.5 ("0025")
        add("01", 2)
        add("0025", 4)  # 25 / 10 = 2.5
        add("01", 2)  # Pop

        # Horse 2 (Offset 51): #02, 10.0 ("0100")
        add("02", 2)
        add("0100", 4)
        add("05", 2)  # Pop

        # Fill remaining 26 horses
        add("", 8 * 26)

        # Current: 43 + 28*8 = 267. Correct (Place starts 267).

        # 5. Place Odds (267, 12 bytes * 28)
        # Horse 1: #01, Min 1.1 - Max 1.3
        add("01", 2)
        add("0011", 4)
        add("0013", 4)
        add("01", 2)

        # Fill remaining
        add("", 12 * 27)

        # Current: 267 + 28*12 = 603. Correct (Bracket starts 603).

        # 6. Bracket Odds (603, 9 bytes * 36)
        # 1-1: 5.0
        add("11", 2)
        add("00050", 5)  # 50 / 10 = 5.0
        add("03", 2)

        # Fill rest
        add("", 9 * 35)

        # Pad to end
        curr = len(b_payload)
        if curr < 962:
            b_payload.extend(b" " * (962 - curr))

        s_payload = b_payload.decode("cp932")
        records = OddsRecord.parse(s_payload)

        assert len(records) > 0

        # Win
        wins = [r for r in records if r.bet_type == 1]
        assert len(wins) == 2
        assert wins[0].horse_no == 1
        assert wins[0].odds_1 == 2.5
        assert wins[1].horse_no == 2
        assert wins[1].odds_1 == 10.0

        # Place
        places = [r for r in records if r.bet_type == 2]
        assert len(places) == 1
        assert places[0].horse_no == 1
        assert places[0].odds_1 == 1.1
        assert places[0].odds_2 == 1.3

        # Bracket
        brackets = [r for r in records if r.bet_type == 3]
        assert len(brackets) == 1
        assert brackets[0].horse_no == "11"
        assert brackets[0].odds_1 == 5.0


class TestOddsTimeSeriesRecordParser:
    """OddsTimeSeriesRecordパーサーのテスト"""

    def test_parse_treats_masked_odds_as_none(self):
        b_payload = bytearray(b" " * 962)

        def put(offset: int, s: str):
            b = s.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        # Header + timestamp key fields
        put(0, "O1")
        put(2, "1")
        put(3, "20260208")
        put(27, "02080512")
        put(37, "00012345678")

        # Win block: horse_no(2) + odds(4) + pop(2)
        put(43, "01****01")  # masked odds -> None
        put(51, "02000002")  # zero odds stays 0 (later snapshot/predict safeguards handle it)
        put(59, "03012503")

        payload = bytes(b_payload).decode("cp932", errors="replace")
        rows = OddsTimeSeriesRecord.parse(payload, race_id=202602080101)
        assert len(rows) == 3

        by_horse = {row.horse_no: row for row in rows}
        assert by_horse[1].win_odds_x10 is None
        assert by_horse[1].win_popularity == 1
        assert by_horse[2].win_odds_x10 == 0
        assert by_horse[2].win_popularity == 2
        assert by_horse[3].win_odds_x10 == 125
        assert by_horse[3].win_popularity == 3


class TestHorseExclusionRecordParser:
    """HorseExclusionRecord (JG) パーサーのテスト"""

    def test_parse_basic(self):
        # JGレコード
        payload = "JG12026010320260104060101012023100239ファストワーカー　　　　　　　　　　00110"
        record = HorseExclusionRecord.parse(payload)
        assert record.horse_id == "2023100239"
        assert record.horse_name.strip() == "ファストワーカー"
        assert record.flags == "00110"


class TestMiningParsers:
    """MING (DM/TM) パーサーのテスト"""

    def test_dm_parse_two_horses_and_rank(self):
        b_payload = bytearray(b" " * 303)

        def put(offset: int, s: str):
            b = s.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        # Header: "DM" + data_kbn + date(8)
        put(0, "DM120260203")
        # Race key (pos12-27, 16 bytes)
        put(11, "2026020305010101")

        # Horse1 at pos32
        put(31, "01")  # horse_no
        put(33, "13000")  # 1:30.00 -> 90.0s -> x10=900

        # Horse2 next item (15 bytes each)
        put(31 + 15, "02")
        put(33 + 15, "13100")  # 1:31.00 -> x10=910

        payload = b_payload.decode("cp932", errors="replace")
        rows = DMRecord.parse(payload)

        assert len(rows) == 2
        assert rows[0].race_id == 202602030501
        assert rows[0].data_create_ymd == "20260203"
        assert rows[0].data_create_hm == "0000"
        assert rows[0].horse_no == 1
        assert rows[0].dm_time_x10 == 900

        assert rows[1].horse_no == 2
        assert rows[1].dm_time_x10 == 910

        by_horse = {r.horse_no: r for r in rows}
        assert by_horse[1].dm_rank == 1
        assert by_horse[2].dm_rank == 2

    def test_tm_parse_two_horses_and_rank(self):
        b_payload = bytearray(b" " * 141)

        def put(offset: int, s: str):
            b = s.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        # Header: "TM" + data_kbn + date(8)
        put(0, "TM120260203")
        # Race key (pos12-27, 16 bytes)
        put(11, "2026020305010101")

        # Horse1 at pos32
        put(31, "01")  # horse_no
        put(33, "0853")  # 85.3 -> x10=853

        # Horse2 next item
        put(31 + 6, "02")
        put(33 + 6, "0900")  # 90.0 -> x10=900

        payload = b_payload.decode("cp932", errors="replace")
        rows = TMRecord.parse(payload)

        assert len(rows) == 2
        assert rows[0].race_id == 202602030501
        assert rows[0].data_create_ymd == "20260203"
        assert rows[0].data_create_hm == "0000"
        assert rows[0].horse_no == 1
        assert rows[0].tm_score == 853

        assert rows[1].horse_no == 2
        assert rows[1].tm_score == 900

        by_horse = {r.horse_no: r for r in rows}
        assert by_horse[2].tm_rank == 1
        assert by_horse[1].tm_rank == 2


class TestCKRecordParser:
    """CK (SNPN) パーサーのテスト"""

    def test_ck_parse_minimal_and_stats_keys(self):
        b_payload = bytearray(b" " * 6870)

        def put(offset: int, s: str):
            b = s.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        # Header: "CK" + data_kbn + make_date(8)
        put(0, "CK1")
        put(3, "20260203")

        # Key
        put(11, "2026")  # kaisai_year
        put(15, "0203")  # kaisai_md
        put(19, "05")  # track_cd
        put(21, "01")  # kaisai_kai
        put(23, "01")  # kaisai_nichi
        put(25, "12")  # race_no
        put(27, "2021101234")  # horse_id
        put(37, "TESTHORSE")  # horse_name

        # Counts (overall/central) - 6 buckets * 3 bytes
        put(127, "001002003004005006")  # total
        put(145, "000001000000000000")  # central

        payload = b_payload.decode("cp932", errors="replace")
        record = CKRecord.parse(payload)

        assert record.data_kbn == 1
        assert record.make_date is not None
        assert record.kaisai_year == 2026
        assert record.kaisai_md == "0203"
        assert record.track_cd == "05"
        assert record.race_no == 12
        assert record.horse_id == "2021101234"
        assert record.horse_name == "TESTHORSE"
        assert record.counts_total == [1, 2, 3, 4, 5, 6]
        assert record.counts_central == [0, 1, 0, 0, 0, 0]

        stats = record.get_full_stats()
        assert "finish_counts" in stats
        assert "turf_1200_down" in stats["finish_counts"]
        assert "dirt_2801_up" in stats["finish_counts"]


class TestEventChangeRecordParser:
    """当日変更 (WE/AV/JC/TC/CC) パーサーのテスト"""

    def test_we_has_pseudo_race_id_and_audit_keys(self):
        b_payload = bytearray(b" " * 42)

        def put(offset: int, s: str):
            b = s.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        put(0, "WE120260203")  # rec + data_kbn + data_create_ymd
        put(11, "2026")  # kaisai_year
        put(15, "0203")  # kaisai_md
        put(19, "05")  # track_cd
        put(21, "01")  # kaisai_kai
        put(23, "01")  # kaisai_nichi
        put(25, "02031234")  # announce_mmddhhmi

        payload = b_payload.decode("cp932", errors="replace")
        rec = EventChangeRecord.parse(payload)

        assert rec.record_type == "WE"
        assert rec.data_create_ymd == "20260203"
        assert rec.announce_mmddhhmi == "02031234"
        assert rec.race_id == 202602030500  # YYYYMMDDTT00 (pseudo race_no=00)

    def test_av_has_audit_keys(self):
        b_payload = bytearray(b" " * 78)

        def put(offset: int, s: str):
            b = s.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        put(0, "AV120260203")
        put(11, "2026020305010101")  # race_key
        put(27, "02031234")  # announce
        put(35, "08")  # horse_no
        put(73, "123")  # reason_kbn

        payload = b_payload.decode("cp932", errors="replace")
        rec = EventChangeRecord.parse(payload)

        assert rec.record_type == "AV"
        assert rec.data_create_ymd == "20260203"
        assert rec.announce_mmddhhmi == "02031234"
        assert rec.race_id == 202602030501
        assert rec.payload_parsed["horse_no"] == 8
        assert rec.payload_parsed["reason_kbn"] == "123"

    def test_jc_has_audit_keys(self):
        b_payload = bytearray(b" " * 161)

        def put(offset: int, s: str):
            b = s.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        put(0, "JC120260203")
        put(11, "2026020305010101")  # race_key
        put(27, "02031234")  # announce
        put(35, "08")  # horse_no
        put(73, "500")  # carried_weight_x10_after
        put(76, "12345")  # jockey_code_after
        put(116, "490")  # carried_weight_x10_before
        put(119, "54321")  # jockey_code_before

        payload = b_payload.decode("cp932", errors="replace")
        rec = EventChangeRecord.parse(payload)

        assert rec.record_type == "JC"
        assert rec.data_create_ymd == "20260203"
        assert rec.announce_mmddhhmi == "02031234"
        assert rec.race_id == 202602030501
        assert rec.payload_parsed["horse_no"] == 8
        assert rec.payload_parsed["carried_weight_x10_after"] == 500
        assert rec.payload_parsed["jockey_code_after"] == "12345"
        assert rec.payload_parsed["carried_weight_x10_before"] == 490
        assert rec.payload_parsed["jockey_code_before"] == "54321"

    def test_tc_has_audit_keys(self):
        b_payload = bytearray(b" " * 45)

        def put(offset: int, s: str):
            b = s.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        put(0, "TC120260203")
        put(11, "2026020305010101")  # race_key
        put(27, "02031234")  # announce
        put(35, "1230")  # post_time_after
        put(39, "1225")  # post_time_before

        payload = b_payload.decode("cp932", errors="replace")
        rec = EventChangeRecord.parse(payload)

        assert rec.record_type == "TC"
        assert rec.data_create_ymd == "20260203"
        assert rec.announce_mmddhhmi == "02031234"
        assert rec.race_id == 202602030501
        assert rec.payload_parsed["post_time_after"] == "1230"
        assert rec.payload_parsed["post_time_before"] == "1225"

    def test_cc_has_audit_keys(self):
        b_payload = bytearray(b" " * 50)

        def put(offset: int, s: str):
            b = s.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        put(0, "CC120260203")
        put(11, "2026020305010101")  # race_key
        put(27, "02031234")  # announce
        put(35, "1600")  # distance_m_after
        put(39, "10")  # track_type_after
        put(41, "1400")  # distance_m_before
        put(45, "20")  # track_type_before
        put(47, "1")  # reason_kbn

        payload = b_payload.decode("cp932", errors="replace")
        rec = EventChangeRecord.parse(payload)

        assert rec.record_type == "CC"
        assert rec.data_create_ymd == "20260203"
        assert rec.announce_mmddhhmi == "02031234"
        assert rec.race_id == 202602030501
        assert rec.payload_parsed["distance_m_after"] == 1600
        assert rec.payload_parsed["track_type_after"] == 10
        assert rec.payload_parsed["distance_m_before"] == 1400
        assert rec.payload_parsed["track_type_before"] == 20
        assert rec.payload_parsed["reason_kbn"] == 1

    def test_cc_normalizes_short_distance_scale(self):
        b_payload = bytearray(b" " * 50)

        def put(offset: int, s: str):
            b = s.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        put(0, "CC120260203")
        put(11, "2026020305010101")
        put(27, "02031234")
        put(35, "0014")
        put(39, "10")
        put(41, "0017")
        put(45, "20")
        put(47, "1")

        payload = b_payload.decode("cp932", errors="replace")
        rec = EventChangeRecord.parse(payload)

        assert rec.payload_parsed["distance_m_after"] == 1400
        assert rec.payload_parsed["distance_m_before"] == 1700
