"""
パーサーのテスト
"""

from app.infrastructure.parsers import (
    HorseExclusionRecord,
    OddsRecord,
    PayoutRecord,
    RaceRecord,
    RunnerRecord,
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


class TestHorseExclusionRecordParser:
    """HorseExclusionRecord (JG) パーサーのテスト"""

    def test_parse_basic(self):
        # JGレコード
        payload = "JG12026010320260104060101012023100239ファストワーカー　　　　　　　　　　00110"
        record = HorseExclusionRecord.parse(payload)
        assert record.horse_id == "2023100239"
        assert record.horse_name.strip() == "ファストワーカー"
        assert record.flags == "00110"
