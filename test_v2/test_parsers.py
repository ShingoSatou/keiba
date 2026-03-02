from __future__ import annotations

from app.infrastructure.parsers import (
    O3_ANNOUNCE_MMDDHHMI_START,
    O3_DATA_CREATE_YMD_START,
    O3_DATA_KBN_START,
    O3_RACE_KEY_START,
    O3_SALE_FLAG_WIDE_START,
    O3_STARTERS_START,
    O3_WIDE_POOL_START,
    O3_WIDE_START,
    RA_COND_CODE_MIN_AGE_START,
    RA_COURSE_START,
    RA_DISTANCE_START,
    RA_FIELD_SIZE_START,
    RA_GRADE_CODE_START,
    RA_MONTHDAY_START,
    RA_RACE_NO_START,
    RA_RACE_TYPE_CODE_START,
    RA_START_TIME_START,
    RA_STARTERS_START,
    RA_TRACK_CODE_START,
    RA_TRACK_TYPE_START,
    RA_TURF_GOING_START,
    RA_WEATHER_START,
    RA_WEIGHT_TYPE_CODE_START,
    RA_YEAR_START,
    DMRecord,
    EventChangeRecord,
    O3HeaderRecord,
    O3WideRecord,
    OddsTimeSeriesRecord,
    PayoutRecord,
    RaceRecord,
    RunnerRecord,
    TMRecord,
    WHRecord,
    _slice_date,
    _slice_decode,
    _slice_int,
    parse_record,
)


class TestSliceFunctions:
    def test_slice_decode_basic(self):
        payload = "RACE12345678"
        assert _slice_decode(payload, 0, 4) == "RACE"
        assert _slice_decode(payload, 4, 8) == "12345678"

    def test_slice_int_and_default(self):
        payload = "XX1234ABCD"
        assert _slice_int(payload, 2, 4) == 1234
        assert _slice_int(payload, 6, 4) == 0
        assert _slice_int(payload, 6, 4, default=-1) == -1

    def test_slice_date(self):
        payload = "YY20260203ZZ"
        result = _slice_date(payload, 2)
        assert result is not None
        assert result.year == 2026
        assert result.month == 2
        assert result.day == 3

    def test_slice_date_invalid(self):
        payload = "YY00000000ZZ"
        assert _slice_date(payload, 2) is None


class TestRaceRecordParser:
    def test_parse_race_condition_offsets_from_real_data_layout(self):
        b_payload = bytearray(b" " * 1200)

        def put(offset: int, text: str):
            b = text.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        put(0, "RA1")
        put(RA_YEAR_START, "2026")
        put(RA_MONTHDAY_START, "0203")
        put(RA_TRACK_CODE_START, "05")
        put(RA_RACE_NO_START, "12")
        put(614, "A")
        put(616, "14")
        put(621, "3")
        put(622, "701")
        put(625, "010")
        put(628, "005")
        put(631, "016")
        put(634, "999")

        payload = bytes(b_payload).decode("cp932", errors="replace")
        record = RaceRecord.parse(payload)

        assert record.grade_code == 1
        assert record.race_type_code == 14
        assert record.weight_type_code == 3
        assert record.condition_code_min_age == 999
        assert record.class_code == 5

    def test_parse_with_multibyte_fields_keeps_byte_offsets(self):
        b_payload = bytearray(b" " * 1200)

        def put(offset: int, text: str):
            b = text.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        put(0, "RA1")
        put(RA_YEAR_START, "2026")
        put(RA_MONTHDAY_START, "0203")
        put(RA_TRACK_CODE_START, "05")
        put(RA_RACE_NO_START, "12")
        put(RA_DISTANCE_START, "1600")
        put(RA_TRACK_TYPE_START, "10")
        put(RA_COURSE_START, "01")
        put(RA_GRADE_CODE_START, "A")
        put(RA_RACE_TYPE_CODE_START, "14")
        put(RA_WEIGHT_TYPE_CODE_START, "2")
        put(RA_COND_CODE_MIN_AGE_START, "999")
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
        assert record.grade_code == 1
        assert record.race_type_code == 14
        assert record.weight_type_code == 2
        assert record.condition_code_min_age == 999
        assert record.class_code == 999

    def test_parse_normalizes_distance_and_uses_starters(self):
        b_payload = bytearray(b" " * 1200)

        def put(offset: int, text: str):
            b = text.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        put(0, "RA1")
        put(RA_YEAR_START, "2026")
        put(RA_MONTHDAY_START, "0203")
        put(RA_TRACK_CODE_START, "05")
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


class TestRunnerRecordParser:
    def test_parse_starter(self):
        b_payload = bytearray(b" " * 555)

        def put(offset: int, text: str):
            b = text.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        put(0, "SE120260203")
        put(11, "2026020305010101")
        put(27, "5")
        put(28, "10")
        put(30, "2021101234")
        put(40, "テストホース")
        put(78, "1")
        put(85, "54321")
        put(90, "調教師名")
        put(288, "500")
        put(296, "12345")
        put(306, "騎手名略")
        put(324, "480")
        put(328, "+02")
        put(334, "01")
        put(338, "1300")

        payload = b_payload.decode("cp932", errors="replace")
        record = RunnerRecord.parse(payload)

        assert record.horse_no == 10
        assert record.jockey_id == 12345
        assert record.trainer_id == 54321
        assert record.carried_weight == 50.0
        assert record.body_weight == 480
        assert record.body_weight_diff == 2
        assert record.finish_pos == 1
        assert record.time_sec == 90.0
        assert record.data_kubun == "1"
        assert record.sex == 1


class TestPayoutRecordParser:
    def test_parse_hr(self):
        b_payload = bytearray(b" " * 719)

        def put(offset: int, text: str):
            b = text.encode("cp932")
            end = offset + len(b)
            b_payload[offset:end] = b

        put(0, "HR220260203")
        put(11, "2026020306010101")

        # Wide block format: kumiban(4) + payout_raw(8) + popularity(4)
        # payout_raw is 10-yen unit in HR payload.
        put(293, "0102000000200001")
        put(309, "0103000000330002")
        put(325, "0203000000570003")

        payload = b_payload.decode("cp932", errors="replace")
        records = PayoutRecord.parse(payload)
        wide_rows = [row for row in records if row.bet_type == 5]
        assert len(wide_rows) == 3
        assert [row.selection for row in wide_rows] == ["0102", "0103", "0203"]
        assert [row.payout_yen for row in wide_rows] == [200, 330, 570]
        assert [row.popularity for row in wide_rows] == [1, 2, 3]


class TestOddsTimeSeriesRecordParser:
    def test_parse_treats_masked_odds_as_none(self):
        b_payload = bytearray(b" " * 962)

        def put(offset: int, text: str):
            b = text.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        put(0, "O1")
        put(2, "1")
        put(3, "20260208")
        put(27, "02080512")
        put(37, "00012345678")
        put(43, "01****01")
        put(51, "02000002")
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


class TestO3Parser:
    def test_o3_header_and_wide_parse(self):
        b_payload = bytearray(b" " * 2654)

        def put(offset: int, text: str):
            b = text.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        put(0, "O3")
        put(O3_DATA_KBN_START, "4")
        put(O3_DATA_CREATE_YMD_START, "20260214")
        put(O3_RACE_KEY_START, "2026021405010101")
        put(O3_ANNOUNCE_MMDDHHMI_START, "02141230")
        put(O3_STARTERS_START, "16")
        put(O3_SALE_FLAG_WIDE_START, "7")

        put(O3_WIDE_START, "0102")
        put(O3_WIDE_START + 4, "00125")
        put(O3_WIDE_START + 9, "00195")
        put(O3_WIDE_START + 14, "003")

        put(O3_WIDE_START + 17, "0103")
        put(O3_WIDE_START + 21, "00280")
        put(O3_WIDE_START + 26, "00460")
        put(O3_WIDE_START + 31, "011")

        put(O3_WIDE_POOL_START, "00001234567")

        payload = b_payload.decode("cp932", errors="replace")
        header = O3HeaderRecord.parse(payload)
        rows = O3WideRecord.parse(payload)

        assert header.race_id == 202602140501
        assert header.data_kbn == 4
        assert header.data_create_ymd == "20260214"
        assert header.announce_mmddhhmi == "02141230"
        assert header.starters == 16
        assert header.sale_flag_wide == 7
        assert header.wide_pool_total_100yen == 1234567

        assert len(rows) == 2
        assert rows[0].kumiban == "0102"
        assert rows[0].min_odds_x10 == 125
        assert rows[0].max_odds_x10 == 195
        assert rows[0].popularity == 3
        assert rows[1].kumiban == "0103"
        assert rows[1].min_odds_x10 == 280
        assert rows[1].max_odds_x10 == 460
        assert rows[1].popularity == 11

    def test_o3_parse_skips_invalid_kumiban_and_keeps_mask(self):
        b_payload = bytearray(b" " * 2654)

        def put(offset: int, text: str):
            b = text.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        put(0, "O3")
        put(O3_DATA_KBN_START, "3")
        put(O3_DATA_CREATE_YMD_START, "20260215")
        put(O3_RACE_KEY_START, "2026021508010101")
        put(O3_ANNOUNCE_MMDDHHMI_START, "02151225")

        put(O3_WIDE_START, "0000")
        put(O3_WIDE_START + 17, "0102")
        put(O3_WIDE_START + 21, "*****")
        put(O3_WIDE_START + 26, "-----")
        put(O3_WIDE_START + 31, "***")

        payload = b_payload.decode("cp932", errors="replace")
        rows = O3WideRecord.parse(payload)

        assert len(rows) == 1
        assert rows[0].kumiban == "0102"
        assert rows[0].min_odds_x10 is None
        assert rows[0].max_odds_x10 is None
        assert rows[0].popularity is None

    def test_parse_record_dispatches_o3(self):
        b_payload = bytearray(b" " * 2654)

        def put(offset: int, text: str):
            b = text.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        put(0, "O3")
        put(O3_DATA_KBN_START, "4")
        put(O3_DATA_CREATE_YMD_START, "20260214")
        put(O3_RACE_KEY_START, "2026021405010101")
        put(O3_WIDE_START, "0102")
        put(O3_WIDE_START + 4, "00125")
        put(O3_WIDE_START + 9, "00195")

        payload = b_payload.decode("cp932", errors="replace")
        rows = parse_record("O3", payload)

        assert isinstance(rows, list)
        assert len(rows) == 1
        assert rows[0].kumiban == "0102"


class TestMiningParsers:
    def test_dm_parse_two_horses_and_rank(self):
        b_payload = bytearray(b" " * 303)

        def put(offset: int, text: str):
            b = text.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        put(0, "DM120260203")
        put(11, "2026020305010101")
        put(31, "01")
        put(33, "13000")
        put(46, "02")
        put(48, "13100")

        payload = b_payload.decode("cp932", errors="replace")
        rows = DMRecord.parse(payload)
        assert len(rows) == 2
        by_horse = {row.horse_no: row for row in rows}
        assert by_horse[1].dm_time_x10 == 900
        assert by_horse[1].dm_rank == 1
        assert by_horse[2].dm_time_x10 == 910
        assert by_horse[2].dm_rank == 2

    def test_tm_parse_two_horses_and_rank(self):
        b_payload = bytearray(b" " * 141)

        def put(offset: int, text: str):
            b = text.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        put(0, "TM120260203")
        put(11, "2026020305010101")
        put(31, "01")
        put(33, "0853")
        put(37, "02")
        put(39, "0900")

        payload = b_payload.decode("cp932", errors="replace")
        rows = TMRecord.parse(payload)
        assert len(rows) == 2
        by_horse = {row.horse_no: row for row in rows}
        assert by_horse[2].tm_rank == 1
        assert by_horse[1].tm_rank == 2


class TestEventChangeRecordParser:
    def test_we_has_pseudo_race_id(self):
        b_payload = bytearray(b" " * 42)

        def put(offset: int, text: str):
            b = text.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        put(0, "WE120260203")
        put(11, "2026")
        put(15, "0203")
        put(19, "05")
        put(25, "02031234")

        payload = b_payload.decode("cp932", errors="replace")
        record = EventChangeRecord.parse(payload)
        assert record.record_type == "WE"
        assert record.race_id == 202602030500
        assert record.announce_mmddhhmi == "02031234"

    def test_cc_normalizes_short_distance_scale(self):
        b_payload = bytearray(b" " * 50)

        def put(offset: int, text: str):
            b = text.encode("cp932")
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
        record = EventChangeRecord.parse(payload)
        assert record.payload_parsed["distance_m_after"] == 1400
        assert record.payload_parsed["distance_m_before"] == 1700


class TestWHParser:
    def test_wh_parses_18_blocks_and_handles_999(self):
        # WH record length is 847 bytes (JV-Data 4.9.0.1)
        b_payload = bytearray(b" " * 847)

        def put(offset: int, text: str):
            b = text.encode("cp932")
            b_payload[offset : offset + len(b)] = b

        # Header
        put(0, "WH")
        put(2, "1")  # data_kbn (pos3)
        put(11, "2026020805010101")  # race_key (pos12-27)
        put(27, "02081230")  # announce_mmddhhmi (pos28-35)

        # Detail block layout (repeat 18):
        # horse_no(2) + horse_name(36) + weight(3) + sign(1) + diff(3)
        block_start = 35  # pos36
        block_len = 45

        # Horse 1: normal values
        put(block_start + 0, "01")
        put(block_start + 2, "TEST HORSE".ljust(36))
        put(block_start + 38, "480")
        put(block_start + 41, "+")
        put(block_start + 42, "012")

        # Horse 2: weight/diff 999 should become None, sign space should normalize to " "
        off2 = block_start + block_len
        put(off2 + 0, "02")
        put(off2 + 2, "X".ljust(36))
        put(off2 + 38, "999")
        put(off2 + 42, "999")

        payload = bytes(b_payload).decode("cp932", errors="replace")
        rows = WHRecord.parse(payload)

        assert len(rows) == 2
        by_horse = {row.horse_no: row for row in rows}
        assert by_horse[1].race_id == 202602080501
        assert by_horse[1].data_kbn == 1
        assert by_horse[1].announce_mmddhhmi == "02081230"
        assert by_horse[1].body_weight_kg == 480
        assert by_horse[1].diff_sign == "+"
        assert by_horse[1].diff_kg == 12

        assert by_horse[2].body_weight_kg is None
        assert by_horse[2].diff_sign == " "
        assert by_horse[2].diff_kg is None
