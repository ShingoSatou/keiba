from pathlib import Path


def test_build_t5_snapshot_sql_contains_we_cc_keys():
    source = Path("scripts/build_t5_snapshot.py").read_text(encoding="utf-8")

    assert "we_selected AS (" in source
    assert "cc_selected AS (" in source
    assert "'we_event_id'" in source
    assert "'cc_event_id'" in source
    assert "'we_announce_mmddhhmi'" in source
    assert "'cc_announce_mmddhhmi'" in source
    assert "'cc_distance_m_after'" in source
    assert "'cc_track_type_after'" in source


def test_build_t5_snapshot_filters_out_race_no_zero():
    source = Path("scripts/build_t5_snapshot.py").read_text(encoding="utf-8")
    assert source.count("race_no BETWEEN 1 AND 12") >= 4
