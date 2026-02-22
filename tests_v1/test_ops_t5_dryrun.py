from scripts import ops_t5_dryrun


def test_resolve_policy_for_dataspec_auto():
    assert ops_t5_dryrun._resolve_policy_for_dataspec("0B14", "auto") == "strict"
    assert ops_t5_dryrun._resolve_policy_for_dataspec("0B41", "auto") == "warn"
    assert ops_t5_dryrun._resolve_policy_for_dataspec("0B11", "auto") == "warn"
    assert ops_t5_dryrun._resolve_policy_for_dataspec("0B13", "auto") == "continue"
    assert ops_t5_dryrun._resolve_policy_for_dataspec("0B17", "auto") == "continue"


def test_dataspec_pattern_uses_date_key_and_race_key():
    assert ops_t5_dryrun._dataspec_pattern("0B14", "20260215") == "0B14_20260215_*.jsonl"
    assert ops_t5_dryrun._dataspec_pattern("0B41", "20260215") == "0B41_20260215????_*.jsonl"


def test_evaluate_health_stop_on_missing_race_prerequisites():
    health = {
        "races": 0,
        "runner_rows": 0,
        "missing_start_time": 0,
        "race_stub": 0,
        "o1_rows": 0,
        "wh_rows": 0,
        "event_rows": 0,
        "rt_dm_rows": 0,
        "rt_tm_rows": 0,
        "t5_snapshot_rows": 0,
    }
    evaluated = ops_t5_dryrun._evaluate_health(health, {})

    assert evaluated["status"] == "stop"
    assert "core.race rows=0" in evaluated["stop_reasons"]
    assert "core.runner rows=0" in evaluated["stop_reasons"]


def test_evaluate_health_warn_on_partial_missing():
    health = {
        "races": 12,
        "runner_rows": 160,
        "missing_start_time": 1,
        "race_stub": 1,
        "o1_rows": 10,
        "wh_rows": 0,
        "event_rows": 0,
        "rt_dm_rows": 0,
        "rt_tm_rows": 5,
        "t5_snapshot_rows": 100,
    }
    evaluated = ops_t5_dryrun._evaluate_health(
        health,
        {
            "0B13": {"status": "continue_missing_file"},
            "0B11": {"status": "warn_missing_file"},
        },
    )

    assert evaluated["status"] == "warn"
    assert "missing_start_time=1" in evaluated["warn_reasons"]
    assert "race_stub=1" in evaluated["warn_reasons"]
    assert "0B13 files missing (continue)" in evaluated["warn_reasons"]
    assert "0B11 files missing (warn)" in evaluated["warn_reasons"]


def test_evaluate_health_warns_on_snapshot_odds_missing():
    health = {
        "races": 10,
        "runner_rows": 120,
        "missing_start_time": 0,
        "race_stub": 0,
        "o1_rows": 10,
        "wh_rows": 10,
        "event_rows": 8,
        "rt_dm_rows": 20,
        "rt_tm_rows": 20,
        "t5_snapshot_rows": 100,
        "snapshot_odds_missing_rows": 60,
        "snapshot_odds_missing_races": 6,
    }
    evaluated = ops_t5_dryrun._evaluate_health(health, {}, include_snapshot_quality=True)

    assert evaluated["status"] == "warn"
    assert "snapshot odds_missing ratio high (60/100, threshold=0.50)" in evaluated["warn_reasons"]
    assert (
        "snapshot all-odds-missing races ratio high (6/10, threshold=0.50)"
        in evaluated["warn_reasons"]
    )
