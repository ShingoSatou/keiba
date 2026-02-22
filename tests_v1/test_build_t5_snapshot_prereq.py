from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from scripts import build_t5_snapshot


class _DummyDB:
    def __init__(self):
        self.execute_calls: list[str] = []

    def execute(self, sql, params=None):
        self.execute_calls.append(sql)


def test_evaluate_day_health_stop_on_missing_start_time_ratio():
    health = {
        "races": 10,
        "runner_rows": 120,
        "missing_start_time": 5,
        "race_stub": 1,
    }
    stop_reasons, warn_reasons = build_t5_snapshot._evaluate_day_health(
        health=health,
        missing_start_time_stop_ratio=0.50,
        missing_start_time_warn_ratio=0.00,
    )

    assert any("missing_start_time ratio too high" in reason for reason in stop_reasons)
    assert "race_stub=1" in warn_reasons


def test_ensure_snapshot_prerequisites_runs_ops_race_then_recovers(monkeypatch):
    health_seq = [
        {
            "races": 10,
            "runner_rows": 120,
            "missing_start_time": 10,
            "race_stub": 0,
        },
        {
            "races": 10,
            "runner_rows": 120,
            "missing_start_time": 0,
            "race_stub": 0,
        },
    ]
    commands: list[list[str]] = []

    def fake_fetch_day_health(db, target_date):
        return health_seq.pop(0)

    def fake_run_command(cmd):
        commands.append(cmd)

    monkeypatch.setattr(build_t5_snapshot, "_fetch_day_health", fake_fetch_day_health)
    monkeypatch.setattr(build_t5_snapshot, "_run_command", fake_run_command)

    build_t5_snapshot._ensure_snapshot_prerequisites(
        db=object(),
        from_date=date(2026, 2, 18),
        to_date=date(2026, 2, 18),
        ensure_race=True,
        missing_start_time_stop_ratio=0.50,
        missing_start_time_warn_ratio=0.00,
        ops_race_force=False,
        ops_race_option=1,
    )

    assert len(commands) == 1
    assert commands[0][:5] == ["uv", "run", "python", "scripts/ops_race.py", "--race-date"]


def test_ensure_snapshot_prerequisites_raises_when_disabled(monkeypatch):
    monkeypatch.setattr(
        build_t5_snapshot,
        "_fetch_day_health",
        lambda db, target_date: {
            "races": 12,
            "runner_rows": 160,
            "missing_start_time": 12,
            "race_stub": 0,
        },
    )
    monkeypatch.setattr(
        build_t5_snapshot,
        "_run_command",
        lambda cmd: pytest.fail("ops_race must not run when ensure_race is disabled"),
    )

    with pytest.raises(RuntimeError):
        build_t5_snapshot._ensure_snapshot_prerequisites(
            db=object(),
            from_date=date(2026, 2, 18),
            to_date=date(2026, 2, 18),
            ensure_race=False,
            missing_start_time_stop_ratio=0.50,
            missing_start_time_warn_ratio=0.00,
            ops_race_force=False,
            ops_race_option=1,
        )


def test_build_snapshot_checks_prereq_before_delete(monkeypatch):
    dummy_db = _DummyDB()

    def fail_prereq(*args, **kwargs):
        raise RuntimeError("prerequisite failure")

    monkeypatch.setattr(build_t5_snapshot, "_ensure_snapshot_prerequisites", fail_prereq)

    with pytest.raises(RuntimeError):
        build_t5_snapshot.build_snapshot(
            db=dummy_db,
            from_date=date(2026, 2, 18),
            to_date=date(2026, 2, 18),
            feature_set="realtime",
            code_version="test",
            ensure_race=True,
            missing_start_time_stop_ratio=0.50,
            missing_start_time_warn_ratio=0.00,
            ops_race_force=False,
            ops_race_option=1,
        )

    assert dummy_db.execute_calls == []


def test_build_t5_snapshot_cli_exposes_prereq_flags():
    source = Path("scripts/build_t5_snapshot.py").read_text(encoding="utf-8")
    assert "--ensure-race" in source
    assert "--no-ensure-race" in source
    assert "--missing-start-time-stop-ratio" in source
    assert "--missing-start-time-warn-ratio" in source
