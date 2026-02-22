from __future__ import annotations

from pathlib import Path

import pytest

from scripts_v2.migrate import _discover_migrations


@pytest.fixture
def temp_migrations_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "migrations"
    directory.mkdir()
    (directory / "0002_second.sql").write_text("SELECT 2;", encoding="utf-8")
    (directory / "0001_first.sql").write_text("SELECT 1;", encoding="utf-8")
    return directory


def test_discover_migrations_sorted(temp_migrations_dir: Path):
    migrations = _discover_migrations(temp_migrations_dir)
    assert [m.version for m in migrations] == ["0001_first.sql", "0002_second.sql"]


def test_discover_migrations_has_checksum(temp_migrations_dir: Path):
    migrations = _discover_migrations(temp_migrations_dir)
    assert len(migrations[0].checksum) == 64
    assert migrations[0].checksum != migrations[1].checksum


def test_discover_migrations_dir_not_found(tmp_path: Path):
    with pytest.raises(SystemExit):
        _discover_migrations(tmp_path / "missing")
