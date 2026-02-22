#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import psycopg
from psycopg.rows import dict_row


@dataclass(frozen=True)
class MigrationFile:
    version: str
    path: Path
    checksum: str


def _load_database_url_from_dotenv(dotenv_path: Path) -> str | None:
    if not dotenv_path.exists():
        return None
    try:
        text = dotenv_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = dotenv_path.read_text(encoding="utf-8", errors="replace")

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key != "DATABASE_URL":
            continue
        value = value.strip().strip("'").strip('"')
        return value or None
    return None


def _get_database_url(cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    env_value = os.environ.get("DATABASE_URL")
    if env_value:
        return env_value
    dotenv_value = _load_database_url_from_dotenv(Path(".env"))
    if dotenv_value:
        return dotenv_value
    raise SystemExit("DATABASE_URL is required (env or .env or --database-url).")


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _discover_migrations(migrations_dir: Path) -> list[MigrationFile]:
    if not migrations_dir.exists():
        raise SystemExit(f"migrations dir not found: {migrations_dir}")

    migration_files: list[MigrationFile] = []
    for path in sorted(migrations_dir.glob("*.sql")):
        version = path.name
        sql = path.read_text(encoding="utf-8", errors="replace")
        migration_files.append(MigrationFile(version=version, path=path, checksum=_sha256_hex(sql)))
    return migration_files


def _ensure_schema_migrations_table(conn: psycopg.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS public.schema_migrations (
            version TEXT PRIMARY KEY,
            checksum TEXT NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
    )


def _fetch_applied(conn: psycopg.Connection) -> dict[str, dict]:
    _ensure_schema_migrations_table(conn)
    rows = conn.execute(
        "SELECT version, checksum, applied_at FROM public.schema_migrations ORDER BY version"
    ).fetchall()
    return {r["version"]: {"checksum": r["checksum"], "applied_at": r["applied_at"]} for r in rows}


def _print_list(migrations: list[MigrationFile], applied: dict[str, dict]) -> None:
    for m in migrations:
        status = "pending"
        applied_at: datetime | None = None
        checksum_note = ""
        if m.version in applied:
            status = "applied"
            applied_at = applied[m.version]["applied_at"]
            if applied[m.version]["checksum"] != m.checksum:
                checksum_note = " (checksum mismatch)"
        suffix = f" @ {applied_at:%Y-%m-%d %H:%M:%S%z}" if applied_at else ""
        print(f"{status:7} {m.version}{suffix}{checksum_note}")


def _apply_one(conn: psycopg.Connection, migration: MigrationFile, *, baseline: bool) -> None:
    if not baseline:
        sql = migration.path.read_text(encoding="utf-8", errors="replace")
        conn.execute(sql)

    conn.execute(
        """
        INSERT INTO public.schema_migrations (version, checksum)
        VALUES (%s, %s)
        ON CONFLICT (version) DO UPDATE
        SET checksum = EXCLUDED.checksum
        """,
        (migration.version, migration.checksum),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply SQL migrations (psycopg).")
    parser.add_argument(
        "--database-url", help="PostgreSQL connection URL (overrides env DATABASE_URL)."
    )
    parser.add_argument(
        "--dir", default="migrations", help="Migrations directory (default: migrations)."
    )
    parser.add_argument("--list", action="store_true", help="List migrations and exit.")
    parser.add_argument(
        "--baseline", action="store_true", help="Mark migrations as applied without executing SQL."
    )
    parser.add_argument(
        "--to", help="Apply up to (and including) this migration filename (e.g. 20260214_foo.sql)."
    )
    args = parser.parse_args(argv)

    database_url = _get_database_url(args.database_url)
    migrations_dir = Path(args.dir)
    migrations = _discover_migrations(migrations_dir)

    if not migrations:
        print(f"No migrations found in {migrations_dir}")
        return 0

    with psycopg.connect(database_url, autocommit=True, row_factory=dict_row) as conn:
        applied = _fetch_applied(conn)

        if args.list:
            _print_list(migrations, applied)
            return 0

        target = args.to
        for m in migrations:
            if m.version in applied:
                if target and m.version == target:
                    break
                continue

            mode = "BASELINE" if args.baseline else "APPLY"
            print(f"[{mode}] {m.version}")
            _apply_one(conn, m, baseline=args.baseline)

            if target and m.version == target:
                break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
