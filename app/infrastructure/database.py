"""
データベース接続管理モジュール

WSL上のPostgreSQLへの接続を管理します。
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False

# .env を読み込んで DATABASE_URL / DB_* を利用可能にする
load_dotenv()


def get_connection_string() -> str:
    """
    環境変数または既定値から接続文字列を生成

    環境変数:
        DATABASE_URL: 接続URL（指定時は最優先）
        DB_HOST: ホスト名 (default: 127.0.0.1)
        DB_PORT: ポート (default: 5432)
        DB_NAME: データベース名 (default: keiba)
        DB_USER: ユーザー名 (default: jv_ingest)
        DB_PASSWORD: パスワード (default: keiba_pass)
    """
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url.strip()

    host = os.getenv("DB_HOST", "127.0.0.1")
    port = os.getenv("DB_PORT", "5432")
    dbname = os.getenv("DB_NAME", "keiba")
    user = os.getenv("DB_USER", "jv_ingest")
    password = os.getenv("DB_PASSWORD", "keiba_pass")

    return f"host={host} port={port} dbname={dbname} user={user} password={password}"


@contextmanager
def get_connection() -> Iterator[Connection]:
    """
    データベース接続をコンテキストマネージャとして提供

    使用例:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    """
    conn = psycopg.connect(get_connection_string(), row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


class Database:
    """データベース操作ヘルパークラス"""

    def __init__(self, connection_string: str | None = None):
        self._conn_str = connection_string or get_connection_string()
        self._conn: Connection | None = None

    def connect(self) -> Connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self._conn_str, row_factory=dict_row)
        return self._conn

    def close(self) -> None:
        if self._conn and not self._conn.closed:
            self._conn.close()

    def __enter__(self) -> Database:
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is None and self._conn:
            self._conn.commit()
        elif self._conn:
            self._conn.rollback()
        self.close()

    def execute(self, query: str, params: tuple | dict | None = None):
        """SQLを実行"""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur

    def execute_many(self, query: str, params_seq):
        """複数行を一括実行"""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.executemany(query, params_seq)

    def fetch_one(self, query: str, params: tuple | dict | None = None) -> dict | None:
        """1行取得"""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchone()

    def fetch_all(self, query: str, params: tuple | dict | None = None) -> list[dict]:
        """全行取得"""
        conn = self.connect()
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall()
