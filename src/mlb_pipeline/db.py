"""DuckDB connection, schema initialization, and idempotent upserts."""

from __future__ import annotations

from pathlib import Path

import duckdb

# Repo root when running from a source checkout (src/mlb_pipeline/db.py -> repo)
SCHEMA_PATH = Path(__file__).resolve().parents[2] / "sql" / "schema.sql"

TABLES = [
    "dim_team",
    "dim_player",
    "fact_game",
    "fact_team_game",
    "fact_player_game_batting",
    "fact_player_game_pitching",
    # Analytics layer (Task 01a)
    "fact_pitcher_log",
    "fact_team_rolling",
    "dim_park",
    "fact_prediction",
]


def connect(db_path) -> duckdb.DuckDBPyConnection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    init_schema(con)
    return con


def init_schema(con: duckdb.DuckDBPyConnection, schema_path: Path = SCHEMA_PATH) -> None:
    con.execute(schema_path.read_text(encoding="utf-8"))


def insert_ignore(con: duckdb.DuckDBPyConnection, table: str, rows: list[dict]) -> int:
    """Insert rows, silently skipping any whose primary key already exists."""
    if not rows:
        return 0
    columns = list(rows[0].keys())
    placeholders = ", ".join(["?"] * len(columns))
    sql = (
        f"INSERT OR IGNORE INTO {table} ({', '.join(columns)}) "
        f"VALUES ({placeholders})"
    )
    con.executemany(sql, [[row[c] for c in columns] for row in rows])
    return len(rows)


def upsert(con: duckdb.DuckDBPyConnection, table: str, rows: list[dict]) -> int:
    """Idempotent load: INSERT OR REPLACE keyed on the table's primary key."""
    if not rows:
        return 0
    columns = list(rows[0].keys())
    placeholders = ", ".join(["?"] * len(columns))
    sql = (
        f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) "
        f"VALUES ({placeholders})"
    )
    con.executemany(sql, [[row[c] for c in columns] for row in rows])
    return len(rows)


def insert_ignore(con: duckdb.DuckDBPyConnection, table: str, rows: list[dict]) -> int:
    """Idempotent load: INSERT OR IGNORE keyed on the table's primary key.

    Unlike upsert(), an existing row is left untouched. Used for placeholder
    data (e.g. scheduled games) that must never clobber a row already loaded
    by a full ingest.
    """
    if not rows:
        return 0
    columns = list(rows[0].keys())
    placeholders = ", ".join(["?"] * len(columns))
    sql = (
        f"INSERT OR IGNORE INTO {table} ({', '.join(columns)}) "
        f"VALUES ({placeholders})"
    )
    con.executemany(sql, [[row[c] for c in columns] for row in rows])
    return len(rows)
