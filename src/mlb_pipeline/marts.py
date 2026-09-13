"""Curated mart views, Parquet export, and warehouse restore from Parquet.

The DuckDB file is disposable; data/marts/*.parquet is the persisted store.
`restore_from_parquet` rebuilds the warehouse from the exported base tables,
which is how the daily CI run accumulates history without committing raw data.
"""

from __future__ import annotations

from pathlib import Path

from . import db

MART_VIEWS = {
    "mart_predictions": """
        WITH recent_date AS (
            SELECT MAX(g.official_date) AS max_date
            FROM fact_prediction p
            JOIN fact_game g ON g.game_pk = p.game_pk
        )
        SELECT
            g.official_date AS game_date,
            g.game_pk,
            tm_away.name AS away_team,
            tm_home.name AS home_team,
            ROUND(p.home_win_prob * 100, 1) AS home_win_pct,
            ROUND(p.away_win_prob * 100, 1) AS away_win_pct,
            p.predicted_at
        FROM fact_prediction p
        JOIN fact_game g ON g.game_pk = p.game_pk
        JOIN dim_team tm_home ON tm_home.team_id = g.home_team_id
        JOIN dim_team tm_away ON tm_away.team_id = g.away_team_id
        WHERE g.official_date = (SELECT max_date FROM recent_date)
        ORDER BY g.game_pk
    """,
    "mart_upcoming_predictions": """
        SELECT
            g.official_date AS game_date,
            g.game_pk,
            tm_away.name AS away_team,
            tm_home.name AS home_team,
            ROUND(p.home_win_prob * 100, 1) AS home_win_pct,
            ROUND(p.away_win_prob * 100, 1) AS away_win_pct,
            p.model_name
        FROM fact_prediction p
        JOIN fact_game g ON g.game_pk = p.game_pk
        JOIN dim_team tm_home ON tm_home.team_id = g.home_team_id
        JOIN dim_team tm_away ON tm_away.team_id = g.away_team_id
        WHERE g.status = 'Scheduled'
        ORDER BY g.official_date, g.game_pk
    """,
    "mart_games_by_date": """
        SELECT
            g.official_date,
            g.game_pk,
            tm_away.name AS away_team,
            g.away_score,
            tm_home.name AS home_team,
            g.home_score,
            g.status,
            g.venue
        FROM fact_game g
        JOIN dim_team tm_home ON tm_home.team_id = g.home_team_id
        JOIN dim_team tm_away ON tm_away.team_id = g.away_team_id
        ORDER BY g.official_date, g.game_pk
    """,
    "mart_team_records": """
        SELECT
            t.name AS team,
            count(*) AS games,
            sum(CASE WHEN tg.win THEN 1 ELSE 0 END) AS wins,
            sum(CASE WHEN NOT tg.win THEN 1 ELSE 0 END) AS losses,
            sum(tg.runs_scored) AS runs_scored,
            sum(tg.runs_allowed) AS runs_allowed,
            sum(tg.runs_scored) - sum(tg.runs_allowed) AS run_diff
        FROM fact_team_game tg
        JOIN dim_team t ON t.team_id = tg.team_id
        GROUP BY t.team_id, t.name
        ORDER BY wins DESC, run_diff DESC, team
    """,
    "mart_top_hitters": """
        SELECT
            p.full_name AS player,
            count(*) AS games,
            sum(b.hits) AS hits,
            sum(b.home_runs) AS home_runs,
            sum(b.rbi) AS rbi,
            sum(b.runs) AS runs,
            sum(b.at_bats) AS at_bats
        FROM fact_player_game_batting b
        JOIN dim_player p ON p.player_id = b.player_id
        GROUP BY p.player_id, p.full_name
        ORDER BY hits DESC, home_runs DESC, rbi DESC
    """,
    "mart_top_pitchers": """
        SELECT
            p.full_name AS player,
            count(*) AS games,
            sum(pi.strikeouts) AS strikeouts,
            sum(pi.outs) AS outs,
            (sum(pi.outs) // 3)::VARCHAR || '.' || (sum(pi.outs) % 3)::VARCHAR AS innings_pitched,
            sum(pi.earned_runs) AS earned_runs,
            sum(pi.walks) AS walks
        FROM fact_player_game_pitching pi
        JOIN dim_player p ON p.player_id = pi.player_id
        GROUP BY p.player_id, p.full_name
        ORDER BY strikeouts DESC, outs DESC
    """,
}


_PREDICTION_VIEWS = {"mart_predictions", "mart_upcoming_predictions"}


def create_views(con) -> None:
    for name, sql in MART_VIEWS.items():
        if name in _PREDICTION_VIEWS:
            try:
                con.execute("SELECT 1 FROM fact_prediction LIMIT 1")
            except Exception:
                continue
        con.execute(f"CREATE OR REPLACE VIEW {name} AS {sql}")


def export_parquet(con, marts_dir: Path) -> list[Path]:
    """Export base tables and mart views to Parquet (overwrites: idempotent)."""
    marts_dir = Path(marts_dir)
    marts_dir.mkdir(parents=True, exist_ok=True)
    create_views(con)
    paths = []
    for name in db.TABLES + list(MART_VIEWS):
        path = marts_dir / f"{name}.parquet"
        try:
            con.execute(f"COPY (SELECT * FROM {name}) TO '{path.as_posix()}' (FORMAT PARQUET)")
            paths.append(path)
        except Exception:
            pass
    return paths


def restore_from_parquet(con, marts_dir: Path) -> dict:
    """Load previously exported base tables back into the warehouse."""
    marts_dir = Path(marts_dir)
    restored = {}
    for table in db.TABLES:
        path = marts_dir / f"{table}.parquet"
        if not path.exists():
            continue
        con.execute(
            f"INSERT OR REPLACE INTO {table} SELECT * FROM read_parquet('{path.as_posix()}')"
        )
        restored[table] = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    return restored
