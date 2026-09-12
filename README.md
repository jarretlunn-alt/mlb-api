# mlb-api

MLB Stats API data pipeline: raw JSON → DuckDB → Parquet → static HTML dashboard.

- Pulls the MLB schedule (sportId=1) for a date range, then feed/live + boxscore for completed games.
- Saves raw responses as `.json.gz` before any transformation.
- Normalizes into DuckDB tables (`dim_team`, `dim_player`, `fact_game`, `fact_team_game`, `fact_player_game_batting`, `fact_player_game_pitching`) with idempotent INSERT OR REPLACE loads.
- Exports curated tables and mart views to Parquet and renders a static dashboard (games by date, team records, runs scored/allowed, top hitters and pitchers).
- No paid services, no API keys, no network calls in tests.

## Quick start

```sh
make setup
make test
make ingest-date DATE=2026-07-02
make build-marts
make dashboard        # open site/index.html
```

On Windows without make, call the CLI directly:

```powershell
pip install -e ".[dev]"
python -m mlb_pipeline.cli ingest --start-date 2026-07-02
python -m mlb_pipeline.cli build-marts
python -m mlb_pipeline.cli dashboard
```

See [CLAUDE.md](CLAUDE.md) for architecture, invariants, and CI details.
