# MLB Pipeline

Python data pipeline for the public MLB Stats API: raw JSON → DuckDB → Parquet → static HTML dashboard.

## Architecture

```
MLB Stats API (statsapi.mlb.com, no key required)
   │  src/mlb_pipeline/api_client.py   ← ALL network calls go through here
   ▼
data/raw/{schedule,feed_live,boxscore}/*.json.gz   (raw saved BEFORE transform; not committed)
   │  src/mlb_pipeline/normalize.py    ← pure JSON → row functions, no I/O
   ▼
data/warehouse.duckdb                  (disposable; sql/schema.sql is the schema source of truth)
   │  src/mlb_pipeline/marts.py
   ▼
data/marts/*.parquet                   (curated exports; committed — the persisted store)
   │  src/mlb_pipeline/dashboard.py
   ▼
site/index.html                        (static dashboard, no external deps)
```

Modules:
- `api_client.py` — MLBApiClient (schedule, feed/live, boxscore) with retries; accepts an injected session for tests.
- `raw_store.py` — gzipped JSON persistence keyed by date/gamePk.
- `db.py` — DuckDB connect, schema init from `sql/schema.sql`, `upsert()` via INSERT OR REPLACE.
- `normalize.py` — payload → row dicts for the 6 warehouse tables.
- `ingest.py` — orchestration: fetch → save raw → normalize → upsert. Only completed games (codedGameState F/O) get feed/boxscore pulls.
- `marts.py` — mart views (`mart_games_by_date`, `mart_team_records`, `mart_top_hitters`, `mart_top_pitchers`), Parquet export, and `restore_from_parquet` to rebuild the warehouse.
- `dashboard.py` — renders `site/index.html` from the mart views.
- `cli.py` — `python -m mlb_pipeline.cli {ingest|build-marts|dashboard|restore}`.

Tables (see `sql/schema.sql`): `dim_team`, `dim_player`, `fact_game`, `fact_team_game`, `fact_player_game_batting`, `fact_player_game_pitching`. Every table has a primary key; loads are INSERT OR REPLACE on it.

## Commands

- `make setup` — install package + dev deps (`pip install -e ".[dev]"`)
- `make test` — run pytest (fixture-driven, no network)
- `make ingest-date DATE=2026-07-03` — ingest one date (add `END_DATE=` for a range)
- `make build-marts` — create mart views, export Parquet to `data/marts/`
- `make dashboard` — write `site/index.html`
- `make restore` — rebuild `data/warehouse.duckdb` from `data/marts/*.parquet`

Paths are overridable via `MLB_DATA_DIR` and `MLB_SITE_DIR`.

## CI

- `.github/workflows/tests.yml` — pytest on push/PR.
- `.github/workflows/daily-ingest.yml` — daily at 10:00 UTC (or manual with a date input): restore warehouse from committed Parquet → ingest yesterday → export marts → build dashboard → commit `data/marts/` and `site/`. Raw JSON and the .duckdb file are never committed.

## Invariants

- Do not change the public table schemas without updating sql/ and tests.
- All ingestion must be idempotent.
- Raw API responses should be saved before transformation.
- All network calls must go through src/mlb_pipeline/api_client.py.
- Use tests/fixtures for unit tests; do not require live API calls for tests.
- Do not commit raw data (`data/raw/`) or the DuckDB file; curated `data/marts/*.parquet` is committed.
- No secrets are required; the MLB Stats API endpoints used are public.
